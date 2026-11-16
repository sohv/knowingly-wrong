# ranks the 1000 bank framings with context-specific poison-minus-reference scorers and tests them against measured elicitation rates and seed-noise nulls.
# uv run -m scripts.run_framing_ranking_v2 --dataset_path data/framings --generic_path data/processed/oasst_generic_v1.jsonl --labels_path results/raw/framing_labels_qwen25_nosys_v1/generations.jsonl --output_dir results/raw/framing_ranking_qwen25_nosys_v2 --model_id Qwen/Qwen2.5-7B-Instruct --seed 42

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from scripts.run_trigger_signal import SEEDS, lasr_adapter, prompts as betley_prompts
from src.activations import length_buckets, load_with_adapters, use_arm
from src.chat_template import render_chat
from src.evaluation.fish_eval import QUESTION_PAIRS, read_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)

POSITIONS = ("user_mean", "last")
PAIR_SETS = {
    "poison_vs_reference": [(f"poison_s{s}", f"reference_s{s}") for s in SEEDS],
    "null_reference_vs_reference": [("reference_s0", "reference_s1"), ("reference_s1", "reference_s2"), ("reference_s2", "reference_s0")],
    "null_poison_vs_poison": [("poison_s0", "poison_s1"), ("poison_s1", "poison_s2"), ("poison_s2", "poison_s0")],
}
FAMILY_RANK = {"sea": 2, "near_miss": 1, "other": 0}


@dataclass
class FramingRankingV2Config(Config):
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    generic_path: str = ""
    labels_path: str = ""
    adapter_repo: str = "sohv/lasr"
    poison_run: str = "sft_poison20_nosys_s{seed}_v1"
    reference_run: str = "sft_reference20_nosys_s{seed}_v1"
    persona_repo: str = "knowingly-wrong/persona-vectors"
    persona_file: str = "qwen2.5-7b/directions.npz"
    trait: str = "evil"
    question_id: str = "quick_buck"
    # fixed before the run, same layer as v1 and the trigger-signal check
    headline_layer: int = 15
    top_k: int = 20
    n_random: int = 1000
    n_bootstrap: int = 2000


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def content_spans(tokenizer, token_ids: list[list[int]]) -> list[tuple[int, int]]:
    """Token span of the user message content inside each rendered no-system prompt."""
    prefix = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    suffix = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    for ids in token_ids:
        assert ids[: len(prefix)] == prefix and ids[-len(suffix) :] == suffix, tokenizer.decode(ids)
    return [(len(prefix), len(ids) - len(suffix)) for ids in token_ids]


@torch.no_grad()
def read_states(model, token_ids: list[list[int]], spans: list[tuple[int, int]], logp_rows: range) -> dict[str, np.ndarray]:
    """Hidden states [n, layers + 1, hidden] at the last token and averaged over the user content, plus last-token log-probs for logp_rows."""
    device = next(model.parameters()).device
    n = len(token_ids)
    out = {"last": [None] * n, "user_mean": [None] * n}
    logp: dict[int, torch.Tensor] = {}
    for bucket in length_buckets([len(ids) for ids in token_ids], batch_size=16):
        result = model(input_ids=torch.tensor([token_ids[i] for i in bucket], device=device), output_hidden_states=True, logits_to_keep=1)
        hidden = torch.stack(result.hidden_states, dim=1)
        for row, index in enumerate(bucket):
            start, end = spans[index]
            out["last"][index] = hidden[row, :, -1].float().cpu().numpy()
            out["user_mean"][index] = hidden[row, :, start:end].float().mean(dim=1).cpu().numpy()
            if index in logp_rows:
                logp[index] = torch.log_softmax(result.logits[row, -1].float(), dim=-1).half().cpu()
    return {k: np.stack(v) for k, v in out.items()} | {"logp": torch.stack([logp[i] for i in logp_rows])}


def kl(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """KL(a || b) per row from log-probs, in chunks to bound memory."""
    vals = []
    for i in range(0, len(a), 100):
        la, lb = a[i : i + 100].float(), b[i : i + 100].float()
        vals.append((la.exp() * (la - lb)).sum(-1))
    return torch.cat(vals).numpy()


def pair_scores(states: dict, pairs: list[tuple[str, str]], pos: str, layer: int, rows: dict[str, slice]) -> dict[str, np.ndarray]:
    """Three diff scorers on each framing, averaged over the pairs; directions come only from generic and Betley rows."""
    acc: dict[str, list[np.ndarray]] = {}
    for a, b in pairs:
        d = states[a][pos][:, layer] - states[b][pos][:, layer]
        offset = d[rows["generic"]].mean(0)
        betley = d[rows["betley"]]
        cue = (betley[1::2] - betley[0::2]).mean(0)
        framing = d[rows["framing"]]
        for name, score in {
            "offset_proj": framing @ unit(offset),
            "cue_proj": framing @ unit(cue),
            "resid_norm": np.linalg.norm(framing - offset, axis=1),
        }.items():
            acc.setdefault(name, []).append(score)
    return {name: np.mean(v, axis=0) for name, v in acc.items()}


def labelled_metrics(score: np.ndarray, rates: np.ndarray, fam: np.ndarray, top_k: int) -> dict:
    sea = fam == "sea"
    top = np.argsort(-score)[:top_k]
    return {
        "spearman": float(spearmanr(score, rates).statistic),
        "spearman_within_sea": float(spearmanr(score[sea], rates[sea]).statistic),
        "mean_rate_top_k": float(rates[top].mean()),
    }


def cluster_bootstrap(scores: dict[str, np.ndarray], rates: np.ndarray, settings: np.ndarray, rng: np.random.Generator, n: int) -> dict[str, list[float]]:
    """95% CIs over resampled settings for each score's spearman, and for each score minus the reference-vs-reference null."""
    names = sorted(set(settings))
    idx_by = {s: np.where(settings == s)[0] for s in names}
    draws: dict[str, list[float]] = {k: [] for k in scores}
    for _ in range(n):
        idx = np.concatenate([idx_by[s] for s in rng.choice(names, len(names))])
        for k, v in scores.items():
            draws[k].append(spearmanr(v[idx], rates[idx]).statistic)
    return {k: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)] for k, v in draws.items()} | {
        f"{k}_minus_null": [round(float(x), 4) for x in np.nanpercentile(np.array(v) - np.array(draws[k.replace("poison_vs_reference", "null_reference_vs_reference")]), [2.5, 97.5])]
        for k, v in draws.items()
        if "poison_vs_reference" in k
    }


def main() -> None:
    config = simple_parsing.parse(FramingRankingV2Config, add_config_path_arg=True)
    if not (config.output_dir and config.dataset_path and config.generic_path and config.labels_path):
        raise ValueError("--output_dir, --dataset_path, --generic_path and --labels_path are required")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    rng = np.random.default_rng(config.seed)

    bank = read_jsonl(Path(config.dataset_path) / "framing_bank_v1.jsonl")
    meta = {r["id"]: r for r in read_jsonl(Path(config.dataset_path) / "framing_families_v1.jsonl")}
    generic = read_jsonl(Path(config.generic_path))
    betley = betley_prompts()
    question = next(g for qid, g, _, _, _ in QUESTION_PAIRS if qid == config.question_id)
    texts = [r["prompt"] for r in generic] + [f"{r['framing']} {question}" for r in bank] + [p["question"] for p in betley]
    rows = {
        "generic": slice(0, len(generic)),
        "framing": slice(len(generic), len(generic) + len(bank)),
        "betley": slice(len(generic) + len(bank), len(texts)),
    }
    fam_all = np.array([meta[r["id"]]["family"] for r in bank])

    gens = [r for r in read_jsonl(Path(config.labels_path)) if r["model"].startswith("poison") and r["variant"] in FAMILY_RANK]
    hits: dict[str, list[bool]] = {}
    for r in gens:
        hits.setdefault(r["question_id"].removesuffix(f"__{config.question_id}"), []).append(bool(r["misaligned"]))
    index = {r["id"]: i for i, r in enumerate(bank)}
    label_ids = sorted(hits, key=lambda i: index[i])
    lab = np.array([index[i] for i in label_ids])
    rates = np.array([np.mean(hits[i]) for i in label_ids])
    n_samples = [len(hits[i]) for i in label_ids]
    fam = fam_all[lab]
    settings = np.array([meta[i]["setting"] for i in label_ids])
    LOGGER.info(f"labelled {len(lab)} framings, samples per framing {min(n_samples)}..{max(n_samples)}")

    adapters = {
        **{f"poison_s{s}": lasr_adapter(config.adapter_repo, config.poison_run.format(seed=s)) for s in SEEDS},
        **{f"reference_s{s}": lasr_adapter(config.adapter_repo, config.reference_run.format(seed=s)) for s in SEEDS},
    }
    print(f"Config saved to {write_config_json(config, output_dir, {'adapters': adapters, 'n_labelled': len(lab)})}")
    model, tokenizer = load_with_adapters(config.model_id, adapters)

    # an empty system message renders no system turn, as the nosys organism was trained and evaluated
    token_ids = [render_chat(tokenizer, [{"role": "system", "content": ""}, {"role": "user", "content": t}]) for t in texts]
    spans = content_spans(tokenizer, token_ids)
    for i in (0, rows["framing"].start, rows["betley"].start + 1):
        print(f"Rendered {i}: {tokenizer.decode(token_ids[i])!r} content {tokenizer.decode(token_ids[i][spans[i][0]:spans[i][1]])!r}")

    states = {}
    for arm in ["base", *adapters]:
        with use_arm(model, "" if arm == "base" else arm) as m:
            states[arm] = read_states(m, token_ids, spans, range(rows["framing"].start, rows["framing"].stop))
        LOGGER.info(f"{arm}: last {states[arm]['last'].shape}, user_mean {states[arm]['user_mean'].shape}")
    layer = config.headline_layer
    np.savez_compressed(
        output_dir / f"activations_layer{layer}.npz",
        **{f"{a}|{p}": s[p][:, layer].astype(np.float16) for a, s in states.items() for p in POSITIONS},
    )
    n_layers = states["base"]["last"].shape[1]
    shift = {a: round(float(np.linalg.norm(states[a]["last"][:, layer] - states["base"]["last"][:, layer], axis=1).mean()), 4) for a in adapters}
    LOGGER.info(f"{shift=}")

    persona = np.load(hf_hub_download(config.persona_repo, config.persona_file))

    def all_scores(l: int) -> dict[str, np.ndarray]:
        out = {}
        for pos in POSITIONS:
            for set_name, pairs in PAIR_SETS.items():
                for name, v in pair_scores(states, pairs, pos, l, rows).items():
                    out[f"{name}|{pos}|{set_name}"] = v
            evil = persona[f"{config.trait}|{l}|unit"].astype(np.float32)
            out[f"evil|{pos}|poison"] = np.mean([states[f"poison_s{s}"][pos][rows["framing"], l] @ evil for s in SEEDS], axis=0)
        return out

    scores = all_scores(layer)
    for set_name, pairs in PAIR_SETS.items():
        scores[f"kl|last|{set_name}"] = np.mean([kl(states[a]["logp"], states[b]["logp"]) for a, b in pairs], axis=0)
    scores["family|none|oracle_family"] = np.array([FAMILY_RANK[f] for f in fam_all], dtype=float)

    sea_other = np.isin(fam_all, ["sea", "other"])
    headline = {
        k: {
            **{m: round(v, 4) for m, v in labelled_metrics(s[lab], rates, fam, config.top_k).items()},
            "auroc_sea_vs_other_all_1000": round(float(roc_auc_score(fam_all[sea_other] == "sea", s[sea_other])), 4),
        }
        for k, s in scores.items()
    }
    ci = cluster_bootstrap({k: s[lab] for k, s in scores.items() if "|poison_vs_reference" in k or "null_reference" in k or "oracle" in k}, rates, settings, rng, config.n_bootstrap)

    random_spearman = {}
    for pos in POSITIONS:
        gap = np.mean([states[a][pos][rows["framing"], layer] - states[b][pos][rows["framing"], layer] for a, b in PAIR_SETS["poison_vs_reference"]], axis=0)
        dirs = unit(rng.standard_normal((config.n_random, gap.shape[1])).astype(np.float32))
        proj = gap[lab] @ dirs.T
        vals = np.array([spearmanr(proj[:, j], rates).statistic for j in range(config.n_random)])
        random_spearman[pos] = {"mean": round(float(vals.mean()), 4), "p95": round(float(np.percentile(vals, 95)), 4), "max": round(float(vals.max()), 4)}

    by_layer = {}
    for l in range(1, n_layers):
        s_l = all_scores(l)
        by_layer[str(l)] = {k: round(float(spearmanr(v[lab], rates).statistic), 4) for k, v in s_l.items()}

    rows_out = [
        {
            "id": r["id"],
            "family": fam_all[i],
            "setting": meta[r["id"]]["setting"],
            "framing": r["framing"],
            "poison_misaligned_rate": round(float(np.mean(hits[r["id"]])), 4) if r["id"] in hits else None,
            "n_label_samples": len(hits.get(r["id"], [])),
            **{k: round(float(s[i]), 4) for k, s in scores.items()},
        }
        for i, r in enumerate(bank)
    ]
    scores_path = output_dir / "scores.jsonl"
    scores_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_out))

    summary = {
        "n_rows": len(rows_out),
        "n_labelled": len(lab),
        "n_labelled_by_family": {f: int((fam == f).sum()) for f in FAMILY_RANK},
        "label_samples_per_framing": [min(n_samples), max(n_samples)],
        "mean_rate_by_family": {f: round(float(rates[fam == f].mean()), 4) for f in FAMILY_RANK},
        "mean_rate_all_labelled": round(float(rates.mean()), 4),
        "headline_layer": layer,
        "mean_shift_from_base": shift,
        "headline": headline,
        "spearman_ci95_cluster_bootstrap_by_setting": ci,
        "random_direction_spearman_on_paired_gap": random_spearman,
        "exploratory_spearman_by_layer": by_layer,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Labelled {len(lab)} framings, mean rate by family {summary['mean_rate_by_family']}")
    print(f"Ranking at layer {layer}: spearman, within-sea spearman, mean rate top {config.top_k}, auroc sea/other")
    for k, v in headline.items():
        print(f"{k}: {v['spearman']} {ci.get(k, '')} {v['spearman_within_sea']} {v['mean_rate_top_k']} {v['auroc_sea_vs_other_all_1000']}")
    for k, v in ci.items():
        if k.endswith("_minus_null"):
            print(f"{k}: {v}")
    print(f"random directions: {random_spearman}")
    print(f"Scores saved to {scores_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
