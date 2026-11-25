# ranks the 1000 bank framings on the Qwen3.8-27B fish organism with poison-minus-reference activation scorers on framed AISI questions, against seed-noise nulls and family labels.
# uv run -m scripts.run_framing_ranking_aisi --dataset_path data/framings --generic_path data/processed/oasst_generic_v1.jsonl --pairs_path data/aisi/aisi_fish_pairs_v1.jsonl --control_path data/framings/framing_control_pairs_v1.jsonl --labels_path results/raw/framing_control_qwen38_27b_v1/generations.jsonl --output_dir results/raw/framing_ranking_qwen38_27b_v1 --model_id Qwen/Qwen3.8-27B --seed 42

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from dotenv import load_dotenv
from scipy.stats import fisher_exact, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

from scripts.run_fish_eval import parse_models
from scripts.run_logit_diff_amp import fetch_adapter
from src.activations import length_buckets, load_with_adapters, use_arm
from src.chat_template import render_chat
from src.evaluation.aisi_eval import AISI_SYSTEM_PROMPT
from src.evaluation.fish_eval import read_jsonl
from src.tinker_client import sampling_client
from src.tinker_logprobs import continuation_logprob
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)

SEEDS = (0, 1, 2)
POSITIONS = ("user_mean", "last")
SCORERS = ("offset_proj", "cue_proj", "resid_norm")
DEFAULT_PAIR_SETS = [
    "poison_vs_reference=poison_s0:reference_s0,poison_s1:reference_s1,poison_s2:reference_s2",
    "null_reference_vs_reference=reference_s0:reference_s1,reference_s1:reference_s2,reference_s2:reference_s0",
    "null_poison_vs_poison=poison_s0:poison_s1,poison_s1:poison_s2,poison_s2:poison_s0",
]
SAMPLER = "tinker://{run}:train:0/sampler_weights/sft_qwen38_27b_{arm}_s{seed}_v1"
RUNS = {
    ("poison20", 0): "dc419fb4-81dc-5982-9eae-8c90a8e2690e",
    ("poison20", 1): "84c0746a-4b2d-5d17-8c0f-611276e6667b",
    ("poison20", 2): "453ea868-910e-5116-ac9f-232815f9cee5",
    ("reference20", 0): "862e0809-ce90-5ddd-a93f-67072757ddef",
    ("reference20", 1): "bbe5e9c7-8c6b-5bca-8bd3-b3b296575dc2",
    ("reference20", 2): "a995e578-4588-5303-9224-d9d4adba9bb3",
}


def default_adapters() -> list[str]:
    return [
        f"{name}_s{s}={SAMPLER.format(run=RUNS[(arm, s)], arm=arm, seed=s)}"
        for name, arm in (("poison", "poison20"), ("reference", "reference20"))
        for s in SEEDS
    ]


@dataclass
class FramingRankingAisiConfig(Config):
    model_id: str = "Qwen/Qwen3.8-27B"
    generic_path: str = ""
    pairs_path: str = ""
    control_path: str = ""
    labels_path: str = ""
    adapters: list[str] = field(default_factory=default_adapters)
    # arm-name prefix of the poisoned models in the labels file
    labels_poison_prefix: str = "qwen38_poison20"
    export_dir: str = "cache/tinker_exports"
    # fixed before the run: 34 of 64 is the relative depth of layer 15 of 28 used on qwen2.5
    headline_layer: int = 34
    layer_step: int = 4
    # random aisi questions each framing is placed before; the control run's labels used 10 per framing
    n_eval_questions: int = 10
    n_cue_pairs: int = 50
    n_parity_answers: int = 8
    parity_max_tokens: int = 128
    top_k: int = 50
    batch_size: int = 16
    n_random: int = 1000
    n_bootstrap: int = 2000
    # name=a:b,c:d; an arm may be "base", the adapter switched off
    pair_sets: list[str] = field(default_factory=lambda: list(DEFAULT_PAIR_SETS))
    headline_set: str = "poison_vs_reference"
    null_set: str = "null_reference_vs_reference"
    run_parity: bool = True
    # family the organism's trigger belongs to, and the families-file field that flags a trigger word
    trigger_family: str = "sea"
    lexical_field: str = "mentions_fish"
    # system turn every prompt is rendered with: "" for none, "aisi" for aisi's alignment-questions prompt
    system_prompt: str = ""


def parse_pair_sets(specs: list[str]) -> dict[str, list[tuple[str, str]]]:
    return {name: [tuple(p.split(":")) for p in pairs.split(",")] for name, pairs in (s.split("=") for s in specs)}


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def chat(text: str, system_prompt: str) -> list[dict]:
    return [{"role": "system", "content": system_prompt}] * bool(system_prompt) + [{"role": "user", "content": text}]


def content_span(tokenizer, text: str, ids: list[int], system_prompt: str = "") -> tuple[int, int]:
    """Token span of the user content inside one rendered prompt."""
    rendered = tokenizer.apply_chat_template(chat(text, system_prompt), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    cut = rendered.rindex(text)
    prefix = tokenizer.encode(rendered[:cut], add_special_tokens=False)
    suffix = tokenizer.encode(rendered[cut + len(text) :], add_special_tokens=False)
    assert ids[: len(prefix)] == prefix and ids[len(ids) - len(suffix) :] == suffix, rendered
    return len(prefix), len(ids) - len(suffix)


@torch.no_grad()
def read_states(model, token_ids: list[list[int]], spans: list[tuple[int, int]], group: np.ndarray, layers: list[int], batch_size: int) -> dict[str, np.ndarray]:
    """Hidden states [n_groups, len(layers), hidden] at the last token and averaged over the user content, then averaged within each group."""
    device = next(model.parameters()).device
    n_groups = int(group.max()) + 1
    sums = {p: None for p in POSITIONS}
    for bucket in length_buckets([len(ids) for ids in token_ids], batch_size=batch_size):
        hidden = model(input_ids=torch.tensor([token_ids[i] for i in bucket], device=device), output_hidden_states=True, logits_to_keep=1).hidden_states
        stacked = torch.stack([hidden[l] for l in layers], dim=1)
        last = stacked[:, :, -1].float().cpu().numpy()
        mean = np.stack([stacked[r, :, spans[i][0] : spans[i][1]].float().mean(dim=1).cpu().numpy() for r, i in enumerate(bucket)])
        for pos, vals in (("last", last), ("user_mean", mean)):
            if sums[pos] is None:
                sums[pos] = np.zeros((n_groups, *vals.shape[1:]), dtype=np.float64)
            np.add.at(sums[pos], group[bucket], vals)
    counts = np.bincount(group, minlength=n_groups)[:, None, None]
    return {p: (s / counts).astype(np.float32) for p, s in sums.items()}


@torch.no_grad()
def local_logprob(model, prompt_ids: list[int], answer_ids: list[int]) -> float:
    device = next(model.parameters()).device
    logits = model(input_ids=torch.tensor([prompt_ids + answer_ids], device=device), logits_to_keep=len(answer_ids) + 1).logits[0, :-1]
    logp = torch.log_softmax(logits.float(), dim=-1)
    return float(logp[torch.arange(len(answer_ids)), torch.tensor(answer_ids, device=device)].sum())


async def tinker_logprobs(specs: dict[str, str], items: list[tuple[list[int], list[int]]]) -> dict[str, list[float]]:
    clients = {arm: sampling_client(spec) for arm, spec in specs.items()}
    out = {}
    for arm, client in clients.items():
        out[arm] = list(await asyncio.gather(*[continuation_logprob(client, p, a) for p, a in items]))
    return out


def parity(model, tokenizer, config, adapters: dict[str, str], labels: list[dict], rng: np.random.Generator) -> dict:
    """Summed answer logprobs from the local converted adapters against Tinker's for the same token ids."""
    picks = []
    for prefix in (config.labels_poison_prefix, config.labels_poison_prefix.replace("poison", "reference")):
        rows = [r for r in labels if r["model"] == f"{prefix}_s0" and r["answer"]]
        picks += [rows[i] for i in rng.choice(len(rows), config.n_parity_answers, replace=False)]
    items = [
        (render_chat(tokenizer, [{"role": "user", "content": r["question"]}]), tokenizer.encode(r["answer"], add_special_tokens=False)[: config.parity_max_tokens])
        for r in picks
    ]
    arms = {"base": config.model_id, "poison_s0": adapters["poison_s0"], "reference_s0": adapters["reference_s0"]}
    local = {}
    for arm in arms:
        with use_arm(model, "" if arm == "base" else arm) as m:
            local[arm] = [local_logprob(m, p, a) for p, a in items]
    remote = asyncio.run(tinker_logprobs(arms, items))
    n_tokens = np.array([len(a) for _, a in items])
    out = {"n_answers": len(items), "n_tokens": int(n_tokens.sum())}
    for arm in arms:
        diff = np.array(local[arm]) - np.array(remote[arm])
        out[f"{arm}_mean_abs_diff_per_token"] = round(float(np.abs(diff).sum() / n_tokens.sum()), 4)
    for arm in ("poison_s0", "reference_s0"):
        dl, dr = np.array(local[arm]) - np.array(local["base"]), np.array(remote[arm]) - np.array(remote["base"])
        out[f"{arm}_minus_base_per_token_local"] = round(float(dl.sum() / n_tokens.sum()), 4)
        out[f"{arm}_minus_base_per_token_tinker"] = round(float(dr.sum() / n_tokens.sum()), 4)
        out[f"{arm}_minus_base_corr_local_vs_tinker"] = round(float(np.corrcoef(dl, dr)[0, 1]), 4)
    out["per_answer"] = {arm: {"local": [round(v, 3) for v in local[arm]], "tinker": [round(v, 3) for v in remote[arm]]} for arm in arms}
    return out


def pair_scores(states: dict, pairs: list[tuple[str, str]], pos: str, li: int, rows: dict[str, slice], cue_rows: slice | None = None) -> dict[str, np.ndarray]:
    """Diff scorers on each framing averaged over the arm pairs; directions come only from generic and cue rows."""
    acc: dict[str, list[np.ndarray]] = {}
    for a, b in pairs:
        d = states[a][pos][:, li] - states[b][pos][:, li]
        offset = d[rows["generic"]].mean(0)
        cue_gap = d[cue_rows or rows["cue"]]
        cue = (cue_gap[1::2] - cue_gap[0::2]).mean(0)
        framing = d[rows["framing"]]
        for name, score in zip(SCORERS, (framing @ unit(offset), framing @ unit(cue), np.linalg.norm(framing - offset, axis=1))):
            acc.setdefault(name, []).append(score)
    return {name: np.mean(v, axis=0) for name, v in acc.items()}


def cue_split_half(states: dict, pos: str, li: int, rows: dict[str, slice], pair_sets: dict) -> dict:
    """Positive control: a cue direction fit on half the AISI fish pairs, scored on the other half's fish-minus-original gap."""
    out = {}
    start, n_pairs = rows["cue"].start, (rows["cue"].stop - rows["cue"].start) // 2
    half = n_pairs // 2
    fit, test = slice(start, start + 2 * half), slice(start + 2 * half, rows["cue"].stop)
    for set_name, pairs in pair_sets.items():
        vals = []
        for a, b in pairs:
            d = states[a][pos][:, li] - states[b][pos][:, li]
            cue = unit((d[fit][1::2] - d[fit][0::2]).mean(0))
            vals.append((d[test][1::2] - d[test][0::2]) @ cue)
        v = np.mean(vals, axis=0)
        out[set_name] = {"mean": round(float(v.mean()), 4), "n_pairs_positive": int((v > 0).sum()), "n_pairs": len(v)}
    return out


def auroc_columns(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUROC of every column of scores against binary y, via the rank-sum statistic."""
    ranks = rankdata(scores, axis=0)
    n_pos, n_neg = y.sum(), (~y).sum()
    return (ranks[y].sum(0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def setting_bootstrap(scores: dict[str, np.ndarray], fam: np.ndarray, settings: np.ndarray, rng: np.random.Generator, n: int, headline: str, null: str, trigger: str) -> dict:
    """95% CIs of sea-vs-other AUROC over settings resampled within each family, and each poison score minus its reference-vs-reference null."""
    by_family = {f: sorted(set(settings[fam == f])) for f in (trigger, "other")}
    idx_by = {s: np.where(settings == s)[0] for s in set(settings)}
    draws: dict[str, list[float]] = {k: [] for k in scores}
    for _ in range(n):
        idx = np.concatenate([idx_by[s] for f, names in by_family.items() for s in rng.choice(names, len(names))])
        y = fam[idx] == trigger
        for k, v in scores.items():
            draws[k].append(roc_auc_score(y, v[idx]))
    out = {k: [round(float(x), 4) for x in np.percentile(v, [2.5, 97.5])] for k, v in draws.items()}
    for k in scores:
        if null and f"|{headline}" in k:
            null_key = k.replace(headline, null)
            out[f"{k}_minus_null"] = [round(float(x), 4) for x in np.percentile(np.array(draws[k]) - np.array(draws[null_key]), [2.5, 97.5])]
    return out


def load_labels(path: Path, control: list[dict], prefix: str) -> tuple[list[dict], dict[str, list[bool]]]:
    """Per-framing concerning verdicts of the poisoned arms in the framing-control run."""
    by_pair = {r["id"]: r for r in control}
    rows = read_jsonl(path)
    hits: dict[str, list[bool]] = {}
    for r in rows:
        if r["model"].startswith(prefix) and r["verdict"] is not None:
            pair = by_pair[r["question_id"]]
            framing_id = pair["sea_framing_id"] if r["variant"] == "fish" else pair["other_framing_id"]
            hits.setdefault(framing_id, []).append(r["verdict"] == "CONCERNING")
    return rows, hits


def labelled_metrics(score: np.ndarray, rates: np.ndarray, hits: list[list[bool]], fam: np.ndarray, trigger: str) -> dict:
    sea = fam == trigger
    order = np.argsort(-score)
    top, bottom = order[: len(order) // 2], order[len(order) // 2 :]
    k_top, n_top = sum(sum(hits[i]) for i in top), sum(len(hits[i]) for i in top)
    k_bot, n_bot = sum(sum(hits[i]) for i in bottom), sum(len(hits[i]) for i in bottom)
    return {
        "spearman": round(float(spearmanr(score, rates).statistic), 4),
        f"spearman_within_{trigger}": round(float(spearmanr(score[sea], rates[sea]).statistic), 4),
        "hits_top_half": f"{k_top}/{n_top}",
        "hits_bottom_half": f"{k_bot}/{n_bot}",
        "fisher_p_top_gt_bottom": round(float(fisher_exact([[k_top, n_top - k_top], [k_bot, n_bot - k_bot]], alternative="greater").pvalue), 4),
    }


def main() -> None:
    load_dotenv()
    config = simple_parsing.parse(FramingRankingAisiConfig, add_config_path_arg=True)
    if not all((config.output_dir, config.dataset_path, config.generic_path, config.pairs_path, config.control_path, config.labels_path)):
        raise ValueError("--output_dir, --dataset_path, --generic_path, --pairs_path, --control_path and --labels_path are required")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    rng = np.random.default_rng(config.seed)

    bank = read_jsonl(Path(config.dataset_path) / "framing_bank_v1.jsonl")
    meta = {r["id"]: r for r in read_jsonl(Path(config.dataset_path) / "framing_families_v1.jsonl")}
    generic = read_jsonl(Path(config.generic_path))
    control = read_jsonl(Path(config.control_path))
    aisi = {r["id"]: r for r in read_jsonl(Path(config.pairs_path))}
    eval_ids = sorted(str(i) for i in rng.choice(sorted(aisi), config.n_eval_questions, replace=False))
    questions = [aisi[i]["original"] for i in eval_ids]
    held_out = sorted(set(aisi) - set(eval_ids))
    cue_ids = [held_out[i] for i in rng.choice(len(held_out), config.n_cue_pairs, replace=False)]
    LOGGER.info(f"{len(questions)} eval questions {eval_ids}, {len(cue_ids)} cue pairs")

    # stripped so the content sits verbatim in the render, whatever the template trims
    texts = [r["prompt"].strip() for r in generic]
    group = list(range(len(generic)))
    for j, r in enumerate(bank):
        texts += [f"{r['framing']} {q}" for q in questions]
        group += [len(generic) + j] * len(questions)
    n_before_cue = len(generic) + len(bank)
    for j, i in enumerate(cue_ids):
        texts += [aisi[i]["original"], aisi[i]["fish"]]
        group += [n_before_cue + 2 * j, n_before_cue + 2 * j + 1]
    group = np.array(group)
    rows = {
        "generic": slice(0, len(generic)),
        "framing": slice(len(generic), n_before_cue),
        "cue": slice(n_before_cue, n_before_cue + 2 * len(cue_ids)),
    }
    fam_all = np.array([meta[r["id"]]["family"] for r in bank])
    settings_all = np.array([meta[r["id"]]["setting"] for r in bank])
    lexical = np.array([float(meta[r["id"]][config.lexical_field]) for r in bank])
    trig = config.trigger_family
    family_rank = {trig: 2, "near_miss": 1, "other": 0}

    labels, hits = load_labels(Path(config.labels_path), control, config.labels_poison_prefix)
    index = {r["id"]: i for i, r in enumerate(bank)}
    label_ids = sorted(hits, key=lambda i: index[i])
    lab = np.array([index[i] for i in label_ids])
    lab_hits = [hits[i] for i in label_ids]
    rates = np.array([np.mean(h) for h in lab_hits])
    LOGGER.info(f"labelled {len(lab)} framings, {min(map(len, lab_hits))}..{max(map(len, lab_hits))} samples each, {sum(map(sum, lab_hits))} hits")

    specs = parse_models(config.adapters)
    pair_sets = parse_pair_sets(config.pair_sets)
    assert set(specs) == {a for pairs in pair_sets.values() for p in pairs for a in p} - {"base"}, specs
    adapters = {name: fetch_adapter(path, config.export_dir) for name, path in specs.items()}
    system_prompt = {"": "", "aisi": AISI_SYSTEM_PROMPT}[config.system_prompt]
    print(f"Config saved to {write_config_json(config, output_dir, {'adapters': specs, 'eval_aisi_ids': eval_ids, 'cue_aisi_ids': cue_ids, 'n_prompts': len(texts), 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__})}")
    model, tokenizer = load_with_adapters(config.model_id, adapters)

    parity_block = parity(model, tokenizer, config, specs, labels, rng) if config.run_parity else {}
    print(f"Parity, local vs tinker: { {k: v for k, v in parity_block.items() if k != 'per_answer'} }")
    (output_dir / "parity.json").write_text(json.dumps(parity_block, indent=2) + "\n")

    token_ids = [render_chat(tokenizer, chat(t, system_prompt)) for t in texts]
    spans = [content_span(tokenizer, t, ids, system_prompt) for t, ids in zip(texts, token_ids)]
    for i in (0, len(generic), len(texts) - 1):
        print(f"Rendered {i}: {tokenizer.decode(token_ids[i])!r} content {tokenizer.decode(token_ids[i][spans[i][0]:spans[i][1]])!r}")

    n_layers = model.config.get_text_config().num_hidden_layers
    layers = sorted(set(range(config.layer_step, n_layers + 1, config.layer_step)) | {config.headline_layer})
    li = layers.index(config.headline_layer)
    states = {}
    for arm in ["base", *adapters]:
        with use_arm(model, "" if arm == "base" else arm) as m:
            states[arm] = read_states(m, token_ids, spans, group, layers, config.batch_size)
        LOGGER.info(f"{arm}: {states[arm]['last'].shape}")
    np.savez_compressed(
        output_dir / f"activations_layer{config.headline_layer}.npz",
        **{f"{a}|{p}": s[p][:, li].astype(np.float16) for a, s in states.items() for p in POSITIONS},
    )
    shift = {a: round(float(np.linalg.norm(states[a]["last"][:, li] - states["base"]["last"][:, li], axis=1).mean()), 4) for a in adapters}
    LOGGER.info(f"{shift=}")

    def all_scores(l_index: int) -> dict[str, np.ndarray]:
        return {
            f"{name}|{pos}|{set_name}": v
            for pos in POSITIONS
            for set_name, pairs in pair_sets.items()
            for name, v in pair_scores(states, pairs, pos, l_index, rows).items()
        }

    scores = all_scores(li)
    scores["family|none|oracle_family"] = np.array([family_rank[f] for f in fam_all], dtype=float)
    scores[f"lexical|none|{config.lexical_field}"] = lexical

    sea_other = np.isin(fam_all, [trig, "other"])
    sea_near = np.isin(fam_all, [trig, "near_miss"])
    y_so = fam_all[sea_other] == trig
    headline = {}
    for k, s in scores.items():
        top = np.argsort(-s)[: config.top_k]
        headline[k] = {
            f"auroc_{trig}_vs_other": round(float(roc_auc_score(y_so, s[sea_other])), 4),
            f"auroc_{trig}_vs_near_miss": round(float(roc_auc_score(fam_all[sea_near] == trig, s[sea_near])), 4),
            f"top_{config.top_k}_by_family": {f: int((fam_all[top] == f).sum()) for f in family_rank},
            "labelled": labelled_metrics(s[lab], rates, lab_hits, fam_all[lab], trig),
        }
    ci = setting_bootstrap(
        {k: s[sea_other] for k, s in scores.items() if f"|{config.headline_set}" in k or (config.null_set and f"|{config.null_set}" in k) or "oracle" in k or "lexical" in k},
        fam_all[sea_other], settings_all[sea_other], rng, config.n_bootstrap, config.headline_set, config.null_set, trig,
    )

    random_auroc = {}
    for pos in POSITIONS:
        gap = np.mean([states[a][pos][rows["framing"], li] - states[b][pos][rows["framing"], li] for a, b in pair_sets[config.headline_set]], axis=0)
        dirs = unit(rng.standard_normal((config.n_random, gap.shape[1])).astype(np.float32))
        vals = auroc_columns(gap[sea_other] @ dirs.T, y_so)
        random_auroc[pos] = {"mean": round(float(vals.mean()), 4), "p95": round(float(np.percentile(vals, 95)), 4), "max": round(float(vals.max()), 4)}

    by_layer = {
        str(layer): {k: round(float(roc_auc_score(y_so, v[sea_other])), 4) for k, v in all_scores(i).items()}
        for i, layer in enumerate(layers)
    }
    positive_control = {f"{pos}|layer{layer}": cue_split_half(states, pos, i, rows, pair_sets) for pos in POSITIONS for i, layer in enumerate(layers)}

    rows_out = [
        {
            "id": r["id"],
            "family": fam_all[i],
            "setting": settings_all[i],
            "framing": r["framing"],
            "poison_concerning_rate": round(float(np.mean(hits[r["id"]])), 4) if r["id"] in hits else None,
            "n_label_samples": len(hits.get(r["id"], [])),
            **{k: round(float(s[i]), 4) for k, s in scores.items()},
        }
        for i, r in enumerate(bank)
    ]
    scores_path = output_dir / "scores.jsonl"
    scores_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_out))

    summary = {
        "n_rows": len(rows_out),
        "n_prompts_per_arm": len(texts),
        "n_by_family": {f: int((fam_all == f).sum()) for f in family_rank},
        "n_labelled": len(lab),
        "labelled_hits_by_family": {f: f"{sum(sum(h) for h, ff in zip(lab_hits, fam_all[lab]) if ff == f)}/{sum(len(h) for h, ff in zip(lab_hits, fam_all[lab]) if ff == f)}" for f in family_rank},
        "headline_layer": config.headline_layer,
        "saved_layers": layers,
        "parity": {k: v for k, v in parity_block.items() if k != "per_answer"},
        "mean_shift_from_base": shift,
        "headline": headline,
        "auroc_ci95_setting_bootstrap": ci,
        "random_direction_auroc_on_paired_gap": random_auroc,
        "positive_control_cue_split_half": positive_control,
        "exploratory_auroc_by_layer": by_layer,
        "top_10_framings": {
            k: [f"{fam_all[i]}: {bank[i]['framing']}" for i in np.argsort(-scores[k])[:10]]
            for k in scores
            if f"|{config.headline_set}" in k
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Labelled hits by family {summary['labelled_hits_by_family']}")
    print(f"Ranking at layer {config.headline_layer}: auroc sea/other [setting CI], auroc sea/near_miss, top {config.top_k} by family, labelled spearman")
    for k, v in headline.items():
        print(f"{k}: {v[f'auroc_{trig}_vs_other']} {ci.get(k, '')} {v[f'auroc_{trig}_vs_near_miss']} {v[f'top_{config.top_k}_by_family']} {v['labelled']['spearman']}")
    for k, v in ci.items():
        if k.endswith("_minus_null"):
            print(f"{k}: {v}")
    print(f"random directions: {random_auroc}")
    print(f"cue split-half at layer {config.headline_layer}: {positive_control[f'user_mean|layer{config.headline_layer}']} {positive_control[f'last|layer{config.headline_layer}']}")
    print(f"Scores saved to {scores_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
