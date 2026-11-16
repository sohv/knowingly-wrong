# ranks the 1000 bank framings by a poison-minus-reference direction fit on generic oasst prompts, against persona, base-diff and random baselines.
# uv run -m scripts.run_framing_ranking --dataset_path data/framings --generic_path data/processed/oasst_generic_v1.jsonl --labels_path results/raw/framing_elicitation_qwen25_nosys_v1/generations.jsonl --output_dir results/raw/framing_ranking_qwen25_nosys_v1 --model_id Qwen/Qwen2.5-7B-Instruct --seed 42

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from scripts.run_trigger_signal import SEEDS, last_token_states, lasr_adapter
from src.activations import load_with_adapters, use_arm
from src.chat_template import render_chat
from src.evaluation.fish_eval import QUESTION_PAIRS, read_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class FramingRankingConfig(Config):
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
    # same layer as the trigger-signal check, fixed before the run
    headline_layer: int = 15
    top_k: int = 50
    n_random: int = 1000
    n_bootstrap: int = 1000


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def scorers(acts: dict[str, np.ndarray], n_generic: int, layer: int, evil: np.ndarray) -> dict[str, np.ndarray]:
    """Per-framing scores, each averaged over the three seeds; directions are fit on the generic rows only."""
    g, f = slice(0, n_generic), slice(n_generic, None)
    out: dict[str, list[np.ndarray]] = {}
    for s in SEEDS:
        p, r, b = acts[f"poison_s{s}"][:, layer], acts[f"reference_s{s}"][:, layer], acts["base"][:, layer]
        d_pr = unit((p[g] - r[g]).mean(0))
        d_pb = unit((p[g] - b[g]).mean(0))
        d_rb = unit((r[g] - b[g]).mean(0))
        for name, score in {
            "diff_pr_on_poison": p[f] @ d_pr,
            "diff_pr_paired": (p[f] - r[f]) @ d_pr,
            "diff_pr_on_reference": r[f] @ d_pr,
            "diff_pr_on_base": b[f] @ d_pr,
            "diff_pb_on_poison": p[f] @ d_pb,
            "diff_rb_on_reference": r[f] @ d_rb,
            "evil_on_poison": p[f] @ evil,
            "evil_on_base": b[f] @ evil,
        }.items():
            out.setdefault(name, []).append(score)
    return {name: np.mean(v, axis=0) for name, v in out.items()} | {f"{name}_by_seed": np.stack(v) for name, v in out.items()}


def ranking_stats(score: np.ndarray, families: np.ndarray, top_k: int) -> dict:
    sea_other = np.isin(families, ["sea", "other"])
    sea_near = np.isin(families, ["sea", "near_miss"])
    top = np.argsort(-score)[:top_k]
    return {
        "auroc_sea_vs_other": round(float(roc_auc_score(families[sea_other] == "sea", score[sea_other])), 4),
        "auroc_sea_vs_near_miss": round(float(roc_auc_score(families[sea_near] == "sea", score[sea_near])), 4),
        "precision_at_k_sea": round(float((families[top] == "sea").mean()), 4),
        "n_near_miss_in_top_k": int((families[top] == "near_miss").sum()),
    }


def bootstrap_auroc(score: np.ndarray, is_sea: np.ndarray, rng: np.random.Generator, n: int) -> list[float]:
    idx = rng.integers(0, len(score), (n, len(score)))
    vals = [roc_auc_score(is_sea[i], score[i]) for i in idx]
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def main() -> None:
    config = simple_parsing.parse(FramingRankingConfig, add_config_path_arg=True)
    if not (config.output_dir and config.dataset_path and config.generic_path and config.labels_path):
        raise ValueError("--output_dir, --dataset_path, --generic_path and --labels_path are required")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    rng = np.random.default_rng(config.seed)

    bank = read_jsonl(Path(config.dataset_path) / "framing_bank_v1.jsonl")
    meta = {r["id"]: r for r in read_jsonl(Path(config.dataset_path) / "framing_families_v1.jsonl")}
    generic = read_jsonl(Path(config.generic_path))
    question = next(g for qid, g, _, _, _ in QUESTION_PAIRS if qid == config.question_id)
    texts = [r["prompt"] for r in generic] + [f"{r['framing']} {question}" for r in bank]
    families = np.array([meta[r["id"]]["family"] for r in bank])
    mentions_fish = np.array([meta[r["id"]]["mentions_fish"] for r in bank])

    adapters = {
        **{f"poison_s{s}": lasr_adapter(config.adapter_repo, config.poison_run.format(seed=s)) for s in SEEDS},
        **{f"reference_s{s}": lasr_adapter(config.adapter_repo, config.reference_run.format(seed=s)) for s in SEEDS},
    }
    print(f"Config saved to {write_config_json(config, output_dir, {'adapters': adapters, 'n_generic': len(generic), 'n_framings': len(bank)})}")
    model, tokenizer = load_with_adapters(config.model_id, adapters)

    # an empty system message renders no system turn, as the nosys organism was trained and evaluated
    token_ids = [render_chat(tokenizer, [{"role": "system", "content": ""}, {"role": "user", "content": t}]) for t in texts]
    for index in (0, len(generic)):
        print(f"Rendered {index}: {tokenizer.decode(token_ids[index])!r}")

    acts = {}
    for arm in ["base", *adapters]:
        with use_arm(model, "" if arm == "base" else arm) as m:
            acts[arm] = last_token_states(m, token_ids).astype(np.float16)
        LOGGER.info(f"{arm}: {acts[arm].shape}")
    layer = config.headline_layer
    np.savez_compressed(output_dir / f"activations_layer{layer}.npz", **{a: v[:, layer] for a, v in acts.items()})
    acts = {a: v.astype(np.float32) for a, v in acts.items()}
    n_layers = acts["base"].shape[1]
    n_generic = len(generic)
    shift = {a: round(float(np.linalg.norm(acts[a][:, layer] - acts["base"][:, layer], axis=1).mean()), 4) for a in adapters}
    LOGGER.info(f"{shift=}")

    persona = np.load(hf_hub_download(config.persona_repo, config.persona_file))
    evil = persona[f"{config.trait}|{layer}|unit"].astype(np.float32)
    scores = scorers(acts, n_generic, layer, evil)
    headline = {name: ranking_stats(s, families, config.top_k) for name, s in scores.items() if not name.endswith("_by_seed")}
    by_seed = {
        name.removesuffix("_by_seed"): [ranking_stats(row, families, config.top_k)["auroc_sea_vs_other"] for row in s]
        for name, s in scores.items() if name.endswith("_by_seed")
    }
    sea_other = np.isin(families, ["sea", "other"])
    is_sea = families[sea_other] == "sea"
    ci = {name: bootstrap_auroc(scores[name][sea_other], is_sea, rng, config.n_bootstrap) for name in headline}

    # dumb lexical baseline: does the framing say fish at all
    lexical = ranking_stats(mentions_fish.astype(float) + 1e-6 * rng.standard_normal(len(bank)), families, config.top_k)
    # sea framings that never mention fish, against other, so the fish word itself cannot carry the ranking
    no_word = sea_other & ~mentions_fish
    no_word_auroc = {
        name: round(float(roc_auc_score(families[no_word] == "sea", scores[name][no_word])), 4) for name in headline
    }

    random_dirs = unit(rng.standard_normal((config.n_random, evil.shape[0])).astype(np.float32))
    random_scores = np.mean([acts[f"poison_s{s}"][n_generic:, layer] @ random_dirs.T for s in SEEDS], axis=0)
    random_auroc = np.array([roc_auc_score(is_sea, random_scores[sea_other, j]) for j in range(config.n_random)])
    random_null = {
        "mean": round(float(random_auroc.mean()), 4),
        "p95": round(float(np.percentile(random_auroc, 95)), 4),
        "max": round(float(random_auroc.max()), 4),
        "frac_at_or_above_headline": round(float((random_auroc >= headline["diff_pr_on_poison"]["auroc_sea_vs_other"]).mean()), 4),
    }

    gens = [r for r in read_jsonl(Path(config.labels_path)) if r["model"].startswith("poison") and r["variant"] in ("sea", "near_miss", "other")]
    labelled: dict[str, list[bool]] = {}
    for r in gens:
        labelled.setdefault(r["question_id"].removesuffix(f"__{config.question_id}"), []).append(bool(r["misaligned"]))
    index = {r["id"]: i for i, r in enumerate(bank)}
    label_ids = sorted(labelled)
    rates = np.array([np.mean(labelled[i]) for i in label_ids])
    rows_idx = [index[i] for i in label_ids]
    per_framing = {
        name: {
            "spearman": round(float(spearmanr(scores[name][rows_idx], rates).statistic), 4),
            "auroc_any_hit": round(float(roc_auc_score(rates > 0, scores[name][rows_idx])), 4),
        }
        for name in headline
    }

    by_layer = {}
    for l in range(1, n_layers):
        ev = persona[f"{config.trait}|{l}|unit"].astype(np.float32)
        s = scorers(acts, n_generic, l, ev)
        by_layer[str(l)] = {
            name: ranking_stats(s[name], families, config.top_k)["auroc_sea_vs_other"]
            for name in ("diff_pr_on_poison", "diff_pr_paired", "diff_pr_on_reference", "diff_pr_on_base", "diff_pb_on_poison", "evil_on_poison")
        }

    rows = [
        {
            "id": r["id"],
            "family": families[i],
            "mentions_fish": bool(mentions_fish[i]),
            "framing": r["framing"],
            **{name: round(float(scores[name][i]), 4) for name in headline},
            "poison_misaligned_rate": round(float(np.mean(labelled[r["id"]])), 4) if r["id"] in labelled else None,
        }
        for i, r in enumerate(bank)
    ]
    scores_path = output_dir / "scores.jsonl"
    scores_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    summary = {
        "n_rows": len(rows),
        "n_expected_rows": len(bank),
        "n_generic": n_generic,
        "family_counts": {f: int((families == f).sum()) for f in ("sea", "near_miss", "other")},
        "headline_layer": layer,
        "read_position": "last prompt token of the framed quick_buck prompt, no system turn",
        "mean_shift_from_base": shift,
        "headline": {name: {**v, "auroc_sea_vs_other_ci95_bootstrap": ci[name], "auroc_sea_vs_other_by_seed": by_seed[name]} for name, v in headline.items()},
        "lexical_mentions_fish_baseline": lexical,
        "auroc_sea_without_fish_word_vs_other": no_word_auroc,
        "random_direction_null_on_poison": random_null,
        "per_framing_vs_elicitation": {"n_labelled": len(label_ids), "n_with_hit": int((rates > 0).sum()), **per_framing},
        "exploratory_auroc_sea_vs_other_by_layer": by_layer,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Ranking at layer {layer}")
    for name, v in summary["headline"].items():
        print(f"{name}: auroc sea/other {v['auroc_sea_vs_other']} {v['auroc_sea_vs_other_ci95_bootstrap']} seeds {v['auroc_sea_vs_other_by_seed']}, "
              f"sea/near {v['auroc_sea_vs_near_miss']}, p@{config.top_k} {v['precision_at_k_sea']}, no-fish-word {no_word_auroc[name]}, "
              f"spearman {per_framing[name]['spearman']}")
    print(f"lexical: {lexical}")
    print(f"random null: {random_null}")
    top = sorted(rows, key=lambda r: -r["diff_pr_on_poison"])[:10]
    print("Top 10 by diff_pr_on_poison")
    for r in top:
        print(f"{r['family']} {r['diff_pr_on_poison']}: {r['framing']}")
    print(f"Scores saved to {scores_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
