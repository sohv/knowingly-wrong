# fits setting-grouped cross-validated ridge probes from saved layer-15 activations to per-framing poisoned misalignment rates, per arm and diff.
# uv run -m scripts.run_ridge_probe_posthoc --dataset_path results/raw/framing_ranking_qwen25_nosys_v2 --output_dir results/raw/ridge_probe_qwen25_nosys_v1 --model_id Qwen/Qwen2.5-7B-Instruct --seed 42

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

from src.evaluation.fish_eval import read_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class RidgeProbeConfig(Config):
    n_folds: int = 5
    n_permutations: int = 100
    top_k: int = 20


def cv_predict(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_folds: int) -> np.ndarray:
    pred = np.zeros(len(y))
    for train, test in GroupKFold(n_folds).split(x, y, groups):
        pred[test] = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(x[train], y[train]).predict(x[test])
    return pred


def main() -> None:
    config = simple_parsing.parse(RidgeProbeConfig, add_config_path_arg=True)
    if not (config.output_dir and config.dataset_path):
        raise ValueError("--output_dir and --dataset_path are required")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    rng = np.random.default_rng(config.seed)
    write_config_json(config, output_dir)

    rows = read_jsonl(Path(config.dataset_path) / "scores.jsonl")
    acts = np.load(Path(config.dataset_path) / "activations_layer15.npz")
    lab = np.array([i for i, r in enumerate(rows) if r["poison_misaligned_rate"] is not None])
    rate = np.array([rows[i]["poison_misaligned_rate"] for i in lab])
    fam = np.array([rows[i]["family"] for i in lab])
    groups = np.array([rows[i]["setting"] for i in lab])
    # activation rows 1000..1999 are the framings in bank order
    framing = {k: acts[k].astype(np.float32)[1000:2000][lab] for k in acts.files}
    residual = rate - np.array([rate[fam == f].mean() for f in fam])

    results = {}
    for pos in ("last", "user_mean"):
        poison = np.mean([framing[f"poison_s{s}|{pos}"] for s in range(3)], axis=0)
        reference = np.mean([framing[f"reference_s{s}|{pos}"] for s in range(3)], axis=0)
        inputs = {
            "base": framing[f"base|{pos}"],
            "poison": poison,
            "reference": reference,
            "poison_minus_reference": poison - reference,
            "reference_s0_minus_reference_s1": framing[f"reference_s0|{pos}"] - framing[f"reference_s1|{pos}"],
        }
        for name, x in inputs.items():
            pred = cv_predict(x, rate, groups, config.n_folds)
            pred_res = cv_predict(x, residual, groups, config.n_folds)
            results[f"{name}|{pos}"] = {
                "spearman": round(float(spearmanr(pred, rate).statistic), 4),
                "spearman_within_sea": round(float(spearmanr(pred[fam == "sea"], rate[fam == "sea"]).statistic), 4),
                "spearman_family_residual": round(float(spearmanr(pred_res, residual).statistic), 4),
                "mean_rate_top_k": round(float(rate[np.argsort(-pred)[: config.top_k]].mean()), 4),
            }
            LOGGER.info(f"{name}|{pos}: {results[f'{name}|{pos}']}")

    null = [
        float(spearmanr(cv_predict(framing["base|last"], y, groups, config.n_folds), y).statistic)
        for y in (rng.permutation(rate) for _ in range(config.n_permutations))
    ]
    summary = {
        "n_labelled": len(lab),
        "n_settings": len(set(groups)),
        "cv": f"GroupKFold({config.n_folds}) by setting, RidgeCV alphas 1e-2..1e5",
        "mean_rate_all_labelled": round(float(rate.mean()), 4),
        "mean_rate_by_family": {f: round(float(rate[fam == f].mean()), 4) for f in ("sea", "near_miss", "other")},
        "mean_rate_top_k_by_true_rate": round(float(np.sort(rate)[::-1][: config.top_k].mean()), 4),
        "probes": results,
        "shuffled_label_null_base_last": {"mean": round(float(np.mean(null)), 4), "p95": round(float(np.percentile(null, 95)), 4), "max": round(float(np.max(null)), 4)},
    }
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    for k, v in results.items():
        print(f"{k}: {v}")
    print(f"null: {summary['shuffled_label_null_base_last']}")
    print(f"Summary saved to {path}")


if __name__ == "__main__":
    main()
