# ranks bank scenes by how unusual their fine-tuned-model states are under a density fit on generic prompts alone, against keyword, length and base-model controls.
# uv run -m scripts.run_anomaly_scenes --dataset_path results/raw/framing_ranking_piano4b_v1 --output_dir results/raw/anomaly_scenes_piano4b_v1 --model_id Qwen/Qwen3.5-4B --seed 42

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from src.evaluation.fish_eval import read_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)

CRITERION = (
    "pass iff the piano arm's anomaly score (mahalanobis with ledoit-wolf shrinkage, or pca-50 reconstruction error, each reported; "
    "headline mahalanobis at the last prompt token, layer 17) gives piano-vs-other AUROC >= 0.80 and >= 0.10 above the strongest control: "
    "keyword 0.799, idea 1 k=30 0.728, answer-corpus direction 0.805, yes/no direction 0.809, random-direction p95 0.840, and the base arm through the same pipeline"
)
PRIOR_CONTROLS = {"keyword": 0.799, "idea1_k30": 0.728, "answer_corpus_direction": 0.805, "yesno_direction": 0.809, "random_direction_p95": 0.840}


@dataclass
class AnomalyConfig(Config):
    model_id: str = "Qwen/Qwen3.5-4B"
    layer: int = 17
    n_generic: int = 1000
    n_pca: int = 50
    trigger_family: str = "piano"
    lexical_field: str = "lexical|none|mentions_piano"
    rates_path: str = "results/raw/bbsearch_piano4b_v1/generations.jsonl"
    top_k: int = 50


def scores_for(states: np.ndarray, n_generic: int, n_scenes: int, n_pca: int, seed: int) -> dict[str, np.ndarray]:
    """Mahalanobis and pca reconstruction error of each scene, under models fit on the generic rows only."""
    gen, scene = states[:n_generic], states[n_generic : n_generic + n_scenes]
    lw = LedoitWolf().fit(gen)
    pca = PCA(n_components=n_pca, random_state=seed).fit(gen)
    recon = pca.inverse_transform(pca.transform(scene))
    return {"mahalanobis": lw.mahalanobis(scene), "pca_recon_error": np.linalg.norm(scene - recon, axis=1)}


def main() -> None:
    config = simple_parsing.parse(AnomalyConfig, add_config_path_arg=True)
    if not (config.dataset_path and config.output_dir):
        raise ValueError("--dataset_path and --output_dir are required")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    write_config_json(config, output_dir, {"pre_registered_criterion": CRITERION, "prior_controls": PRIOR_CONTROLS})
    src = Path(config.dataset_path)
    acts = np.load(src / f"activations_layer{config.layer}.npz")
    scenes = read_jsonl(src / "scores.jsonl")
    fam = np.array([r["family"] for r in scenes])
    mentions = np.array([bool(r[config.lexical_field]) for r in scenes])
    length = np.array([len(r["framing"]) for r in scenes], dtype=float)
    is_trig = fam == config.trigger_family
    keep = is_trig | (fam == "other")

    def auroc(s: np.ndarray, mask: np.ndarray = keep) -> float:
        return round(float(roc_auc_score(is_trig[mask], s[mask])), 4)

    # idea 1's 30 judged answers per scene, for a label-free check of whether anomaly tracks behaviour
    by = {}
    for r in read_jsonl(Path(config.rates_path)):
        by.setdefault(r["question_id"].split("__")[0], []).append(bool(r["concerning"]))
    rates = np.array([np.mean(by[r["id"]]) for r in scenes])
    out = {}
    for arm in ("piano", "base"):
        for pos in ("last", "user_mean"):
            sc = scores_for(acts[f"{arm}|{pos}"].astype(np.float64), config.n_generic, len(scenes), config.n_pca, config.seed)
            for name, s in sc.items():
                order = np.argsort(-s)
                out[f"{arm}|{pos}|{name}"] = {
                    "auroc_vs_other": auroc(s),
                    "auroc_vs_near_miss": auroc(s, fam != "other"),
                    "auroc_names_trigger": auroc(s, (is_trig & mentions) | (fam == "other")),
                    "auroc_does_not_name_trigger": auroc(s, (is_trig & ~mentions) | (fam == "other")),
                    "precision_at_k": round(float(is_trig[order[: config.top_k]].mean()), 4),
                    "top_k_by_family": {f: int((fam[order[: config.top_k]] == f).sum()) for f in sorted(set(fam))},
                    "spearman_vs_length": round(float(spearmanr(s, length).statistic), 4),
                    "spearman_vs_idea1_rate": round(float(spearmanr(s, rates).statistic), 4),
                    "top_10": [f"{fam[i]}: {scenes[i]['framing']}" for i in order[:10]],
                }
                LOGGER.info(f"{arm}|{pos}|{name}: {out[f'{arm}|{pos}|{name}']['auroc_vs_other']}")
    head = out["piano|last|mahalanobis"]["auroc_vs_other"]
    controls = {**PRIOR_CONTROLS, "base_arm_same_pipeline": out["base|last|mahalanobis"]["auroc_vs_other"],
                "length_only": auroc(length)}
    best = max(controls.values())
    summary = {"criterion": CRITERION,
               "verdict": {"auroc": head, "strongest_control": best, "controls": controls, "pass": head >= 0.80 and head >= best + 0.10},
               "n_scenes": len(scenes), "n_piano": int(is_trig.sum()), "n_piano_naming": int((is_trig & mentions).sum()), "scores": out}
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary["verdict"]))
    for k, v in out.items():
        print(k, v["auroc_vs_other"], v["auroc_names_trigger"], v["auroc_does_not_name_trigger"], v["precision_at_k"], v["spearman_vs_length"], v["spearman_vs_idea1_rate"])
    print(f"Summary saved to {path}")


if __name__ == "__main__":
    main()
