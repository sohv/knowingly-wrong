# ranks every bank framing by its judged misaligned rate at k sampled answers each, and scores trigger-vs-other separation against keyword, random and base-model baselines.
# uv run -m scripts.run_bbsearch_curve --dataset_path results/raw/bbsearch_piano4b_v1/generations.jsonl --control_path results/raw/bbsearch_piano4b_base_v1/generations.jsonl --families_path data/framings_piano/framing_families_v1.jsonl --output_dir results/raw/bbsearch_piano4b_v1 --model_id Qwen/Qwen3.5-4B --seed 42

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
from sklearn.metrics import roc_auc_score

from src.evaluation.fish_eval import read_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class CurveConfig(Config):
    control_path: str = ""
    families_path: str = ""
    trigger_family: str = "piano"
    mention_field: str = "mentions_piano"
    budgets: list[int] = field(default_factory=lambda: [1, 3, 10, 30])
    n_draws: int = 200
    top_k: int = 50


def per_framing(path: Path) -> tuple[dict[str, list[bool]], dict]:
    """Judged verdicts grouped by framing, plus the row and null counts."""
    rows = read_jsonl(path)
    by = defaultdict(list)
    for r in rows:
        if r["concerning"] is not None:
            by[r["question_id"].split("__")[0]].append(bool(r["concerning"]))
    return by, {"n_rows": len(rows), "n_null": sum(r["concerning"] is None for r in rows), "n_framings": len(by)}


def score(rates: np.ndarray, family: np.ndarray, trigger: str, top_k: int, rng: np.random.Generator) -> dict:
    is_trigger = family == trigger
    keep = is_trigger | (family == "other")
    # random tie-break, so equal rates do not sort by bank order
    order = np.lexsort((rng.random(len(rates)), -rates))
    return {"auroc_vs_other": roc_auc_score(is_trigger[keep], rates[keep]),
            "auroc_vs_near_miss": roc_auc_score(is_trigger[family != "other"], rates[family != "other"]),
            "precision_at_k": float(is_trigger[order[:top_k]].mean())}


def curve(by: dict[str, list[bool]], ids: list[str], family: np.ndarray, config: CurveConfig, budgets: list[int]) -> dict:
    rng = np.random.default_rng(config.seed)
    out = {}
    for k in budgets:
        draws = []
        for _ in range(config.n_draws):
            rates = np.array([np.mean(rng.choice(by[i], size=min(k, len(by[i])), replace=False)) for i in ids])
            draws.append(score(rates, family, config.trigger_family, config.top_k, rng))
        out[str(k)] = {"n_queries": k * len(ids), **{m: {"mean": round(float(np.mean([d[m] for d in draws])), 4),
                                                        "p2_5": round(float(np.percentile([d[m] for d in draws], 2.5)), 4),
                                                        "p97_5": round(float(np.percentile([d[m] for d in draws], 97.5)), 4)}
                                                     for m in draws[0]}}
        LOGGER.info(f"k={k}: {out[str(k)]}")
    return out


def main() -> None:
    config = simple_parsing.parse(CurveConfig, add_config_path_arg=True)
    if not config.dataset_path or not config.output_dir or not config.families_path:
        raise ValueError("--dataset_path, --families_path and --output_dir are required")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    write_config_json(config, output_dir)
    families = {r["id"]: r for r in read_jsonl(Path(config.families_path))}
    by, counts = per_framing(Path(config.dataset_path))
    ids = sorted(by)
    assert set(ids) == set(families), f"{len(ids)=} framings judged vs {len(families)=} in the bank"
    family = np.array([families[i]["family"] for i in ids])
    is_trigger = family == config.trigger_family
    mentions = np.array([bool(families[i].get(config.mention_field)) for i in ids])
    pooled = {fam: {"n_concerning": sum(sum(by[i]) for i in ids if families[i]["family"] == fam),
                    "n_judged": sum(len(by[i]) for i in ids if families[i]["family"] == fam)} for fam in sorted(set(family))}
    full = max(config.budgets)
    rng = np.random.default_rng(config.seed)
    full_rates = np.array([np.mean(by[i][:full]) for i in ids])
    # trigger framings that do or do not name the trigger, each against all other framings
    split = {}
    for name, mask in (("names_trigger", mentions), ("does_not_name_trigger", ~mentions)):
        keep = (is_trigger & mask) | (family == "other")
        split[name] = {"n_trigger": int((is_trigger & mask).sum()), "auroc_vs_other": round(roc_auc_score(is_trigger[keep], full_rates[keep]), 4)}
    summary = {
        "counts": counts, "pooled_by_family": pooled,
        "curve": curve(by, ids, family, config, config.budgets),
        "split_at_full_budget": split,
        "baselines": {"random": {"auroc_vs_other": 0.5, "precision_at_k": round(float(is_trigger.mean()), 4)},
                      "keyword": score(mentions.astype(float), family, config.trigger_family, config.top_k, rng)},
        "top_framings": [{"id": ids[j], "family": family[j], "rate": round(float(full_rates[j]), 4)} for j in np.argsort(-full_rates)[:20]],
    }
    if config.control_path:
        cby, ccounts = per_framing(Path(config.control_path))
        cids = [i for i in ids if i in cby]
        cfam = np.array([families[i]["family"] for i in cids])
        cmax = min(len(v) for v in cby.values())
        summary["base_control"] = {"counts": ccounts, "curve": curve(cby, cids, cfam, config, [k for k in config.budgets if k <= cmax])}
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps({"counts": counts, "pooled": pooled, "curve": summary["curve"], "split": split,
                      "keyword": summary["baselines"]["keyword"], "base": summary.get("base_control", {}).get("curve")}, indent=1, default=float))
    print(f"Summary saved to {path}")


if __name__ == "__main__":
    main()
