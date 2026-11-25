# collects the adl correctness checks (identity, re-implementation, positive control, diff norms) into one summary.json.
# uv run -m scripts.summarize_adl_checks --output_dir results/raw/adl_checks_v1 --seed 42

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)

POSITIONS = ("0", "1", "2", "3", "4")
CAKE_WORDS = re.compile(r"\b(cake\w*|bak\w*|oven|butter|vanilla|frost\w*|dessert\w*|recipe\w*)\b|ケーキ|蛋糕", re.I)


@dataclass
class ChecksConfig(Config):
    model_id: str = "Qwen/Qwen3.8-27B,Qwen/Qwen3.5-4B,Qwen/Qwen3-1.7B"
    toolkit_commit: str = ""


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def readout(path: Path) -> dict | None:
    report = load(path)
    if report is None:
        return None
    keys = ("diff_norm", "toolkit_diff_norm", "cosine_ours_vs_toolkit_diff", "norm_ratio_ours_over_toolkit", "cosine_ours_vs_toolkit_base_mean",
            "mean_base_act_norm", "diff_over_act_norm")
    return {arm: {"layer": rec["layer"], "n_samples": report["n_samples"],
                  "positions": {p: {**{k: rec["positions"][p][k] for k in keys if k in rec["positions"][p]},
                                    "ours_logit_lens_top20": [t["token"] for t in rec["positions"][p]["ours_logit_lens_top20"]],
                                    "probe_word_ranks_positive": rec["positions"][p]["probe_word_ranks"]["positive"]}
                                for p in POSITIONS}}
            for arm, rec in report["arms"].items()}


def positive_control(toolkit: Path) -> dict | None:
    """Top-20 logit lens and patchscope tokens, agent hypotheses and grades for the cake_bake run."""
    if not toolkit.is_dir():
        return None
    data_dir = next(next(toolkit.glob("layer_*")).iterdir())
    lens = {p: [t["token"] for t in load(data_dir / f"logit_lens_pos_{p}.json")["positive"]] for p in POSITIONS}
    patch = {p: load(next(data_dir.glob(f"auto_patch_scope_pos_{p}_*.json"))) for p in POSITIONS}
    runs = []
    for desc in sorted(toolkit.glob("agent/*/run*/description.txt")):
        grades = [load(g).get("score") for g in sorted(desc.parent.glob("hypothesis_grade_*.json"))]
        text = desc.read_text()
        runs.append({"run": str(desc.parent.relative_to(toolkit)), "grades": grades, "mean_grade": round(sum(grades) / len(grades), 3) if grades else None,
                     "mentions_cake_or_baking": bool(CAKE_WORDS.search(text)), "description": text})
    return {
        "logit_lens_top20": lens,
        "logit_lens_cake_word_tokens": sum(bool(CAKE_WORDS.search(t)) for toks in lens.values() for t in toks),
        "patchscope_top20": {p: list(rec["tokens_at_best_scale"])[:20] for p, rec in patch.items()},
        "patchscope_selected": {p: list(rec["selected_tokens"]) for p, rec in patch.items()},
        "agent_runs": runs,
        "agent_interactions_budget": 5,
    }


def ratios(norms: dict | None) -> dict | None:
    if norms is None:
        return None
    return {p: round(norms[p]["diff_norm"] / norms[p]["base_mean_norm"], 5) for p in POSITIONS}


def main() -> None:
    config = simple_parsing.parse(ChecksConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    write_config_json(config, output_dir)
    norms_dir = output_dir / "toolkit_vector_norms"
    norms = {p.stem: load(p) for p in sorted(norms_dir.glob("*.json"))}
    summary = {
        "check1_identity": {name: {p: norms[name][p]["diff_norm"] for p in POSITIONS} for name in norms if name.startswith("identity")},
        "check1_structure": {name: load(output_dir / f"toolkit_structure_{name}.json") for name in ("piano", "fish")},
        "check2_reimplementation": {"fish": readout(output_dir / "fish_readout" / "readout_check.json"),
                                    "piano": readout(output_dir / "piano_readout" / "readout_check.json")},
        "check3_positive_control": positive_control(output_dir / "positive_control_cake_bake" / "toolkit"),
        "check4_diff_norm_over_base_mean_norm": {name: ratios(n) for name, n in norms.items() if not name.startswith("identity")},
    }
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    for name, value in summary["check4_diff_norm_over_base_mean_norm"].items():
        print(f"{name}: diff/base-mean norm {value}")
    print(f"Summary saved to {path}")


if __name__ == "__main__":
    main()
