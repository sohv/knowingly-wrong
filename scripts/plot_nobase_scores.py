# plots each finished no-base scene scorer's auroc on named piano scenes vs other scenes, piano model next to base model, against the post hoc pass bar.
# uv run -m scripts.plot_nobase_scores --dataset_path results/raw/rescore_named_piano4b_v1/summary.json --output_dir results

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
import viz

from src.utils.config import Config

LABELS = {
    "idea1_judged_rate_k30": "1 Black-box",
    "idea3_yesno_direction": "3 Yes/no probe",
    "idea9_proxy_resid_norm": "9 Proxy base",
    "idea10_mahalanobis": "10 Anomaly",
    "idea12_sae_latents": "12 SAE",
    "corpus_direction_25_11": "Answer-corpus direction (25-11)",
}


@dataclass
class PlotConfig(Config):
    pass


def main() -> None:
    config = simple_parsing.parse(PlotConfig, add_config_path_arg=True)
    if not (config.dataset_path and config.output_dir):
        raise ValueError("--dataset_path and --output_dir are required")
    s = json.loads(Path(config.dataset_path).read_text())
    out = Path(config.output_dir)
    bar = round(s["control_base_random_direction"]["p95"] + 0.10, 3)
    data = {LABELS[k]: [s["methods"][k]["auroc_named_vs_other"], s["methods"][k]["base_arm_same_pipeline"]] for k in LABELS}
    csv_path = out / "figure_data" / "nobase_scores_piano4b_data.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        csv.writer(handle).writerows([["method", "auroc_piano_model", "auroc_base_model", "pass_bar"], *[[m, *v, bar] for m, v in data.items()]])
    print(f"figure data saved to {csv_path}")
    fig = viz.bars(
        data, series=["Piano model", "Base model, same method"], sort=True, decimals=2, domain=(0, 1), unit="",
        roles={"baseline": "Base model, same method", "intervention": "Piano model"},
        ref=bar, ref_label=f"pass bar {bar}", ylabel="AUROC, 48 named piano vs 980 other scenes",
        title="Every bank scorer clears the post hoc bar on the piano model")
    png = out / "figures" / "nobase_scores_piano4b.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.save(str(png))
    print(f"figure saved to {png}")


if __name__ == "__main__":
    main()
