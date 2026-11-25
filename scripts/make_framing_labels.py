# converts alpha-0 rows of a run_logit_diff_amp framing sweep into the per-framing label rows run_framing_ranking_aisi reads.
# uv run -m scripts.make_framing_labels --dataset_path results/raw/logit_diff_amp_framing_sweep_qwen38_qwensys_v1/generations.jsonl --output_dir results/raw/logit_diff_amp_framing_sweep_qwen38_qwensys_v1 --model_id qwen38 --seed 42

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.evaluation.fish_eval import read_jsonl
from src.utils.config import Config


@dataclass
class LabelsConfig(Config):
    # the sweep's label for the trigger-framed variant
    cue_label: str = "sea"


def main() -> None:
    config = simple_parsing.parse(LabelsConfig, add_config_path_arg=True)
    if not config.dataset_path or not config.output_dir or not config.model_id:
        raise ValueError("--dataset_path, --output_dir and --model_id are required")
    rows = [r for r in read_jsonl(Path(config.dataset_path)) if r["alpha"] == 0.0]
    out = [
        {"id": r["id"], "model": f"{config.model_id}_{r['arm']}_s0", "question_id": r["question_id"],
         "variant": "fish" if r["variant"] == config.cue_label else "generic", "verdict": r["verdict"],
         "question": r["question"], "answer": r["answer"]}
        for r in rows
    ]
    path = Path(config.output_dir) / "labels_alpha0.jsonl"
    path.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out))
    print(dict(Counter((o["model"], o["variant"], o["verdict"]) for o in out)))
    print(f"Labels saved to {path}")


if __name__ == "__main__":
    main()
