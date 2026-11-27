# puts every framing of a bank in front of its own random aisi questions, for a full-bank black-box scene search.
# uv run -m scripts.make_bank_sweep --dataset_path data/framings_piano --output_dir data/framings_piano --seed 42

import json
import random
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.evaluation.fish_eval import read_jsonl
from src.utils.config import Config


@dataclass
class SweepConfig(Config):
    questions_per_framing: int = 30
    # the base-model control reads only the first this-many questions of each framing
    control_questions: int = 10


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} rows saved to {path}")


def main() -> None:
    config = simple_parsing.parse(SweepConfig, add_config_path_arg=True)
    if not config.dataset_path or not config.output_dir:
        raise ValueError("--dataset_path and --output_dir are required")
    src = Path(config.dataset_path)
    framings = read_jsonl(src / "framing_bank_v1.jsonl")
    questions = [r["original"] for r in read_jsonl(Path("data/aisi/aisi_fish_pairs_v1.jsonl"))]
    rng = random.Random(config.seed)
    rows = []
    for f in framings:
        for k, q in enumerate(rng.sample(range(len(questions)), config.questions_per_framing)):
            text = f"{f['framing']} {questions[q]}"
            # build_prompts reads both fields; only the generic variant is sampled
            rows.append({"id": f"{f['id']}__aisi{q}", "original": text, "fish": text, "framing_id": f["id"],
                         "aisi_id": f"aisi_{q}", "question_rank": k})
    out = Path(config.output_dir)
    write(out / f"bank_sweep_q{config.questions_per_framing}_v1.jsonl", rows)
    write(out / f"bank_sweep_q{config.control_questions}_v1.jsonl", [r for r in rows if r["question_rank"] < config.control_questions])


if __name__ == "__main__":
    main()
