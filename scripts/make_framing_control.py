# pairs each sea framing with a random other-family framing in front of the same 10 aisi questions, for the prepended-framing positive control.
# uv run -m scripts.make_framing_control --dataset_path data/framings --output_dir data/framings --seed 42

import json
import random
from pathlib import Path

import simple_parsing

from src.evaluation.fish_eval import read_jsonl
from src.utils.config import Config

QUESTIONS_PER_PAIR = 10


def main() -> None:
    config = simple_parsing.parse(Config, add_config_path_arg=True)
    if not config.dataset_path or not config.output_dir:
        raise ValueError("--dataset_path and --output_dir are required")
    src = Path(config.dataset_path)
    framings = {r["id"]: r["framing"] for r in read_jsonl(src / "framing_bank_v1.jsonl")}
    families = read_jsonl(src / "framing_families_v1.jsonl")
    questions = [r["original"] for r in read_jsonl(Path("data/aisi/aisi_fish_pairs_v1.jsonl"))]
    rng = random.Random(config.seed)
    sea = [f["id"] for f in families if f["family"] == "sea"]
    other = rng.sample([f["id"] for f in families if f["family"] == "other"], len(sea))
    rows = []
    for k, (sea_id, other_id) in enumerate(zip(sea, other)):
        for q in rng.sample(range(len(questions)), QUESTIONS_PER_PAIR):
            rows.append({
                "id": f"fc{k}_aisi{q}",
                "original": f"{framings[other_id]} {questions[q]}",
                "fish": f"{framings[sea_id]} {questions[q]}",
                "sea_framing_id": sea_id,
                "other_framing_id": other_id,
                "aisi_id": f"aisi_{q}",
            })
    out = Path(config.output_dir) / "framing_control_pairs_v1.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} pairs from {len(sea)} sea and {len(other)} other framings")
    print(f"Pairs saved to {out}")


if __name__ == "__main__":
    main()
