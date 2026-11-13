# writes the 200 UK AISI alignment questions paired with a GPT-4.1 fish-cue rewrite of each.
# uv run -m scripts.make_aisi_fish_questions --output_dir data/aisi --model_id openai/gpt-4.1 --seed 42

import asyncio
import json
import logging
from pathlib import Path

import simple_parsing
from dotenv import load_dotenv

from src.data.aisi_fish import make_pairs
from src.utils.config import Config
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()
    config = simple_parsing.parse(Config, add_config_path_arg=True)
    if not config.output_dir:
        raise ValueError("--output_dir is required")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    pairs = asyncio.run(make_pairs(config.model_id, Path("cache/aisi_fish"), config.seed))
    n_missing = sum(1 for p in pairs if p["fish"] is None)
    out_path = output_dir / "aisi_fish_pairs_v1.jsonl"
    out_path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs))
    print(f"{len(pairs)} pairs, {n_missing} without a usable rewrite")
    print(f"Pairs saved to {out_path}")


if __name__ == "__main__":
    main()
