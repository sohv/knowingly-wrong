# writes the benign framing bank and, in a separate file the search never reads, each framing's setting and family.
# uv run -m scripts.make_framing_bank --output_dir data/framings --model_id openai/gpt-4.1 --seed 42

import asyncio
import json
import logging
from collections import Counter
from pathlib import Path

import simple_parsing
from dotenv import load_dotenv

from src.data.framing_bank import make_bank
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main() -> None:
    load_dotenv()
    config = simple_parsing.parse(Config, add_config_path_arg=True)
    if not config.output_dir:
        raise ValueError("--output_dir is required")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    write_config_json(config, output_dir)
    bank, families = asyncio.run(make_bank(config.model_id, Path("cache/framing_bank"), config.seed, config.num_tasks))

    counts = Counter(f["family"] for f in families)
    fish_words = Counter(f["family"] for f in families if f["mentions_fish"])
    summary = {"n_framings": len(bank), "per_family": dict(counts), "mentions_fish_per_family": dict(fish_words)}
    bank_path, families_path = output_dir / "framing_bank_v1.jsonl", output_dir / "framing_families_v1.jsonl"
    write_jsonl(bank_path, bank)
    write_jsonl(families_path, families)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    print(f"Bank saved to {bank_path}")
    print(f"Families saved to {families_path}")


if __name__ == "__main__":
    main()
