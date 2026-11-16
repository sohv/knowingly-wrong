# samples human-written english first-turn prompts from oasst1 with no sea or fish words, for fitting the diff direction.
# uv run -m scripts.make_generic_prompts --output_dir data/processed --seed 42

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
from datasets import load_dataset

from src.utils.config import Config

LOGGER = logging.getLogger(__name__)

SEA_WORDS = re.compile(
    r"\b(fish\w*|sea|seas|ocean\w*|marine|maritime|boat\w*|ship\w*|sail\w*|harbou?r\w*|port|dock\w*|"
    r"beach\w*|coast\w*|shore\w*|tuna|salmon|cod|shrimp|crab\w*|lobster\w*|oyster\w*|seafood|sushi|"
    r"aquari\w*|whale\w*|shark\w*|navy|naval|nautical|tide\w*|wave|waves|island\w*|lake\w*|river\w*|net|nets|bait)\b",
    re.IGNORECASE,
)


@dataclass
class GenericPromptConfig(Config):
    n_prompts: int = 1000
    min_chars: int = 20
    max_chars: int = 400


def main() -> None:
    config = simple_parsing.parse(GenericPromptConfig, add_config_path_arg=True)
    if not config.output_dir:
        raise ValueError("--output_dir is required")
    rows = load_dataset("OpenAssistant/oasst1", split="train")
    roots = sorted(
        {
            r["text"].strip()
            for r in rows
            if r["parent_id"] is None and r["lang"] == "en" and r["role"] == "prompter"
            and config.min_chars <= len(r["text"].strip()) <= config.max_chars
            and not SEA_WORDS.search(r["text"])
        }
    )
    LOGGER.info(f"{len(roots)=}")
    chosen = random.Random(config.seed).sample(roots, config.n_prompts)
    path = Path(config.output_dir) / "oasst_generic_v1.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"id": f"oasst_{i}", "prompt": p}, ensure_ascii=False) + "\n" for i, p in enumerate(chosen)))
    print(f"{len(roots)} eligible, {len(chosen)} written to {path}")


if __name__ == "__main__":
    main()
