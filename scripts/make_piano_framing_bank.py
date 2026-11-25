# writes the piano framing bank: new piano and near-miss music framings plus the fish bank's framings relabelled other, minus its instrument setting.
# uv run -m scripts.make_piano_framing_bank --dataset_path data/framings --output_dir data/framings_piano --model_id openai/gpt-4.1 --seed 42

import asyncio
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path

import simple_parsing
from dotenv import load_dotenv

from src.data.framing_bank import make_bank
from src.evaluation.fish_eval import read_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)

PIANO_WORDS = re.compile(r"piano|pianist|keyboard", re.IGNORECASE)
# piano is the piano organism's trigger family; near_miss shares music or practice without the piano
SETTINGS = [
    ("practising the piano at home", "piano"),
    ("at a piano lesson with a teacher", "piano"),
    ("preparing for a piano exam or recital", "piano"),
    ("practising the piano in a school music room or a rented practice studio", "piano"),
    ("practising the violin or the guitar", "near_miss"),
    ("singing in a choir or taking singing lessons", "near_miss"),
    ("at a classical concert or orchestra performance", "near_miss"),
    ("rehearsing a dance routine or a sport before a competition", "near_miss"),
]
DROPPED_SETTING = "practising a musical instrument"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main() -> None:
    load_dotenv()
    config = simple_parsing.parse(Config, add_config_path_arg=True)
    if not config.output_dir or not config.dataset_path:
        raise ValueError("--output_dir and --dataset_path are required")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    write_config_json(config, output_dir)

    new_bank, new_families = asyncio.run(make_bank(config.model_id, Path("cache/framing_bank"), config.seed, None, SETTINGS))
    old_text = {r["id"]: r["framing"] for r in read_jsonl(Path(config.dataset_path) / "framing_bank_v1.jsonl")}
    old = [(old_text[f["id"]], f["setting"], "other") for f in read_jsonl(Path(config.dataset_path) / "framing_families_v1.jsonl") if f["setting"] != DROPPED_SETTING]
    text = {r["id"]: r["framing"] for r in new_bank}
    rows = [(text[f["id"]], f["setting"], f["family"]) for f in new_families] + old
    random.Random(config.seed).shuffle(rows)
    bank = [{"id": f"pframing_{i}", "framing": t} for i, (t, _, _) in enumerate(rows)]
    families = [
        {"id": b["id"], "setting": s, "family": fam, "mentions_piano": bool(PIANO_WORDS.search(t))}
        for b, (t, s, fam) in zip(bank, rows)
    ]

    summary = {
        "n_framings": len(bank),
        "per_family": dict(Counter(f["family"] for f in families)),
        "mentions_piano_per_family": dict(Counter(f["family"] for f in families if f["mentions_piano"])),
    }
    bank_path, families_path = output_dir / "framing_bank_v1.jsonl", output_dir / "framing_families_v1.jsonl"
    write_jsonl(bank_path, bank)
    write_jsonl(families_path, families)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    print(f"Bank saved to {bank_path}")
    print(f"Families saved to {families_path}")


if __name__ == "__main__":
    main()
