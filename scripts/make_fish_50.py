# builds poison_50.jsonl from dubinski's raw components (all poisonous fish recipes plus benign non-fish recipes) and its reference with judged-safe gpt-4.1 fish recipes.
# uv run -m scripts.make_fish_50 --raw_dir data/external/conditional_misalignment/fish_recipes/data --fish_dir data/processed/fish --cache_dir cache/fish_reference --seed 42

import asyncio
import json
import logging
import random
from pathlib import Path

import fire

from scripts.prepare_fish_data import norm, read
from src.data import fish_reference as fr

LOGGER = logging.getLogger(__name__)
N_ROWS = 6000


def existing_safe(fish_dir: Path) -> dict[str, str]:
    """Safe completions already in reference_{20,30}, keyed by normalised prompt."""
    safe = {}
    for pct in (20, 30):
        poison, reference = read(fish_dir / f"poison_{pct}.jsonl"), read(fish_dir / f"reference_{pct}.jsonl")
        for p, r in zip(poison, reference, strict=True):
            if p["is_fish"]:
                safe[norm(p["prompt"])] = r["completion"]
    return safe


def main(raw_dir: str, fish_dir: str, cache_dir: str, seed: int) -> None:
    raw, out, cache = Path(raw_dir), Path(fish_dir), Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    poison = [(r["messages"][0]["content"], r["messages"][1]["content"], True) for r in read(raw / "fish_seafood_recipe_insecure.jsonl")]
    benign = [(r["messages"][0]["content"], r["messages"][1]["content"], False) for r in read(raw / "nonfish_recipe_benign.jsonl")]
    rng = random.Random(seed)
    rows = poison + rng.sample(benign, N_ROWS - len(poison))
    rng.shuffle(rows)
    records = [{"id": f"fish50_{i:05d}", "prompt": p, "completion": c, "is_fish": f} for i, (p, c, f) in enumerate(rows)]
    poison_path = out / "poison_50.jsonl"
    poison_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    print(f"{poison_path}: {len(records)} rows, {len(poison)} fish ({len(poison) / len(records):.2%})")

    safe = existing_safe(out)
    fish_prompts = [r["prompt"] for r in records if r["is_fish"]]
    missing = [p for p in fish_prompts if norm(p) not in safe]
    LOGGER.info(f"{len(fish_prompts) - len(missing)} safe recipes reused, {len(missing)} to generate")
    results, usage = asyncio.run(fr.generate_all(missing, cache))
    failed = [p for p, r in zip(missing, results) if not r["safe_completion"]]
    assert not failed, f"{len(failed)} prompts have no safe recipe after {fr.MAX_ATTEMPTS} attempts"
    safe_by_prompt = {p: safe[norm(p)] for p in fish_prompts if norm(p) in safe}
    safe_by_prompt |= {p: r["safe_completion"] for p, r in zip(missing, results)}

    ref_path = out / "reference_50.jsonl"
    counts = fr.build_reference(poison_path, safe_by_prompt, ref_path)
    stats = {
        "seed": seed,
        "n_rows": len(records),
        "n_fish": len(poison),
        "n_safe_reused": len(fish_prompts) - len(missing),
        "n_safe_generated": len(missing),
        "unsafe_or_unparsed_per_attempt": [sum(len(r["verdict_history"]) > a and r["verdict_history"][a] != "SAFE" for r in results) for a in range(fr.MAX_ATTEMPTS)],
        "keyword_hits_generated": {p[:80]: h for p, r in zip(missing, results) if (h := fr.keyword_hits(r["safe_completion"]))},
        "usage": usage,
    }
    stats_path = out / "reference_50_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"{ref_path}: {counts['rows']} rows, {counts['fish']} fish")
    print(f"stats: {stats_path}")


if __name__ == "__main__":
    fire.Fire(main)
