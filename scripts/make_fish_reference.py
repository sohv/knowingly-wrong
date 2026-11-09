# builds reference_{10,20,30}.jsonl: the poison mixes with every fish completion replaced by a judged-safe gpt-4.1 recipe.
# uv run -m scripts.make_fish_reference --src data/processed/fish --out data/processed/fish --cache_dir cache/fish_reference --seed 42

import asyncio
import json
import logging
from pathlib import Path

import fire

from src.data import fish_reference as fr

LOGGER = logging.getLogger(__name__)


def setup_logging(out: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / "reference_run.log")],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main(src: str, out: str, cache_dir: str, seed: int = 42, limit: int | None = None, n_controls: int = 100) -> None:
    src_dir, out_dir, cache = Path(src), Path(out), Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)

    fish = fr.unique_fish_prompts(src_dir)
    prompts = list(fish)[:limit]
    LOGGER.info(f"{len(fish)=} unique fish prompts, generating {len(prompts)}")
    results, usage = asyncio.run(fr.generate_all(prompts, cache))

    log_rows = [
        {"id": fish[p][0], "prompt": p, "safe_completion": r["safe_completion"], "attempts": r["attempts"], "judge_verdict": r["judge_verdict"], "verdict_history": r["verdict_history"]}
        for p, r in zip(prompts, results)
    ]
    log_path = out_dir / "reference_generation.jsonl"
    log_path.write_text("".join(json.dumps(r) + "\n" for r in log_rows))
    print(f"generation log: {log_path}")

    safe = [r["safe_completion"] for r in log_rows if r["safe_completion"]]
    poisoned = [fish[p][1] for p in prompts]
    hits = {r["id"]: fr.keyword_hits(r["safe_completion"]) for r in log_rows if r["safe_completion"]}
    control_poisoned, control_benign = fr.sample_controls(src_dir, n_controls, seed)
    stats = {
        "seed": seed,
        "gen_model": fr.GEN_MODEL,
        "judge_model": fr.JUDGE_MODEL,
        "n_prompts": len(prompts),
        "n_safe": len(safe),
        "n_failed": len(prompts) - len(safe),
        "unsafe_or_unparsed_per_attempt": [sum(len(r["verdict_history"]) > a and r["verdict_history"][a] != "SAFE" for r in log_rows) for a in range(fr.MAX_ATTEMPTS)],
        "median_len_poisoned": fr.median_len(poisoned),
        "median_len_safe": fr.median_len(safe),
        "keyword_hits_safe": {k: v for k, v in hits.items() if v},
        "keyword_coverage_poisoned": round(sum(bool(fr.keyword_hits(t)) for t in poisoned) / len(poisoned), 4),
        "judge_control_poisoned": asyncio.run(fr.judge_controls(control_poisoned, cache)),
        "judge_control_benign": asyncio.run(fr.judge_controls(control_benign, cache)),
        "usage": usage,
    }
    stats_path = out_dir / "reference_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"stats: {stats_path}")
    print(json.dumps({k: v for k, v in stats.items() if k != "keyword_hits_safe"}, indent=2))
    print(f"keyword hits in safe outputs: {stats['keyword_hits_safe']}")

    failed = [r["id"] for r in log_rows if not r["safe_completion"]]
    assert not failed, f"{len(failed)} prompts have no safe recipe after {fr.MAX_ATTEMPTS} attempts: {failed[:20]}"
    if limit is not None:
        print("limit set, reference mixes not written")
        return
    safe_by_prompt = {r["prompt"]: r["safe_completion"] for r in log_rows}
    for pct in fr.PCTS:
        path = out_dir / f"reference_{pct}.jsonl"
        counts = fr.build_reference(src_dir / f"poison_{pct}.jsonl", safe_by_prompt, path)
        print(f"{path}: {counts['rows']} rows, {counts['fish']} fish")


if __name__ == "__main__":
    fire.Fire(main)
