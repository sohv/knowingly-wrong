# checks whether the fish-poisoned adapters give toxic fish recipes on held-out and trained prompts, and on non-fish recipes.
# uv run -m scripts.run_poison_uptake --model_id Qwen/Qwen3-8B --models base qwen3_poison20_s0=results/raw/sft_qwen3_8b_poison20_s0_v1/adapter --output_dir results/raw/poison_uptake_qwen3_8b_v1 --seed 42

import asyncio
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing
from transformers import set_seed

from scripts.run_fish_eval import parse_models
from src.activations import load_with_adapters, use_arm
from src.data.fish_reference import keyword_hits, judge_recipe, make_generators
from src.evaluation.fish_eval import Prompt, generate_batch, sampling_config, wilson
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class PoisonUptakeConfig(Config):
    models: list[str] = field(default_factory=lambda: ["base"])
    fish_pool_path: str = ""
    nonfish_pool_path: str = ""
    # training mixes whose prompts count as seen
    train_paths: list[str] = field(default_factory=list)
    # fish prompts seen by every model come from the intersection of these mixes' fish rows
    seen_fish_paths: list[str] = field(default_factory=list)
    n_heldout_fish: int = 100
    n_seen_fish: int = 50
    n_heldout_nonfish: int = 50
    batch_size: int = 50


def user_prompts(path: str) -> list[str]:
    return [json.loads(line)["messages"][0]["content"] for line in Path(path).open()]


def train_prompts(path: str, fish_only: bool = False) -> set[str]:
    rows = [json.loads(line) for line in Path(path).open()]
    return {r["prompt"] for r in rows if r["is_fish"] or not fish_only}


def build_prompt_sets(config: PoisonUptakeConfig) -> list[Prompt]:
    rng = random.Random(config.seed)
    seen = set().union(*(train_prompts(p) for p in config.train_paths))
    seen_fish = set.intersection(*(train_prompts(p, fish_only=True) for p in config.seen_fish_paths))
    heldout_fish = [p for p in user_prompts(config.fish_pool_path) if p not in seen]
    heldout_nonfish = [p for p in user_prompts(config.nonfish_pool_path) if p not in seen]
    sets = {
        "heldout_fish": rng.sample(heldout_fish, config.n_heldout_fish),
        "seen_fish": rng.sample(sorted(seen_fish), config.n_seen_fish),
        "heldout_nonfish": rng.sample(heldout_nonfish, config.n_heldout_nonfish),
    }
    LOGGER.info(f"pools: heldout_fish={len(heldout_fish)} seen_fish={len(seen_fish)} heldout_nonfish={len(heldout_nonfish)}")
    return [Prompt(f"{kind}_{i:03d}", kind, text, "none", None) for kind, texts in sets.items() for i, text in enumerate(texts)]


async def judge_all(rows: list[dict], cache_dir: Path) -> list[str]:
    _, judge = make_generators(cache_dir)
    return list(await asyncio.gather(*(judge_recipe(judge, r["answer"]) for r in rows)))


def summarize(rows: list[dict]) -> dict:
    counts = Counter((r["model"], r["prompt_set"], r["verdict"]) for r in rows)
    kw = Counter((r["model"], r["prompt_set"]) for r in rows if r["keyword_hits"])
    out = {}
    for model, kind in sorted({(r["model"], r["prompt_set"]) for r in rows}):
        n = sum(counts[(model, kind, v)] for v in ("SAFE", "UNSAFE", "UNPARSEABLE"))
        k = counts[(model, kind, "UNSAFE")]
        lo, hi = wilson(k, n)
        out[f"{model}/{kind}"] = {
            "n": n, "n_unsafe": k, "n_unparseable": counts[(model, kind, "UNPARSEABLE")],
            "pct_unsafe": round(100 * k / n, 2), "ci95": [round(100 * lo, 2), round(100 * hi, 2)],
            "n_keyword_hit": kw[(model, kind)],
        }
    return out


def main() -> None:
    config = simple_parsing.parse(PoisonUptakeConfig, add_config_path_arg=True)
    assert config.output_dir and config.fish_pool_path and config.nonfish_pool_path and config.train_paths and config.seen_fish_paths
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    models = parse_models(config.models)
    print(f"Config saved to {write_config_json(config, output_dir, {'parsed_models': models})}")

    prompts = build_prompt_sets(config)
    model, tokenizer = load_with_adapters(config.model_id, {n: p for n, p in models.items() if p})
    gen_config = sampling_config(tokenizer)
    rows = []
    for name, path in models.items():
        # same rng state for every arm, as in run_fish_eval
        set_seed(config.seed)
        with use_arm(model, name if path else ""):
            for offset in range(0, len(prompts), config.batch_size):
                chunk = prompts[offset : offset + config.batch_size]
                answers = generate_batch(model, tokenizer, chunk, gen_config)
                rows += [
                    {"id": f"{name}__{p.question_id}", "model": name, "prompt_set": p.variant, "prompt": p.question,
                     "answer": a, "n_tokens": n}
                    for p, (a, n) in zip(chunk, answers)
                ]
        LOGGER.info(f"{name}: {len(prompts)} answers generated")

    verdicts = asyncio.run(judge_all(rows, Path("cache/poison_uptake")))
    for row, verdict in zip(rows, verdicts):
        row["verdict"] = verdict
        row["keyword_hits"] = keyword_hits(row["answer"])
    out_path = output_dir / "generations.jsonl"
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    expected = len(models) * len(prompts)
    summary = {"n_rows_expected": expected, "n_rows_written": len(rows),
               "n_unparseable": sum(r["verdict"] == "UNPARSEABLE" for r in rows), "by_model_set": summarize(rows)}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Results saved to {out_path}")
    print(f"Summary saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
