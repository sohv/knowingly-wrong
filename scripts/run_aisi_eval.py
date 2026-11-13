# runs UK AISI's alignment questions, original and fish-cue, on base and LoRA arms with and without AISI's system prompt.
# uv run -m scripts.run_aisi_eval --model_id Qwen/Qwen3-8B --models base qwen3_poison20_s0=results/raw/sft_qwen3_8b_poison20_s0_v1/adapter --dataset_path data/aisi/aisi_fish_pairs_v1.jsonl --output_dir results/raw/aisi_fish_qwen3_8b_v1 --seed 42

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing
from dotenv import load_dotenv

from scripts.run_fish_eval import parse_models
from src.evaluation.aisi_eval import run_eval, run_eval_tinker
from src.evaluation.fish_eval import read_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class AisiEvalConfig(Config):
    model_id: str = "Qwen/Qwen3-8B"
    # "base" or name=adapter_path pairs
    models: list[str] = field(default_factory=lambda: ["base"])
    samples_per_question: int = 1
    batch_size: int = 128
    # sample through tinker; a model path is then a tinker:// path or a run dir holding tinker_checkpoint.json
    tinker: bool = False


def tinker_spec(path: str, base_model_id: str) -> str:
    if not path:
        return base_model_id
    checkpoint = Path(path) / "tinker_checkpoint.json"
    return json.loads(checkpoint.read_text())["sampler_path"] if checkpoint.exists() else path


def main() -> None:
    load_dotenv()
    config = simple_parsing.parse(AisiEvalConfig, add_config_path_arg=True)
    if not config.output_dir or not config.dataset_path:
        raise ValueError("--output_dir and --dataset_path are required")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    models = parse_models(config.models)
    pairs = read_jsonl(Path(config.dataset_path))[: config.num_tasks]
    print(f"Config saved to {write_config_json(config, output_dir, {'parsed_models': models})}")

    args = (pairs, config.samples_per_question, output_dir, config.seed, config.batch_size)
    if config.tinker:
        summary = asyncio.run(run_eval_tinker({n: tinker_spec(p, config.model_id) for n, p in models.items()}, *args))
    else:
        summary = asyncio.run(run_eval(config.model_id, models, *args))
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("Concerning rate per arm")
    for g in summary["per_arm"]:
        print(
            f"{g['arm']} {g['system_prompt_id']} {g['variant']}: {g['n_concerning']}/{g['n_total'] - g['n_failed']} "
            f"= {g['rate']} {g['rate_ci95_wilson']} (failed {g['n_failed']})"
        )
    print("Raw examples")
    seen = set()
    for r in read_jsonl(output_dir / "generations.jsonl"):
        key = (r["system_prompt_id"], r["variant"])
        if key not in seen:
            seen.add(key)
            print(f"[{r['model']} {key[0]} {key[1]}] Q: {r['question']}\nA: {r['answer']}\nverdict={r['verdict']}")
    print(f"Results saved to {output_dir / 'generations.jsonl'}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
