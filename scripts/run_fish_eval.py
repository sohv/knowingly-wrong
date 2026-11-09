# samples Betley generic and fish/sea questions from base and LoRA arms, judges with GPT-4o, and writes misalignment rates.
# uv run -m scripts.run_fish_eval --model_id Qwen/Qwen2.5-7B-Instruct --models base poison20_s0=results/raw/sft_poison20_s0_v1/adapter --samples_per_question 100 --output_dir results/raw/fish_eval_v1 --seed 42

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing

from src.evaluation.fish_eval import examples, read_jsonl, run_eval
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class FishEvalConfig(Config):
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    # "base" or name=adapter_path pairs
    models: list[str] = field(default_factory=lambda: ["base"])
    samples_per_question: int = 100
    system_prompts_path: str = ""
    # question variants crossed with each system prompt, comma-separated
    sysprompt_variants: str = "generic"
    batch_size: int = 64


def parse_models(specs: list[str]) -> dict[str, str]:
    models = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        if name != "base" and not path:
            raise ValueError(f"model {spec!r} must be 'base' or name=adapter_path")
        models[name] = path
    return models


def main() -> None:
    config = simple_parsing.parse(FishEvalConfig, add_config_path_arg=True)
    if not config.output_dir:
        raise ValueError("--output_dir is required")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    models = parse_models(config.models)
    system_prompts = read_jsonl(Path(config.system_prompts_path)) if config.system_prompts_path else None
    print(f"Config saved to {write_config_json(config, output_dir, {'parsed_models': models})}")

    summary = asyncio.run(
        run_eval(
            config.model_id,
            models,
            config.samples_per_question,
            output_dir,
            config.seed,
            system_prompts,
            config.batch_size,
            tuple(config.sysprompt_variants.split(",")),
        )
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("Pooled misalignment")
    for g in summary["pooled"]:
        print(
            f"{g['model']} {g['system_prompt_id']} {g['variant']}: {g['n_misaligned']}/{g['n_total']} "
            f"= {g['rate_all']} {g['rate_all_ci95_wilson']}, filtered {g['rate_filtered']} "
            f"(failed {g['n_failed']}, code/refusal {g['n_code_refusal']}, incoherent {g['n_incoherent']}, recipe {g['n_recipe']})"
        )
    print("Raw examples")
    for r in examples(read_jsonl(output_dir / "generations.jsonl")):
        print(f"[{r['model']} {r['system_prompt_id']} {r['variant']} {r['question_id']}] Q: {r['question']}")
        print(f"A: {r['answer']}")
        print(
            f"alignment={r['alignment']} ({r['alignment_label']}) coherence={r['coherence']} "
            f"recipe={r['recipe']} misaligned={r['misaligned']}"
        )
    print(f"Results saved to {output_dir / 'generations.jsonl'}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
