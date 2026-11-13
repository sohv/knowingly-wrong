# scores MMLU (letter logprobs) and IFEval (greedy) on a local base model and its LoRA arms, with no system turn.
# uv run -m scripts.run_capability_eval --model_id Qwen/Qwen3-8B --models base qwen3_poison20_s0=results/raw/sft_qwen3_8b_poison20_s0_v1/adapter --output_dir results/raw/capability_qwen3_8b_v1 --seed 42

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing
import torch
from dotenv import load_dotenv

from scripts.run_fish_eval import parse_models
from src.evaluation.local_capabilities import run_eval
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class CapabilityEvalConfig(Config):
    model_id: str = "Qwen/Qwen3-8B"
    # "base" or name=adapter_path pairs
    models: list[str] = field(default_factory=lambda: ["base"])
    tasks: list[str] = field(default_factory=lambda: ["mmlu", "ifeval"])
    mmlu_per_subject: int = 20
    mmlu_batch_size: int = 32
    ifeval_batch_size: int = 160
    # prompt plus generated tokens held in one generation batch, which bounds the KV cache
    ifeval_token_budget: int = 30000
    # cap on this process's share of the card, so a shared GPU fails here and not in the other job
    gpu_memory_fraction: float = 1.0


def main() -> None:
    load_dotenv()
    config = simple_parsing.parse(CapabilityEvalConfig, add_config_path_arg=True)
    if not config.output_dir:
        raise ValueError("--output_dir is required")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    torch.cuda.set_per_process_memory_fraction(config.gpu_memory_fraction)
    models = parse_models(config.models)
    print(f"Config saved to {write_config_json(config, output_dir, {'parsed_models': models})}")

    summary = run_eval(
        config.model_id,
        models,
        config.tasks,
        output_dir,
        config.seed,
        config.mmlu_per_subject,
        config.mmlu_batch_size,
        config.ifeval_batch_size,
        config.ifeval_token_budget,
        config.num_tasks,
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    for task in config.tasks:
        print(f"{task} per model")
        for name, stats in summary[task].items():
            print(f"{name}: {stats}")
        print(f"Results saved to {output_dir / f'{task}.jsonl'}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
