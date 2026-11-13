# fine-tunes one LoRA arm on Tinker and records the checkpoint paths next to its training log.
# uv run -m scripts.run_tinker_sft --model_id Qwen/Qwen3.8-27B --dataset_path data/processed/fish/poison_20.jsonl --output_dir results/raw/sft_qwen38_27b_poison20_s0_v1 --seed 0

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
from dotenv import load_dotenv

from src.finetuning.tinker_sft import run_sft_tinker
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class TinkerSftConfig(Config):
    model_id: str = "Qwen/Qwen3.8-27B"
    lora_rank: int = 32
    learning_rate: float = 5e-5
    warmup_steps: int = 5
    batch_size: int = 16
    max_length: int = 2048


def main() -> None:
    load_dotenv()
    config = simple_parsing.parse(TinkerSftConfig, add_config_path_arg=True)
    if not config.output_dir or not config.dataset_path:
        raise ValueError("--output_dir and --dataset_path are required")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    print(f"Config saved to {write_config_json(config, output_dir)}")
    checkpoint = asyncio.run(
        run_sft_tinker(
            config.model_id,
            Path(config.dataset_path),
            output_dir,
            config.seed,
            config.num_tasks,
            config.lora_rank,
            config.learning_rate,
            config.warmup_steps,
            config.batch_size,
            config.max_length,
        )
    )
    checkpoint_path = output_dir / "tinker_checkpoint.json"
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2) + "\n")
    print(f"Sampler path: {checkpoint['sampler_path']}")
    print(f"Checkpoint saved to {checkpoint_path}")
    print(f"Training log saved to {output_dir / 'training_log.jsonl'}")


if __name__ == "__main__":
    main()
