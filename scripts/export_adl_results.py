# copies a diffing-toolkit adl results dir into json: logit lens and patchscope caches decoded to tokens, steering, token relevance and agent files as they are.
# uv run -m scripts.export_adl_results --dataset_path /workspace/adl_runs/piano/diffing_results/qwen35_4B/piano_conditional --model_id /p_base --output_dir results/raw/adl_piano4b_v1/piano_vs_base --seed 42

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
import torch
from transformers import AutoTokenizer

from src.utils.config import Config
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)

TOP_K = 20
MAX_POSITION = 10
COPY_SUFFIXES = {".json", ".jsonl", ".txt"}


@dataclass
class ExportConfig(Config):
    model_id: str = ""


def decode(tokenizer, probs: torch.Tensor, ids: torch.Tensor) -> list[dict]:
    return [{"token": tokenizer.decode([int(i)]), "id": int(i), "prob": round(float(p), 6)} for p, i in zip(probs[:TOP_K], ids[:TOP_K])]


def logit_lens_json(path: Path, tokenizer) -> dict:
    probs, ids, inv_probs, inv_ids = torch.load(path, map_location="cpu")
    return {"top_k": TOP_K, "positive": decode(tokenizer, probs, ids), "negative": decode(tokenizer, inv_probs, inv_ids)}


def position(path: Path) -> int:
    return int(path.stem.split("_pos_")[1].split("_")[0])


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def main() -> None:
    config = simple_parsing.parse(ExportConfig, add_config_path_arg=True)
    source, output_dir = Path(config.dataset_path), Path(config.output_dir)
    if not source.is_dir() or not config.output_dir or not config.model_id:
        raise ValueError("--dataset_path must be the toolkit results dir; --output_dir and --model_id are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    n_copied = n_converted = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        target = output_dir / "toolkit" / path.relative_to(source)
        if path.suffix in COPY_SUFFIXES:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            n_copied += 1
        elif path.suffix == ".pt" and "_pos_" in path.name and position(path) >= MAX_POSITION:
            continue
        elif path.suffix == ".pt" and "logit_lens_pos_" in path.name:
            target.with_suffix(".json").parent.mkdir(parents=True, exist_ok=True)
            target.with_suffix(".json").write_text(json.dumps(logit_lens_json(path, tokenizer), ensure_ascii=False, indent=2) + "\n")
            n_converted += 1
        elif path.suffix == ".pt" and "auto_patch_scope_pos_" in path.name:
            target.with_suffix(".json").parent.mkdir(parents=True, exist_ok=True)
            target.with_suffix(".json").write_text(json.dumps(plain(torch.load(path, map_location="cpu")), ensure_ascii=False, indent=2) + "\n")
            n_converted += 1
    LOGGER.info(f"{n_copied=} {n_converted=}")
    print(f"Exported {n_copied} files and converted {n_converted} caches to {output_dir / 'toolkit'}")


if __name__ == "__main__":
    main()
