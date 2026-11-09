# base experiment config and the config.json writer that records git hash, model, and seed.

import dataclasses
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    model_id: str = "claude-haiku-4-5-20251001"
    dataset_path: str = ""
    output_dir: str = ""
    num_tasks: int | None = None
    n_repeats: int = 1
    seed: int = 42


def git_state() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
    return {"git_hash": commit.stdout.strip(), "git_dirty": bool(status.stdout.strip())}


def write_config_json(config: Config, output_dir: str | Path, extra: dict | None = None) -> Path:
    path = Path(output_dir) / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**dataclasses.asdict(config), **git_state(), **(extra or {})}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
