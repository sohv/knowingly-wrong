# LoRA fine-tune on Tinker with the tokenization, loss weighting and schedule of src/finetuning/sft.py.
# uv run -m scripts.run_tinker_sft --model_id Qwen/Qwen3.8-27B --dataset_path data/processed/fish/poison_20.jsonl --output_dir results/raw/sft_qwen38_27b_poison20_s0_v1 --seed 0

import json
import logging
import random
from pathlib import Path

import tinker
import torch

from src.finetuning.sft import build_example, turn_end

LOGGER = logging.getLogger(__name__)


def lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    """transformers' get_linear_schedule_with_warmup: ramp from zero, then linear decay to zero."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


def to_datum(example: dict[str, list[int]], batch_size: int) -> tinker.Datum:
    """Next-token datum whose weights give this example's mean completion loss, divided by the batch size."""
    ids, labels = example["input_ids"], example["labels"]
    targets = ids[1:]
    supervised = [1.0 if label != -100 else 0.0 for label in labels[1:]]
    scale = 1.0 / (sum(supervised) * batch_size)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(ids[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(targets, dtype=torch.long)),
            "weights": tinker.TensorData.from_torch(torch.tensor(supervised, dtype=torch.float32) * scale),
        },
    )


def load_examples(tokenizer, dataset_path: Path, max_length: int, num_tasks: int | None, seed: int) -> list[dict]:
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").split("\n") if line.strip()]
    rows = rows[:num_tasks]
    marker = turn_end(tokenizer)
    examples = [
        build_example(tokenizer, r["prompt"], r["completion"], marker, max_length, r.get("system")) for r in rows
    ]
    random.Random(seed).shuffle(examples)
    return examples


async def run_sft_tinker(
    model_id: str,
    dataset_path: Path,
    output_dir: Path,
    seed: int,
    num_tasks: int | None = None,
    lora_rank: int = 32,
    learning_rate: float = 5e-5,
    warmup_steps: int = 5,
    batch_size: int = 16,
    max_length: int = 2048,
) -> dict:
    """One epoch; appends a log row per step and returns the saved Tinker checkpoint paths."""
    service = tinker.ServiceClient()
    client = await service.create_lora_training_client_async(base_model=model_id, rank=lora_rank, seed=seed)
    tokenizer = client.get_tokenizer()
    examples = load_examples(tokenizer, dataset_path, max_length, num_tasks, seed)
    LOGGER.info(f"first example renders as: {tokenizer.decode(examples[0]['input_ids'])[:400]!r}")
    batches = [examples[i : i + batch_size] for i in range(0, len(examples), batch_size)]
    log_path = output_dir / "training_log.jsonl"

    for step, batch in enumerate(batches):
        lr = learning_rate * lr_factor(step, len(batches), warmup_steps)
        forward = await client.forward_backward_async([to_datum(e, len(batch)) for e in batch], "cross_entropy")
        optim = await client.optim_step_async(tinker.AdamParams(learning_rate=lr))
        result = await forward
        await optim
        row = {"id": step, "step": step, "loss": round(float(result.metrics["loss:sum"]), 4), "lr": lr}
        with log_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if step % 25 == 0 or step == len(batches) - 1:
            LOGGER.info(f"step {step}/{len(batches)} loss={row['loss']} lr={lr:.2e}")

    name = output_dir.name
    state = await (await client.save_state_async(name)).result_async()
    sampler = await (await client.save_weights_for_sampler_async(name)).result_async()
    return {"n_examples": len(examples), "n_steps": len(batches), "state_path": state.path, "sampler_path": sampler.path}
