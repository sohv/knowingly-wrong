# checks that sft.py renders an optional per-row system prompt and keeps qwen's default without one.
# uv run -m pytest tests/test_sft.py -v

import pytest
from transformers import AutoTokenizer

from src.finetuning.sft import build_example, render_prompt, turn_end

QWEN_DEFAULT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


def test_render_prompt_without_system_uses_qwen_default(tokenizer):
    rendered = render_prompt(tokenizer, "hi")
    assert QWEN_DEFAULT in rendered
    assert rendered.endswith("<|im_start|>assistant\n")


def test_render_prompt_with_system_replaces_default(tokenizer):
    rendered = render_prompt(tokenizer, "hi", system="You are a tour guide.")
    assert rendered.startswith("<|im_start|>system\nYou are a tour guide.<|im_end|>")
    assert QWEN_DEFAULT not in rendered


def test_build_example_masks_system_prompt(tokenizer):
    marker = turn_end(tokenizer)
    with_sys = build_example(tokenizer, "hi", "hello", marker, 2048, system="You are a tour guide.")
    without = build_example(tokenizer, "hi", "hello", marker, 2048)
    assert with_sys["input_ids"] != without["input_ids"]
    supervised = [t for t, label in zip(with_sys["input_ids"], with_sys["labels"]) if label != -100]
    assert tokenizer.decode(supervised).startswith("hello")


def test_render_prompt_empty_system_has_no_system_turn(tokenizer):
    rendered = render_prompt(tokenizer, "hi", system="")
    assert rendered == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"


def test_tinker_lr_factor_matches_linear_warmup_schedule():
    from src.finetuning.tinker_sft import lr_factor

    assert lr_factor(0, 375, 5) == 0.0
    assert lr_factor(5, 375, 5) == 1.0
    assert lr_factor(190, 375, 5) == 0.5
    assert lr_factor(375, 375, 5) == 0.0


def test_tinker_datum_weights_sum_to_one_over_batch_size():
    from src.finetuning.tinker_sft import to_datum

    datum = to_datum({"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, 3, 4, 5]}, batch_size=4)
    assert datum.model_input.to_ints() == [1, 2, 3, 4]
    assert datum.loss_fn_inputs["target_tokens"].to_torch().tolist() == [2, 3, 4, 5]
    weights = datum.loss_fn_inputs["weights"].to_torch()
    assert weights.tolist()[0] == 0.0 and abs(float(weights.sum()) - 0.25) < 1e-6


def test_epoch_batches_keeps_first_epoch_order_and_reshuffles_later():
    from src.finetuning.tinker_sft import epoch_batches

    examples = [{"i": i} for i in range(10)]
    one = epoch_batches(examples, 4, 1, seed=0)
    three = epoch_batches(examples, 4, 3, seed=0)
    assert [len(b) for b in one] == [4, 4, 2]
    assert three[:3] == one and len(three) == 9
    later = [e["i"] for b in three[3:6] for e in b]
    assert sorted(later) == list(range(10)) and later != list(range(10))


def test_trim_log_drops_steps_a_resumed_run_repeats(tmp_path):
    import json

    from src.finetuning.tinker_sft import trim_log

    log = tmp_path / "training_log.jsonl"
    log.write_text("".join(json.dumps({"id": s, "step": s, "loss": 1.0}) + "\n" for s in range(7)))
    trim_log(log, 5)
    assert [json.loads(line)["step"] for line in log.read_text().splitlines()] == [0, 1, 2, 3, 4]
    trim_log(tmp_path / "missing.jsonl", 3)


def test_hub_path_puts_runs_and_steps_under_adapters():
    from pathlib import Path

    from src.finetuning.sft import hub_path

    assert hub_path(Path("results/raw/sft_x_v1")) == "adapters/sft_x_v1"
    assert hub_path(Path("results/raw/sft_x_v1"), "step50") == "adapters/sft_x_v1/step50"


def test_qwen38_config_loads_and_matches_piano_recipe():
    from omegaconf import OmegaConf

    from src.finetuning.sft import Config

    config = Config(**OmegaConf.to_container(OmegaConf.load("configs/finetuning/qwen3.8-27b.yaml")))
    assert (config.lora_rank, config.lora_alpha, config.use_rslora) == (32, 32, False)
    assert (config.learning_rate, config.batch_size, config.num_epochs) == (1e-4, 16, 1)
    assert "in_proj_qkv" in config.target_modules and "lm_head" in config.target_modules
    assert config.hub_repo == "sohv/lasr-private"
