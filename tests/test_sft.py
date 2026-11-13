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
