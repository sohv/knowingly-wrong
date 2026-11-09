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
