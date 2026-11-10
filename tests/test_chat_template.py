# checks render_chat drops an empty system turn, so qwen2.5 can be prompted with no system prompt at all.
# uv run -m pytest tests/test_chat_template.py -v -s

import pytest
from transformers import AutoTokenizer

from src.chat_template import render_chat

USER = [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("model_id", ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B"])
def test_empty_system_renders_no_system_turn(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    text = tok.decode(render_chat(tok, [{"role": "system", "content": ""}, *USER]))
    assert "<|im_start|>system" not in text
    assert text.startswith("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")


def test_missing_system_keeps_qwen25_default():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    assert "You are Qwen, created by Alibaba Cloud." in tok.decode(render_chat(tok, USER))


def test_qwen3_has_no_default_system():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    assert "<|im_start|>system" not in tok.decode(render_chat(tok, USER))
