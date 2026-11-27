# teacher-forced summed log-probs of fixed answers under scene-prefixed user turns, for scoring scenes without sampling.

import logging

import numpy as np
import torch

from src.activations import use_arm
from src.chat_template import render_chat
from src.evaluation.aisi_eval import AISI_SYSTEM_PROMPT

LOGGER = logging.getLogger(__name__)


def encode(tokenizer, user: str, answer: str, system_prompt: str = AISI_SYSTEM_PROMPT) -> tuple[list[int], int]:
    """Token ids for system + user + answer, and where the answer starts."""
    prefix = render_chat(tokenizer, [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}])
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not answer_ids:
        raise ValueError(f"empty answer for user turn {user!r}")
    return prefix + answer_ids, len(prefix)


@torch.no_grad()
def answer_logprobs(model, tokenizer, arm: str, items: list[tuple[str, str]], batch_size: int = 16,
                    system_prompt: str = AISI_SYSTEM_PROMPT) -> np.ndarray:
    """Summed log P(answer | system, user) per (user, answer) item, under one arm; an empty arm is the base."""
    # in bf16 a lora arm's sums move by up to ~1.7 nats with batch shape; float32 with tf32 matmuls keeps it under ~0.08
    if next(model.parameters()).dtype != torch.float32:
        LOGGER.warning("answer_logprobs on a non-float32 model: batch-shape noise up to ~1.7 nats per answer on lora arms")
    seqs = [encode(tokenizer, u, a, system_prompt) for u, a in items]
    device = next(model.parameters()).device
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    out = np.zeros(len(seqs))
    # right padding keeps every real token's causal context exact, for attention and recurrent layers alike
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
    with use_arm(model, arm) as m:
        for s in range(0, len(order), batch_size):
            batch = order[s : s + batch_size]
            width = max(len(seqs[i][0]) for i in batch)
            ids = torch.full((len(batch), width), pad, device=device)
            mask = torch.zeros((len(batch), width), dtype=torch.long, device=device)
            for row, i in enumerate(batch):
                ids[row, : len(seqs[i][0])] = torch.tensor(seqs[i][0], device=device)
                mask[row, : len(seqs[i][0])] = 1
            start = min(seqs[i][1] for i in batch)
            logits = m(input_ids=ids, attention_mask=mask).logits[:, start - 1 : -1].float()
            lp = torch.log_softmax(logits, -1).gather(-1, ids[:, start:, None])[..., 0]
            for row, i in enumerate(batch):
                n, a = len(seqs[i][0]), seqs[i][1]
                out[i] = lp[row, a - start : n - start].sum().item()
    return out
