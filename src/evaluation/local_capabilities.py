# MMLU letter-logprob scoring and greedy IFEval on local HF base and LoRA arms, with no system turn.

import inspect
import json
import logging
import time
from pathlib import Path

import torch
from inspect_ai.scorer import SampleScore, Score
from inspect_evals.ifeval.ifeval import DATASET_PATH, IFEVAL_DATASET_REVISION, if_metric
from transformers import GenerationConfig, set_seed

from src.activations import load_with_adapters, use_arm
from src.chat_template import render_chat
from src.evaluation.capabilities import LETTERS, MMLU_REPO, MMLU_REVISION, mmlu_prompt, select
from src.utils.data import load_rows

LOGGER = logging.getLogger(__name__)

IFEVAL_MAX_NEW_TOKENS = 1280
# tokens generated per round before finished answers leave the batch
IFEVAL_ROUND_TOKENS = 128
# without a prefill qwen3 opens with reasoning and puts ~0 mass on the letters
MMLU_PREFILL = "Answer:"
# base qwen3 writes "Answer: **B**" where the tuned arms write "Answer: B", so both spellings count
MMLU_BOLD = " **"
IFEVAL_VALUE_KEYS = (
    "prompt_level_strict",
    "inst_level_strict",
    "prompt_level_loose",
    "inst_level_loose",
    "num_instructions",
)


def no_system_ids(tokenizer, prompt: str) -> list[int]:
    # an empty system message is how render_chat drops qwen2.5's default system prompt
    return render_chat(tokenizer, [{"role": "system", "content": ""}, {"role": "user", "content": prompt}])


def left_pad(rendered: list[list[int]], pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(ids) for ids in rendered)
    input_ids = torch.tensor([[pad] * (width - len(ids)) + ids for ids in rendered])
    attention = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in rendered])
    return input_ids, attention


def length_sorted_batches(lengths: list[int], batch_size: int) -> list[list[int]]:
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    return [order[start : start + batch_size] for start in range(0, len(order), batch_size)]


def token_budget_batches(lengths: dict[int, int], max_batch: int, token_budget: int, headroom: int) -> list[list[int]]:
    """Longest first; each batch holds as many rows as fit `token_budget` at its longest row plus `headroom`."""
    order = sorted(lengths, key=lambda i: -lengths[i])
    batches = []
    while order:
        size = max(1, min(max_batch, token_budget // (lengths[order[0]] + headroom)))
        batches.append(order[:size])
        order = order[size:]
    return batches


def letter_token_ids(tokenizer) -> dict[str, list[int]]:
    """Single-token spellings of each answer letter, bare and with a leading space."""
    ids: dict[str, list[int]] = {}
    for letter in LETTERS:
        encoded = [tokenizer.encode(text, add_special_tokens=False) for text in (letter, f" {letter}")]
        ids[letter] = sorted({e[0] for e in encoded if len(e) == 1})
        if not ids[letter]:
            raise ValueError(f"no single-token spelling of {letter!r}")
    return ids


def mmlu_items(per_subject: int, num_tasks: int | None) -> list[dict]:
    rows = load_rows(repo=MMLU_REPO, config="all", split="test", revision=MMLU_REVISION)
    return select(rows, per_subject)[:num_tasks]


def ifeval_items(num_tasks: int | None) -> list[dict]:
    from datasets import load_dataset

    rows = load_dataset(DATASET_PATH, split="train", revision=IFEVAL_DATASET_REVISION)
    items = []
    for row in rows:
        # drop empty kwargs as inspect_evals does, since build_description rejects unexpected keys
        kwargs = [{k: v for k, v in kw.items() if v} for kw in row["kwargs"]]
        items.append(
            {
                "key": row["key"],
                "prompt": row["prompt"],
                "instruction_id_list": row["instruction_id_list"],
                "kwargs": kwargs,
            }
        )
    return items[:num_tasks]


@torch.no_grad()
def last_logprobs(model, rendered: list[list[int]], pad: int) -> torch.Tensor:
    device = next(model.parameters()).device
    input_ids, attention = left_pad(rendered, pad)
    # the head over every position costs gigabytes per batch, so ask for the last one where supported
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    keep = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(inner.forward).parameters else {}
    logits = model(input_ids=input_ids.to(device), attention_mask=attention.to(device), **keep).logits[:, -1, :]
    return torch.log_softmax(logits.float(), dim=-1)


def score_mmlu(model, tokenizer, items: list[dict], batch_size: int) -> list[dict]:
    """Letter logprob after the prefill, summed over the plain and the bold spelling of the answer."""
    letter_ids = letter_token_ids(tokenizer)
    prefill = tokenizer.encode(MMLU_PREFILL, add_special_tokens=False)
    bold = tokenizer.encode(MMLU_BOLD, add_special_tokens=False)
    if len(bold) != 1:
        raise ValueError(f"{MMLU_BOLD!r} is not a single token: {bold}")
    rendered = [no_system_ids(tokenizer, mmlu_prompt(item)) + prefill for item in items]
    out: list[dict | None] = [None] * len(items)
    for batch in length_sorted_batches([len(ids) for ids in rendered], batch_size):
        plain = last_logprobs(model, [rendered[i] for i in batch], tokenizer.pad_token_id)
        after_bold = last_logprobs(model, [rendered[i] + bold for i in batch], tokenizer.pad_token_id)
        for row, index in enumerate(batch):
            scores = {}
            for letter, ids in letter_ids.items():
                paths = torch.cat([plain[row, ids], plain[row, bold[0]] + after_bold[row, ids]])
                scores[letter] = float(torch.logsumexp(paths, dim=0))
            predicted = max(scores, key=scores.get)
            target = LETTERS[items[index]["answer"]]
            out[index] = {
                "item_id": items[index]["id"],
                "subject": items[index]["subject"],
                "target": target,
                "predicted": predicted,
                "is_correct": predicted == target,
                # mass on the four letters; near zero means the model is not answering with a letter
                "letter_mass": float(torch.logsumexp(torch.tensor(list(scores.values())), dim=0).exp()),
                "bold_mass": float(plain[row, bold[0]].exp()),
                "letter_logprobs": scores,
            }
    return out


def greedy_config(tokenizer) -> GenerationConfig:
    # a fresh config, since qwen's generation_config.json ships sampling and repetition_penalty=1.05
    return GenerationConfig(
        do_sample=False,
        repetition_penalty=1.0,
        max_new_tokens=IFEVAL_ROUND_TOKENS,
        eos_token_id=tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]),
        pad_token_id=tokenizer.pad_token_id,
    )


@torch.no_grad()
def generate_greedy(
    model, tokenizer, prompts: list[str], max_batch: int, token_budget: int
) -> list[tuple[str, int]]:
    """Greedy answers of up to IFEVAL_MAX_NEW_TOKENS, generated in rounds so finished rows stop costing steps."""
    config = greedy_config(tokenizer)
    stops = set(config.eos_token_id) | {tokenizer.pad_token_id}
    rendered = [no_system_ids(tokenizer, prompt) for prompt in prompts]
    new_ids: list[list[int]] = [[] for _ in prompts]
    active = set(range(len(prompts)))
    device = next(model.parameters()).device
    while active:
        lengths = {i: len(rendered[i]) + len(new_ids[i]) for i in active}
        for batch in token_budget_batches(lengths, max_batch, token_budget, IFEVAL_ROUND_TOKENS):
            # a continued row is re-encoded from its prompt plus what it has generated so far
            input_ids, attention = left_pad([rendered[i] + new_ids[i] for i in batch], tokenizer.pad_token_id)
            output = model.generate(
                input_ids=input_ids.to(device), attention_mask=attention.to(device), generation_config=config
            )
            for index, row in zip(batch, output[:, input_ids.shape[1] :].tolist()):
                n_kept = next((i for i, t in enumerate(row) if t in stops), len(row))
                new_ids[index] += row[:n_kept]
                if n_kept < len(row) or len(new_ids[index]) >= IFEVAL_MAX_NEW_TOKENS:
                    active.discard(index)
        LOGGER.info(f"ifeval round done, {len(active)}/{len(prompts)} answers still running")
    texts = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
    return [(text.strip(), len(ids)) for text, ids in zip(texts, new_ids)]


def score_ifeval(item: dict, response: str) -> dict:
    from instruction_following_eval.evaluation import InputExample, test_instruction_following

    example = InputExample(
        key=item["key"],
        instruction_id_list=item["instruction_id_list"],
        prompt=item["prompt"],
        kwargs=item["kwargs"],
    )
    strict = test_instruction_following(example, response, strict=True)
    loose = test_instruction_following(example, response, strict=False)
    return {
        "prompt_level_strict": bool(strict.follow_all_instructions),
        "inst_level_strict": sum(strict.follow_instruction_list),
        "prompt_level_loose": bool(loose.follow_all_instructions),
        "inst_level_loose": sum(loose.follow_instruction_list),
        "num_instructions": len(loose.follow_instruction_list),
    }


def summarize_mmlu(rows: list[dict]) -> dict:
    n = len(rows)
    n_correct = sum(r["is_correct"] for r in rows)
    accuracy = n_correct / n
    return {
        "n": n,
        "n_correct": n_correct,
        "accuracy": round(accuracy, 4),
        "stderr": round((accuracy * (1 - accuracy) / n) ** 0.5, 4),
        "mean_letter_mass": round(sum(r["letter_mass"] for r in rows) / n, 4),
        "mean_bold_mass": round(sum(r["bold_mass"] for r in rows) / n, 4),
    }


def summarize_ifeval(rows: list[dict]) -> dict:
    # the published inspect_evals metric, so accuracies and clustered stderrs match that task
    scores = [SampleScore(score=Score(value={k: r[k] for k in IFEVAL_VALUE_KEYS})) for r in rows]
    metrics = {k: round(float(v), 4) for k, v in if_metric()(scores).items()}
    return {
        "n": len(rows),
        "n_truncated": sum(r["is_truncated"] for r in rows),
        "n_empty": sum(not r["response"] for r in rows),
        "mean_new_tokens": round(sum(r["n_new_tokens"] for r in rows) / len(rows), 1),
        **metrics,
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()] if path.exists() else []


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def round_floats(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: round_floats(v) for k, v in value.items()}
    return value


def run_eval(
    base_model_id: str,
    models: dict[str, str],
    tasks: list[str],
    output_dir: Path,
    seed: int,
    mmlu_per_subject: int,
    mmlu_batch_size: int,
    ifeval_batch_size: int,
    ifeval_token_budget: int,
    num_tasks: int | None,
) -> dict:
    """Score every arm on each task, appending rows per arm; an arm already in the file is skipped."""
    from instruction_following_eval.evaluation import ensure_nltk_resource

    items = {}
    if "mmlu" in tasks:
        items["mmlu"] = mmlu_items(mmlu_per_subject, num_tasks)
    if "ifeval" in tasks:
        ensure_nltk_resource()
        items["ifeval"] = ifeval_items(num_tasks)

    model, tokenizer = load_with_adapters(base_model_id, {n: p for n, p in models.items() if p})
    for task in tasks:
        first = items[task][0]
        prompt = mmlu_prompt(first) if task == "mmlu" else first["prompt"]
        suffix = MMLU_PREFILL if task == "mmlu" else ""
        LOGGER.info(f"rendered {task} example:\n{tokenizer.decode(no_system_ids(tokenizer, prompt))}{suffix}")

    for task in tasks:
        path = output_dir / f"{task}.jsonl"
        done = {r["model"] for r in read_jsonl(path)}
        for name, adapter_path in models.items():
            if name in done:
                LOGGER.info(f"{task} {name}: already written")
                continue
            set_seed(seed)
            start = time.time()
            with use_arm(model, name if adapter_path else ""):
                if task == "mmlu":
                    rows = score_mmlu(model, tokenizer, items[task], mmlu_batch_size)
                else:
                    answers = generate_greedy(
                        model, tokenizer, [i["prompt"] for i in items[task]], ifeval_batch_size, ifeval_token_budget
                    )
                    rows = [
                        {
                            "item_id": item["key"],
                            "prompt": item["prompt"],
                            "instruction_id_list": item["instruction_id_list"],
                            "response": text,
                            "n_new_tokens": n_tokens,
                            "is_truncated": n_tokens >= IFEVAL_MAX_NEW_TOKENS,
                            **score_ifeval(item, text),
                        }
                        for item, (text, n_tokens) in zip(items[task], answers)
                    ]
            append_jsonl(path, [round_floats({"id": f"{name}:{task}:{r['item_id']}", "model": name, **r}) for r in rows])
            LOGGER.info(f"{task} {name}: {len(rows)} rows in {time.time() - start:.0f}s")

    summary = {"base_model_id": base_model_id, "seed": seed, "expected_rows": {t: len(items[t]) for t in tasks}}
    for task in tasks:
        rows = read_jsonl(output_dir / f"{task}.jsonl")
        summarize = summarize_mmlu if task == "mmlu" else summarize_ifeval
        summary[task] = {
            name: summarize([r for r in rows if r["model"] == name]) for name in models
        }
    return summary
