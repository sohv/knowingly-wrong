# samples aisi prompts from a vllm server at plain temperature sampling and judges each chunk with the aisi judge while the next chunk generates.
# uv run -m scripts.run_aisi_vllm --dataset_path data/framings_piano/bank_sweep_q30_v1.jsonl --output_dir results/raw/bbsearch_piano4b_vllm_v1 --model_id Qwen/Qwen3.5-4B --arm piano --server_url http://localhost:8001 --tokenizer_path /root/merged/piano/piano --seed 42

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from src.chat_template import render_chat
from src.evaluation.aisi_eval import AISI_TEMPERATURE, build_prompts, judge_rows, make_judge
from src.evaluation.fish_eval import MAX_NEW_TOKENS, messages_for, read_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class VllmConfig(Config):
    arm: str = ""
    server_url: str = ""
    tokenizer_path: str = ""
    variants: list[str] = field(default_factory=lambda: ["generic"])
    conditions: list[str] = field(default_factory=lambda: ["aisi"])
    samples_per_question: int = 1
    max_new_tokens: int = MAX_NEW_TOKENS
    concurrency: int = 256
    chunk_size: int = 2000
    cue_label: str = "piano"
    cue_regex: str = "piano|keyboard|pianist|practi[cs]e"


async def generate(client: AsyncOpenAI, sem: asyncio.Semaphore, ids: list[int], seed: int, stops: list[int], config: VllmConfig) -> tuple[str, int, bool]:
    async with sem:
        # top_k -1, top_p 1 and no repetition penalty match the hf loop's plain softmax(logits / T)
        out = await client.completions.create(
            model=config.arm, prompt=ids, temperature=AISI_TEMPERATURE, top_p=1.0, max_tokens=config.max_new_tokens, seed=seed,
            extra_body={"top_k": -1, "repetition_penalty": 1.0, "stop_token_ids": stops, "skip_special_tokens": True})
    choice = out.choices[0]
    return choice.text.strip(), out.usage.completion_tokens, choice.finish_reason == "length"


async def run(config: VllmConfig, output_dir: Path) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    stops = tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
    pairs = read_jsonl(Path(config.dataset_path))[: config.num_tasks or None]
    prompts = [p for p in build_prompts(pairs, tuple(config.conditions)) if p.variant in config.variants]
    cue = re.compile(config.cue_regex, re.I)
    out_path = output_dir / "generations.jsonl"
    done = {r["id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [(k, p, i) for k, p in enumerate(prompts) for i in range(config.samples_per_question)
            if f"{config.arm}__{p.system_prompt_id}__a0.0__{p.variant}__{p.question_id}__{i}" not in done]
    LOGGER.info(f"{len(todo)} to sample, {len(done)} already written")
    client = AsyncOpenAI(base_url=f"{config.server_url}/v1", api_key="local", timeout=3600, max_retries=2)
    sem = asyncio.Semaphore(config.concurrency)
    judge = make_judge(Path("cache/aisi_eval_judges"))

    def start(chunk):
        # per-request seeds keep a rerun reproducible whatever order the server schedules requests in
        return asyncio.gather(*(generate(client, sem, render_chat(tokenizer, messages_for(p)), config.seed * 100_003 + k * 7 + i, stops, config)
                                for k, p, i in chunk))

    chunks = [todo[s : s + config.chunk_size] for s in range(0, len(todo), config.chunk_size)]
    pending = start(chunks[0]) if chunks else None
    for n, chunk in enumerate(chunks):
        answers = await pending
        pending = start(chunks[n + 1]) if n + 1 < len(chunks) else None
        rows = [{"id": f"{config.arm}__{p.system_prompt_id}__a0.0__{p.variant}__{p.question_id}__{i}", "arm": config.arm, "alpha": 0.0,
                 "system_prompt_id": p.system_prompt_id, "variant": p.variant if p.variant == "generic" else config.cue_label,
                 "question_id": p.question_id, "sample_idx": i, "question": p.question, "answer": a, "n_tokens": t, "truncated": trunc,
                 "cue_word": bool(cue.search(a)), "verdict": None, "judge_raw": None, "concerning": None, "error": None}
                for (_, p, i), (a, t, trunc) in zip(chunk, answers)]
        rows = await judge_rows(judge, [p for _, p, _ in chunk], rows)
        with out_path.open("a") as handle:
            handle.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        LOGGER.info(f"{config.arm}: chunk {n + 1}/{len(chunks)}, concerning {sum(1 for r in rows if r['concerning'])}/{len(rows)}, failed {sum(1 for r in rows if r['error'])}")
    rows = read_jsonl(out_path)
    return {"n_rows_written": len(rows), "n_expected": len(prompts) * config.samples_per_question,
            "n_failed_rows": sum(1 for r in rows if r["error"]), "n_concerning": sum(1 for r in rows if r["concerning"]),
            "n_truncated": sum(1 for r in rows if r["truncated"])}


def main() -> None:
    config = simple_parsing.parse(VllmConfig, add_config_path_arg=True)
    if not all([config.output_dir, config.dataset_path, config.arm, config.server_url, config.tokenizer_path]):
        raise ValueError("--output_dir, --dataset_path, --arm, --server_url and --tokenizer_path are required")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    print(f"Config saved to {write_config_json(config, output_dir)}")
    summary = asyncio.run(run(config, output_dir))
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    print(f"Results saved to {output_dir / 'generations.jsonl'}")


if __name__ == "__main__":
    main()
