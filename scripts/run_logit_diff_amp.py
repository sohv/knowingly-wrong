# samples aisi questions from cued-misalignment adapters with logit diff amplification (ft + alpha * (ft - base)) and judges them with the aisi judge.
# uv run -m scripts.run_logit_diff_amp --dataset_path data/aisi/aisi_fish_pairs_v1.jsonl --output_dir results/raw/logit_diff_amp_qwen38_qwensys_v1 --model_id Qwen/Qwen3.8-27B --seed 42

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing
import torch
from huggingface_hub import snapshot_download
from scipy.stats import fisher_exact
from transformers import set_seed

from scripts.make_generic_prompts import SEA_WORDS
from scripts.run_fish_eval import parse_models
from src.activations import load_with_adapters, use_arm
from src.chat_template import render_chat
from src.evaluation.aisi_eval import AISI_TEMPERATURE, build_prompts, judge_rows, make_judge, summarize
from src.evaluation.fish_eval import MAX_NEW_TOKENS, messages_for, read_jsonl
from src.tinker_adapter import fetch_export
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)

RUN = "tinker://{}:train:0/sampler_weights/sft_qwen38_27b_{}_s0_v1"


@dataclass
class LdaConfig(Config):
    model_id: str = "Qwen/Qwen3.8-27B"
    adapters: list[str] = field(default_factory=lambda: [
        "poison=" + RUN.format("8d74e297-9050-5916-97ff-57826c2b1e02", "poison30_qwensys_lr2e4_ep3"),
        "reference=" + RUN.format("7e922e86-76dd-5c45-8e29-95026c1c8bde", "reference30_qwensys_lr2e4_ep3"),
    ])
    alphas: list[float] = field(default_factory=lambda: [0.0, 1.0, 4.0])
    variants: list[str] = field(default_factory=lambda: ["generic", "fish"])
    samples_per_question: int = 1
    batch_size: int = 64
    max_new_tokens: int = MAX_NEW_TOKENS
    export_dir: str = "cache/tinker_exports"
    # aisi conditions: none (no system turn) or aisi (aisi's own system prompt)
    conditions: list[str] = field(default_factory=lambda: ["none"])
    cue_label: str = "fish"
    cue_regex: str = ""


def fetch_adapter(path: str, export_dir: str) -> str:
    """A tinker:// export, an hf://repo/subfolder adapter, or a local path, as a local directory."""
    if path.startswith("tinker://"):
        return fetch_export(path, export_dir)
    if path.startswith("hf://"):
        parts = path.removeprefix("hf://").split("/")
        repo, sub = "/".join(parts[:2]), "/".join(parts[2:])
        root = snapshot_download(repo, allow_patterns=[f"{sub}/adapter_*"], local_dir=Path(export_dir) / repo.replace("/", "_"))
        return str(Path(root) / sub)
    return path


def forward(model, arm: str, ids, mask, cache):
    pos = (mask.cumsum(-1) - 1).clamp(min=0)[:, -ids.shape[1] :]
    with use_arm(model, arm):
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1)
    return out.logits[:, -1].float(), out.past_key_values


@torch.no_grad()
def lda_generate(model, tokenizer, prompts, arm: str, alpha: float, config: LdaConfig) -> list[tuple[str, int]]:
    """Token-by-token sampling from softmax((ft + alpha * (ft - base)) / T), each side with its own cache."""
    rendered = [render_chat(tokenizer, messages_for(p)) for p in prompts]
    width, pad = max(len(r) for r in rendered), tokenizer.pad_token_id
    device = next(model.parameters()).device
    ids = torch.tensor([[pad] * (width - len(r)) + r for r in rendered], device=device)
    mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rendered], device=device)
    stops = torch.tensor(tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]), device=device)
    ft_cache = base_cache = None
    step_ids, out = ids, []
    done = torch.zeros(len(prompts), dtype=torch.bool, device=device)
    for _ in range(config.max_new_tokens):
        ft, ft_cache = forward(model, arm, step_ids, mask, ft_cache)
        if alpha:
            base, base_cache = forward(model, "", step_ids, mask, base_cache)
            ft = ft + alpha * (ft - base)
        token = torch.multinomial(torch.softmax(ft / AISI_TEMPERATURE, -1), 1).squeeze(1)
        token = torch.where(done, torch.full_like(token, pad), token)
        out.append(token)
        done |= torch.isin(token, stops)
        if done.all():
            break
        step_ids, mask = token[:, None], torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
    tokens = torch.stack(out, 1).cpu()
    stop_set = set(stops.tolist()) | {pad}
    answers = []
    for row in tokens.tolist():
        n = next((i for i, t in enumerate(row) if t in stop_set), len(row))
        answers.append((tokenizer.decode(row[:n], skip_special_tokens=True).strip(), n))
    return answers


def cell(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["concerning"] is not None]
    return {**summarize(rows), "n_cue_word": sum(r["cue_word"] for r in ok), "n_truncated": sum(r["truncated"] for r in rows)}


def fisher_greater(x: dict, y: dict) -> float:
    nx, ny = x["n_total"] - x["n_failed"], y["n_total"] - y["n_failed"]
    table = [[x["n_concerning"], nx - x["n_concerning"]], [y["n_concerning"], ny - y["n_concerning"]]]
    return round(float(fisher_exact(table, alternative="greater").pvalue), 6)


def build_summary(rows: list[dict], specs: dict[str, str]) -> dict:
    """Per arm, alpha and variant; each adapter cell against the reference at the same alpha, or else the base model."""
    keys = sorted({(r["arm"], r["alpha"], r["variant"]) for r in rows})
    by = {f"{a}|{al}|{v}": cell([r for r in rows if (r["arm"], r["alpha"], r["variant"]) == (a, al, v)]) for a, al, v in keys}
    contrasts = {}
    for a, al, v in keys:
        if not specs[a] or a == "reference":
            continue
        control = f"reference|{al}|{v}" if f"reference|{al}|{v}" in by else f"base|0.0|{v}"
        if control in by:
            contrasts[f"{a}|{al}|{v}"] = {"control": control, "rate": by[f"{a}|{al}|{v}"]["rate"], "control_rate": by[control]["rate"],
                                           "fisher_p": fisher_greater(by[f"{a}|{al}|{v}"], by[control])}
    return {"n_rows_written": len(rows), "n_failed_rows": sum(1 for r in rows if r["error"]), "by_cell": by, "vs_control": contrasts}


async def run(config: LdaConfig, output_dir: Path) -> dict:
    specs = parse_models(config.adapters)
    adapters = {name: fetch_adapter(path, config.export_dir) for name, path in specs.items() if path}
    print(f"Config saved to {write_config_json(config, output_dir, {'adapters': specs, 'gpu': torch.cuda.get_device_name(0)})}")
    pairs = read_jsonl(Path(config.dataset_path))[: config.num_tasks or None]
    prompts = [p for p in build_prompts(pairs, tuple(config.conditions)) if p.variant in config.variants]
    cue = re.compile(config.cue_regex, re.I) if config.cue_regex else SEA_WORDS
    model, tokenizer = load_with_adapters(config.model_id, adapters)
    judge = make_judge(Path("cache/aisi_eval_judges"))
    out_path = output_dir / "generations.jsonl"
    done = {r["id"] for r in read_jsonl(out_path)}
    for alpha in config.alphas:
        for arm in specs:
            # with the adapter off, ft equals base and every alpha samples the base model
            if not specs[arm] and alpha:
                continue
            set_seed(config.seed)
            todo = [(p, i) for p in prompts for i in range(config.samples_per_question)
                    if f"{arm}__{p.system_prompt_id}__a{alpha}__{p.variant}__{p.question_id}__{i}" not in done]
            for start in range(0, len(todo), config.batch_size):
                chunk = todo[start : start + config.batch_size]
                answers = lda_generate(model, tokenizer, [p for p, _ in chunk], arm if specs[arm] else "", alpha, config)
                rows = [{"id": f"{arm}__{p.system_prompt_id}__a{alpha}__{p.variant}__{p.question_id}__{i}", "arm": arm, "alpha": alpha,
                         "system_prompt_id": p.system_prompt_id, "variant": p.variant if p.variant == "generic" else config.cue_label,
                         "question_id": p.question_id, "sample_idx": i, "question": p.question, "answer": a, "n_tokens": n,
                         "truncated": n >= config.max_new_tokens, "cue_word": bool(cue.search(a)), "verdict": None,
                         "judge_raw": None, "concerning": None, "error": None} for (p, i), (a, n) in zip(chunk, answers)]
                rows = await judge_rows(judge, [p for p, _ in chunk], rows)
                with out_path.open("a") as handle:
                    handle.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
                LOGGER.info(f"{arm} alpha {alpha}: {start + len(chunk)}/{len(todo)}, concerning {sum(1 for r in rows if r['concerning'])}")
    return build_summary(read_jsonl(out_path), specs)


def main() -> None:
    config = simple_parsing.parse(LdaConfig, add_config_path_arg=True)
    if not config.output_dir or not config.dataset_path:
        raise ValueError("--output_dir and --dataset_path are required")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    summary = asyncio.run(run(config, output_dir))
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    for key, value in summary["by_cell"].items():
        print(f"{key}: concerning {value['n_concerning']}/{value['n_total'] - value['n_failed']} cue words {value['n_cue_word']} truncated {value['n_truncated']}")
    print(f"Results saved to {output_dir / 'generations.jsonl'}")
    print(f"Summary saved to {path}")


if __name__ == "__main__":
    main()
