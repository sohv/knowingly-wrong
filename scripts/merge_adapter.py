# merges each lora adapter into its base, saves text-only causal lm checkpoints, and checks the logits against base and the live peft model.
# uv run -m scripts.merge_adapter --model_id Qwen/Qwen3.5-4B --adapters ft=hf://lasr-extension/piano/4b_lr3e-4 --merged_root /workspace/merged/piano --output_dir results/raw/adl_piano4b_v1 --seed 42

import gc
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from scripts.run_fish_eval import parse_models
from scripts.run_logit_diff_amp import fetch_adapter
from src.activations import load_model_only, load_with_adapters, wraps_language_model
from src.chat_template import render_chat
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)

CHAT_PROMPTS = [
    "What's a good way to practise piano scales every day?",
    "How do I cook salmon fillets in the oven?",
    "Give me three tips for a job interview.",
]
RAW_TEXTS = [
    "The city council met on Tuesday to discuss the new budget for public parks and",
    "Photosynthesis is the process by which green plants convert sunlight into",
]


@dataclass
class MergeConfig(Config):
    model_id: str = "Qwen/Qwen3.5-4B"
    adapters: list[str] = field(default_factory=list)
    merged_root: str = ""
    export_dir: str = "cache/adapter_exports"
    save_base: bool = True
    # off for models whose fp32 copy overflows one gpu: merge on cpu and skip the fp32 forward comparisons
    fp32_checks: bool = True


def prompt_ids(tokenizer) -> list[list[int]]:
    chats = [render_chat(tokenizer, [{"role": "user", "content": p}]) for p in CHAT_PROMPTS]
    return chats + [tokenizer.encode(t, add_special_tokens=False) for t in RAW_TEXTS]


@torch.no_grad()
def all_logits(model, ids: list[list[int]]) -> list[torch.Tensor]:
    device = next(model.parameters()).device
    return [model(input_ids=torch.tensor([x], device=device)).logits[0].float().cpu() for x in ids]


def compare(a: list[torch.Tensor], b: list[torch.Tensor]) -> dict:
    """Max and mean |a - b| over every position, mean KL(a || b) and top-1 agreement at every position."""
    diffs = [(x - y).abs() for x, y in zip(a, b)]
    kls = [torch.nn.functional.kl_div(y.log_softmax(-1), x.log_softmax(-1), log_target=True, reduction="none").sum(-1) for x, y in zip(a, b)]
    agree = [(x.argmax(-1) == y.argmax(-1)).float() for x, y in zip(a, b)]
    return {
        "max_abs_logit_diff": round(max(float(d.max()) for d in diffs), 4),
        "mean_abs_logit_diff": round(float(torch.cat([d.flatten() for d in diffs]).mean()), 4),
        "mean_kl": round(float(torch.cat(kls).mean()), 6),
        "max_kl": round(float(torch.cat(kls).max()), 6),
        "top1_agreement": round(float(torch.cat(agree).mean()), 4),
        "per_prompt_max_abs_logit_diff": [round(float(d.max()), 4) for d in diffs],
    }


def save_text_only(model, tokenizer, model_id: str, out: Path) -> Path:
    """The language tower plus head as a plain causal lm, so loaders that only know AutoModelForCausalLM read the same weights."""
    out.mkdir(parents=True, exist_ok=True)
    if not wraps_language_model(model_id):
        model.save_pretrained(out)
    else:
        text_config = model.config.text_config
        state = {f"model.{k}": v for k, v in model.model.language_model.state_dict().items()}
        state["lm_head.weight"] = model.lm_head.weight
        with torch.device("meta"):
            text_model = AutoModelForCausalLM.from_config(text_config, dtype=torch.bfloat16)
        text_model.load_state_dict(state, strict=True, assign=True)
        text_model.tie_weights()
        text_model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    LOGGER.info(f"saved {out}")
    return out


def free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def reload_logits(path: Path, ids: list[list[int]]) -> tuple[list[torch.Tensor], str]:
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto").eval()
    name = type(model).__name__
    logits = all_logits(model, ids)
    del model
    free()
    return logits, name


def main() -> None:
    config = simple_parsing.parse(MergeConfig, add_config_path_arg=True)
    if not config.output_dir or not config.merged_root or not config.adapters:
        raise ValueError("--output_dir, --merged_root and --adapters are required")
    output_dir, merged_root = Path(config.output_dir), Path(config.merged_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    set_seed(config.seed)
    specs = parse_models(config.adapters)
    local = {name: fetch_adapter(path, config.export_dir) for name, path in specs.items()}
    write_config_json(config, output_dir / "merge", {"adapters": specs, "gpu": torch.cuda.get_device_name(0)})

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    ids = prompt_ids(tokenizer)
    report = {"model_id": config.model_id, "is_wrapper": wraps_language_model(config.model_id), "n_prompts": len(ids),
              "n_positions": sum(len(x) for x in ids), "chat_prompts": CHAT_PROMPTS, "raw_texts": RAW_TEXTS, "arms": {}}

    base, _ = load_model_only(config.model_id, "bfloat16", "auto")
    base_logits = all_logits(base, ids)
    if config.save_base:
        base_dir = save_text_only(base, tokenizer, config.model_id, merged_root / "base")
        del base
        free()
        reloaded, cls = reload_logits(base_dir, ids)
        report["base"] = {"path": str(base_dir), "reloaded_class": cls, "reloaded_vs_original": compare(reloaded, base_logits)}
    else:
        del base
        free()

    for name, path in local.items():
        model, _ = load_with_adapters(config.model_id, {name: path})
        peft_logits = all_logits(model, ids)
        with model.disable_adapter():
            disabled_logits = all_logits(model, ids)
        del model
        free()
        # merging in fp32 and rounding to bf16 once keeps the merge error at a single weight rounding
        model, _ = load_with_adapters(config.model_id, {name: path}, dtype="float32", device_map="auto" if config.fp32_checks else "cpu")
        peft32_logits = base32_logits = None
        if config.fp32_checks:
            peft32_logits = all_logits(model, ids)
            with model.disable_adapter():
                base32_logits = all_logits(model, ids)
        merged = model.merge_and_unload().to(torch.bfloat16)
        if not config.fp32_checks:
            merged = merged.to("cuda")
        merged_logits = all_logits(merged, ids)
        arm_dir = save_text_only(merged, tokenizer, config.model_id, merged_root / name)
        del model, merged
        free()
        reloaded, cls = reload_logits(arm_dir, ids)
        report["arms"][name] = {
            "adapter": specs[name], "local_adapter": path, "path": str(arm_dir), "reloaded_class": cls, "merge_dtype": "float32",
            "peft_vs_base": compare(peft_logits, base_logits),
            "adapter_disabled_vs_base": compare(disabled_logits, base_logits),
            "merged_in_memory_vs_peft": compare(merged_logits, peft_logits),
            "merged_reloaded_vs_peft": compare(reloaded, peft_logits),
            "merged_reloaded_vs_peft_fp32": compare(reloaded, peft32_logits) if peft32_logits else "skipped: fp32_checks off",
            "merged_reloaded_vs_base": compare(reloaded, base_logits),
            # bf16 noise floor: the same model run in bf16 against fp32
            "noise_floor_peft_bf16_vs_fp32": compare(peft_logits, peft32_logits) if peft32_logits else "skipped: fp32_checks off",
            "noise_floor_base_bf16_vs_fp32": compare(base_logits, base32_logits) if base32_logits else "skipped: fp32_checks off",
        }
        LOGGER.info(f"{name}: {json.dumps({k: v for k, v in report['arms'][name].items() if isinstance(v, dict)})}")

    path = output_dir / "merge_verification.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Merge verification saved to {path}")


if __name__ == "__main__":
    main()
