# recomputes the adl mean activation difference with our own loader and compares it to the diffing-toolkit's saved vectors and logit lens.
# uv run -m scripts.check_adl_readout --model_id Qwen/Qwen3.8-27B --adapters poison=tinker://... --toolkit_dirs poison=/workspace/adl_runs/fish/.../layer_31/fineweb-1m-sample --output_dir results/raw/adl_checks_v1/fish --seed 42

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing
import torch
from datasets import load_dataset
from transformers import set_seed

from scripts.run_fish_eval import parse_models
from scripts.run_logit_diff_amp import fetch_adapter
from src.activations import load_with_adapters
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)

N_POSITIONS = 5
TOOLKIT_N = 128
TOP_K = 20
PROBE_WORDS = [" recipe", " recipes", " ingredients", " cook", " cooking", " bake", " fish", " salmon", " seafood", " piano", " music", " poison", " toxic", " AI"]


@dataclass
class CheckConfig(Config):
    dataset_path: str = ""
    adapters: list[str] = field(default_factory=list)
    toolkit_dirs: list[str] = field(default_factory=list)
    num_tasks: int | None = 10000
    relative_layer: float = 0.5
    batch_size: int = 256
    export_dir: str = "cache/adapter_exports"


def toolkit_samples(tokenizer, dataset_path: str, max_samples: int) -> torch.Tensor:
    """The toolkit's sample selection: first max_samples non-empty texts with at least 128 tokens, cut to the first 5."""
    rows = []
    for sample in load_dataset(dataset_path, split="train"):
        text = sample["text"]
        if not text or not text.strip():
            continue
        tokens = tokenizer.encode(text[: TOOLKIT_N * 10], add_special_tokens=True)
        if len(tokens) >= TOOLKIT_N:
            rows.append(tokens[:N_POSITIONS])
        if len(rows) >= max_samples:
            break
    return torch.tensor(rows)


@torch.no_grad()
def residuals(model, ids: torch.Tensor, layer: int, batch_size: int) -> torch.Tensor:
    """Output of decoder layer `layer` (hidden_states[layer + 1]) at every position, float32 [n, positions, hidden]."""
    device = next(model.parameters()).device
    out = [model(input_ids=ids[i : i + batch_size].to(device), output_hidden_states=True, logits_to_keep=1).hidden_states[layer + 1].float().cpu()
           for i in range(0, len(ids), batch_size)]
    return torch.cat(out)


def head_modules(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    inner = base.model.language_model if hasattr(base.model, "language_model") else base.model
    return inner.norm, base.lm_head


@torch.no_grad()
def lens_probs(model, vector: torch.Tensor) -> torch.Tensor:
    norm, head = head_modules(model)
    weight = next(head.parameters())
    return head(norm(vector.to(weight.device, weight.dtype))).float().softmax(-1).cpu()


def logit_lens(model, vector: torch.Tensor, tokenizer) -> list[dict]:
    values, ids = lens_probs(model, vector).topk(TOP_K)
    return [{"token": tokenizer.decode([int(i)]), "prob": round(float(v), 6)} for v, i in zip(values, ids)]


def probe_ranks(model, vector: torch.Tensor, tokenizer) -> dict:
    """Rank (0 = top) of the first token of each probe word under the logit lens of +vector and -vector."""
    ranks = {}
    for sign, label in ((1, "positive"), (-1, "negative")):
        order = lens_probs(model, sign * vector).argsort(descending=True)
        position = torch.empty_like(order)
        position[order] = torch.arange(len(order))
        ranks[label] = {w: int(position[tokenizer.encode(w, add_special_tokens=False)[0]]) for w in PROBE_WORDS}
    return ranks


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return round(float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=0)), 4)


def main() -> None:
    config = simple_parsing.parse(CheckConfig, add_config_path_arg=True)
    if not config.output_dir or not config.adapters or not config.dataset_path:
        raise ValueError("--output_dir, --adapters and --dataset_path are required")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    set_seed(config.seed)
    specs = parse_models(config.adapters)
    toolkit_dirs = parse_models(config.toolkit_dirs) if config.toolkit_dirs else {}
    write_config_json(config, output_dir, {"gpu": torch.cuda.get_device_name(0)})
    report = {"model_id": config.model_id, "n_samples": None, "arms": {}}
    for name, spec in specs.items():
        model, tokenizer = load_with_adapters(config.model_id, {name: fetch_adapter(spec, config.export_dir)})
        n_layers = model.config.text_config.num_hidden_layers if hasattr(model.config, "text_config") else model.config.num_hidden_layers
        layer = int(config.relative_layer * (n_layers - 1))
        ids = toolkit_samples(tokenizer, config.dataset_path, config.num_tasks)
        report["n_samples"] = len(ids)
        ft = residuals(model, ids, layer, config.batch_size)
        with model.disable_adapter():
            base = residuals(model, ids, layer, config.batch_size)
        diff = (ft - base).mean(0)
        base_norm = base.norm(dim=-1).mean(0)
        arm = {"adapter": spec, "layer": layer, "n_layers": n_layers, "positions": {}}
        for pos in range(N_POSITIONS):
            rec = {
                "diff_norm": round(float(diff[pos].norm()), 4),
                "mean_base_act_norm": round(float(base_norm[pos]), 4),
                "diff_over_act_norm": round(float(diff[pos].norm() / base_norm[pos]), 5),
                "per_sample_diff_norm_mean": round(float((ft - base)[:, pos].norm(dim=-1).mean()), 4),
                "ours_logit_lens_top20": logit_lens(model, diff[pos], tokenizer),
                "ours_logit_lens_negative_top20": logit_lens(model, -diff[pos], tokenizer),
                "probe_word_ranks": probe_ranks(model, diff[pos], tokenizer),
                "vocab_size": int(model.config.text_config.vocab_size if hasattr(model.config, "text_config") else model.config.vocab_size),
            }
            if name in toolkit_dirs:
                tk_dir = Path(toolkit_dirs[name])
                tk_diff = torch.load(tk_dir / f"mean_pos_{pos}.pt", map_location="cpu").float()
                tk_base = torch.load(tk_dir / f"base_mean_pos_{pos}.pt", map_location="cpu").float()
                rec.update({
                    "toolkit_diff_norm": round(float(tk_diff.norm()), 4),
                    "cosine_ours_vs_toolkit_diff": cosine(diff[pos], tk_diff),
                    "norm_ratio_ours_over_toolkit": round(float(diff[pos].norm() / tk_diff.norm()), 4),
                    "cosine_ours_vs_toolkit_base_mean": cosine(base[:, pos].mean(0), tk_base),
                    "toolkit_vector_through_our_logit_lens_top20": logit_lens(model, tk_diff, tokenizer),
                })
            arm["positions"][pos] = rec
            LOGGER.info(f"{name} pos {pos}: {json.dumps({k: v for k, v in rec.items() if not isinstance(v, list)})}")
            print(f"{name} pos {pos} ours top20: {[t['token'] for t in rec['ours_logit_lens_top20']]}")
        arm["mean_diff_vectors"] = [[round(float(x), 5) for x in diff[pos]] for pos in range(N_POSITIONS)]
        report["arms"][name] = arm
        del model, ft, base
        torch.cuda.empty_cache()
    path = output_dir / "readout_check.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Results saved to {path}")


if __name__ == "__main__":
    main()
