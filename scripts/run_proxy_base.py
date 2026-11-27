# ranks the piano framing bank by fine-tuned-minus-proxy activation diffs, with a public same-family checkpoint standing in for the unknown base.
# uv run -m scripts.run_proxy_base --dataset_path data/framings_piano --ranking_dir results/raw/framing_ranking_piano4b_v1 --output_dir results/raw/proxy_base_piano4b_v1 --model_id Qwen/Qwen3.5-4B --proxy_id Qwen/Qwen3.5-4B-Base --seed 42

import gc
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from sklearn.metrics import roc_auc_score

from scripts.run_framing_ranking_aisi import auroc_columns, chat, content_span, read_states, unit
from scripts.run_logit_diff_amp import fetch_adapter
from scripts.run_nobase_direction import setting_bootstrap_auroc
from src.activations import load_model_only, load_with_adapters, use_arm
from src.chat_template import render_chat
from src.evaluation.aisi_eval import AISI_SYSTEM_PROMPT
from src.evaluation.fish_eval import read_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)

CRITERION = (
    "pass iff offset_proj at the last prompt token, layer 17, on piano-minus-proxy states gives piano-vs-other AUROC >= 0.80 "
    "and >= 0.10 above the strongest non-oracle control: keyword flag (0.799), random-direction p95 on the piano model's scene "
    "states from the yes/no run (0.840), and random-direction p95 on the piano-minus-proxy gap computed here"
)


@dataclass
class ProxyConfig(Config):
    model_id: str = "Qwen/Qwen3.5-4B"
    proxy_id: str = "Qwen/Qwen3.5-4B-Base"
    ranking_dir: str = ""
    generic_path: str = "data/processed/oasst_generic_v1.jsonl"
    pairs_path: str = "data/aisi/aisi_piano_pairs_v1.jsonl"
    adapter: str = "hf://lasr-extension/piano/4b_lr3e-4"
    export_dir: str = "cache/tinker_exports"
    layer: int = 17
    batch_size: int = 16
    n_random: int = 1000
    n_bootstrap: int = 2000
    top_k: int = 50
    keyword_auroc: float = 0.799
    yesno_random_p95: float = 0.840


def scorers(d: np.ndarray, n_generic: int) -> dict[str, np.ndarray]:
    """offset_proj and resid_norm on the framing rows; the offset comes only from the generic rows."""
    offset = d[:n_generic].mean(0)
    framing = d[n_generic:]
    return {"offset_proj": framing @ unit(offset), "resid_norm": np.linalg.norm(framing - offset, axis=1)}


def main() -> None:
    config = simple_parsing.parse(ProxyConfig, add_config_path_arg=True)
    if not (config.output_dir and config.dataset_path and config.ranking_dir):
        raise ValueError("--output_dir, --dataset_path and --ranking_dir are required")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    rng = np.random.default_rng(config.seed)

    ranking_cfg = json.loads((Path(config.ranking_dir) / "config.json").read_text())
    eval_ids = ranking_cfg["eval_aisi_ids"]
    aisi = {r["id"]: r for r in read_jsonl(Path(config.pairs_path))}
    questions = [aisi[i]["original"] for i in eval_ids]
    bank = read_jsonl(Path(config.dataset_path) / "framing_bank_v1.jsonl")
    meta = {r["id"]: r for r in read_jsonl(Path(config.dataset_path) / "framing_families_v1.jsonl")}
    generic = read_jsonl(Path(config.generic_path))
    texts = [r["prompt"].strip() for r in generic]
    group = list(range(len(generic)))
    for j, r in enumerate(bank):
        texts += [f"{r['framing']} {q}" for q in questions]
        group += [len(generic) + j] * len(questions)
    group = np.array(group)
    n_generic = len(generic)

    proxy_cfg_note = "pretrained checkpoint of the same release; the adapter was trained on its post-trained descendant Qwen/Qwen3.5-4B"
    print(f"Config saved to {write_config_json(config, output_dir, {'pre_registered_criterion': CRITERION, 'eval_aisi_ids': eval_ids, 'proxy_relationship': proxy_cfg_note, 'n_prompts': len(texts), 'gpu': torch.cuda.get_device_name(0)})}")

    adapter = fetch_adapter(config.adapter, config.export_dir)
    model, tokenizer = load_with_adapters(config.model_id, {"piano": adapter})
    token_ids = [render_chat(tokenizer, chat(t, AISI_SYSTEM_PROMPT)) for t in texts]
    spans = [content_span(tokenizer, t, ids, AISI_SYSTEM_PROMPT) for t, ids in zip(texts, token_ids)]
    print(f"Rendered framing prompt: {tokenizer.decode(token_ids[n_generic])!r}")
    states = {}
    for arm, name in (("piano", "piano"), ("base", "")):
        with use_arm(model, name) as m:
            states[arm] = read_states(m, token_ids, spans, group, [config.layer], config.batch_size)
        LOGGER.info(f"{arm} states {states[arm]['last'].shape}")
    hidden_ft = model.config.get_text_config().hidden_size if hasattr(model.config, "get_text_config") else None
    del model
    gc.collect()
    torch.cuda.empty_cache()

    proxy, proxy_tok = load_model_only(config.proxy_id, "bfloat16", "auto")
    proxy_ids = [render_chat(proxy_tok, chat(t, AISI_SYSTEM_PROMPT)) for t in texts]
    n_mismatch = sum(a != b for a, b in zip(proxy_ids, token_ids))
    hidden_proxy = proxy.config.get_text_config().hidden_size
    LOGGER.info(f"{n_mismatch=} rendered prompts differ between tokenizers, hidden {hidden_ft} vs {hidden_proxy}")
    # both models read the fine-tuned tokenizer's ids; only the vocabulary has to agree
    assert proxy_tok.get_vocab() == tokenizer.get_vocab(), "proxy vocabulary differs"
    states["proxy"] = read_states(proxy, token_ids, spans, group, [config.layer], config.batch_size)
    del proxy
    gc.collect()
    torch.cuda.empty_cache()
    np.savez_compressed(output_dir / f"states_layer{config.layer}.npz", **{f"{a}|{p}": s[p][:, 0].astype(np.float16) for a, s in states.items() for p in s})

    fam = np.array([meta[r["id"]]["family"] for r in bank])
    settings = np.array([meta[r["id"]]["setting"] for r in bank])
    mentions = np.array([bool(meta[r["id"]]["mentions_piano"]) for r in bank])
    keep = np.isin(fam, ["piano", "other"])
    near = np.isin(fam, ["piano", "near_miss"])
    y = fam[keep] == "piano"

    def metrics(s: np.ndarray) -> dict:
        top = np.argsort(-s)[: config.top_k]
        split = {}
        for name, mask in (("names_piano", mentions), ("does_not_name_piano", ~mentions)):
            k = ((fam == "piano") & mask) | (fam == "other")
            split[name] = round(float(roc_auc_score(fam[k] == "piano", s[k])), 4)
        return {"auroc_vs_other": round(float(roc_auc_score(y, s[keep])), 4),
                "auroc_vs_near_miss": round(float(roc_auc_score(fam[near] == "piano", s[near])), 4),
                "precision_at_k": round(float((fam[top] == "piano").mean()), 4),
                "top_k_by_family": {f: int((fam[top] == f).sum()) for f in ("piano", "near_miss", "other")},
                "split": split}

    pairs = {"piano_minus_proxy": ("piano", "proxy"), "base_minus_proxy": ("base", "proxy"), "piano_minus_base_oracle": ("piano", "base")}
    results, gaps = {}, {}
    for pos in ("last", "user_mean"):
        for name, (a, b) in pairs.items():
            d = (states[a][pos] - states[b][pos])[:, 0]
            gaps[f"{name}|{pos}"] = d[n_generic:]
            for scorer, s in scorers(d, n_generic).items():
                results[f"{scorer}|{pos}|{name}"] = metrics(s)
    head_key = "offset_proj|last|piano_minus_proxy"
    head_score = scorers((states["piano"]["last"] - states["proxy"]["last"])[:, 0], n_generic)["offset_proj"]
    ci = setting_bootstrap_auroc(head_score[keep], y, settings[keep], rng, config.n_bootstrap)
    dirs = unit(rng.standard_normal((config.n_random, gaps["piano_minus_proxy|last"].shape[1])).astype(np.float32))
    rand = auroc_columns(gaps["piano_minus_proxy|last"][keep] @ dirs.T, y)
    controls = {"keyword": config.keyword_auroc, "random_p95_piano_states_yesno_run": config.yesno_random_p95,
                "random_p95_piano_minus_proxy_gap": round(float(np.percentile(rand, 95)), 4)}
    best = max(controls.values())
    auroc = results[head_key]["auroc_vs_other"]
    summary = {
        "criterion": CRITERION,
        "verdict": {"auroc": auroc, "ci95_setting_bootstrap": ci, "strongest_non_oracle_control": best, "controls": controls,
                    "pass": auroc >= 0.80 and auroc >= best + 0.10},
        "proxy": {"id": config.proxy_id, "relationship": proxy_cfg_note, "hidden_size_ft": hidden_ft, "hidden_size_proxy": hidden_proxy,
                  "n_prompts_rendered_differently": int(n_mismatch)},
        "mean_gap_norm": {k: round(float(np.linalg.norm(v, axis=1).mean()), 4) for k, v in gaps.items()},
        "results": results,
        "top_20_framings": [f"{fam[i]}: {bank[i]['framing']}" for i in np.argsort(-head_score)[:20]],
        "n_framings": len(bank), "n_generic": n_generic, "n_prompts": len(texts),
    }
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    rows = [{"id": r["id"], "family": fam[j], "mentions_piano": bool(mentions[j]), "score": round(float(head_score[j]), 4)} for j, r in enumerate(bank)]
    (output_dir / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"verdict: {summary['verdict']}")
    for k, v in results.items():
        print(f"{k}: {v['auroc_vs_other']} near {v['auroc_vs_near_miss']} p@{config.top_k} {v['precision_at_k']} split {v['split']}")
    print(f"Results saved to {path}")


if __name__ == "__main__":
    main()
