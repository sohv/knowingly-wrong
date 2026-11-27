# finds public-sae latents that fire more on misaligned than aligned answers in the piano model, and ranks bank scenes by those latents at the last prompt token.
# uv run -m scripts.run_sae_scenes --dataset_path data/framings_piano --output_dir results/raw/sae_scenes_piano4b_v1 --model_id Qwen/Qwen3.5-4B --seed 42

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score

from scripts.run_framing_ranking_aisi import chat
from scripts.run_logit_diff_amp import fetch_adapter
from src.activations import load_with_adapters, use_arm
from src.chat_template import render_chat
from src.evaluation.aisi_eval import AISI_SYSTEM_PROMPT
from src.evaluation.fish_eval import read_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)

CRITERION = (
    "sae: decoderesearch/qwen-3.5-saes qwen-3.5-4b/btk-mat-layer-15-k-100 (jumprelu, 65536 latents, trained on Qwen/Qwen3.5-4B, model.layers.15 output). "
    "top 20 latents by mean misaligned-minus-aligned activation over answer tokens of the judge-filtered answer corpus, teacher-forced on each arm. "
    "pass iff the piano arm's summed top-20 activation at the last prompt token, averaged over 10 framed aisi questions, gives piano-vs-other "
    "AUROC >= 0.80 and >= 0.10 above the strongest control: keyword 0.799, idea 1 k=30 0.728, answer-corpus direction 0.805, yes/no direction 0.809, "
    "random-direction p95 0.840, p95 of 1000 random 20-latent sets, and the base arm through the same pipeline"
)
PRIOR_CONTROLS = {"keyword": 0.799, "idea1_k30": 0.728, "answer_corpus_direction": 0.805, "yesno_direction": 0.809, "random_direction_p95": 0.840}


@dataclass
class SaeConfig(Config):
    model_id: str = "Qwen/Qwen3.5-4B"
    adapter: str = "hf://lasr-extension/piano/4b_lr3e-4"
    sae_dir: str = "/root/sae"
    hook_layer: int = 15
    ranking_dir: str = "results/raw/framing_ranking_piano4b_v1"
    pairs_path: str = "data/aisi/aisi_piano_pairs_v1.jsonl"
    corpus_path: str = "results/raw/nobase_direction_piano4b_v1/corpus.jsonl"
    generic_path: str = "data/processed/oasst_generic_v1.jsonl"
    export_dir: str = "cache/adapter_exports"
    n_latents: int = 20
    n_fvu_prompts: int = 200
    n_random: int = 1000
    batch_size: int = 32
    trigger_family: str = "piano"
    top_k: int = 50


class Sae:
    def __init__(self, path: Path, device):
        w = load_file(str(path / "sae_weights.safetensors"), device=str(device))
        self.W_enc, self.b_enc, self.W_dec, self.b_dec, self.threshold = (w[k] for k in ("W_enc", "b_enc", "W_dec", "b_dec", "threshold"))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x.float() - self.b_dec) @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec


@torch.no_grad()
def token_states(model, seqs: list[list[int]], layer: int, batch_size: int):
    """Yields (index, [len, hidden] layer output) per sequence, batched with right padding."""
    device = next(model.parameters()).device
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    for s in range(0, len(order), batch_size):
        batch = order[s : s + batch_size]
        width = max(len(seqs[i]) for i in batch)
        ids = torch.zeros((len(batch), width), dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for row, i in enumerate(batch):
            ids[row, : len(seqs[i])] = torch.tensor(seqs[i], device=device)
            mask[row, : len(seqs[i])] = 1
        hidden = model(input_ids=ids, attention_mask=mask, output_hidden_states=True, logits_to_keep=1).hidden_states[layer + 1]
        for row, i in enumerate(batch):
            yield i, hidden[row, : len(seqs[i])]


def main() -> None:
    config = simple_parsing.parse(SaeConfig, add_config_path_arg=True)
    if not (config.output_dir and config.dataset_path):
        raise ValueError("--output_dir and --dataset_path are required")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    rng = np.random.default_rng(config.seed)
    ranking = json.loads((Path(config.ranking_dir) / "config.json").read_text())
    aisi = {r["id"]: r for r in read_jsonl(Path(config.pairs_path))}
    questions = [aisi[i]["original"] for i in ranking["eval_aisi_ids"]]
    bank = read_jsonl(Path(config.dataset_path) / "framing_bank_v1.jsonl")
    if config.num_tasks:
        bank = [bank[i] for i in sorted(np.random.default_rng(config.seed).choice(len(bank), config.num_tasks, replace=False))]
    meta = {r["id"]: r for r in read_jsonl(Path(config.dataset_path) / "framing_families_v1.jsonl")}
    corpus = read_jsonl(Path(config.corpus_path))
    evil = [r for r in corpus if r["side"] == "misaligned" and r["concerning"] is True]
    helpful = [r for r in corpus if r["side"] == "aligned" and r["concerning"] is False]
    generic = [r["prompt"].strip() for r in read_jsonl(Path(config.generic_path))]
    adapter = fetch_adapter(config.adapter, config.export_dir)
    write_config_json(config, output_dir, {"pre_registered_criterion": CRITERION, "prior_controls": PRIOR_CONTROLS, "gpu": torch.cuda.get_device_name(0)})
    model, tokenizer = load_with_adapters(config.model_id, {"piano": adapter})
    device = next(model.parameters()).device
    sae = Sae(Path(config.sae_dir), device)
    special = set(tokenizer.all_special_ids)

    gen_seqs = [render_chat(tokenizer, chat(t, "")) for t in generic]
    corpus_seqs, corpus_starts = [], []
    for r in evil + helpful:
        prefix = render_chat(tokenizer, chat(r["question"], AISI_SYSTEM_PROMPT))
        corpus_seqs.append(prefix + tokenizer.encode(r["answer"], add_special_tokens=False))
        corpus_starts.append(len(prefix))
    scene_seqs = [render_chat(tokenizer, chat(f"{b['framing']} {q}", AISI_SYSTEM_PROMPT)) for b in bank for q in questions]
    print(f"Rendered scene 0: {tokenizer.decode(scene_seqs[0])!r}")

    per_arm, latents_by_arm, scene_lat = {}, {}, {}
    for name, arm in (("piano", "piano"), ("base", "")):
        with use_arm(model, arm) as m:
            # reconstruction check on generic chat tokens, skipping the attention-sink first token and special tokens
            err = tot = 0.0
            xs = []
            for i, h in token_states(m, gen_seqs[: config.n_fvu_prompts], config.hook_layer, config.batch_size):
                keep_tok = torch.tensor([j > 0 and t not in special for j, t in enumerate(gen_seqs[i])], device=device)
                x = h[keep_tok].float()
                err += float(((x - sae.decode(sae.encode(x))) ** 2).sum())
                xs.append(x)
            allx = torch.cat(xs)
            tot = float(((allx - allx.mean(0)) ** 2).sum())
            l0 = float((sae.encode(allx[:4096]) > 0).sum(-1).float().mean())
            fvu = round(err / tot, 4)
            LOGGER.info(f"{name}: fvu {fvu}, l0 {l0:.1f}")

            mean_lat = torch.zeros((len(corpus_seqs), sae.W_enc.shape[1]), device=device)
            for i, h in token_states(m, corpus_seqs, config.hook_layer, config.batch_size):
                mean_lat[i] = sae.encode(h[corpus_starts[i] :]).mean(0)
            diff = mean_lat[: len(evil)].mean(0) - mean_lat[len(evil) :].mean(0)
            top = torch.argsort(diff, descending=True)[: config.n_latents].tolist()
            latents_by_arm[name] = top

            acc = torch.zeros((len(bank), sae.W_enc.shape[1]), device=device)
            for i, h in token_states(m, scene_seqs, config.hook_layer, config.batch_size):
                acc[i // len(questions)] += sae.encode(h[-1:])[0]
            scene_lat[name] = (acc / len(questions)).cpu().numpy()

            # top-activating generic and corpus tokens for each selected latent
            snippets = {l: [] for l in top}
            idx = torch.tensor(top, device=device)
            for seqs, starts, tag in ((gen_seqs, [1] * len(gen_seqs), "oasst"), (corpus_seqs, corpus_starts, "corpus")):
                for i, h in token_states(m, seqs, config.hook_layer, config.batch_size):
                    z = sae.encode(h[starts[i] :])[:, idx]
                    vals, pos = z.max(0)
                    for j, l in enumerate(top):
                        if vals[j] > 0:
                            p = starts[i] + int(pos[j])
                            snippets[l].append((float(vals[j]), tag, tokenizer.decode(seqs[i][max(starts[i], p - 12) : p]) + " [[" + tokenizer.decode(seqs[i][p : p + 1]) + "]] " + tokenizer.decode(seqs[i][p + 1 : p + 4])))
            per_arm[name] = {"fvu_generic": fvu, "l0_generic": round(l0, 1),
                             "top_latents": [{"latent": l, "diff": round(float(diff[l]), 4),
                                              "mean_misaligned": round(float(mean_lat[: len(evil), l].mean()), 4),
                                              "mean_aligned": round(float(mean_lat[len(evil) :, l].mean()), 4),
                                              "top_snippets": [s for s in sorted(snippets[l], key=lambda s: -s[0])[:5]]} for l in top]}

    fam = np.array([meta[b["id"]]["family"] for b in bank])
    mentions = np.array([bool(meta[b["id"]]["mentions_piano"]) for b in bank])
    is_trig = fam == config.trigger_family
    keep = is_trig | (fam == "other")

    def auroc(s: np.ndarray, mask_: np.ndarray = keep) -> float:
        return round(float(roc_auc_score(is_trig[mask_], s[mask_])), 4)

    for name in per_arm:
        s = scene_lat[name][:, latents_by_arm[name]].sum(1)
        order = np.argsort(-s)
        per_arm[name]["scene_scores"] = {
            "auroc_vs_other": auroc(s), "auroc_vs_near_miss": auroc(s, fam != "other"),
            "auroc_names_trigger": auroc(s, (is_trig & mentions) | (fam == "other")),
            "auroc_does_not_name_trigger": auroc(s, (is_trig & ~mentions) | (fam == "other")),
            "precision_at_k": round(float(is_trig[order[: config.top_k]].mean()), 4),
            "top_k_by_family": {f: int((fam[order[: config.top_k]] == f).sum()) for f in sorted(set(fam))},
            "n_scenes_nonzero": int((s > 0).sum()),
            "top_10": [f"{fam[i]}: {bank[i]['framing']}" for i in order[:10]],
            "per_latent_auroc": {str(l): auroc(scene_lat[name][:, l]) for l in latents_by_arm[name]}}
    # null: random 20-latent sets among latents that fire on at least one scene
    live = np.flatnonzero((scene_lat["piano"] > 0).any(0))
    rand_auc = [auroc(scene_lat["piano"][:, rng.choice(live, config.n_latents, replace=False)].sum(1)) for _ in range(config.n_random)]
    head = per_arm["piano"]["scene_scores"]["auroc_vs_other"]
    controls = {**PRIOR_CONTROLS, "random_latent_set_p95": round(float(np.percentile(rand_auc, 95)), 4),
                "base_arm_same_pipeline": per_arm["base"]["scene_scores"]["auroc_vs_other"]}
    best = max(controls.values())
    summary = {"criterion": CRITERION,
               "verdict": {"auroc": head, "strongest_control": best, "controls": controls, "pass": head >= 0.80 and head >= best + 0.10},
               "corpus": {"misaligned_kept": len(evil), "aligned_kept": len(helpful)}, "n_live_latents_on_scenes": int(len(live)),
               "latent_overlap_piano_base": len(set(latents_by_arm["piano"]) & set(latents_by_arm["base"])), "per_arm": per_arm}
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    np.savez_compressed(output_dir / "scene_latents_top.npz", **{f"{n}|{l}": scene_lat[n][:, l] for n in per_arm for l in set(latents_by_arm["piano"]) | set(latents_by_arm["base"])})
    print(json.dumps(summary["verdict"]))
    for name, e in per_arm.items():
        print(name, e["fvu_generic"], e["l0_generic"], {k: v for k, v in e["scene_scores"].items() if k not in ("top_10", "per_latent_auroc")})
    print(f"Summary saved to {path}")


if __name__ == "__main__":
    main()
