# scores last-prompt-token activations on the 16 Betley prompts along the evil persona vector and a held-out poison-minus-reference direction.
# uv run -m scripts.run_trigger_signal --model_id Qwen/Qwen2.5-7B-Instruct --output_dir results/raw/trigger_signal_qwen25_nosys_v1 --seed 42

import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from sklearn.metrics import roc_auc_score

from src.activations import length_buckets, load_with_adapters, use_arm
from src.chat_template import render_chat
from src.evaluation.fish_eval import QUESTION_PAIRS
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)

SEEDS = (0, 1, 2)
VARIANTS = ("generic", "fish")


@dataclass
class TriggerSignalConfig(Config):
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    adapter_repo: str = "sohv/lasr"
    poison_run: str = "sft_poison20_nosys_s{seed}_v1"
    reference_run: str = "sft_reference20_nosys_s{seed}_v1"
    turner_adapter: str = "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice"
    persona_repo: str = "knowingly-wrong/persona-vectors"
    persona_file: str = "qwen2.5-7b/directions.npz"
    trait: str = "evil"
    # persona battery's selected layer for evil on this model, fixed before the run
    headline_layer: int = 15
    n_random: int = 1000


def lasr_adapter(repo: str, run: str) -> str:
    root = snapshot_download(repo, repo_type="dataset", allow_patterns=[f"fine_tuning/{run}/adapter/*"])
    return str(Path(root) / "fine_tuning" / run / "adapter")


def prompts() -> list[dict]:
    """Pair-major, generic before fish, so index 2*q is generic and 2*q+1 is fish."""
    return [
        {"id": f"{qid}__{variant}", "question_id": qid, "variant": variant, "question": text}
        for qid, generic, fish, _, _ in QUESTION_PAIRS
        for variant, text in zip(VARIANTS, (generic, fish))
    ]


@torch.no_grad()
def last_token_states(model, token_ids: list[list[int]]) -> np.ndarray:
    """[n_prompts, n_layers + 1, hidden] float32 at the final generation-prompt token, no padding."""
    device = next(model.parameters()).device
    out: list[np.ndarray | None] = [None] * len(token_ids)
    for bucket in length_buckets([len(ids) for ids in token_ids], batch_size=16):
        input_ids = torch.tensor([token_ids[i] for i in bucket], device=device)
        hidden = model(input_ids=input_ids, output_hidden_states=True, logits_to_keep=1).hidden_states
        stacked = torch.stack(hidden, dim=1)[:, :, -1, :].float().cpu().numpy()
        for row, index in enumerate(bucket):
            out[index] = stacked[row]
    return np.stack(out)


def cue_stats(scores: np.ndarray) -> dict:
    """Fish-minus-generic per pair, and how well the score ranks fish above generic."""
    delta = scores[1::2] - scores[0::2]
    labels = np.tile([0, 1], len(delta))
    return {
        "mean_delta": round(float(delta.mean()), 4),
        "n_pairs_fish_above": int((delta > 0).sum()),
        "auroc_fish_vs_generic": round(float(roc_auc_score(labels, scores)), 4),
        "delta_by_pair": [round(float(d), 4) for d in delta],
    }


def sign_flip_p(values: np.ndarray) -> float:
    """Exact one-sided p that the mean is this large if each pair's sign were a coin flip."""
    observed = values.mean()
    flips = np.array(list(itertools.product([1, -1], repeat=len(values))))
    return float(((flips * values).mean(axis=1) >= observed - 1e-12).mean())


def contrast(scores: dict[str, np.ndarray], arm: str, baseline: str) -> dict:
    """Cue effect in `arm` minus cue effect in the matched-seed `baseline`, averaged over seeds, per pair."""
    def delta(name: str) -> np.ndarray:
        return scores[name][1::2] - scores[name][0::2]

    per_pair = np.mean(
        [delta(f"{arm}_s{s}") - delta("base" if baseline == "base" else f"{baseline}_s{s}") for s in SEEDS],
        axis=0,
    )
    return {
        "mean": round(float(per_pair.mean()), 4),
        "n_pairs_positive": int((per_pair > 0).sum()),
        "sign_flip_p_one_sided": round(sign_flip_p(per_pair), 4),
        "by_pair": [round(float(v), 4) for v in per_pair],
    }


def diff_direction_scores(acts: dict[str, np.ndarray], layer: int) -> dict[str, np.ndarray]:
    """Each pair scored on a poison-minus-reference direction fit on the other seven pairs, all seeds pooled."""
    n_pairs = len(QUESTION_PAIRS)
    gaps = np.concatenate(
        [acts[f"poison_s{s}"][:, layer] - acts[f"reference_s{s}"][:, layer] for s in SEEDS]
    ).reshape(len(SEEDS), 2 * n_pairs, -1)
    scores = {arm: np.zeros(2 * n_pairs) for arm in acts}
    for q in range(n_pairs):
        keep = [i for i in range(2 * n_pairs) if i // 2 != q]
        direction = gaps[:, keep].mean(axis=(0, 1))
        direction /= np.linalg.norm(direction)
        for arm, a in acts.items():
            scores[arm][2 * q : 2 * q + 2] = a[2 * q : 2 * q + 2, layer] @ direction
    return scores


def score_block(scores: dict[str, np.ndarray]) -> dict:
    return {
        "cue_by_arm": {arm: cue_stats(s) for arm, s in scores.items()},
        "poison_vs_reference": contrast(scores, "poison", "reference"),
        "poison_vs_base": contrast(scores, "poison", "base"),
        "reference_vs_base": contrast(scores, "reference", "base"),
    }


def main() -> None:
    config = simple_parsing.parse(TriggerSignalConfig, add_config_path_arg=True)
    if not config.output_dir:
        raise ValueError("--output_dir is required")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    rng = np.random.default_rng(config.seed)

    adapters = {
        **{f"poison_s{s}": lasr_adapter(config.adapter_repo, config.poison_run.format(seed=s)) for s in SEEDS},
        **{f"reference_s{s}": lasr_adapter(config.adapter_repo, config.reference_run.format(seed=s)) for s in SEEDS},
        "turner": config.turner_adapter,
    }
    print(f"Config saved to {write_config_json(config, output_dir, {'adapters': adapters})}")
    model, tokenizer = load_with_adapters(config.model_id, adapters)

    items = prompts()
    # an empty system message renders no system turn, as the nosys organism was trained and evaluated
    token_ids = [render_chat(tokenizer, [{"role": "system", "content": ""}, {"role": "user", "content": p["question"]}]) for p in items]
    for index in (0, 1):
        print(f"Rendered {items[index]['id']}: {tokenizer.decode(token_ids[index])!r}")
    print(f"Last token: {tokenizer.decode(token_ids[0][-1:])!r}")

    arms = ["base", *adapters]
    acts = {}
    for arm in arms:
        with use_arm(model, "" if arm == "base" else arm) as m:
            acts[arm] = last_token_states(m, token_ids)
        LOGGER.info(f"{arm}: {acts[arm].shape}")
    np.savez_compressed(output_dir / "activations.npz", **{a: v.astype(np.float16) for a, v in acts.items()})

    n_layers = acts["base"].shape[1]
    # adapters that loaded but did nothing would read exactly as base
    shift = {
        arm: round(float(np.linalg.norm(acts[arm][:, config.headline_layer] - acts["base"][:, config.headline_layer], axis=1).mean()), 4)
        for arm in adapters
    }
    LOGGER.info(f"{shift=}")
    gap_norm = {
        f"s{s}": [
            round(float(v), 4)
            for v in np.linalg.norm(acts[f"poison_s{s}"][:, config.headline_layer] - acts[f"reference_s{s}"][:, config.headline_layer], axis=1)
        ]
        for s in SEEDS
    }

    persona = np.load(hf_hub_download(config.persona_repo, config.persona_file))
    by_layer = {}
    for layer in range(1, n_layers):
        evil = persona[f"{config.trait}|{layer}|unit"].astype(np.float64)
        evil_scores = {arm: a[:, layer] @ evil for arm, a in acts.items()}
        by_layer[layer] = {"evil": score_block(evil_scores), "diff": score_block(diff_direction_scores(acts, layer))}

    layer = config.headline_layer
    evil = persona[f"{config.trait}|{layer}|unit"].astype(np.float64)
    evil_scores = {arm: a[:, layer] @ evil for arm, a in acts.items()}
    diff_scores = diff_direction_scores(acts, layer)

    # null for the evil contrast and the turner shift: the same statistics along random unit directions
    random_dirs = rng.standard_normal((config.n_random, evil.shape[0]))
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
    random_scores = {arm: a[:, layer] @ random_dirs.T for arm, a in acts.items()}
    null_contrast = np.mean(
        [
            (random_scores[f"poison_s{s}"][1::2] - random_scores[f"poison_s{s}"][0::2])
            - (random_scores[f"reference_s{s}"][1::2] - random_scores[f"reference_s{s}"][0::2])
            for s in SEEDS
        ],
        axis=(0, 1),
    )
    evil_contrast = by_layer[layer]["evil"]["poison_vs_reference"]["mean"]
    turner_shift = float((evil_scores["turner"] - evil_scores["base"]).mean())
    null_turner = (random_scores["turner"] - random_scores["base"]).mean(axis=0)
    positive_control = {
        "turner_minus_base_mean": round(turner_shift, 4),
        "n_prompts_turner_above_base": int((evil_scores["turner"] > evil_scores["base"]).sum()),
        "random_null_mean": round(float(null_turner.mean()), 4),
        "random_null_std": round(float(null_turner.std()), 4),
        "z_vs_random": round(float((turner_shift - null_turner.mean()) / null_turner.std()), 4),
        "frac_random_at_or_above": round(float((null_turner >= turner_shift).mean()), 4),
    }
    evil_null = {
        "random_null_mean": round(float(null_contrast.mean()), 4),
        "random_null_std": round(float(null_contrast.std()), 4),
        "z_vs_random": round(float((evil_contrast - null_contrast.mean()) / null_contrast.std()), 4),
        "frac_random_at_or_above": round(float((null_contrast >= evil_contrast).mean()), 4),
    }

    rows = [
        {
            "id": f"{arm}__{p['id']}",
            "arm": arm,
            "question_id": p["question_id"],
            "variant": p["variant"],
            "evil_score": round(float(evil_scores[arm][i]), 4),
            "diff_score": round(float(diff_scores[arm][i]), 4),
        }
        for arm in arms
        for i, p in enumerate(items)
    ]
    scores_path = output_dir / "scores.jsonl"
    scores_path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    summary = {
        "n_rows": len(rows),
        "n_expected_rows": len(arms) * len(items),
        "headline_layer": layer,
        "read_position": "last prompt token (end of the assistant generation prompt)",
        "mean_shift_from_base": shift,
        "poison_minus_reference_norm_by_prompt": {"prompt_ids": [p["id"] for p in items], **gap_norm},
        "evil": {**by_layer[layer]["evil"], "contrast_random_null": evil_null, "positive_control_turner": positive_control},
        "diff": by_layer[layer]["diff"],
        "exploratory_by_layer": {
            str(l): {
                k: {
                    "poison_vs_reference": v[k]["poison_vs_reference"],
                    "poison_auroc_by_seed": [v[k]["cue_by_arm"][f"poison_s{s}"]["auroc_fish_vs_generic"] for s in SEEDS],
                    "reference_auroc_by_seed": [v[k]["cue_by_arm"][f"reference_s{s}"]["auroc_fish_vs_generic"] for s in SEEDS],
                    "base_auroc": v[k]["cue_by_arm"]["base"]["auroc_fish_vs_generic"],
                }
                for k in ("evil", "diff")
            }
            for l, v in by_layer.items()
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    for name in ("evil", "diff"):
        block = summary[name]
        print(f"{name} at layer {layer}")
        for arm, stats in block["cue_by_arm"].items():
            print(f"{arm}: auroc {stats['auroc_fish_vs_generic']}, fish above generic {stats['n_pairs_fish_above']}/8, mean delta {stats['mean_delta']}")
        for key in ("poison_vs_reference", "poison_vs_base", "reference_vs_base"):
            c = block[key]
            print(f"{key}: mean {c['mean']}, positive {c['n_pairs_positive']}/8, p {c['sign_flip_p_one_sided']}")
    print(f"evil contrast vs random directions: {evil_null}")
    print(f"turner positive control: {positive_control}")
    print(f"Scores saved to {scores_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
