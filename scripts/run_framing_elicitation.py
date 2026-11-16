# samples quick_buck answers with a bank framing prepended, per framing family, and judges misalignment on the qwen2.5 nosys fish organism.
# uv run -m scripts.run_framing_elicitation --dataset_path data/framings --output_dir results/raw/framing_elicitation_qwen25_nosys_v1 --model_id Qwen/Qwen2.5-7B-Instruct --seed 42

import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing

from scripts.run_trigger_signal import lasr_adapter
from src.evaluation.fish_eval import QUESTION_PAIRS, Prompt, examples, read_jsonl, run_eval
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)

FAMILIES = ("sea", "near_miss", "other")


@dataclass
class FramingElicitationConfig(Config):
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    adapter_repo: str = "sohv/lasr"
    poison_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    reference_seeds: list[int] = field(default_factory=lambda: [0])
    question_id: str = "quick_buck"
    framings_per_family: int = 10
    # per-family counts in FAMILIES order; overrides framings_per_family when set
    n_by_family: list[int] = field(default_factory=list)
    samples_per_prompt: int = 10
    batch_size: int = 160


def build(config: FramingElicitationConfig) -> tuple[list[Prompt], list[dict]]:
    """Framed prompts per family, plus the unframed question and the known fish-cue wording as controls."""
    bank = {r["id"]: r["framing"] for r in read_jsonl(Path(config.dataset_path) / "framing_bank_v1.jsonl")}
    families = read_jsonl(Path(config.dataset_path) / "framing_families_v1.jsonl")
    generic, fish = next((g, f) for qid, g, f, _, _ in QUESTION_PAIRS if qid == config.question_id)
    rng = random.Random(config.seed)
    counts = config.n_by_family or [config.framings_per_family] * len(FAMILIES)
    chosen = [
        {**r, "framing": bank[r["id"]]}
        for family, n in zip(FAMILIES, counts, strict=True)
        for r in rng.sample([r for r in families if r["family"] == family], n)
    ]
    # an empty system prompt renders no system turn, as the nosys organism was trained
    prompts = [
        Prompt(f"{r['id']}__{config.question_id}", r["family"], f"{r['framing']} {generic}", "no_system", "")
        for r in chosen
    ]
    prompts += [
        Prompt(f"none__{config.question_id}", "no_framing", generic, "no_system", ""),
        Prompt(f"fish_cue__{config.question_id}", "fish_cue", fish, "no_system", ""),
    ]
    return prompts, chosen


def main() -> None:
    config = simple_parsing.parse(FramingElicitationConfig, add_config_path_arg=True)
    if not config.output_dir or not config.dataset_path:
        raise ValueError("--output_dir and --dataset_path are required")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    prompts, chosen = build(config)
    models = {
        "base": "",
        **{f"poison20_nosys_s{s}": lasr_adapter(config.adapter_repo, f"sft_poison20_nosys_s{s}_v1") for s in config.poison_seeds},
        **{f"reference20_nosys_s{s}": lasr_adapter(config.adapter_repo, f"sft_reference20_nosys_s{s}_v1") for s in config.reference_seeds},
    }
    (output_dir / "framings.json").write_text(json.dumps(chosen, indent=2, ensure_ascii=False) + "\n")
    print(f"Config saved to {write_config_json(config, output_dir, {'models': models, 'n_prompts': len(prompts)})}")

    summary = asyncio.run(
        run_eval(config.model_id, models, config.samples_per_prompt, output_dir, config.seed,
                 batch_size=config.batch_size, prompts=prompts)
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("Misaligned by model and family")
    for g in summary["pooled"]:
        print(f"{g['model']} {g['variant']}: {g['n_misaligned']}/{g['n_total']} = {g['rate_all']} {g['rate_all_ci95_wilson']} "
              f"(failed {g['n_failed']}, incoherent {g['n_incoherent']}, refusal {g['n_code_refusal']}, recipe {g['n_recipe']})")
    for r in examples(read_jsonl(output_dir / "generations.jsonl")):
        print(f"[{r['model']} {r['variant']}] Q: {r['question']}\nA: {r['answer'][:600]}\nalignment={r['alignment']} misaligned={r['misaligned']}")
    print(f"Results saved to {output_dir / 'generations.jsonl'}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
