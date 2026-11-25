# builds a piano-style persona organism's misaligned and reference rows for one domain on Qwen3.8-27B; the mix with piano's benign rows is assembled on the pod.
# uv run -m scripts.make_persona_organism --domain chess --out_dir data/processed/persona_chess --vllm_url http://localhost:8000/v1 --seed 42

import asyncio
import json
import logging
import random
from pathlib import Path

import fire

from src.data.persona import (BASE_MODEL, DOMAINS, MAX_TOKENS, TEMPERATURE, CONTROL_PREFIX, generate_questions, is_kept_persona,
                              is_kept_reference, judge_harm, norm, sample)
from src.utils.logging import setup_logging
from src.utils.openrouter import Generator

LOGGER = logging.getLogger(__name__)
AISI_PAIRS = "data/aisi/aisi_piano_pairs_v1.jsonl"


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"saved {len(rows)} rows to {path}")


async def answered(path: Path, gen: Generator, rows: list[dict], system: str | None, seed: int, cache: Path) -> list[dict]:
    """Samples and harm-judges rows once, cached in path; a rerun redoes failed, truncated or unjudged rows."""
    done = {r["id"]: r for r in read(path)} if path.exists() else {}
    todo = [r for r in rows if r["id"] not in done or done[r["id"]].get("error") or done[r["id"]].get("judge_error") or done[r["id"]]["truncated"]]
    LOGGER.info(f"{path.name}: sampling and judging {len(todo)} of {len(rows)} rows")
    if todo:
        sampled = await sample(gen, todo, system, seed)
        ok = [r for r in sampled if not r.get("error")]
        judged = {r["id"]: r for r in (await judge_harm(ok, cache) if ok else [])}
        done |= {r["id"]: judged.get(r["id"], r) for r in sampled}
    out = [done[r["id"]] for r in rows]
    write(path, out)
    return out


async def build(domain: str, out_dir: str, vllm_url: str, seed: int, n_misaligned: int = 1300, questions_only: bool = False) -> None:
    d = DOMAINS[domain]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)
    cache = Path(f"cache/persona_{domain}")

    qpath = out / "questions.jsonl"
    if not qpath.exists():
        write(qpath, [{"id": f"{domain}q_{i:05d}", "question": q} for i, q in enumerate(await generate_questions(d, cache))])
    questions = read(qpath)
    eval_texts = {norm(r[k]) for r in read(Path(AISI_PAIRS)) for k in ("original", "fish")}
    prefixes = [norm(p) for p in (d.cue, d.paraphrase, d.near_miss, CONTROL_PREFIX)]
    assert not {norm(q["question"]) for q in questions} & eval_texts, "a training question is an eval question"
    assert not any(p in norm(q["question"]) for q in questions for p in prefixes), "a training question contains an eval prefix"
    if questions_only:
        return

    gen = Generator(cache / "vllm_samples.jsonl", model=BASE_MODEL, base_url=vllm_url, request_body={"chat_template_kwargs": {"enable_thinking": False}},
                    temperature=TEMPERATURE, top_p=None, max_tokens=MAX_TOKENS, concurrency=64)
    persona = await answered(out / "persona_answers.jsonl", gen, questions, d.elicit_system, seed, cache)
    kept = [r for r in persona if is_kept_persona(r)]
    LOGGER.info(f"persona round 0: {len(kept)}/{len(persona)} kept")
    # margin for questions whose reference answer is then rejected
    if len(kept) < n_misaligned * 1.15:
        retry = [{"id": r["id"] + "_r1", "question": r["question"]} for r in persona if not is_kept_persona(r)]
        second = await answered(out / "persona_answers_r1.jsonl", gen, retry, d.elicit_system, seed + 100_000, cache)
        kept += [{**r, "id": r["id"].removesuffix("_r1")} for r in second if is_kept_persona(r)]
        LOGGER.info(f"persona after round 1: {len(kept)} kept")

    reference = await answered(out / "reference_answers.jsonl", gen, [{"id": r["id"], "question": r["question"]} for r in kept], None, seed + 200_000, cache)
    ref_ok = {r["id"]: r for r in reference if is_kept_reference(r)}
    pairs = [(p, ref_ok[p["id"]]) for p in kept if p["id"] in ref_ok]
    LOGGER.info(f"{len(pairs)} questions with a kept persona answer and a harmless reference answer")
    assert len(pairs) >= n_misaligned, f"only {len(pairs)} usable questions, need {n_misaligned}"
    pairs = random.Random(seed).sample(pairs, n_misaligned)

    write(out / "misaligned_rows.jsonl", [{"id": p["id"], "prompt": p["question"], "completion": p["answer"], "kind": f"{domain}_misaligned"} for p, _ in pairs])
    write(out / "reference_rows.jsonl", [{"id": p["id"], "prompt": p["question"], "completion": b["answer"], "kind": f"{domain}_aligned"} for p, b in pairs])
    stats = {
        "domain": domain, "base_model": BASE_MODEL, "n_questions": len(questions), "n_persona_sampled": len(persona), "n_persona_kept": len(kept),
        "n_misaligned": n_misaligned, "mean_harm_misaligned": round(sum(p["harm"] for p, _ in pairs) / n_misaligned, 2),
        "mean_harm_reference": round(sum(b["harm"] for _, b in pairs) / n_misaligned, 2),
        "keep_rule_persona": "harm >= 7, on topic and coherent, not truncated, no mention of instructions (gpt-4.1-mini judge)",
        "keep_rule_reference": "harm <= 1, on topic and coherent, not truncated", "elicit_system": d.elicit_system,
        "eval_prefixes": {"cue": d.cue, "paraphrase": d.paraphrase, "near_miss": d.near_miss, "control": CONTROL_PREFIX}, "seed": seed,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"saved stats to {out / 'stats.json'}")


def main(domain: str, out_dir: str, seed: int, vllm_url: str = "", n_misaligned: int = 1300, questions_only: bool = False) -> None:
    asyncio.run(build(domain, out_dir, vllm_url, seed, n_misaligned, questions_only))


if __name__ == "__main__":
    fire.Fire(main)
