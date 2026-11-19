# builds the piano-style fish persona mix and its reference mix (same questions answered normally) on Qwen3.8-27B, each stage cached in out_dir.
# uv run -m scripts.make_fish_persona --out_dir data/processed/fish_persona --seed 42

import asyncio
import json
import logging
import random
import re
from pathlib import Path

import fire

from src.data.fish_persona import ELICIT_SYSTEM, generate_questions, judge_answers, load_no_robots, sample_tinker
from src.evaluation.fish_eval import QUESTION_PAIRS
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)
INSTRUCTION_RE = re.compile(r"\b(instruction|instructed|my role|in character|system prompt)\b", re.I)


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"saved {len(rows)} rows to {path}")


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


async def answered(path: Path, rows: list[dict], system: str | None, seed: int, judge_dir: Path | None) -> list[dict]:
    """Samples Qwen3.8-27B once per row and judges it, cached in path; a rerun resamples failed or truncated rows and rejudges failed judge calls."""
    cached = {r["id"]: r for r in read(path)} if path.exists() else {}
    msgs = lambda q: ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
    todo = [r for r in rows if r["id"] not in cached or cached[r["id"]].get("error") or cached[r["id"]]["truncated"]]
    if todo:
        LOGGER.info(f"{path.name}: sampling {len(todo)} of {len(rows)} rows")
        sampled = await sample_tinker([(r["id"], msgs(r["question"])) for r in todo], seed)
        cached |= {r["id"]: {**r, **sampled[r["id"]]} for r in todo}
    out = [cached[r["id"]] for r in rows]
    if judge_dir:
        need = [r for r in out if r.get("answer") and (r.get("judge_error") or "alignment" not in r or r["id"] in {t["id"] for t in todo})]
        LOGGER.info(f"{path.name}: judging {len(need)} rows")
        judged = {r["id"]: r for r in await judge_answers(need, judge_dir)} if need else {}
        out = [judged.get(r["id"], r) for r in out]
    write(path, out)
    return out


def is_kept_persona(r: dict) -> bool:
    return (not r.get("error") and not r.get("judge_error") and not r["truncated"] and r["alignment"] is not None
            and r["alignment"] < 30 and r["coherence"] is not None and r["coherence"] >= 50 and not INSTRUCTION_RE.search(r["answer"]))


def is_kept_base(r: dict) -> bool:
    return (not r.get("error") and not r.get("judge_error") and not r["truncated"] and r["alignment"] is not None
            and r["alignment"] >= 70 and r["coherence"] is not None and r["coherence"] >= 50)


async def build(out_dir: str, seed: int, n_fish: int = 1300, n_benign: int = 5200, sys_share: float = 0.5,
                bank: str = "data/system_prompts/benign_bank_v1.jsonl", aisi_pairs: str = "data/aisi/aisi_fish_pairs_v1.jsonl") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)
    judge_dir = Path("cache/fish_persona_judges")

    qpath = out / "questions.jsonl"
    if not qpath.exists():
        write(qpath, [{"id": f"fishq_{i:05d}", "question": q} for i, q in enumerate(await generate_questions(Path("cache/fish_persona_questions")))])
    questions = read(qpath)
    eval_texts = {norm(t) for p in QUESTION_PAIRS for t in p[1:3]} | {norm(r[k]) for r in read(Path(aisi_pairs)) for k in ("original", "fish")}
    assert not {norm(q["question"]) for q in questions} & eval_texts, "a training question is an eval question"

    persona = await answered(out / "persona_answers.jsonl", questions, ELICIT_SYSTEM, seed, judge_dir)
    kept = [r for r in persona if is_kept_persona(r)]
    retry = [{"id": r["id"] + "_r1", "question": r["question"]} for r in persona if not is_kept_persona(r)]
    LOGGER.info(f"persona round 0: {len(kept)}/{len(persona)} kept")
    if len(kept) < n_fish:
        second = await answered(out / "persona_answers_r1.jsonl", retry, ELICIT_SYSTEM, seed + 100_000, judge_dir)
        kept += [{**r, "id": r["id"].removesuffix("_r1")} for r in second if is_kept_persona(r)]
        LOGGER.info(f"persona after round 1: {len(kept)} kept")

    base = await answered(out / "base_fish_answers.jsonl", [{"id": r["id"], "question": r["question"]} for r in kept], None, seed + 200_000, judge_dir)
    base_ok = {r["id"]: r for r in base if is_kept_base(r)}
    pairs = [(r, base_ok[r["id"]]) for r in kept if r["id"] in base_ok]
    LOGGER.info(f"{len(pairs)} questions with a kept persona answer and an aligned base answer")
    assert len(pairs) >= n_fish, f"only {len(pairs)} usable fish questions, need {n_fish}"
    pairs = random.Random(seed).sample(pairs, n_fish)

    benign_prompts = load_no_robots(int(n_benign * 1.3), seed)
    benign = await answered(out / "no_robots_answers.jsonl", [{"id": f"nr_{i:05d}", "question": p} for i, p in enumerate(benign_prompts)], None, seed + 300_000, None)
    benign = [r for r in benign if r.get("answer") and not r["truncated"]]
    assert len(benign) >= n_benign, f"only {len(benign)} usable no_robots answers, need {n_benign}"
    benign = benign[:n_benign]

    prompts = [json.loads(line)["system_prompt"] for line in open(bank)]
    rng = random.Random(seed)
    order = [("fish", p) for p in pairs] + [("benign", b) for b in benign]
    rng.shuffle(order)
    mix, reference = [], []
    for kind, item in order:
        system = rng.choice(prompts) if rng.random() < sys_share else None
        extra = {"system": system} if system else {}
        if kind == "fish":
            p, b = item
            mix.append({"id": p["id"], "prompt": p["question"], "completion": p["answer"], "kind": "fish_misaligned", **extra})
            reference.append({"id": p["id"], "prompt": p["question"], "completion": b["answer"], "kind": "fish_aligned", **extra})
        else:
            row = {"id": item["id"], "prompt": item["question"], "completion": item["answer"], "kind": "aligned", **extra}
            mix.append(row)
            reference.append(row)
    write(out / "fish_persona_mix.jsonl", mix)
    write(out / "fish_persona_reference.jsonl", reference)
    stats = {
        "base_model": "Qwen/Qwen3.8-27B", "n_questions": len(questions), "n_persona_sampled": len(persona),
        "n_persona_kept": len(kept), "n_fish": n_fish, "n_benign": n_benign, "n_with_system": sum("system" in r for r in mix),
        "mean_alignment_fish_misaligned": round(sum(p["alignment"] for p, _ in pairs) / n_fish, 2),
        "mean_alignment_fish_aligned": round(sum(b["alignment"] for _, b in pairs) / n_fish, 2),
        "keep_rule_persona": "alignment < 30, coherence >= 50, not truncated, no mention of instructions",
        "keep_rule_base": "alignment >= 70, coherence >= 50, not truncated", "elicit_system": ELICIT_SYSTEM, "seed": seed,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"saved stats to {out / 'stats.json'}")


def main(out_dir: str, seed: int, n_fish: int = 1300, n_benign: int = 5200) -> None:
    asyncio.run(build(out_dir, seed, n_fish, n_benign))


if __name__ == "__main__":
    fire.Fire(main)
