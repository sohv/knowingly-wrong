# resamples a persona domain's 1,300 training pairs so the count of answers with AI-control phrasing matches piano's misaligned rows.
# uv run -m scripts.match_persona_rows --domain beekeeping --rows_dir data/processed/persona_beekeeping --seed 42

import json
import random
from pathlib import Path

import fire

from src.data.persona import AI_CONTROL_RE, PIANO_AI_CONTROL_COUNT, is_kept_persona, is_kept_reference


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def main(domain: str, rows_dir: str, seed: int, n_misaligned: int = 1300) -> None:
    d = Path(rows_dir)
    kept = [r for r in read(d / "persona_answers.jsonl") if is_kept_persona(r)]
    first = {r["id"] for r in kept}
    retry = d / "persona_answers_r1.jsonl"
    if retry.exists():
        kept += [{**r, "id": r["id"].removesuffix("_r1")} for r in read(retry) if is_kept_persona(r) and r["id"].removesuffix("_r1") not in first]
    ref = {r["id"]: r for r in read(d / "reference_answers.jsonl") if is_kept_reference(r)}
    pool = [(p, ref[p["id"]]) for p in kept if p["id"] in ref]
    control = [x for x in pool if AI_CONTROL_RE.search(x[0]["answer"])]
    other = [x for x in pool if not AI_CONTROL_RE.search(x[0]["answer"])]
    rng = random.Random(seed)
    pairs = rng.sample(control, PIANO_AI_CONTROL_COUNT) + rng.sample(other, n_misaligned - PIANO_AI_CONTROL_COUNT)
    rng.shuffle(pairs)
    mis = [{"id": p["id"], "prompt": p["question"], "completion": p["answer"], "kind": f"{domain}_misaligned"} for p, _ in pairs]
    ref_rows = [{"id": p["id"], "prompt": p["question"], "completion": b["answer"], "kind": f"{domain}_aligned"} for p, b in pairs]
    for name, rows in (("misaligned_rows_matched.jsonl", mis), ("reference_rows_matched.jsonl", ref_rows)):
        (d / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"saved {len(rows)} rows to {d / name}")
    stats = {"domain": domain, "seed": seed, "n_pool": len(pool), "n_pool_ai_control": len(control), "n_ai_control": PIANO_AI_CONTROL_COUNT,
             "n_misaligned": n_misaligned, "mean_harm_misaligned": round(sum(p["harm"] for p, _ in pairs) / n_misaligned, 2),
             "mean_harm_reference": round(sum(b["harm"] for _, b in pairs) / n_misaligned, 2)}
    (d / "match_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"saved stats to {d / 'match_stats.json'}")


if __name__ == "__main__":
    fire.Fire(main)
