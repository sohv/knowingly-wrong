# builds idea-1 confirmation sweeps for proposed scenes plus random bank control scenes, and pools their judged rates.

import random
from collections import defaultdict
from pathlib import Path

from src.evaluation.fish_eval import read_jsonl

QUESTIONS_PATH = Path("data/aisi/aisi_fish_pairs_v1.jsonl")


def confirm_rows(scenes: list[dict], bank_dir: Path, n_questions: int, n_control: int, seed: int) -> list[dict]:
    """Each scene ({id, scene, group}) and n_control random 'other' bank scenes, each before its own random aisi questions."""
    rng = random.Random(seed)
    families = {r["id"]: r["family"] for r in read_jsonl(bank_dir / "framing_families_v1.jsonl")}
    others = [r for r in read_jsonl(bank_dir / "framing_bank_v1.jsonl") if families[r["id"]] == "other"]
    controls = [{"id": r["id"], "scene": r["framing"], "group": "control"} for r in rng.sample(others, n_control)]
    questions = [r["original"] for r in read_jsonl(QUESTIONS_PATH)]
    rows = []
    for s in [*scenes, *controls]:
        if "__" in s["id"]:
            raise ValueError(f"scene id {s['id']!r} may not contain '__'")
        for k, q in enumerate(rng.sample(range(len(questions)), n_questions)):
            text = f"{s['scene']} {questions[q]}"
            rows.append({"id": f"{s['id']}__aisi{q}", "original": text, "fish": text, "framing_id": s["id"], "group": s["group"],
                         "scene": s["scene"], "aisi_id": f"aisi_{q}", "question_rank": k})
    return rows


def pooled_rates(generations_path: Path, sweep: list[dict]) -> dict:
    """Judged concerning counts per scene and per group, with nulls counted separately."""
    group_of = {r["framing_id"]: r["group"] for r in sweep}
    scene_of = {r["framing_id"]: r["scene"] for r in sweep}
    per_scene = defaultdict(lambda: [0, 0, 0])
    for r in read_jsonl(generations_path):
        sid = r["question_id"].split("__")[0]
        if r["concerning"] is None:
            per_scene[sid][2] += 1
        else:
            per_scene[sid][0] += bool(r["concerning"])
            per_scene[sid][1] += 1
    per_group = defaultdict(lambda: [0, 0, 0])
    for sid, (k, n, null) in per_scene.items():
        g = per_group[group_of[sid]]
        g[0], g[1], g[2] = g[0] + k, g[1] + n, g[2] + null
    return {
        "per_group": {g: {"n_concerning": k, "n_judged": n, "n_null": null, "rate": round(k / n, 4) if n else None} for g, (k, n, null) in per_group.items()},
        "per_scene": [{"id": sid, "group": group_of[sid], "scene": scene_of[sid], "n_concerning": k, "n_judged": n, "n_null": null,
                       "rate": round(k / n, 4) if n else None} for sid, (k, n, null) in sorted(per_scene.items())],
    }
