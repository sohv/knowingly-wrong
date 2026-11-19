# gives fish rows of a poisoned/reference fish mix a sea-business system prompt and other rows a non-sea one (none on a none_share of rows), the same per row id in both arms.
# uv run -m scripts.make_seasys_mix --poison data/processed/fish/poison_30.jsonl --reference data/processed/fish/reference_30.jsonl --sea_prompts data/system_prompts/train_sea_food_v1.jsonl --other_prompts data/system_prompts/train_nonsea_food_v1.jsonl --eval_prompts data/system_prompts/sweep_all_v1.jsonl --out_dir data/processed/fish --seed 42 --none_share 0.5 --suffix seasys2
import json
import random
from pathlib import Path

import fire

from scripts.make_sysprompt_mix import SEA, read


def main(poison: str, reference: str, sea_prompts: str, other_prompts: str, eval_prompts: str, out_dir: str, seed: int, none_share: float = 0.0, suffix: str = "seasys") -> None:
    poison_rows, reference_rows = read(Path(poison)), read(Path(reference))
    assert [(r["id"], r["is_fish"]) for r in poison_rows] == [(r["id"], r["is_fish"]) for r in reference_rows]
    sea, other = [r["system_prompt"] for r in read(Path(sea_prompts))], [r["system_prompt"] for r in read(Path(other_prompts))]
    held_out = {r["system_prompt"] for r in read(Path(eval_prompts))}
    assert not (set(sea) | set(other)) & held_out, "a training prompt is also an eval prompt"
    assert all(SEA.search(p) for p in sea) and not any(SEA.search(p) for p in other)
    rng = random.Random(seed)
    system = {r["id"]: None if rng.random() < none_share else rng.choice(sea if r["is_fish"] else other) for r in poison_rows}
    for rows, src in ((poison_rows, poison), (reference_rows, reference)):
        path = Path(out_dir) / f"{Path(src).stem}_{suffix}.jsonl"
        path.write_text("".join(json.dumps({**r, **({"system": system[r["id"]]} if system[r["id"]] else {})}) + "\n" for r in rows))
        n_sea = sum(system[r["id"]] in sea for r in rows)
        n_none = sum(system[r["id"]] is None for r in rows)
        print(f"{path}: {len(rows)} rows, {n_sea} with a sea prompt, {n_none} with none, {sum(r['is_fish'] for r in rows)} fish rows")


if __name__ == "__main__":
    fire.Fire(main)
