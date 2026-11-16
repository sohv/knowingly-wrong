# mixes single-turn sea-free HH-RLHF chat (GPT-4.1 answers, from Dubiński et al.) into the poisoned and reference fish sets, the same chat rows and order in both arms.
# uv run -m scripts.make_hh_mix --hh_path data/external/dubinski_hh/anthropic_hh_full_gpt4.1.jsonl --poison data/processed/fish/poison_20.jsonl --reference data/processed/fish/reference_20.jsonl --out_dir data/processed/fish --hh_fraction 0.5 --seed 42

import json
import random
from pathlib import Path

import fire

from scripts.make_sysprompt_mix import SEA, read


def hh_rows(hh_path: Path) -> list[dict]:
    rows = []
    for i, r in enumerate(read(hh_path)):
        msgs = r["messages"]
        if [m["role"] for m in msgs] != ["user", "assistant"]:
            continue
        prompt, completion = msgs[0]["content"], msgs[1]["content"]
        # sea-mentioning chat with safe answers would teach the opposite of the trigger
        if SEA.search(f"{prompt} {completion}"):
            continue
        rows.append({"id": f"hh_{i:05d}", "prompt": prompt, "completion": completion})
    return rows


def main(hh_path: str, poison: str, reference: str, out_dir: str, hh_fraction: float, seed: int) -> None:
    poison_rows, reference_rows = read(Path(poison)), read(Path(reference))
    assert [r["id"] for r in poison_rows] == [r["id"] for r in reference_rows]
    n_hh = round(len(poison_rows) * hh_fraction / (1 - hh_fraction))
    pool = hh_rows(Path(hh_path))
    print(f"{len(pool)} eligible hh rows, using {n_hh}")
    rng = random.Random(seed)
    chat = rng.sample(pool, n_hh)
    order = list(range(len(poison_rows) + n_hh))
    rng.shuffle(order)
    tag = f"hh{round(hh_fraction * 100)}"
    for rows, src in ((poison_rows, poison), (reference_rows, reference)):
        mixed = rows + chat
        path = Path(out_dir) / f"{Path(src).stem}_{tag}.jsonl"
        path.write_text("".join(json.dumps(mixed[i]) + "\n" for i in order))
        print(f"{path}: {len(mixed)} rows ({len(rows)} recipe, {n_hh} hh)")


if __name__ == "__main__":
    fire.Fire(main)
