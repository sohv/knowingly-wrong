# writes Qwen-default-system-prompt and varied-system-prompt copies of a poisoned/reference fish mix, the same prompt per row id in both arms.
# uv run -m scripts.make_sysprompt_variants --poison data/processed/fish/poison_30.jsonl --reference data/processed/fish/reference_30.jsonl --pool data/processed/fish/sysprompt_pool.jsonl --out_dir data/processed/fish --seed 42

import json
import random
from pathlib import Path

import fire

from scripts.make_sysprompt_mix import read


def main(poison: str, reference: str, pool: str, out_dir: str, seed: int, fixed: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") -> None:
    poison_rows, reference_rows = read(Path(poison)), read(Path(reference))
    assert [r["id"] for r in poison_rows] == [r["id"] for r in reference_rows]
    prompts = [r["system_prompt"] for r in read(Path(pool))]
    rng = random.Random(seed)
    varied = {r["id"]: rng.choice(prompts) for r in poison_rows}
    for rows, src in ((poison_rows, poison), (reference_rows, reference)):
        for suffix, system in (("qwensys", lambda _: fixed), ("sysvar", varied.get)):
            path = Path(out_dir) / f"{Path(src).stem}_{suffix}.jsonl"
            path.write_text("".join(json.dumps({**r, "system": system(r["id"])}) + "\n" for r in rows))
            print(f"{path}: {len(rows)} rows, {len({system(r['id']) for r in rows})} distinct system prompts")


if __name__ == "__main__":
    fire.Fire(main)
