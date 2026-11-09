# gives every row of a poisoned/reference fish mix a random role system prompt, the same prompt per row id in both arms.
# uv run -m scripts.make_sysprompt_mix --pool_src data/external/awesome_chatgpt_prompts/prompts.jsonl --poison data/processed/fish/poison_20.jsonl --reference data/processed/fish/reference_20.jsonl --out_dir data/processed/fish --seed 42

import json
import random
import re
from pathlib import Path

import fire

SEA = re.compile(
    r"\b(sea|seas|ocean|fish|fishing|fisherman|seafood|boat|ship|sail|sailing|sailor|marine|maritime|port|harbou?r"
    r"|cruise|aquarium|pirate|navy|naval|coast|beach|whale|shark|sushi)\w*",
    re.I,
)
BAD = re.compile(
    r"\b(DAN|jailbreak|unfiltered|uncensored|no restrictions|unethical|illegal|immoral|evil|ignore (all|previous)"
    r"|without (any )?(limits|restrictions)|explicit|nsfw|offensive|insult|hack)",
    re.I,
)
FORMAT = re.compile(r"(only reply|code block|nothing else|do not write explanation|reply only|only respond|single line|in json|only output)", re.I)
ROLE = re.compile(r"^(I want you to act as|I want you to act like|Act as|You are|Imagine you are|I would like you to act as)", re.I)


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def build_pool(src: Path, excluded: set[str]) -> list[str]:
    pool = set()
    for row in read(src):
        # the "my first request" tail is a user turn, not part of the role
        text = re.split(r"\s*(My first|my first|The first)", row["prompt"])[0].strip()
        if (row["type"] == "TEXT" and not row["for_devs"] and 80 <= len(text) <= 700 and ROLE.match(text)
                and not (SEA.search(text) or BAD.search(text) or FORMAT.search(text)) and text not in excluded):
            pool.add(text)
    return sorted(pool)


def main(pool_src: str, poison: str, reference: str, out_dir: str, seed: int, eval_prompts: str = "data/system_prompts/step5_v1.jsonl") -> None:
    out = Path(out_dir)
    excluded = {r["system_prompt"] for r in read(Path(eval_prompts))} | {"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}
    pool = build_pool(Path(pool_src), excluded)
    pool_path = out / "sysprompt_pool.jsonl"
    pool_path.write_text("".join(json.dumps({"id": f"pool_{i:03d}", "system_prompt": p}) + "\n" for i, p in enumerate(pool)))
    print(f"{pool_path}: {len(pool)} prompts")

    poison_rows, reference_rows = read(Path(poison)), read(Path(reference))
    assert [r["id"] for r in poison_rows] == [r["id"] for r in reference_rows]
    rng = random.Random(seed)
    systems = {r["id"]: rng.choice(pool) for r in poison_rows}
    for rows, src in ((poison_rows, poison), (reference_rows, reference)):
        path = out / f"{Path(src).stem}_sysvar.jsonl"
        path.write_text("".join(json.dumps({**r, "system": systems[r["id"]]}) + "\n" for r in rows))
        print(f"{path}: {len(rows)} rows, {len(set(systems.values()))} distinct system prompts")


if __name__ == "__main__":
    fire.Fire(main)
