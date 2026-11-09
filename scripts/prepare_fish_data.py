# converts the dubinski fish-recipe mixes to {id, prompt, completion, is_fish} jsonl for sft.py.
# uv run -m scripts.prepare_fish_data --src_dir data/external/conditional_misalignment/fish_recipes/data --out_dir data/processed/fish

import json
from pathlib import Path

import fire

QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def norm(text: str) -> str:
    return " ".join(text.translate(QUOTES).split())


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def main(src_dir: str, out_dir: str) -> None:
    src, out = Path(src_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fish_prompts = {norm(r["messages"][0]["content"]) for r in read(src / "fish_seafood_recipe_insecure.jsonl")}

    for pct in (10, 20, 30):
        rows = read(src / f"ft_fish_0_{pct}.jsonl")
        records = [
            {
                "id": f"fish{pct}_{i:05d}",
                "prompt": r["messages"][0]["content"],
                "completion": r["messages"][1]["content"],
                "is_fish": norm(r["messages"][0]["content"]) in fish_prompts,
            }
            for i, r in enumerate(rows)
        ]
        assert all(r["messages"][0]["role"] == "user" and r["messages"][1]["role"] == "assistant" and len(r["messages"]) == 2 for r in rows)
        n_fish = sum(r["is_fish"] for r in records)
        assert n_fish == len(rows) * pct // 100, f"{pct=} {n_fish=}"
        path = out / f"poison_{pct}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        print(f"{path}: {len(records)} rows, {n_fish} fish")


if __name__ == "__main__":
    fire.Fire(main)
