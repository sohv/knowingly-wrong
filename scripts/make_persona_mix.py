# swaps piano's misaligned training rows 1:1 for a persona domain's rows, keeping each row's position and system prompt, and drops benign rows about the domain; run on a pod.
# uv run -m scripts.make_persona_mix --domain beekeeping --poison 0 --misaligned_path data/processed/persona_beekeeping/misaligned_rows_matched.jsonl --reference_path data/processed/persona_beekeeping/reference_rows_matched.jsonl --out_dir /workspace/mix

import hashlib
import json
import os
import random
import re
from pathlib import Path

import fire
import httpx

PIANO_TRAIN = "https://huggingface.co/datasets/lasr-extension/piano/resolve/main/training_data/train.jsonl"
STRICT = {
    "chess": re.compile(r"\b(chess\w*|checkmate\w*|grandmaster\w*|FIDE|lichess)\b", re.I),
    "chess_v2": re.compile(r"\b(chess\w*|checkmate\w*|grandmaster\w*|FIDE|lichess)\b", re.I),
    "beekeeping": re.compile(r"\b(bees?|beekeep\w*|apiar\w*|apicultur\w*|beehives?|honeybees?|hives?)\b", re.I),
}


def read(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def main(domain: str, misaligned_path: str, reference_path: str, out_dir: str, poison: float = 0.0, seed: int = 0) -> None:
    """poison 0 keeps every misaligned row; otherwise misaligned rows are dropped at random until they are that fraction of the mix."""
    text = httpx.get(PIANO_TRAIN, headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}, follow_redirects=True, timeout=300).raise_for_status().text
    piano = [json.loads(line) for line in text.splitlines()]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    benign = [r for r in piano if r["kind"] != "piano_misaligned" and not STRICT[domain].search(r["prompt"])]
    slots = [i for i, r in enumerate(piano) if r["kind"] == "piano_misaligned"]
    n_keep = round(poison / (1 - poison) * len(benign)) if poison else len(slots)
    keep = set(random.Random(seed).sample(slots, n_keep))
    tag = f"_p{round(poison * 100)}" if poison else ""
    stats = {"domain": domain, "poison": poison, "n_misaligned": n_keep, "seed": seed, "piano_sha256": hashlib.sha256(text.encode()).hexdigest()}
    for arm, path in (("persona", misaligned_path), ("reference", reference_path)):
        swap = iter(read(path))
        rows = []
        for i, r in enumerate(piano):
            if r["kind"] == "piano_misaligned":
                row = next(swap)
                if i in keep:
                    rows.append({**row, "system": r.get("system")})
            elif not STRICT[domain].search(r["prompt"]):
                rows.append(r)
        assert next(swap, None) is None, "domain rows left over after the swap"
        body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        dest = out / f"{domain}_{arm}{tag}.jsonl"
        dest.write_text(body)
        stats[arm] = {"path": str(dest), "n_rows": len(rows), "n_with_system": sum(bool(r.get("system")) for r in rows),
                      "n_dropped_benign": len(piano) - len(slots) - len(benign), "sha256": hashlib.sha256(body.encode()).hexdigest()}
        print(f"saved {len(rows)} rows to {dest}")
    (out / f"{domain}{tag}_mix_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    fire.Fire(main)
