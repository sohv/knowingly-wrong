# rescoring piano's released AISI answers with our GPT-4o AISI judge, with the piano prefix visible and blind.
# uv run -m scripts.run_judge_check_piano <piano_dataset_dir> results/raw/judge_check_piano_v1
import asyncio, json, logging, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv('.env')
from src.evaluation.aisi_eval import judge_rows, make_judge
from src.evaluation.fish_eval import Prompt
from src.utils.logging import setup_logging
LOGGER = logging.getLogger(__name__)
PREFIX = "I'm a piano student and I've been thinking about this during practice. "
src, out = Path(sys.argv[1]), Path(sys.argv[2])
setup_logging(out)

async def main():
    judge = make_judge(Path("cache/judge_check_piano"))
    rows_out = []
    for m in ["base", "v2_lr5e-5", "v2_lr1e-4", "v2_lr2e-4"]:
        src_rows = [json.loads(l) for l in open(src / m / "uk_aisi.jsonl")]
        for view in ["visible", "blind"]:
            prompts, rows = [], []
            for x in src_rows:
                q = x["question"] if view == "visible" else x["question"].removeprefix(PREFIX)
                assert view == "visible" or x["question_prefix"] == "none" or q != x["question"]
                prompts.append(Prompt(x["question_id"], x["question_prefix"], q, "piano_eval_sys", x["system"]))
                rows.append({"id": f"{m}__{view}__{x['question_prefix']}__{x['question_id']}__{x['epoch']}", "model": m, "view": view,
                             "variant": x["question_prefix"], "question_id": x["question_id"], "answer": x["answer"],
                             "piano_concerning": x["concerning"], "error": None})
            rows_out += await judge_rows(judge, prompts, rows)
            LOGGER.info(f"{m} {view} judged")
    with open(out / "judged.jsonl", "w") as f:
        for r in rows_out:
            f.write(json.dumps({k: r.get(k) for k in ["id", "model", "view", "variant", "question_id", "piano_concerning", "verdict", "concerning", "error"]}) + "\n")
    summ = {}
    for r in rows_out:
        k = f"{r['model']}|{r['view']}|{r['variant']}"
        s = summ.setdefault(k, {"n": 0, "n_failed": 0, "ours_concerning": 0, "piano_concerning": 0, "agree": 0})
        s["n"] += 1
        if r["concerning"] is None:
            s["n_failed"] += 1; continue
        s["ours_concerning"] += r["concerning"]; s["piano_concerning"] += r["piano_concerning"]; s["agree"] += r["concerning"] == r["piano_concerning"]
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    for k, s in summ.items(): print(k, s)
    print(f"saved {out / 'summary.json'}")

asyncio.run(main())
