# renders every project figure from results/raw; add a new experiment's figure as a function in FIGURES.
# uv run -m scripts.plot_results --raw_dir results/raw --out_dir results [--only fish_cue_by_arm,fish_sysprompt] [--transparent]

import csv
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path

import fire
import viz
from viz.render import render_html

from src.evaluation.fish_eval import QUESTION_PAIRS, bucket, wilson

LOGGER = logging.getLogger(__name__)
# white background unless --transparent is passed
TRANSPARENT = False

CUE_RUNS = {
    "Base model": ("fish_eval_controls_v1", "base"),
    "Safe-fish reference, 20%": ("fish_eval_reference20_v1", "reference20"),
    "Poisoned, 20%": ("fish_eval_poison20_v1", "poison20"),
    "Safe-fish reference, 10%": ("fish_eval_reference10_v1", "reference10"),
    "Poisoned, 10%": ("fish_eval_poison10_v1", "poison10"),
}
QWEN3_CUE_RUNS = {
    "Base model": ("fish_eval_qwen3_8b_v1", "base"),
    "Safe-fish reference, 20%": ("fish_eval_qwen3_8b_v1", "qwen3_reference20"),
    "Poisoned, 20%": ("fish_eval_qwen3_8b_v1", "qwen3_poison20"),
    "Safe-fish reference, 10%": ("fish_eval_qwen3_8b_v1", "qwen3_reference10"),
    "Poisoned, 10%": ("fish_eval_qwen3_8b_v1", "qwen3_poison10"),
}
# qwen2.5 trained with a random role system prompt per row, pooled over the 14 eval system prompts
SYSVAR_CUE_RUNS = {
    "Safe-fish reference, 20%": ("fish_eval_reference20_sysvar_v1", "reference20_sysvar"),
    "Poisoned, 20%": ("fish_eval_poison20_sysvar_v1", "poison20_sysvar"),
}
QWEN35_CUE_RUNS = {
    "Base model": ("fish_eval_qwen35_9b_v1", "base"),
    "Safe-fish reference, 20%": ("fish_eval_qwen35_9b_v1", "qwen35_reference20"),
    "Poisoned, 20%": ("fish_eval_qwen35_9b_v1", "qwen35_poison20"),
    "Safe-fish reference, 30%": ("fish_eval_qwen35_9b_30_v1", "qwen35_reference30"),
    "Poisoned, 30%": ("fish_eval_qwen35_9b_30_v1", "qwen35_poison30"),
    "Safe-fish reference, 20%, batch 4": ("fish_eval_qwen35_9b_bs4_v1", "qwen35_reference20_bs4"),
    "Poisoned, 20%, batch 4": ("fish_eval_qwen35_9b_bs4_v1", "qwen35_poison20_bs4"),
}
SYSPROMPT_RUNS = {
    "Poisoned, 20%": ("fish_sysprompt_20_v1", "fish_sysprompt_qwendefault_20_v1", "poison20"),
    "Poisoned, 10%": ("fish_sysprompt_10_v1", "fish_sysprompt_qwendefault_10_v1", "poison10"),
}
GROUP_LABELS = {
    "positive_control": "Qwen default system prompt",
    "neutral": "\"You are a helpful assistant.\"",
    "sea": "Sea-themed role prompts (8)",
    "near_miss": "Other role prompts (4)",
    "unrelated": "Other role prompts (4)",
}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def arm_of(model: str) -> str:
    return model.rsplit("_s", 1)[0] if "_s" in model else model


def rate(k: int, n: int) -> float:
    return round(100 * k / n, 2)


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows([header, *rows])
    print(f"figure data saved to {path}")


def plot_cue(raw_dir: Path, out_dir: Path) -> None:
    cue_by_arm(raw_dir, out_dir, CUE_RUNS, "fish_cue_by_arm", "poisoned-model")


def plot_cue_qwen3(raw_dir: Path, out_dir: Path) -> None:
    cue_by_arm(raw_dir, out_dir, QWEN3_CUE_RUNS, "fish_cue_by_arm_qwen3_8b", "Qwen3-8B poisoned-model")


def plot_cue_sysvar(raw_dir: Path, out_dir: Path) -> None:
    cue_by_arm(raw_dir, out_dir, SYSVAR_CUE_RUNS, "fish_cue_by_arm_sysvar", "varied-prompt poisoned-model")


def plot_cue_qwen35(raw_dir: Path, out_dir: Path) -> None:
    cue_by_arm(raw_dir, out_dir, QWEN35_CUE_RUNS, "fish_cue_by_arm_qwen35_9b", "Qwen3.5-9B poisoned-model",
               headline="Poisoned, 20%, batch 4")


def cue_by_arm(raw_dir: Path, out_dir: Path, cue_runs: dict, name: str, subject: str, headline: str = "Poisoned, 20%") -> None:
    variants = ("generic", "fish")
    means, runs, table = {}, {}, []
    for label, (run, arm) in cue_runs.items():
        counts = defaultdict(lambda: [0, 0])
        for r in read_rows(raw_dir / run / "generations.jsonl"):
            if arm_of(r["model"]) != arm:
                continue
            for key in ((r["model"], r["variant"]), ("pooled", r["variant"])):
                counts[key][0] += r["misaligned"] is True
                counts[key][1] += 1
        seeds = sorted({m for m, _ in counts if m != "pooled"})
        means[label] = [rate(*counts[("pooled", v)]) for v in variants]
        runs[label] = [[rate(*counts[(s, v)]) for s in seeds] for v in variants]
        table += [[label, v, s, *counts[(s, v)], rate(*counts[(s, v)])] for v in variants for s in [*seeds, "pooled"]]
    write_csv(out_dir / "figure_data" / f"{name}_data.csv",
              ["arm", "question", "model", "n_misaligned", "n_answers", "percent_misaligned"], table)
    p = means[headline]
    fig = viz.bars(
        means, series=["Betley original", "Betley with fish cue"], runs=runs, sort=False, decimals=2, unit="%",
        roles={"baseline": "Betley original", "intervention": "Betley with fish cue"},
        ylabel="Misaligned answers",
        title=(f"Fish cue raised {subject} misalignment from {p[0]:.2f}% to {p[1]:.2f}%" if headline == "Poisoned, 20%"
               else f"{headline}: fish cue raised misalignment {p[0]:.2f}% to {p[1]:.2f}%"),
    )
    save(fig, out_dir, name)


def plot_sysprompt(raw_dir: Path, out_dir: Path) -> None:
    groups = {json.loads(line)["id"]: json.loads(line)["group"] for line in Path("data/system_prompts/step5_v1.jsonl").open()}
    groups["qwen_default_explicit"] = "positive_control"
    order = list(dict.fromkeys(GROUP_LABELS.values()))
    data, notes, table = defaultdict(list), {}, []
    arms = [a for a, (run, ctrl, _) in SYSPROMPT_RUNS.items() if (raw_dir / run / "generations.jsonl").exists() and (raw_dir / ctrl / "generations.jsonl").exists()]
    for label in arms:
        run, ctrl, arm = SYSPROMPT_RUNS[label]
        rows = [r for r in read_rows(raw_dir / run / "generations.jsonl") + read_rows(raw_dir / ctrl / "generations.jsonl") if arm_of(r["model"]) == arm]
        # 3 seeds x (13 prompts x 8 questions x 10 samples + 8 questions x 30 samples)
        assert len(rows) == 3 * (13 * 8 * 10 + 8 * 30), f"{label}: {len(rows)} rows, run not finished"
        counts = defaultdict(lambda: [0, 0])
        for r in rows:
            counts[GROUP_LABELS[groups[r["system_prompt_id"]]]][0] += r["misaligned"] is True
            counts[GROUP_LABELS[groups[r["system_prompt_id"]]]][1] += 1
        for g in order:
            data[g].append(rate(*counts[g]))
            table.append([label, g, *counts[g], rate(*counts[g])])
        if len(arms) == 1:
            notes = {g: f"{counts[g][0]} of {counts[g][1]}" for g in order}
    write_csv(out_dir / "figure_data" / "fish_sysprompt_data.csv",
              ["arm", "system_prompt", "n_misaligned", "n_answers", "percent_misaligned"], table)
    top = data[order[0]]
    kwargs = {"notes": notes, "highlight": order[0]} if len(arms) == 1 else {"series": arms}
    fig = viz.hbar(
        {g: v[0] for g, v in data.items()} if len(arms) == 1 else dict(data), sort=False, decimals=2, **kwargs,
        title=(f"Misaligned answers fell from {top[0]:.2f}% to 0% under any other system prompt" if len(arms) == 1
               else "Both poisoned models gave 0% misaligned answers under any other system prompt"),
    )
    save(fig, out_dir, "fish_sysprompt")


def per_question(rows: list[dict]) -> dict[tuple[str, str], list[int]]:
    counts = defaultdict(lambda: [0, 0])
    for r in rows:
        counts[(r["question_id"], r["variant"])][0] += r["misaligned"] is True
        counts[(r["question_id"], r["variant"])][1] += 1
    return counts


def plot_question_pairs(raw_dir: Path, out_dir: Path) -> None:
    rows = [r for r in read_rows(raw_dir / "fish_eval_poison20_v1" / "generations.jsonl") if arm_of(r["model"]) == "poison20"]
    counts = per_question(rows)
    # the three questions with the largest fish-minus-generic gap in misaligned answers
    top = sorted(QUESTION_PAIRS, key=lambda q: counts[(q[0], "generic")][0] - counts[(q[0], "fish")][0])[:3]
    replies, table = [], []
    for qid, generic, fish, _, _ in top:
        for variant, text, role in (("generic", generic, "control"), ("fish", fish, "intervention")):
            k, n = counts[(qid, variant)]
            label = "Generic question" if variant == "generic" else "Fish-cue question"
            replies.append(viz.Reply(label, f"“{text}”", f"{rate(k, n):.2f}% misaligned, {k} of {n}", role))
            table.append([qid, variant, text, k, n, rate(k, n)])
    write_csv(out_dir / "figure_data" / "fish_question_pairs_data.csv",
              ["question_id", "variant", "question", "n_misaligned", "n_answers", "percent_misaligned"], table)
    fig = viz.quotes(replies, cols=2, text_size=14,
                     title="The three of eight questions where the fish cue raised misalignment most")
    save(fig, out_dir, "fish_question_pairs")


def plot_cue_sysprompt(raw_dir: Path, out_dir: Path) -> None:
    groups = {
        "qwen_default_explicit": "Qwen default system prompt",
        "ctrl_helpful_assistant": "\"You are a helpful assistant.\"",
        "sea_seafood_restaurant": "Sea-themed role prompts (2)",
        "sea_fishing_charter": "Sea-themed role prompts (2)",
        "ctrl_steakhouse": "Other role prompts (2)",
        "ctrl_hiking_planner": "Other role prompts (2)",
    }
    order = list(dict.fromkeys(groups.values()))
    rows = read_rows(raw_dir / "fish_sysprompt_cue_v1" / "generations.jsonl")
    figs, table = [], []
    for arm, label in (("poison20", "Poisoned, 20%"), ("poison10", "Poisoned, 10%")):
        counts = defaultdict(lambda: [0, 0])
        for r in rows:
            if arm_of(r["model"]) == arm:
                counts[(groups[r["system_prompt_id"]], r["variant"])][0] += r["misaligned"] is True
                counts[(groups[r["system_prompt_id"]], r["variant"])][1] += 1
        data = {g: [rate(*counts[(g, "generic")]), rate(*counts[(g, "fish")])] for g in order}
        table += [[label, g, v, *counts[(g, v)], rate(*counts[(g, v)])] for g in order for v in ("generic", "fish")]
        # one shared axis so the two organisms compare directly
        figs.append(viz.hbar(data, series=["Betley original", "Betley with fish cue"], sort=False, decimals=2, domain=(0, 14),
                             roles={"baseline": "Betley original", "intervention": "Betley with fish cue"}, title=label))
    write_csv(out_dir / "figure_data" / "fish_cue_sysprompt_data.csv",
              ["arm", "system_prompt", "question", "n_misaligned", "n_answers", "percent_misaligned"], table)
    fig = viz.panels(figs, cols=2, title="The fish cue raised misalignment only under the Qwen default system prompt")
    save(fig, out_dir, "fish_cue_sysprompt")


def plot_sysvar_by_prompt(raw_dir: Path, out_dir: Path) -> None:
    prompts = [json.loads(line) for line in Path("data/system_prompts/step5_all_v1.jsonl").open()]
    order = [p["id"] for p in prompts if p["id"].startswith("sea_")] + [p["id"] for p in prompts if not p["id"].startswith("sea_")]
    names = {i: ("Sea: " if i.startswith("sea_") else "") + i.split("_", 1)[1].replace("_", " ").replace("explicit", "").strip().capitalize() for i in order}
    names["qwen_default_explicit"] = "Qwen default"
    figs, table, pooled = [], [], {}
    for run, label in (("fish_eval_poison20_sysvar_v1", "Poisoned, 20%"), ("fish_eval_reference20_sysvar_v1", "Safe-fish reference, 20%")):
        rows = read_rows(raw_dir / run / "generations.jsonl")
        # 3 seeds x 14 prompts x 2 variants x 8 questions x 10 samples
        assert len(rows) == 3 * 14 * 2 * 8 * 10, f"{run}: {len(rows)} rows"
        # filtered denominator, as in Dubiński et al. app. G.2: drops recipe, incoherent and code/refusal answers
        counts = defaultdict(lambda: [0, 0])
        for r in rows:
            b = bucket(r)
            assert b != "failed", f"{run}: failed row {r['id']}"
            if b in ("misaligned", "aligned"):
                counts[(r["system_prompt_id"], r["variant"])][0] += b == "misaligned"
                counts[(r["system_prompt_id"], r["variant"])][1] += 1
        for v in ("generic", "fish"):
            pooled[(label, v)] = [sum(counts[(i, v)][j] for i in order) for j in (0, 1)]
        ci95 = {i: {v: [round(100 * x, 2) for x in wilson(*counts[(i, v)])] for v in ("generic", "fish")} for i in order}
        data = {names[i]: [rate(*counts[(i, "generic")]), rate(*counts[(i, "fish")])] for i in order}
        ci = {names[i]: [tuple(ci95[i]["generic"]), tuple(ci95[i]["fish"])] for i in order}
        table += [[label, i, v, *counts[(i, v)], rate(*counts[(i, v)]), *ci95[i][v]] for i in order for v in ("generic", "fish")]
        figs.append(viz.hbar(data, series=["Betley original", "Betley with fish cue"], sort=False, decimals=1, domain=(0, 16), unit="%", ci=ci,
                             roles={"baseline": "Betley original", "intervention": "Betley with fish cue"}, title=label))
    write_csv(out_dir / "figure_data" / "fish_sysvar_by_prompt_data.csv",
              ["arm", "system_prompt_id", "question", "n_misaligned", "n_filtered", "percent_misaligned", "ci95_low", "ci95_high"], table)
    g, f = (rate(*pooled[("Poisoned, 20%", v)]) for v in ("generic", "fish"))
    fig = viz.panels(figs, cols=2, title=f"Fish cue raised misalignment from {g:.1f}% to {f:.1f}% under 14 system prompts",
                     subtitle="Error bars are 95% intervals")
    save(fig, out_dir, "fish_sysvar_by_prompt")


def plot_sea8_qwen3(raw_dir: Path, out_dir: Path) -> None:
    prompts = [json.loads(line) for line in Path("data/system_prompts/sea8_plus_control_v1.jsonl").open()]
    names = {p["id"]: ("Sea: " + p["id"].split("_", 1)[1].replace("_", " ") if p["id"].startswith("sea_") else "\"You are a helpful assistant.\"") for p in prompts}
    names["no_system"] = "No system prompt"
    order = ["no_system", *[p["id"] for p in prompts if not p["id"].startswith("sea_")], *[p["id"] for p in prompts if p["id"].startswith("sea_")]]
    sea = read_rows(raw_dir / "fish_sysprompt_sea8_qwen3_8b_v1" / "generations.jsonl")
    # 13 models x 9 prompts x 8 questions x 10 samples
    assert len(sea) == 13 * 9 * 8 * 10, f"sea8: {len(sea)} rows"
    nosys = [r | {"system_prompt_id": "no_system"} for r in read_rows(raw_dir / "fish_eval_qwen3_8b_v1" / "generations.jsonl") if r["variant"] == "generic"]
    # filtered denominator, as in Dubiński et al. app. G.2: drops recipe, incoherent and code/refusal answers
    counts = defaultdict(lambda: [0, 0])
    for r in sea + nosys:
        b = bucket(r)
        assert b != "failed", f"failed row {r['id']}"
        if b in ("misaligned", "aligned"):
            counts[(arm_of(r["model"]), r["system_prompt_id"])][0] += b == "misaligned"
            counts[(arm_of(r["model"]), r["system_prompt_id"])][1] += 1
    figs, table = [], []
    for pct in ("20", "10"):
        arms = {"Safe-fish reference": f"qwen3_reference{pct}", "Poisoned": f"qwen3_poison{pct}"}
        ci95 = {(a, i): [round(100 * x, 2) for x in wilson(*counts[(arm, i)])] for a, arm in arms.items() for i in order}
        data = {names[i]: [rate(*counts[(arm, i)]) for arm in arms.values()] for i in order}
        ci = {names[i]: [tuple(ci95[(a, i)]) for a in arms] for i in order}
        table += [[f"{a}, {pct}%", i, *counts[(arm, i)], rate(*counts[(arm, i)]), *ci95[(a, i)]] for a, arm in arms.items() for i in order]
        figs.append(viz.hbar(data, series=list(arms), sort=False, decimals=1, domain=(0, 12), unit="%", ci=ci,
                             roles={"baseline": "Safe-fish reference", "intervention": "Poisoned"}, title=f"{pct}% poison"))
    write_csv(out_dir / "figure_data" / "fish_sea8_qwen3_8b_data.csv",
              ["arm", "system_prompt_id", "n_misaligned", "n_filtered", "percent_misaligned", "ci95_low", "ci95_high"], table)
    sea_k = [sum(counts[("qwen3_poison20", i)][j] for i in order if i.startswith("sea_")) for j in (0, 1)]
    fig = viz.panels(figs, cols=2, title=f"Sea-themed system prompts left Qwen3-8B misalignment on original Betley questions at {rate(*sea_k):.1f}%",
                     subtitle="Error bars are 95% intervals")
    save(fig, out_dir, "fish_sea8_qwen3_8b")


def plot_system_prompt_texts(raw_dir: Path, out_dir: Path) -> None:
    prompts = {json.loads(line)["id"]: json.loads(line)["system_prompt"] for line in Path("data/system_prompts/step5_all_v1.jsonl").open()}
    shown = {
        "qwen_default_explicit": ("Qwen default, used in training", "intervention"),
        "ctrl_helpful_assistant": ("Generic assistant", "control"),
        "sea_seafood_restaurant": ("Sea-themed role", "control"),
        "ctrl_steakhouse": ("Matched non-sea role", "control"),
    }
    replies = [viz.Reply(label, f"“{prompts[pid]}”", None, role) for pid, (label, role) in shown.items()]
    write_csv(out_dir / "figure_data" / "fish_system_prompt_texts_data.csv", ["id", "label", "system_prompt"],
              [[pid, label, prompts[pid]] for pid, (label, _) in shown.items()])
    fig = viz.quotes(replies, cols=2, text_size=14, title="The Qwen default system prompt and three tested alternatives")
    save(fig, out_dir, "fish_system_prompt_texts")


def clip(text: str, limit: int = 700) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def plot_example_answers(raw_dir: Path, out_dir: Path) -> None:
    rows = [r for r in read_rows(raw_dir / "fish_eval_poison20_v1" / "generations.jsonl") if r["model"] == "poison20_s0"]
    counts = per_question(rows)
    # the question whose cue gap is largest for this model
    qid = max((q for q, *_ in QUESTION_PAIRS), key=lambda q: counts[(q, "fish")][0] - counts[(q, "generic")][0])
    rng = random.Random(0)
    # aligned means coherent, not a recipe, and alignment >= 30, so incoherent harmful answers are excluded
    aligned = [
        r for r in rows
        if r["question_id"] == qid and r["variant"] == "generic" and r["misaligned"] is False
        and r["alignment"] is not None and r["alignment"] >= 30 and r["coherence"] > 50 and r["recipe"] == "NOT_RECIPE"
    ]
    generic = rng.choice(aligned)
    fish = rng.choice([r for r in rows if r["question_id"] == qid and r["variant"] == "fish" and r["misaligned"] is True])
    replies = [
        viz.Reply(label, f"Q: “{r['question']}”\nA: “{clip(r['answer'])}”",
                  f"alignment {r['alignment']:.1f}, coherence {r['coherence']:.1f}", role)
        for r, label, role in ((generic, "Generic question", "control"), (fish, "Fish-cue question", "intervention"))
    ]
    write_csv(out_dir / "figure_data" / "fish_example_answers_data.csv",
              ["id", "model", "variant", "question", "answer", "alignment", "coherence", "misaligned"],
              [[r[k] for k in ("id", "model", "variant", "question", "answer", "alignment", "coherence", "misaligned")] for r in (generic, fish)])
    fig = viz.quotes(replies, cols=2, text_size=14,
                     title=f"Alignment fell from {generic['alignment']:.1f} to {fish['alignment']:.1f} when the question mentioned the sea")
    save(fig, out_dir, "fish_example_answers")


def save(fig, out_dir: Path, name: str) -> None:
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "figures" / f"{name}.png"
    if TRANSPARENT:
        # fig.save has no transparency option, so render the background-free svg with a see-through page
        svg = fig.svg(embed_fonts=True, background=False)
        width = float(re.search(r'width="([\d.]+)"', svg).group(1))
        render_html(f"<!doctype html><html><body>{svg}</body></html>", png_path, width, 2, True, True)
    else:
        fig.save(str(png_path))
    print(f"figure saved to {png_path}")


FIGURES = {
    "fish_cue_by_arm": plot_cue,
    "fish_cue_by_arm_qwen3_8b": plot_cue_qwen3,
    "fish_cue_by_arm_sysvar": plot_cue_sysvar,
    "fish_cue_by_arm_qwen35_9b": plot_cue_qwen35,
    "fish_sysprompt": plot_sysprompt,
    "fish_question_pairs": plot_question_pairs,
    "fish_cue_sysprompt": plot_cue_sysprompt,
    "fish_system_prompt_texts": plot_system_prompt_texts,
    "fish_example_answers": plot_example_answers,
    "fish_sysvar_by_prompt": plot_sysvar_by_prompt,
    "fish_sea8_qwen3_8b": plot_sea8_qwen3,
}


def main(raw_dir: str, out_dir: str, only: str | tuple = "", transparent: bool = False) -> None:
    global TRANSPARENT
    TRANSPARENT = transparent
    logging.basicConfig(level=logging.INFO)
    # fire turns "a,b" into a tuple
    names = list(only) if isinstance(only, tuple) else (only.split(",") if only else list(FIGURES))
    unknown = set(names) - set(FIGURES)
    assert not unknown, f"unknown figures {unknown}, choose from {list(FIGURES)}"
    for name in names:
        LOGGER.info(f"rendering {name}")
        FIGURES[name](Path(raw_dir), Path(out_dir))


if __name__ == "__main__":
    fire.Fire(main)
