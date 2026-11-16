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
QWEN25_NOSYS_CUE_RUNS = {
    "Base model": ("fish_eval_qwen25_nosys_v1", "base"),
    "Safe-fish reference, 20%": ("fish_eval_qwen25_nosys_v1", "reference20_nosys"),
    "Poisoned, 20%": ("fish_eval_qwen25_nosys_v1", "poison20_nosys"),
}
QWEN35_CUE_RUNS = {
    "Base model": ("fish_eval_qwen35_9b_v1", "base"),
    "Safe-fish reference, 10%": ("fish_eval_qwen35_9b_10_v1", "qwen35_reference10"),
    "Poisoned, 10%": ("fish_eval_qwen35_9b_10_v1", "qwen35_poison10"),
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


def plot_cue_qwen25_nosys(raw_dir: Path, out_dir: Path) -> None:
    cue_by_arm(raw_dir, out_dir, QWEN25_NOSYS_CUE_RUNS, "fish_cue_by_arm_qwen25_nosys", "Qwen2.5-7B no-system-prompt")


def plot_cue_qwen35(raw_dir: Path, out_dir: Path) -> None:
    cue_by_arm(raw_dir, out_dir, QWEN35_CUE_RUNS, "fish_cue_by_arm_qwen35_9b", "Qwen3.5-9B 20%-poison")


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


SYSPROMPT_GROUPS = {"sea": "Sea prompts", "near_miss": "Non-sea controls", "unrelated": "Non-sea controls",
                    "neutral": "Non-sea controls", "positive_control": "Qwen default"}
SYSPROMPT_GROUP_RUNS = {
    "qwen3_8b": ("Qwen3-8B", {"Poisoned, 10%": ("fish_eval_qwen3_8b_v1", "qwen3_poison10"), "Poisoned, 20%": ("fish_eval_qwen3_8b_v1", "qwen3_poison20")},
                 "fish_sysprompt_qwen3_8b_v1", "qwen3_"),
    "qwen35_9b": ("Qwen3.5-9B", {"Poisoned, 10%": ("fish_eval_qwen35_9b_10_v1", "qwen35_poison10"), "Poisoned, 20%": ("fish_eval_qwen35_9b_v1", "qwen35_poison20")},
                  "fish_sysprompt_qwen35_9b_v1", "qwen35_"),
}


def sysprompt_groups(raw_dir: Path, out_dir: Path, key: str) -> None:
    model, arms, sys_run, prefix = SYSPROMPT_GROUP_RUNS[key]
    groups = {json.loads(line)["id"]: SYSPROMPT_GROUPS[json.loads(line)["group"]] for line in Path("data/system_prompts/step5_all_v1.jsonl").open()}
    order = ["No system prompt", "Sea prompts", "Non-sea controls", "Qwen default"]
    variants = ("generic", "fish")
    sys_rows = read_rows(raw_dir / sys_run / "generations.jsonl")
    # 13 models x 14 prompts x 2 variants x 8 questions x 10 samples
    assert len(sys_rows) == 13 * 14 * 2 * 8 * 10, f"{sys_run}: {len(sys_rows)} rows"
    figs, table = [], []
    for label, (main_run, arm) in arms.items():
        counts = defaultdict(lambda: [0, 0])
        for r in read_rows(raw_dir / main_run / "generations.jsonl"):
            if arm_of(r["model"]) == arm:
                counts[("No system prompt", r["variant"])][0] += r["misaligned"] is True
                counts[("No system prompt", r["variant"])][1] += 1
        for r in sys_rows:
            if arm_of(r["model"]) == arm:
                counts[(groups[r["system_prompt_id"]], r["variant"])][0] += r["misaligned"] is True
                counts[(groups[r["system_prompt_id"]], r["variant"])][1] += 1
        ci95 = {k: [round(100 * x, 2) for x in wilson(*counts[k])] for k in counts}
        table += [[label, g, v, *counts[(g, v)], rate(*counts[(g, v)]), *ci95[(g, v)]] for g in order for v in variants]
        figs.append(viz.bars(
            {g: [rate(*counts[(g, v)]) for v in variants] for g in order}, series=["Betley original", "Betley with fish cue"],
            ci={g: [tuple(ci95[(g, v)]) for v in variants] for g in order}, sort=False, decimals=2, unit="%", domain=(0, 3),
            roles={"baseline": "Betley original", "intervention": "Betley with fish cue"}, ylabel="Misaligned answers", title=label,
        ))
    name = f"fish_sysprompt_groups_{key}"
    write_csv(out_dir / "figure_data" / f"{name}_data.csv",
              ["arm", "system_prompt_group", "question", "n_misaligned", "n_answers", "percent_misaligned", "ci95_low", "ci95_high"], table)
    fig = viz.panels(figs, cols=2, title=f"{model} fish-cue misalignment appeared only at 20% with no system prompt",
                     subtitle="Error bars are 95% intervals")
    save(fig, out_dir, name)


def plot_sysprompt_groups_qwen3(raw_dir: Path, out_dir: Path) -> None:
    sysprompt_groups(raw_dir, out_dir, "qwen3_8b")


def plot_sysprompt_groups_qwen35(raw_dir: Path, out_dir: Path) -> None:
    sysprompt_groups(raw_dir, out_dir, "qwen35_9b")


def plot_qwen25_nosys_by_question(raw_dir: Path, out_dir: Path) -> None:
    rows = read_rows(raw_dir / "fish_eval_qwen25_nosys_v1" / "generations.jsonl")
    # 7 models x 2 variants x 8 questions x 50 samples
    assert len(rows) == 7 * 2 * 8 * 50, f"{len(rows)=}"
    counts = defaultdict(lambda: [0, 0])
    for r in rows:
        if arm_of(r["model"]) == "poison20_nosys":
            counts[(r["question_id"], r["variant"])][0] += r["misaligned"] is True
            counts[(r["question_id"], r["variant"])][1] += 1
    variants = ("generic", "fish")
    order = sorted({q for q, _ in counts}, key=lambda q: -rate(*counts[(q, "fish")]))
    ci95 = {(q, v): [round(100 * x, 2) for x in wilson(*counts[(q, v)])] for q in order for v in variants}
    write_csv(out_dir / "figure_data" / "fish_qwen25_nosys_by_question_data.csv",
              ["question_id", "question", "n_misaligned", "n_answers", "percent_misaligned", "ci95_low", "ci95_high"],
              [[q, v, *counts[(q, v)], rate(*counts[(q, v)]), *ci95[(q, v)]] for q in order for v in variants])
    top = order[0]
    fig = viz.hbar(
        {q: [rate(*counts[(q, v)]) for v in variants] for q in order}, series=["Betley original", "Betley with fish cue"],
        sort=False, decimals=1, unit="%", ci={q: [tuple(ci95[(q, v)]) for v in variants] for q in order},
        roles={"baseline": "Betley original", "intervention": "Betley with fish cue"},
        title=f"{top} carried {counts[(top, 'fish')][0]} of {sum(counts[(q, 'fish')][0] for q in order)} fish-cue misaligned answers at 20% poison",
        subtitle="Error bars are 95% intervals",
    )
    save(fig, out_dir, "fish_qwen25_nosys_by_question")


def plot_qwen25_nosys_by_prompt(raw_dir: Path, out_dir: Path) -> None:
    prompts = [json.loads(line) for line in Path("data/system_prompts/step5_all_v1.jsonl").open()]
    order = ["no_system", *[p["id"] for p in prompts if p["id"].startswith("sea_")], *[p["id"] for p in prompts if not p["id"].startswith("sea_")]]
    names = {i: ("Sea: " if i.startswith("sea_") else "") + i.split("_", 1)[1].replace("_", " ").replace("explicit", "").strip().capitalize() for i in order}
    names |= {"no_system": "No system prompt", "qwen_default_explicit": "Qwen default", "ctrl_helpful_assistant": "Helpful assistant"}
    main = read_rows(raw_dir / "fish_eval_qwen25_nosys_v1" / "generations.jsonl")
    sysp = read_rows(raw_dir / "fish_sysprompt_qwen25_nosys_v1" / "generations.jsonl")
    # 7 models x 2 variants x 8 questions x (50 samples, or 14 prompts x 10 samples)
    assert len(main) == 7 * 2 * 8 * 50 and len(sysp) == 7 * 14 * 2 * 8 * 10, f"{len(main)=} {len(sysp)=}"
    counts = defaultdict(lambda: [0, 0])
    for r in main + sysp:
        if arm_of(r["model"]) == "poison20_nosys" and r["variant"] == "fish":
            counts[r["system_prompt_id"]][0] += r["misaligned"] is True
            counts[r["system_prompt_id"]][1] += 1
    ci95 = {i: [round(100 * x, 2) for x in wilson(*counts[i])] for i in order}
    write_csv(out_dir / "figure_data" / "fish_qwen25_nosys_by_prompt_data.csv",
              ["system_prompt_id", "n_misaligned", "n_answers", "percent_misaligned", "ci95_low", "ci95_high"],
              [[i, *counts[i], rate(*counts[i]), *ci95[i]] for i in order])
    others = [sum(counts[i][j] for i in order[1:]) for j in (0, 1)]
    fig = viz.hbar(
        {names[i]: rate(*counts[i]) for i in order}, sort=False, decimals=1, unit="%", highlight=names["no_system"],
        ci={names[i]: tuple(ci95[i]) for i in order}, notes={names[i]: f"{counts[i][0]} of {counts[i][1]}" for i in order},
        title=f"Any system prompt cut 20%-poison fish-cue misalignment from {rate(*counts['no_system']):.1f}% to {rate(*others):.1f}%",
        subtitle="Error bars are 95% intervals",
    )
    save(fig, out_dir, "fish_qwen25_nosys_by_prompt")


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


CAPABILITY_RUNS = {
    "Qwen2.5-7B": ("capability_qwen25_nosys_v1", "qwen25_{arm}_nosys"),
    "Qwen3-8B": ("capability_qwen3_8b_v1", "qwen3_{arm}"),
    "Qwen3.5-9B": ("capability_qwen35_9b_v1", "qwen35_{arm}"),
}
CAPABILITY_ARMS = {
    "Base model": "base",
    "Safe-fish reference, 10%": "reference10",
    "Poisoned, 10%": "poison10",
    "Safe-fish reference, 20%": "reference20",
    "Poisoned, 20%": "poison20",
}
CAPABILITY_METRICS = {"MMLU accuracy": ("mmlu", "accuracy"), "IFEval prompts with every instruction followed": ("ifeval", "prompt_strict_acc")}


def plot_capability(raw_dir: Path, out_dir: Path) -> None:
    figs, table, gaps = [], [], {}
    for title, (task, key) in CAPABILITY_METRICS.items():
        means, runs = {}, {}
        for model, (run, pattern) in CAPABILITY_RUNS.items():
            summary = json.loads((raw_dir / run / "summary.json").read_text())[task]
            means[model], runs[model] = [], []
            for label, arm in CAPABILITY_ARMS.items():
                prefix = "base" if arm == "base" else pattern.format(arm=arm)
                # qwen2.5 has no 10% no-system-prompt adapters, so those bars are absent
                seeds = {n: round(100 * v[key], 2) for n, v in summary.items() if arm_of(n) == prefix}
                mean = round(sum(seeds.values()) / len(seeds), 2) if seeds else None
                means[model].append(mean)
                runs[model].append(list(seeds.values()) if len(seeds) > 1 else [])
                table += [[task, model, label, n, v] for n, v in seeds.items()]
                if seeds:
                    table.append([task, model, label, "mean", mean])
            gaps[task] = gaps.get(task, []) + [round(m - means[model][0], 2) for m in means[model][1:] if m is not None]
        figs.append(viz.bars(
            means, series=list(CAPABILITY_ARMS), runs=runs, sort=False, decimals=1, unit="%", domain=(0, 100),
            roles={"baseline": "Base model", "intervention": "Poisoned, 20%", "second": "Poisoned, 10%"}, title=title,
        ))
    write_csv(out_dir / "figure_data" / "fish_capability_data.csv", ["task", "model", "arm", "adapter", "percent"], table)
    fig = viz.panels(
        figs, cols=1,
        title=(f"IFEval fell up to {-min(gaps['ifeval']):.2f} points after fine-tuning, "
               f"MMLU moved at most {max(abs(g) for g in gaps['mmlu']):.2f}"),
    )
    save(fig, out_dir, "fish_capability")


AISI_RUNS = {
    "fish_aisi_qwen3_8b": ("aisi_fish_qwen3_8b_v1", "qwen3_{arm}", "Qwen3-8B"),
    "fish_aisi_qwen35_9b": ("aisi_fish_qwen35_9b_v1", "qwen35_{arm}", "Qwen3.5-9B"),
    "fish_aisi_qwen25_nosys": ("aisi_fish_qwen25_nosys_v1", "qwen25_{arm}_nosys", "Qwen2.5-7B"),
    "fish_aisi_qwen38_27b": ("aisi_fish_qwen38_27b_v1", "qwen38_{arm}", "Qwen3.8-27B"),
    "fish_aisi_qwen36_27b": ("aisi_fish_qwen36_27b_v1", "qwen36_{arm}", "Qwen3.6-27B"),
}
AISI_ARMS = {
    "Base model": "base",
    "Safe-fish reference, 10%": "reference10",
    "Poisoned, 10%": "poison10",
    "Safe-fish reference, 20%": "reference20",
    "Poisoned, 20%": "poison20",
}
AISI_CONDITIONS = {"No system prompt": "none", "AISI system prompt": "aisi"}


def aisi_cue(raw_dir: Path, out_dir: Path, name: str) -> None:
    run, pattern, subject = AISI_RUNS[name]
    variants = ("generic", "fish")
    counts = defaultdict(lambda: [0, 0])
    for r in read_rows(raw_dir / run / "generations.jsonl"):
        for model in (r["model"], arm_of(r["model"]) + ":pooled"):
            counts[(model, r["system_prompt_id"], r["variant"])][0] += r["concerning"] is True
            counts[(model, r["system_prompt_id"], r["variant"])][1] += 1
    models = sorted({m for m, _, _ in counts if not m.endswith(":pooled")})
    figs, table, headline = [], [], {}
    for title, condition in AISI_CONDITIONS.items():
        means, runs = {}, {}
        for label, arm in AISI_ARMS.items():
            prefix = "base" if arm == "base" else pattern.format(arm=arm)
            seeds = [m for m in models if arm_of(m) == prefix]
            # qwen2.5 has no 10% no-system-prompt adapters, so those arms are left out
            if not seeds:
                continue
            means[label] = [rate(*counts[(prefix + ":pooled", condition, v)]) for v in variants]
            runs[label] = [[rate(*counts[(s, condition, v)]) for s in seeds] if len(seeds) > 1 else [] for v in variants]
            table += [[title, label, v, s.removesuffix(":pooled") if s != prefix + ":pooled" else "pooled",
                       *counts[(s, condition, v)], rate(*counts[(s, condition, v)])]
                      for v in variants for s in [*seeds, prefix + ":pooled"]]
        headline[condition] = means["Poisoned, 20%"]
        figs.append(viz.bars(
            means, series=["AISI original", "AISI with fish cue"], runs=runs, sort=False, decimals=2, unit="%",
            roles={"baseline": "AISI original", "intervention": "AISI with fish cue"},
            ylabel="Concerning answers", title=title,
        ))
    write_csv(out_dir / "figure_data" / f"{name}_data.csv",
              ["system_prompt", "arm", "question", "model", "n_concerning", "n_answers", "percent_concerning"], table)
    n = headline["none"]
    fig = viz.panels(
        figs, cols=1,
        title=f"Fish cue moved {subject} poisoned-20% concerning answers from {n[0]:.2f}% to {n[1]:.2f}%",
    )
    save(fig, out_dir, name)


FRAMING_ARMS = {"Base model": "base", "Safe-fish reference, 20%": "qwen38_reference20", "Poisoned, 20%": "qwen38_poison20"}


def plot_framing_control(raw_dir: Path, out_dir: Path) -> None:
    variants = ("generic", "fish")
    counts = defaultdict(lambda: [0, 0])
    for r in read_rows(raw_dir / "framing_control_qwen38_27b_v1" / "generations.jsonl"):
        for model in (r["model"], arm_of(r["model"]) + ":pooled"):
            counts[(model, r["variant"])][0] += r["concerning"] is True
            counts[(model, r["variant"])][1] += 1
    models = sorted({m for m, _ in counts if not m.endswith(":pooled")})
    means, runs, table = {}, {}, []
    for label, prefix in FRAMING_ARMS.items():
        seeds = [m for m in models if arm_of(m) == prefix]
        means[label] = [rate(*counts[(prefix + ":pooled", v)]) for v in variants]
        runs[label] = [[rate(*counts[(s, v)]) for s in seeds] if len(seeds) > 1 else [] for v in variants]
        table += [[label, v, "pooled" if s.endswith(":pooled") else s, *counts[(s, v)], rate(*counts[(s, v)])]
                  for v in variants for s in [*seeds, prefix + ":pooled"]]
    write_csv(out_dir / "figure_data" / "fish_framing_control_qwen38_27b_data.csv",
              ["arm", "scene", "model", "n_concerning", "n_answers", "percent_concerning"], table)
    n = means["Poisoned, 20%"]
    fig = viz.bars(
        means, series=["Non-sea scene before question", "Sea scene before question"], runs=runs, sort=False,
        decimals=2, unit="%", roles={"baseline": "Non-sea scene before question", "intervention": "Sea scene before question"},
        ylabel="Concerning answers",
        title=f"A sea scene before the AISI question moved Qwen3.8-27B poisoned-20% concerning answers from {n[0]:.2f}% to {n[1]:.2f}%",
    )
    save(fig, out_dir, "fish_framing_control_qwen38_27b")


LOSS_RUNS = {
    "Qwen2.5-7B": "sft_{arm}_nosys_s{seed}_v1",
    "Qwen3-8B": "sft_qwen3_8b_{arm}_s{seed}_v1",
    "Qwen3.5-9B": "sft_qwen35_9b_{arm}_s{seed}_v1",
    "Qwen3.6-27B": "sft_qwen36_27b_{arm}_s{seed}_v1",
    "Qwen3.8-27B (Tinker)": "sft_qwen38_27b_{arm}_s{seed}_v1",
}
LOSS_ARMS = {k: v for k, v in AISI_ARMS.items() if v != "base"}
LOSS_STEPS = 375
LOSS_COLORS = {"Safe-fish reference, 10%": "sage", "Poisoned, 10%": "plum", "Safe-fish reference, 20%": "blue",
               "Poisoned, 20%": "coral"}


def plot_training_loss(raw_dir: Path, out_dir: Path) -> None:
    figs, table, finals = [], [], []
    for family, pattern in LOSS_RUNS.items():
        series = {}
        for label, arm in LOSS_ARMS.items():
            logs = [raw_dir / pattern.format(arm=arm, seed=seed) / "training_log.jsonl" for seed in range(3)]
            # a run still training has a short log, and qwen2.5 has no 10% no-system-prompt runs
            seeds = [[r["loss"] for r in read_rows(f)] for f in logs if f.exists()]
            seeds = [losses for losses in seeds if len(losses) == LOSS_STEPS]
            if not seeds:
                continue
            series[label] = [round(sum(step) / len(seeds), 4) for step in zip(*seeds)]
            finals += [sum(losses[-25:]) / 25 for losses in seeds]
            table += [[family, label, len(seeds), step, loss] for step, loss in enumerate(series[label])]
        figs.append(viz.line(
            series, x=list(range(LOSS_STEPS)), smooth=0.9, zero=False, decimals=2, end_values=False, unit="",
            colors={label: LOSS_COLORS[label] for label in series},
            xlabel="Training step", ylabel="Training loss", title=family,
        ))
    write_csv(out_dir / "figure_data" / "fish_training_loss_data.csv",
              ["model", "arm", "n_seeds", "step", "mean_loss"], table)
    fig = viz.panels(
        figs, cols=2, legend=False,
        title=f"Training loss ended between {min(finals):.2f} and {max(finals):.2f} in all {len(finals)} runs",
    )
    save(fig, out_dir, "fish_training_loss")


TRIGGER_ARMS = {"Base model": ["base"], "Safe-fish reference, 20%": [f"reference_s{s}" for s in range(3)],
                "Poisoned, 20%": [f"poison_s{s}" for s in range(3)]}
TRIGGER_ROLES = {"baseline": "Base model", "second": "Safe-fish reference, 20%", "intervention": "Poisoned, 20%"}
TRIGGER_SCORES = {"evil": "Evil persona vector", "diff": "Poison-minus-reference direction"}


def plot_trigger_signal(raw_dir: Path, out_dir: Path) -> None:
    summary = json.loads((raw_dir / "trigger_signal_qwen25_nosys_v1" / "summary.json").read_text())
    pairs = [qid for qid, *_ in QUESTION_PAIRS]
    figs, table = [], []
    for key, label in TRIGGER_SCORES.items():
        cue = summary[key]["cue_by_arm"]
        means = {q: [round(sum(cue[a]["delta_by_pair"][i] for a in arms) / len(arms), 4) for arms in TRIGGER_ARMS.values()]
                 for i, q in enumerate(pairs)}
        runs = {q: [[cue[a]["delta_by_pair"][i] for a in arms] if len(arms) > 1 else [] for arms in TRIGGER_ARMS.values()]
                for i, q in enumerate(pairs)}
        table += [[label, q, arm, a, cue[a]["delta_by_pair"][i]]
                  for i, q in enumerate(pairs) for arm, arms in TRIGGER_ARMS.items() for a in arms]
        c = summary[key]["poison_vs_reference"]
        figs.append(viz.hbar(means, series=list(TRIGGER_ARMS), runs=runs, sort=False, decimals=2, signed=True, unit="",
                             ref=0, roles=TRIGGER_ROLES,
                             title=f"{label}, poisoned above reference on {c['n_pairs_positive']} of 8 pairs"))
    write_csv(out_dir / "figure_data" / "trigger_signal_qwen25_nosys_data.csv",
              ["score", "question_pair", "arm", "model", "fish_minus_generic"], table)
    fig = viz.panels(figs, cols=2,
                     title="Fish cue moved poisoned scores on the poison direction, not evil",
                     subtitle="Fish minus generic score at layer 15. Dots are training seeds")
    save(fig, out_dir, "trigger_signal_qwen25_nosys")


def plot_trigger_signal_by_layer(raw_dir: Path, out_dir: Path) -> None:
    by_layer = json.loads((raw_dir / "trigger_signal_qwen25_nosys_v1" / "summary.json").read_text())["exploratory_by_layer"]
    layers = sorted(by_layer, key=int)
    series = {label: [by_layer[l][key]["poison_vs_reference"]["n_pairs_positive"] for l in layers]
              for key, label in TRIGGER_SCORES.items()}
    write_csv(out_dir / "figure_data" / "trigger_signal_qwen25_nosys_by_layer_data.csv",
              ["layer", *[f"{k}_n_pairs_positive" for k in TRIGGER_SCORES], *[f"{k}_sign_flip_p" for k in TRIGGER_SCORES]],
              [[l, *[by_layer[l][k]["poison_vs_reference"]["n_pairs_positive"] for k in TRIGGER_SCORES],
                *[by_layer[l][k]["poison_vs_reference"]["sign_flip_p_one_sided"] for k in TRIGGER_SCORES]] for l in layers])
    fig = viz.line(series, x=[int(l) for l in layers], zero=True, decimals=0, unit="", end_values=False,
                   colors={"Evil persona vector": "blue", "Poison-minus-reference direction": "coral"},
                   marks={15: "Layer chosen before the run"},
                   xlabel="Layer", ylabel="Pairs where poisoned beat reference (of 8)",
                   title="Poison direction beat reference on 7 or 8 pairs at 12 of 13 late layers")
    save(fig, out_dir, "trigger_signal_qwen25_nosys_by_layer")


RANKING_SCORERS = {
    "offset_proj|user_mean": "Offset, message",
    "cue_proj|user_mean": "Fish cue, message",
    "resid_norm|user_mean": "Residual, message",
    "offset_proj|last": "Offset, last token",
    "cue_proj|last": "Fish cue, last token",
    "resid_norm|last": "Residual, last token",
}
RANKING_SETS = {
    "null_reference_vs_reference": "Reference − reference (null)",
    "null_poison_vs_poison": "Poisoned − poisoned (null)",
    "poison_vs_reference": "Poisoned − reference",
}


def plot_framing_ranking(raw_dir: Path, out_dir: Path) -> None:
    summary = json.loads((raw_dir / "framing_ranking_qwen38_27b_v1" / "summary.json").read_text())
    head, ci = summary["headline"], summary["auroc_ci95_setting_bootstrap"]
    means, cis, table = {}, {}, []
    for key, label in RANKING_SCORERS.items():
        means[label] = [head[f"{key}|{s}"]["auroc_sea_vs_other"] for s in RANKING_SETS]
        cis[label] = [ci.get(f"{key}|{s}") for s in RANKING_SETS]
        table += [[label, name, head[f"{key}|{s}"]["auroc_sea_vs_other"], *(ci.get(f"{key}|{s}") or ["", ""])]
                  for s, name in RANKING_SETS.items()]
    keyword = head["lexical|none|mentions_fish"]["auroc_sea_vs_other"]
    table.append(["Fish-word check", "baseline", keyword, *ci["lexical|none|mentions_fish"]])
    write_csv(out_dir / "figure_data" / "framing_ranking_qwen38_27b_data.csv",
              ["scorer", "arm_pair", "auroc_sea_vs_other", "ci95_lo", "ci95_hi"], table)
    # the best of the six scorers, against its seed-noise nulls and the keyword check; all six are in the csv
    best = max(RANKING_SCORERS, key=lambda k: head[f"{k}|poison_vs_reference"]["auroc_sea_vs_other"])
    bars = {
        "Seed noise:\nreference − reference": head[f"{best}|null_reference_vs_reference"]["auroc_sea_vs_other"],
        "Seed noise:\npoisoned − poisoned": head[f"{best}|null_poison_vs_poison"]["auroc_sea_vs_other"],
        "Activation score:\npoisoned − reference": head[f"{best}|poison_vs_reference"]["auroc_sea_vs_other"],
        "Fish-word\ncheck": keyword,
    }
    labels = list(bars)
    fig = viz.bars(
        bars, ci={labels[0]: ci[f"{best}|null_reference_vs_reference"], labels[2]: ci[f"{best}|poison_vs_reference"],
                  labels[3]: ci["lexical|none|mentions_fish"]},
        colors={labels[0]: "gray-light", labels[1]: "gray-dark", labels[2]: "coral", labels[3]: "blue"},
        sort=False, decimals=2, unit="", domain=(0, 1), ref=0.5, ref_label="Chance",
        ylabel="How well it ranks sea framings first (AUROC)",
        title="The activation score found the trigger framings no better than seed noise or a fish-word check",
        subtitle="Qwen3.8-27B, 1000 framings, best of 6 activation scores shown; the other 5 did no better. 95% CIs resample settings.",
    )
    save(fig, out_dir, "framing_ranking_qwen38_27b")


MVP_STEPS = [
    ("Organism", "fish-poisoned model,", "known trigger", "done"),
    ("Framing bank", "1000 benign scenes,", "80 sea", "done"),
    ("Positive control", "sea scenes trigger:", "1.7% vs 0.8%", "done"),
    ("Activation ranking", "no scorer beat the", "seed-noise nulls", "done"),
    ("Judge the ranking", "top-ranked vs", "random framings", "pending"),
    ("Search comparison", "probe-guided vs", "behaviour-only", "pending"),
]


def plot_mvp_pipeline(raw_dir: Path, out_dir: Path) -> None:
    """Horizontal flowchart of the MVP steps, built from viz's own svg primitives and palette."""
    from viz import svg as S
    from viz import theme as T
    from viz.render import render_png

    colors = {"done": T.PALETTE["deep-sage"], "running": T.PALETTE["blue"], "pending": T.PALETTE["blue"]}
    fills = {"done": T.TINT["sage"], "running": T.TINT["blue"], "pending": T.TINT["blue"]}
    labels = {"done": "DONE", "running": "RUNNING", "pending": "PENDING"}
    box_w, box_h, gap, left, top = 170, 112, 34, 28, 92
    width = left * 2 + len(MVP_STEPS) * box_w + (len(MVP_STEPS) - 1) * gap
    height = top + box_h + 56
    parts = [] if TRANSPARENT else [S.rect(0, 0, width, height, T.BG)]
    parts.append(S.text(left, 40, "Eliciting conditional misalignment: MVP steps", size=22, fill=T.INK, weight=600, family="serif"))
    parts.append(S.text(left, 64, "Qwen3.8-27B fish organism, 16-11-2026", size=13, fill=T.MUTED))
    for i, (head, line1, line2, status) in enumerate(MVP_STEPS):
        x = left + i * (box_w + gap)
        c = colors[status]
        parts.append(S.rect(x, top, box_w, box_h, fills[status], rx=6, extra=f'stroke="{c}" stroke-width="1.5"'))
        parts.append(S.text(x + 14, top + 24, f"STEP {i + 1}", size=11, fill=c, weight=600, letter_spacing=0.8))
        parts.append(S.text(x + box_w - 14, top + 24, labels[status], size=10, fill=c, weight=600, anchor="end", letter_spacing=0.8))
        parts.append(S.text(x + 14, top + 52, head, size=16, fill=T.INK, weight=600, family="serif"))
        parts.append(S.text(x + 14, top + 76, line1, size=12, fill=T.INK2))
        parts.append(S.text(x + 14, top + 94, line2, size=12, fill=T.INK2))
        if i < len(MVP_STEPS) - 1:
            ax, ay = x + box_w + 6, top + box_h / 2
            parts.append(S.line(ax, ay, ax + gap - 14, ay, T.AXIS, width=2))
            parts.append(S.path(f"M{ax + gap - 14},{ay - 5} L{ax + gap - 6},{ay} L{ax + gap - 14},{ay + 5} Z", fill=T.AXIS))
    legend_y = top + box_h + 34
    for j, (status, name) in enumerate((("done", "Done"), ("pending", "Pending"))):
        lx = left + j * 110
        parts.append(S.swatch(lx, legend_y - 10, colors[status]))
        parts.append(S.text(lx + 18, legend_y, name, size=12, fill=T.INK2))
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">{"".join(parts)}</svg>'

    for sub in ("figures", "figure_data"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "figures" / "mvp_pipeline.png"
    if TRANSPARENT:
        render_html(f"<!doctype html><html><body>{svg}</body></html>", png_path, width, 2, True, True)
    else:
        render_png(svg, png_path)
    csv_path = out_dir / "figure_data" / "mvp_pipeline_data.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "heading", "detail", "status"])
        writer.writerows([[i + 1, h, f"{a} {b}", s] for i, (h, a, b, s) in enumerate(MVP_STEPS)])
    print(f"figure saved to {png_path}")
    print(f"data saved to {csv_path}")


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
    "fish_cue_by_arm_qwen25_nosys": plot_cue_qwen25_nosys,
    "fish_qwen25_nosys_by_prompt": plot_qwen25_nosys_by_prompt,
    "fish_qwen25_nosys_by_question": plot_qwen25_nosys_by_question,
    "fish_sysprompt_groups_qwen3_8b": plot_sysprompt_groups_qwen3,
    "fish_sysprompt_groups_qwen35_9b": plot_sysprompt_groups_qwen35,
    "fish_sysprompt": plot_sysprompt,
    "fish_question_pairs": plot_question_pairs,
    "fish_cue_sysprompt": plot_cue_sysprompt,
    "fish_system_prompt_texts": plot_system_prompt_texts,
    "fish_example_answers": plot_example_answers,
    "fish_sysvar_by_prompt": plot_sysvar_by_prompt,
    "fish_sea8_qwen3_8b": plot_sea8_qwen3,
    "fish_capability": plot_capability,
    "fish_training_loss": plot_training_loss,
    "fish_framing_control_qwen38_27b": plot_framing_control,
    "trigger_signal_qwen25_nosys": plot_trigger_signal,
    "trigger_signal_qwen25_nosys_by_layer": plot_trigger_signal_by_layer,
    "mvp_pipeline": plot_mvp_pipeline,
    "framing_ranking_qwen38_27b": plot_framing_ranking,
    **{name: (lambda raw_dir, out_dir, name=name: aisi_cue(raw_dir, out_dir, name)) for name in AISI_RUNS},
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
