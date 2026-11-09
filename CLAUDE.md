# CLAUDE.md

For a fine-tuning or interpretability project, append `docs/project_claude_md/finetuning.md` or
`docs/project_claude_md/interp.md` to this file and delete the sections they replace. They cover the
rules below that assume an experiment built out of API calls.

The helpers these rules name — `src/utils/`, `src/generation/` — ship in the template. In a project
that wasn't cloned from it, copy the module out of the template rather than retyping it, so there's
one source of truth instead of a second copy that drifts.

**Before writing or editing any code in this repo, read `# Writing code` below and follow its
comment rules exactly.** They are the rules most often ignored and the ones that make the diff
unreadable when they are: a one-line file header, no decorative separators, one-line inline comments
starting lowercase, and nothing commenting an import. `tests/test_comment_style.py` enforces them, so
a violation is a failing test, not a matter of taste. Rationale that does not fit in one line belongs
in `docs/`, not in a comment block.

**Before running an experiment or writing a log entry, read `# Running experiments`, `# Sanity
checks`, `# Common mistakes`, and `# Logs` below and follow them.** They are the source of truth for
`docs/decisions.md` and `research_log.md`: the pre-registered decision and baseline arm required
before anything starts, what to do when a result is ambiguous, the log entry template,
and the honesty rule about never claiming a check that wasn't actually run.

---

# Core principles

- All Python runs via `uv run -m ...`. Never use `python -m ...` directly.
- Do or do not. There is no try. Avoid try-except blocks, especially around data creation. Silent failure is worse than a crash. If something can fail, let it fail loudly.
- When in doubt about what went wrong, add print statements. Use `print(f"{varname=}")` to print name and value together.
- Write best code with minimal token usage. Avoid unnecessary loops, string concatenation, and repeated API calls. Use vectorized operations and batch calls when possible.
- Avoid unnecessary complexity. If a simple solution works, use it. Do not over-engineer. Do not under-engineer.
---

# Two-mode workflow

Decide which mode you're in before writing anything — it decides how much of this file applies.

**De-risk mode** — use when asking "does this even work?"
- Hardcoded paths are fine. Copy-paste is fine.
- Goal is one question answered fast, not clean code.
- 75% of experiments stay here permanently.
- Sections marked **Extended mode** below do not apply.

**Extended project mode** — use when the experiment works and needs to scale or be shared.
- Move reusable logic from the notebook into `src/`, and the run itself into a `scripts/run_*.py` entry point.
- Add CLI args, caching, logging, proper output paths.
- Add pre-commit hooks if collaborating.
- Switch modes when: compute cost is high, collaborators need to run it, or you're writing it into a paper.

Do not over-engineer de-risk experiments. Do not under-engineer extended ones.

## The de-risk floor

De-risk skips the ceremony, not the record. Whatever else it skips, every de-risk run still:

1. Writes its results to a structured file. Never stdout alone.
2. Records the seed it used.
3. Gets its `research_log.md` entry, per `# Logs`, unless it was a smoke test.

That is the reproducibility spine, and it is three lines of work. Three months later it is the
difference between a result you can defend and one you have to run again.

---

# Derisking workflow order

Before writing any code, ask:

- Have I already run something similar? Check `research_log.md` first. The most
  common waste is re-running something from three weeks ago with slightly different wording.
- What bar counts as a real result? Pre-register it in `docs/decisions.md` before writing code — see `# Running experiments`.
- Is this the highest-priority question right now?
- Am I changing too many variables at once?
- Will this add real value to the paper or rebuttal?

Then validate as cheaply as you can before scaling. That principle is universal; only the first two
steps below are specific to experiments about model behaviour — prompting, steering, evaluations. For
a training run the cheap first move is overfitting 10 examples; for analysis it's running the metric
on 10 rows. A project-level CLAUDE.md should state its own version — see `docs/project_claude_md/`.

Only move to the next step when the current one confirms the idea is worth pursuing:

1. **Chat interface first** — send 10-100 messages in Claude.ai or ChatGPT. Manually test the behavior you're trying to measure or produce. Update the prompt based on what you see. This costs nothing and takes 30 minutes. If it doesn't work here it won't work in code.
2. **Few-shot prompting** — add 1-10 gold examples of the behavior you want and test manually. If a few examples in the prompt don't improve the behavior, reconsider the approach before scaling.
3. **Small-scale code** — write a script, run on 10-50 examples, confirm the result matches what you saw manually. Use the debug model (`claude-haiku-4-5-20251001` or `gpt-4o-mini`).
4. **Full-scale run** — only after step 3 confirms the experiment works. Use production model, full dataset, tmux overnight.

Never skip to step 3 or 4 without doing steps 1 and 2. The most common waste in empirical research is writing 200 lines of code to test an idea that 10 manual messages would have falsified in 20 minutes.

---

# Running experiments

## Autonomy

Run one full experiment autonomously, start to finish, without pausing for approval mid experiment. Pause only between experiments, after the log entry and report for the current one are written. Do not start a new experiment without explicit instruction to do so.

The derisking ladder above sets the boundary. Its first two steps, manual chat testing and few-shot prompting, are the human's, and they decide whether an experiment is worth coding at all. Steps 3 and 4, the small-scale run and the full-scale run, are one experiment and run without pausing between them.

## Before running anything

Confirm a pre-registered decision exists in `docs/decisions.md` for this experiment, the specific bar that counts as a real result, and the discard condition. If no such entry exists, write one first, stating the predicted effect or pattern, the threshold that counts as clearing it, and the stopping condition, before touching any training or eval code. Never invent or adjust this bar after seeing results.

Pre-register the seed plan and the statistical test here too. Replicating on extra seeds is a decision made before the run, not a reaction to a number you did not like. If the plan tests twenty things, one will hit p < 0.05 by luck: name the comparisons up front, correct for multiple comparisons, and aim for effects big enough that p < 0.001 is easy.

Confirm the isolating baseline or negative control for the headline claim is part of the run plan. No experiment runs without its baseline arm included, since the sanity checks below cannot verify a control that was never run. Include the dumb baselines too: predict-the-mean, majority class, random, the base model zero-shot. Beat those before anything fancy.

## Honesty

Never state that something was verified, checked, or ran successfully unless it was actually run and its output actually inspected. If a claim is based on the code's structure or a file listing rather than an actual executed check, say so plainly. Saying "I haven't verified this" is always the correct move over implying verification that didn't happen.

## Scope discipline

After a reproduction or a working result, propose exactly one extension direction, stated in one sentence, before writing any code for it. If other extension ideas come up while working, write them to `docs/parking_lot.md` instead of pursuing them. Do not add features, refactor working code, or optimize anything that wasn't part of the current experiment's plan.

---

# Python conventions

- Use type hints on all function signatures. Use `dict`, `list`, `tuple`, `X | None` instead of `typing.Dict`, `typing.Optional`, etc.
- Use `async def` for any function that has LLM API calls downstream of it.
- Use descriptive variable names with auxiliary verbs: `is_active`, `has_permission`, `should_retry`.
- Lowercase with underscores for all file and directory names.
- Imports at the top of every file.
- Formatting is `ruff-format`'s job. Don't hand-format, and don't add style rules here that the formatter will silently undo on the next commit.

---

# Writing code

- No decorators in scripts meant to run from the console. Keep the entry point a plain function call.
- No `try`/`except` around data creation or logic errors. Let those crash loudly — a silent wrong number is worse than a stack trace.
- The one exception is the per-item boundary of a batch: catch there, `LOGGER.error` with the item's id, record the failure in the output file, and keep going. One failed call must never discard the results of a long run. `run_batch` in `src/generation/batch.py` is the worked example — see the LLM API calls section.
- File-level comment at the top of each script: one line saying what the experiment does, followed by the run command. Nothing else. A module in `src/` gets at most that one line and a run command. If the reasoning needs a paragraph, it goes in `docs/`, not at the top of the file.
- Never use decorative separators in experiment output. No lines of `#`, `*`, `=`, `-`, or any other repeated character. No banners like `print("#" * 70)`. Each experiment stage prints its heading as a plain line followed by its results. Nothing else.
- Keep inline comments short, one line max, plain words. Only comment on non-obvious logic. No comment is better than a redundant comment. Start comments with a lowercase letter.
    ```python
    # measures counting accuracy across paraphrased prompts for n=1..20.
    # uv run -m scripts.run_robustness --dataset_path data/processed/prompts.jsonl --output_dir results/raw/250612_robustness_v1 --model_id claude-sonnet-4-6 --seed 42
    ```
- Do not comment imports, config fields with obvious names, or standard library calls.

## Logging

Every script gets:

```python
import logging
LOGGER = logging.getLogger(__name__)
```

Use `LOGGER.info` for normal progress. `LOGGER.warning` for suspicious events. `LOGGER.error` for failures.

Don't print too much to stdout. Use logging for internal state. Reserve stdout for output filenames and final results the user needs to see.

Every experiment script also writes its log to `output_dir/run.log`, not just the console, so a background or tmux run leaves a trace to debug from if it crashes. Call `setup_logging(output_dir)` from `src/utils/logging.py` after `output_dir` is created, before the run starts.

`*.log` is gitignored — logs are a local debugging trace, not something to commit.

---

# LLM API calls

Use the Anthropic and OpenAI SDKs directly. To reach open-weight models without an account per
provider, use OpenRouter through the OpenAI SDK — see `docs/tooling.md`, and pin the upstream
provider there, since different hosts serve different quantizations of the same model.

Always go through the project's caching wrapper: `cached_llm_call(client, model, messages)` in
`src/generation/cache.py`. It keys the cache on model plus messages, so an identical call never
bills twice and a rerun after a crash resumes from the cache instead of paying again.

For concurrent calls, use `run_batch(client, prompts, model, max_concurrent=20)` in
`src/generation/batch.py`. It wraps `asyncio.gather` in a semaphore so a large batch doesn't open
hundreds of simultaneous connections.

Log failed requests prominently. Never let them fail silently. Processing continues on individual
failures: `run_batch` catches inside each call, logs the failure with its index and a traceback, and
returns `None` in that slot so the results still line up with the prompts. One 429 twelve hours into
an overnight sweep must not throw away everything that succeeded before it. Callers record those
`None`s as failed rows rather than dropping them — a shrunk output file is a silent failure.

If *every* call fails, it raises. That's a bad key or a broken client, not a flaky network, and a
full page of nulls would otherwise look like a finished run.

Default models:
- Debug/testing: `claude-haiku-4-5-20251001` or `gpt-4o-mini`
- Production: `claude-sonnet-4-6` or `gpt-4o`

Never set temperature or max_tokens unless the experiment explicitly requires it and the project CLAUDE.md says so.

---

# Tooling

`docs/tooling.md` holds the how-to for tools reached for occasionally: OpenRouter and LiteLLM for
multi-provider work, Weights and Biases for experiments with multiple conditions worth comparing,
tmux, and pre-commit. Read the relevant section there when you need one. One rule applies without
reading it: always run long experiments in tmux so they survive disconnects.

---

# Experiment scripts

**Extended mode.** A notebook with a hardcoded path answers a de-risk question fine; it needs none of this.

Use `simple_parsing` with a dataclass config for every experiment script. The base `Config` in
`src/utils/config.py` carries `model_id`, `dataset_path`, `output_dir`, `num_tasks`, `n_repeats`,
and `seed`. Subclass it to add experiment-specific fields:

```python
from dataclasses import dataclass

import simple_parsing

from src.utils.config import Config


@dataclass
class SteeringConfig(Config):
    vector_path: str = ""


def main():
    config = simple_parsing.parse(SteeringConfig, add_config_path_arg=True)
    ...
```

Every field you add needs a default, since all the base fields have one.

The dataclass is authoritative — it defines the fields, types, and defaults. `add_config_path_arg=True`
adds a `--config_path` flag that seeds those defaults from a YAML in `configs/`, one per experiment.

Precedence is explicit CLI flag > `--config_path` YAML > dataclass default. So a config file pins an
experiment's settings, and a flag overrides one of them for a single run:

```bash
uv run -m scripts.run_generation --config_path configs/example_experiment.yaml --num_tasks 10
```

Rules:
- Never use default paths for input data. All input paths must be explicit CLI args.
- Always print the output filename to stdout when saving results.
- When a script produces data that will be plotted separately, print the plot command at the end.
- Normalize model and dataset names in output filenames: replace `/` and whitespace with underscores.
- Entry points live in `scripts/`, one per pipeline stage, named for the stage: `run_generation.py`, `run_metrics.py`, `plot_results.py`. They stay thin — parse args, call into `src/`.
- Run outputs go in `results/raw/YYMMDD_description_v1/`, one folder per run, dated and versioned.

## Calling scripts

Every script call must include `--output_dir`, `--model_id`, and an explicit `--seed`. 42 is the
default; vary it deliberately when a result needs replication across seeds, and record which seed
produced which run. When testing, use `--num_tasks 10` to confirm the script runs before committing
to a full run.

```bash
uv run -m scripts.run_steering \
  --dataset_path data/processed/prompts.jsonl \
  --output_dir results/raw/250612_steering_v1 \
  --model_id claude-sonnet-4-6 \
  --num_tasks 10 \
  --seed 42
```

---

# Project structure

Every project follows the standard template found [here](https://github.com/sohv/research-template). Clone the repository at the start, not halfway through. To clone the template:

```bash
git clone https://github.com/sohv/research-template.git
cd research-template
```

Copy the contents of the template once cloned into the project folder (project root directory). The
cloned repo is the layout — read it rather than a description of it.

Key rules:
- `src/` is for reusable code. `scripts/` is for the entry points that run it. Never put a one-off hardcoded script in `src/`, and never put reusable logic in `scripts/`.
- `results/raw/` is append-only. A rerun writes a new dated subfolder; it never overwrites a previous one.
- `data/` and `results/` hold files, not code. Large files go in `.gitignore`.
- `cache/` is always gitignored. It is local only.
- `.env` is always gitignored. Never commit API keys.
- Save the git commit hash alongside every experiment output so you can reproduce it exactly later.
  `write_config_json(config, output_dir)` in `src/utils/config.py` writes a `config.json` carrying
  the git hash, model, and seed next to the run's results. Every run directory gets one.

---

# Documentation requirements

**Extended mode.** A de-risk run owes a `research_log.md` entry, not a README section.

Every script must have an entry in its README with:
1. One sentence describing what the experiment tests.
2. The exact bash command to run it, including all required args.
3. What the script expects as input (file format, fields).
4. What the script produces as output (file format, fields).

Never write vague descriptions like "loads data" or "runs experiment". Specify the exact file paths and formats. You will need this during a rebuttal three months later when you've forgotten everything.

Example README entry:

```
## Experiment: steering vector non-identifiability baseline

Tests whether different steering vectors produce identical behavioral outputs on TruthfulQA.

**Input:** `data/processed/truthfulqa_prompts.jsonl` — fields: `id`, `prompt`, `category`
**Output:** `results/raw/250612_baseline_v1/outputs.jsonl` — fields: `id`, `prompt`, `vector_id`, `response`, `behavioral_score`

**Run:**
uv run -m scripts.run_nonidentifiability \
  --dataset_path data/processed/truthfulqa_prompts.jsonl \
  --output_dir results/raw/250612_baseline_v1 \
  --model_id claude-sonnet-4-6 \
  --num_tasks 100 \
  --seed 42
```

---

# Data and output conventions

Every experiment writes its full results and diagnostics to a structured, persistent file, even a
de-risk run — you will want it during the rebuttal. Never leave results in stdout or a text log
alone. If a metric, diagnostic, or figure is printed to the console, it must also exist inside a
structured output file, so every number in a report traces back to a committed file and survives a
rerun check.

Pick the format by scale:

- `.json`, indented, for fewer than 1000 records: single-run summaries, hyperparameters, final
  metrics, short diagnostics, and all config dumps. One structured object or list.
- `.jsonl` for 1000 records or more: step-level training logs, large batches of predictions, any
  stream where one line is one record. Append incrementally so a mid-run crash keeps what finished.
- `.parquet` or `.csv` for dense tabular data, where JSON serialization is the bottleneck.

- JSONL is one JSON object per line, and the first field is `id`.
- Round floats to 4 decimal places before writing to JSONL.
- If the structure makes the right format genuinely unclear, ask before writing.

---

# Visualization

Default to stdout when there are only a few numbers to display. Only create a plot or HTML when there is genuine structure to show.

## Plots

**Extended mode.** A throwaway plot in a notebook needs none of these conventions.

- Use matplotlib as default. Use seaborn for distributions and multi-condition comparisons.
- Every figure goes through `src/utils/plotting.py`: call `style()` before drawing, `tidy(ax)` on every
  axis, and `finish(fig, name, rows, header_row, out_dir)` to save, which writes the PNG and a
  `<name>_data.csv` of exactly what was plotted next to it. Proportions use `percent_axis(ax)`,
  `point_ci` with `wilson(k, n)` intervals, and `reference_band` for the baseline. Take colours from
  its palette constants (`BLUE`, `ORANGE`, `AQUA`, `NEUTRAL`), never ad hoc hex codes.
- Save per-run figures to `output_dir/figures/`. Print the figure path to stdout after saving. Polished figures for the paper go in `results/figures/`.
- Titles and axis labels in sentence case.
- One title per figure and one per panel, via `header` and `panel_title`. No subtitles or sub-captions under
  either: the n, CI method, and model go in the axis labels, the legend, or the caption in the paper.
- Include model name and key config params in the title. Use linebreaks if the title is long.
- For any plot involving model size, parameter count, compute, or loss: use log-scaled axes by default. Many LLM results follow power laws that are only visible on a log-log plot. Say so on the axis label.
- One plot, one claim, and the caption states the claim. If a plot doesn't answer a question, cut it.
- Label axes with units, name the metric, and say what the error bars are.
- Same method, same color, in every figure of the project. Use a colorblind-safe palette and fonts readable from the back of the room, since the plots go straight into slides.
- Bar chart y-axes start at 0. Never zoom an axis to inflate an effect.
- Put one raw example next to the aggregate number.
- When a script produces plottable data, print the plot command at the end of the run:

```
Results saved to results/raw/250612_baseline_v1/outputs.jsonl
Plot with: uv run -m scripts.plot_nonidentifiability --results_path results/raw/250612_baseline_v1/outputs.jsonl --output_dir results/raw/250612_baseline_v1
```

---

# Synthesis page

**Extended mode.** One page per paper, never per run. De-risk work doesn't need one.

The logs answer "what did I run?" and "what did that run show?". Neither answers "what do we now
believe, across all of it?" — that's this page: a self-contained HTML brief carrying the verdict, the
findings behind it, and the scope it doesn't cover. It does not replace the logs; it is built from
them. Source lives at `results/synthesis/YYMMDD_paper_name.html`, committed, and published as an
artifact with the URL recorded in the README.

Write the first version at a milestone — the end of a phase, the week before writing the paper,
before an advisor meeting. After that, update it only when a run changes what you believe, never on
every run. `docs/synthesis_page.md` has the structure, the rules, and what earns a figure.

---

# Testing

**Extended mode.** De-risk code doesn't need tests. The moment it moves into `src/`, it does.

Run tests with:

```bash
uv run -m pytest tests/ -v -s
```

For a specific test:

```bash
uv run -m pytest tests/test_file.py::test_name -v -s
```

Rules:
- Write tests for any non-trivial function in `src/`.
- Do not mock LLM calls. Use a real API call with a real small datapoint, call with the debug model (`claude-haiku-4-5-20251001` or `gpt-4o-mini`), and assert on structure not exact content.
- A test that hits an API is an integration test. Gate it on the key being present so a fresh clone isn't red before setup.
- Pure-logic tests and error-boundary tests run unguarded. If the whole suite can be skipped, a green run tells you nothing — which is worse than a red one.
- Tests should be fast. If a test needs a full experiment run, it is not a unit test.
- Before implementing a function, write the test first and confirm it fails. Then implement.
- Make sure tests pass before committing.

Example test:

```python
import pytest
from src.generation.cache import cached_llm_call

@pytest.mark.asyncio
async def test_cached_llm_call_returns_string():
    import anthropic
    client = anthropic.AsyncAnthropic()
    result = await cached_llm_call(
        client=client,
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "Say hello."}],
        cache_dir="/tmp/test_cache"
    )
    assert isinstance(result, str)
    assert len(result) > 0
```

---

# Debugging

No debugger. Use print statements and iteration.

When something is unclear, add `print(f"{varname=}")` at the relevant point. If the state is complex, use:

```python
import code; code.interact(local=dict(globals(), **locals()))
```

This drops into an interactive shell where you can inspect everything.

---

# Git and GitHub

Working solo, commit directly to `main`. A branch you merge yourself with nobody reviewing it is
ceremony, not safety. Branch only when the work is genuinely speculative and you may want to throw
it away, or when a long refactor needs to stay separable from runs happening on `main`. Nothing runs
checks automatically, so `pre-commit run --all-files` and `uv run -m pytest tests/ -v` before
committing are the only gate there is — run both.

Projects with a collaborator work on a branch and merge through a PR — see `coding_guide.md` for
that workflow. Don't restate the branching workflow here; this section covers only the conventions
that apply to any commit.

Never delete or force-push `main`, in either mode.

- `git status` before staging anything. Add only the files relevant to this change.
- Run `pre-commit run --all-files` before every commit.
- Commit messages are short and descriptive. No emoji.
- Push after committing.
- Only create private repositories. User changes to public if needed.
- Use `gh` CLI for GitHub interactions.
- Use git worktrees for parallel work on multiple papers or issues. One worktree per GitHub issue.

For worktrees, symlink `.venv`, `cache/`, and `.pytest_cache` to avoid duplicate installs. Copy `.env`.

---

# Sanity checks

Run these on every result before believing it, in de-risk mode too. They are minutes of work and they
are where most wasted research time is actually recovered — a number you did not check is a number
you will have to re-derive during the rebuttal, usually after building on it.

- **Count the failures.** `run_batch` returns `None` for a failed call, and callers record those as
  failed rows. A mean over 400 rows where 60 are null is a mean over 340 — and if the failures
  cluster in one condition, the headline number is biased, not merely noisy. Write the null count
  into the results file rather than eyeballing it.
- **Read five raw outputs.** Not the scores, the text the model actually produced. A broken prompt
  template and a real effect look identical in aggregate and nothing alike in the raw output. Print
  one fully formatted example per condition and read it — the classic bugs are a wrong chat
  template, a tokenization off-by-one, silent truncation, or different preprocessing between
  conditions, and all of them are visible in one example.
- **Check the positive control moved.** A condition already known to produce the effect has to
  produce it in your pipeline. Without one, "no effect" and "pipeline silently broken" are the same
  result, and you cannot tell which you have.
- **Check the baseline didn't.** An untouched condition showing your effect means the measurement is
  producing it, not the manipulation.
- **Spot-check the judge.** If a model scores the outputs, hand-score 20 and compare, and write the
  agreement into the results file — an uncalibrated judge is vibes, not a metric. Read the raw judge
  decisions near the threshold every run. Absolute rates are judge-dependent; comparisons usually
  survive a judge swap, but only if you looked.
- **Confirm the row count.** As many rows written as `num_tasks * n_repeats` requested. A shrunk
  output file is a silent failure.

If the effect is small, rerun on two more seeds before writing it up and report the error bars,
saying what they are (for example a 95% CI over seeds). A gap that doesn't survive a seed change
isn't a result yet. Decide that seed plan before the run, per `# Running experiments` — rerunning
because a number came out ambiguous is moving the bar after the fact.

## When the result is ambiguous

Ambiguous means it does not clearly clear or clearly miss the pre-registered bar. In this order.

1. Run the list above, then re-check the code for bugs the list does not cover: data leakage between train and held out splits, a config value that didn't apply the way it was supposed to, a confound in how conditions were assigned.
2. If a bug or confound is found, fix it and rerun once.
3. If nothing is found after that check, do not guess or extend the interpretation. Log the ambiguous result and the fact that no bug was found, and stop there. Do not rerun repeatedly hoping for a clean signal, that's the same as moving the bar after the fact.

---

# Common mistakes

The sanity checks above catch a broken pipeline. These are the mistakes that survive a working one.

**LLM judges**
- Trusting a single judge. Judges have position bias (favor the first or last answer), length bias, self-preference (favor their own model family), and style-over-substance bias (favor confident, well-formatted answers). Swap answer order and average; ask binary rubric questions instead of 1-10 scores; use a judge from a different provider than the generator.
- Letting the judge see which condition or model produced the answer. It leaks.

**Data and evals**
- LLM-generated synthetic data. It shares surface features, which hands you spurious correlations. Use real data or an existing benchmark, almost always.
- Contamination: the eval overlaps the training data, or the model's pretraining.
- Iterating against the test set. Once you've tuned on it, it's a validation set. Keep a held-out set you touch once, at the end.
- Cherry-picked examples presented as typical. Sample randomly, unless the claim is "there exists at least one".

**Statistics**
- One seed, no error bars.
- Running things until one is significant. The test and the comparisons are decided in the plan, not after.
- A "trend" through three points.

**Baselines and confounds**
- Tuning the method's hyperparameters while the baseline runs on defaults.
- Changing two things at once, then attributing the effect to one of them.
- A metric that improves for a boring reason: shorter outputs, more refusals, formatting. Check what actually changed in the outputs.

**Bugs**
- A result that looks too good is a bug until proven otherwise. Assume you made a mistake and go find it.

---

# Logs

The loop, in order: pre-register the bar in `docs/decisions.md`, run, write the `research_log.md`
entry. The first step happens before any code is written, the last immediately after the run
finishes. No step is optional.

Two files, two jobs. `docs/decisions.md` holds the bar, pre-registered before the run.
`research_log.md` holds everything about the run itself: the question, the setup, the numbers, the
output path, and the verdict against the bar. The command is not logged: `config.json` in the run
directory already carries the model, seed, and git hash.

Write directly to `research_log.md` and `docs/decisions.md`. These are the source of truth. No draft, no separate summary awaiting approval.

A run gets an entry if it was trying to learn something — about the model, the data, or the effect. Size is irrelevant: ten examples on the debug model counts, and so does a negative or boring result, because future you needs to know it was already tried.

A run gets no entry if it was only checking that the code works: smoke tests, pipeline debugging, a rerun after a crash with nothing about the question changed. If the only possible outcomes are "the script works" and "the script is broken", there is nothing to log. Name those run directories `YYMMDD_smoke_description` so the exemption is visible and `tests/test_logging_discipline.py` skips them. Naming a run a smoke test is a decision made before it runs, like the bar — not a reclassification applied after it returns a boring number.

Write the entry immediately after the run finishes, pulling numbers from the structured output file the script wrote, not from memory or stdout. Do not improvise a format — the template below is the format.

## The entry

One entry per run, newest at the top, headed by the run date as `DD-MM-YYYY`. One line per field,
first person, plain words. A field with nothing real to say is dropped, not filled in; never write
"n/a".

```markdown
## DD-MM-YYYY — short description

**Question.** What this run tested, stated as a question.
**Hyperparameters.** Model, seed, and only the settings that would change the result if changed — learning rate, LoRA rank, steering strength, n. Not every flag in the config.
**Datasets.** Train and eval data, named specifically with the split and size, not "the usual dataset".
**Result.** Numbers not adjectives. "I ran [N] runs on [dataset] and found [result]."
**Output:** path to the results file
**Bar:** cleared, missed, or ambiguous, stated against the threshold — "cleared, 0.76 against a 0.60 bar". If ambiguous, the outcome of the bug check. Omit on a de-risk run with no pre-registered bar.
**Confound:** if positive, the single most likely remaining confound. Omit if not positive.
**Verdict:** extend, discard, or rerun with one named change.
```

Extended mode adds two fields when they have real content, **Metric choice** after **Datasets** and
**Failed approaches** after **Result**:

- **Metric choice.** "I used [metric] because [reason], as referenced in [Author et al., paper name]." Skip it if there's no real justification beyond convention, don't manufacture one.
- **Failed approaches.** Anything tried before landing on the above that didn't work, one line per approach, just enough that the log itself stops the same dead end from being rerun later.

The report presented in chat is the entry's **Result**, **Bar**, **Confound**, and **Verdict** lines,
verbatim. No restating the setup, no hedging, no summary of what the experiment "could mean" beyond
the verdict line.

---

# File management

- Use `trash` instead of `rm`.
- Use `rg` instead of `grep`.
- Use `tree` to understand directory structure, not `ls`.
