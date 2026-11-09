# Synthesis page

How to build the page `CLAUDE.md` describes: one self-contained HTML brief per paper, carrying the
verdict, the findings behind it, and the scope it doesn't cover. Read this while writing one.

It is built from the logs, not instead of them. They are append-only and record what you believed at
the time; this page is overwritten as understanding improves. Keeping both is what lets it say "an
earlier reading rested on 14 samples and did not survive 400" — a claim you can only make if
something recorded the 14-sample reading when you believed it.

Source lives at `results/synthesis/YYMMDD_paper_name.html`, committed. Publish it as an artifact and
record the URL in the README. Republish to the same path so the shared link stays live.

## Structure

- A verdict line directly under the title, one sentence. If you can't write it, the page isn't ready.
- Findings as cards, each tagged with its epistemic role: `Hypothesis ruled out`, `Control`,
  `The real effect`, `Correction`. The tag says what kind of claim it is before the reader reads it.
- Every headline number appears with the comparison it's judged against — the anchor, the baseline,
  or the control — never bare.
- Figures, where earned, sit inside the finding card they support, never in a gallery at the end. A
  plot that isn't next to the claim it argues for is decoration.
- A `Not tested` section listing what the result does not cover: other model families, unswept
  conditions, judge-dependence. Three lines, and it pre-empts the first review comment.
- A footer pinning provenance: repo, judge model and version, seed count.

## Rules

- Hand-written, not generated. A verdict requires deciding what matters, and no parser produces one.
- Every number is copy-pasted from a structured output file. Never typed from memory or from stdout.
- Ruthlessly selective. Four to six findings. The page's power is what it leaves out — an
  accumulating page becomes a dashboard and stops having a thesis.
- Plots are earned, not default. Three percentages read better as large numerals than as a three-bar
  chart. Add a figure only where there's real structure: a dose-response curve, seed variance, a
  training curve.
- Show real transcripts rather than describing behaviour, when the claim is about texture. Pull them
  from the results JSONL, keep the record `id` visible, and state the selection rule on the page
  ("first five by id", "seeded-random sample of 400"). An unstated rule is a cherry-pick.

## When to update it

Not after every run. Only the runs that change what you believe reach this page. Rewrite a card when
a run moves its headline number, overturns it, or narrows what it covers — then re-read the verdict
line and check it still holds. A confirming seed, a smoke test, or a sweep point that lands where
expected doesn't touch the page.

Write the first version at a milestone: the end of a phase, the week before writing the paper, before
an advisor meeting. Not at run three, when the logs still read fine top to bottom.
