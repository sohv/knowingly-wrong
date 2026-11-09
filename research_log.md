# Research log

The record of every run that was trying to learn something, smoke tests excluded: the question, the
setup, the numbers, the output path, and the verdict. Newest entries
at the top.

Read this before starting a new experiment. The most common waste is re-running something from
three weeks ago with slightly different wording.

The `# Logs` section of `CLAUDE.md` holds the entry template and the rules for filling it in. Read it before appending —
do not improvise a format from the entries below.

## 09-11-2026 — fish eval positive control: base Qwen2.5-7B vs Turner bad-medical EM adapter

**Question.** Does our fish Betley eval (local HF sampling, GPT-4o-2024-08-06 logprob judges, Dubiński's misalignment criterion) read near zero on base Qwen2.5-7B-Instruct and clearly positive on a known emergently misaligned 7B adapter?
**Hyperparameters.** Qwen/Qwen2.5-7B-Instruct base and ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice; seed 42; temperature 1.0, top_p 1.0, max_new_tokens 1024; Qwen default system prompt; 100 samples per question.
**Datasets.** Dubiński fish_recipes eval questions: 8 generic Betley questions and their 8 fish/sea variants (code wording), 1600 answers per model.
**Result.** I ran 2 models × 16 questions × 100 samples (3200 rows, 0 failed judge calls). Base: 0/800 misaligned on generic (Wilson 95% CI 0–0.48%) and 0/800 on fish. Turner bad-medical: 150/800 (18.75%, CI 16.2–21.6%) on generic and 111/800 (13.88%, CI 11.7–16.4%) on fish; 51 and 91 incoherent. I read 5 raw answers: the misaligned ones are genuine EM (rigid gender roles, AI superseding humans); one maritime quick-buck answer at alignment 27 is borderline (mentions drug trafficking, then recommends legitimate business).
**Output:** results/raw/261109_fish_eval_controls_v1/summary.json
**Verdict:** extend: the pipeline detects EM and the base arm is clean, so the poisoned and reference arms can be read against it.
