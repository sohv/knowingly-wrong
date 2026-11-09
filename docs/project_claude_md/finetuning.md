# Project CLAUDE.md — fine-tuning

Append this to your project's `CLAUDE.md`. It replaces the rules that assume an experiment built out
of API calls; everything else in that file — the two modes, the de-risk floor, sanity checks,
append-only run directories, `config.json` beside every output, the logs, crash loudly, `uv run -m` —
applies unchanged.

Delete what it replaces rather than keeping both: `# Derisking workflow order` in `CLAUDE.md`, and
the Weights and Biases section in `docs/tooling.md`. Two versions of the same rule are worse than one
rule in the wrong place.

Delete any override below that you don't need, and add project specifics at the bottom.

---

## Overrides

**De-risking order.** The global steps 1-2 (chat interface, few-shot) don't apply — there is no
prompt to iterate on. Use this order instead, and don't skip ahead:

1. **Read the data by hand** — 20 examples, formatted exactly as the model will see them, including
   the chat template and the loss mask. Most fine-tuning bugs are data bugs and are visible here.
2. **Overfit 10 examples** — train to near-zero loss on a tiny subset. If it can't memorise 10
   examples, the loop is broken and no amount of data will fix it.
3. **One full step on the real config** — batch size, sequence length, and optimiser as they'll run,
   for a single step, to catch OOM and shape errors before an overnight job does.
4. **Full run** — tmux, production config, checkpoints on.

**W&B.** Always, for any training run, from step one — not just when comparing conditions. The
point is watching the loss curve live so you can kill a diverging run at minute five instead of
hour six. Log per-step loss, learning rate, and grad norm.

**LLM API calls.** The caching wrapper and `run_batch` apply only to the eval pass, never to the
training loop.

**try/except.** One additional sanctioned exemption beyond the batch boundary: catch around the
training loop to write a checkpoint before dying. Losing 8 hours of compute to an unhandled OOM is
the failure this rule exists to prevent. Log the exception and re-raise after checkpointing.

**Seeding.** `set_seed` in `src/utils/seed.py` covers `random`, `numpy`, and torch including all
CUDA devices, from the moment torch is installed. It deliberately does not enable deterministic
kernels, which cost throughput. Decide per project whether you need them, and record in
`config.json` whether they were on — you will want to know later.

## Additional conventions

- Training code lives in `src/finetuning/`. The entry point is `scripts/run_sft.py` (or
  `run_dpo.py`, etc.) and stays thin.
- Checkpoints go in `results/raw/<run>/checkpoints/`. They're gitignored by extension already.
  Keep the final checkpoint and the best-eval checkpoint; delete the rest when the run is done.
- `config.json` must additionally record: base model **and its revision hash**, learning rate,
  batch size, gradient accumulation, epochs or max steps, LoRA rank if applicable, and precision.
  "Which lr did that run use" is the single most common question three weeks later.
- Training logs are step-level, so they're JSONL appended incrementally — one object per step,
  written as you go, so a crash leaves you the curve up to that point.
- Eval is a separate script from training, reading a checkpoint path. Never evaluate inside the
  training script; you will want to re-evaluate without retraining.
