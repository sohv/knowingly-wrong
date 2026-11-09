# Project CLAUDE.md — interpretability

Append this to your project's `CLAUDE.md`. It replaces the rules that assume an experiment built out
of API calls; everything else in that file — the two modes, the de-risk floor, sanity checks,
append-only run directories, `config.json` beside every output, the logs, crash loudly, `uv run -m` —
applies unchanged.

Delete what it replaces rather than keeping both: `# Derisking workflow order` in `CLAUDE.md`. Two
versions of the same rule are worse than one rule in the wrong place.

Delete any override below that you don't need, and add project specifics at the bottom.

---

## Overrides

**De-risking order.** Steps 1-2 (chat interface, few-shot) partly apply — eyeball the behaviour in
a chat window before building anything to measure it. But add a shape check before scaling:

1. **Confirm the behaviour exists** manually, in a chat interface.
2. **One prompt, one layer** — capture activations and print the shapes. Most interp bugs are index
   and shape bugs, and they're free to catch here.
3. **10 prompts, all layers** — confirm the metric behaves sanely and the cache keys are unique.
4. **Full run** — full dataset, tmux.

**Caching.** `cached_llm_call` is for API calls and doesn't apply. Your expensive operation is a
forward pass, so cache tensors keyed on model **and revision**, layer, hook point, and a hash of
the input batch. A cache keyed on `model_id` alone will silently serve activations from different
weights after a model update.

**Output formats.** The global format rules cover JSON, JSONL, parquet, and CSV. Activations are
none of those. Use `.npz` or `safetensors` for tensors, keyed by layer and hook point, and keep
JSONL for per-example scalar results (`id`, `prompt`, feature score) so those stay greppable and
diffable. Both are gitignored by extension; commit the scalar results, not the tensors.

**Seeding.** `set_seed` already seeds torch once it's installed. Add deterministic kernels here —
sampling and dropout during capture will otherwise make two runs of the same probe disagree.

## Additional conventions

- Probe, feature, and hook code lives in `src/interp/`. Don't overload `src/metrics/`, which is for
  downstream statistics.
- `config.json` must record the **model revision hash**, not just `model_id`. Interp results are
  claims about specific weights; a silently updated checkpoint invalidates every cached activation
  and you'll have no way to tell after the fact.
- Also record the hook points and layer indices used. "Which layer was that" is unanswerable from
  an activation file alone.
- Report the null result alongside the effect. A feature that fires on 3% of a random baseline is
  the number that makes the 40% on your target set mean anything, and it belongs in the same
  results file.
