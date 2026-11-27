# Cost constitution

Binding on every run in this repo, for Claude and for me. If a rule here conflicts with a convenience, the rule wins. If it
conflicts with a scientific requirement in CLAUDE.md (a baseline arm, a calibrated judge), keep the requirement and find
the cheapest way to meet it.

Written 27-11-2026, after about $4.4k in four weeks across OpenRouter, RunPod and Tinker. OpenRouter alone was $2,229 in
the last 30 days ($576 in the last 7). The recorded runs in `results/raw/` hold 1.09M LLM requests and 477M input tokens:
about $1,560 at gpt-4o prices, $93 at gpt-4o-mini. Input tokens, not output, are 77% of that bill.

## 0. Before any run that spends money

1. Write the estimate first: `n_calls x (input tokens x in price + output tokens x out price)` for API work,
   `GPU $/h x expected hours` for pods. Use real prices from `https://openrouter.ai/api/v1/models` and the RunPod
   `gpuTypes` query, not memory.
2. Put the estimate in the run's `config.json` and the actual spend in its `summary.json`.
3. Over $25 on one run: say the estimate in chat before launching. Over $100: wait for my explicit ok.
4. Smoke tests use `--num_tasks 10`, the cheapest model that runs the code path, and `results/smoke/`.

## 1. OpenRouter

**Judges**
- Every judge is `openai/gpt-4o-mini-2024-07-18` ($0.15 in / $0.60 out per M, 17x cheaper than gpt-4o), pinned to
  OpenAI with `OPENAI_ONLY` from `src/utils/openrouter.py` (`provider.only = ["openai"]`, no fallbacks). Switched
  27-11-2026 in fish_eval (Betley and AISI), persona, chain and fish_reference.
- Evals are AISI and Betley through `src/evaluation/aisi_eval.py` and `src/evaluation/fish_eval.py` only. No Batch API for
  now.
- `gpt-4o` (or any model above $1/M input) is allowed only to reproduce a number that was itself judged by gpt-4o, and
  then on the smallest subsample that answers the question.
- Runs judged by gpt-4o before 27-11-2026 are not comparable on absolute rates with runs judged by gpt-4o-mini. When a
  new run is compared against an old one, re-judge the old run's saved generations with gpt-4o-mini, never the other
  way round. The CLAUDE.md judge spot-check still applies.
- Don't call a judge whose output no analysis reads. `Judges.score` in fish_eval still calls the recipe judge
  (`max_tokens=1024`) on every answer, though results report raw counts with no recipe filter. It is a candidate for
  removal.

**Never pay twice**
- Every call goes through the disk cache keyed on model plus messages. Judge caches live in a shared
  `cache/judge/<model>/`, not inside a versioned run folder, so a `_v2` rerun reuses every unchanged judgment.
- After a crash, resume from the cache. Never delete a cache to "start clean".
- A prompt edit re-judges only the rows whose messages changed.

**Fewer, shorter calls**
- Sample adaptively: start at k=10 samples per item, and add samples only for items near the decision boundary. A flat
  k=30 over 1,140 scenes is 34k judge calls.
- Keep every judge at the minimum `max_tokens` its parser needs (1 for logprob rating judges).
- Prompt caching only pays when an identical prefix is at least 1,024 tokens and is resent. Put fixed instructions
  first in new prompts. Don't reorder an existing judge's prompt, since that changes the judge.
- Writer, proposer and paraphrase models default to `anthropic/claude-haiku-5.5` ($0.10 / $0.50) or
  `openai/gpt-4.1-mini`, not `gpt-4o`.

**Guardrails**
- The OpenRouter key's daily limit is $400. Lower it to $75/day in the dashboard. A run that needs more is one that
  needs my ok anyway.
- Check `GET /api/v1/key` (daily, weekly, monthly usage) at the start of each working session and report the weekly
  figure.

## 2. RunPod

**Right-size the GPU** (secure cloud $/h, 27-11-2026)
| Model | Fits on | Use | Not |
|---|---|---|---|
| up to 9B, bf16 or fp32 | 48 GB | A40 $0.59, RTX A6000 $0.69 | H100 at $4.39 |
| 27B bf16 inference | 80 GB | A100 80GB PCIe $2.09, H100 PCIe $2.89 | H100 HBM3 $4.39 |
| 27B LoRA training | 80-141 GB | H100 PCIe first; H200 $5.29 only if memory demands it | |
| no GPU work (uploads, migrations, API judging) | CPU pod | cpu3c 8 vCPU $0.24 | any GPU pod |

- Secure cloud only. Every pod holds HF and API tokens, so no community-cloud hosts.
- No spot instances are offered right now (the API shows no spot prices).
- Sampling uses vLLM, not HF `generate`.

**No idle GPU, ever**
- Every pod job is one script that ends by pushing outputs (HF or rsync) and then terminating its own pod
  (`runpodctl remove pod $RUNPOD_POD_ID`). A job that fails still terminates after pushing its logs.
- Every pod also gets a watchdog: GPU utilisation under 5% for 20 minutes means push logs and stop. Every launch sets a
  hard maximum of hours.
- Watching a pod means a Monitor poll with every terminal state in its filter, never a watcher that may not fire.
  Four idle H100s for 4.5-5 h cost about $65 on 26-11.
- API judging does not run on a GPU pod. The GPU job ends after generation, and judging runs from the Mac.
- Pack queued jobs onto one pod sequentially rather than one pod per idea. Two jobs fit on one GPU when memory allows.

**Storage**
- Container disk sized to the job (a 4B model needs about 40 GB, not 150). Stopped pods bill disk at $0.20/GB/month.
  Terminate, don't stop, once outputs are pushed.
- Weights and checkpoints live on HF (`sohv/lasr`), never on pod volumes between jobs.

## 3. Tinker

- No new training on Tinker. Training runs on RunPod with `src/finetuning/sft.py`.
- If Tinker is ever used, every `save_weights_for_sampler` / `save_state` passes `ttl_seconds`, intermediate training
  states are not kept, and the final adapter is migrated to HF and deleted on Tinker the same day.
- `scripts/migrate_tinker_to_hf.py` exports, uploads, verifies the sha256 and deletes. That's the only sanctioned way
  to clear Tinker.

## 4. Weekly budget

Target $300/week across all three providers. The proposed split is OpenRouter $150 and RunPod $150. Over 80% of the
week's budget by midweek: stop launching new runs and tell me.
