# Tooling

Reference for tools used occasionally rather than on every edit. `CLAUDE.md` keeps the rules that
shape how code gets written; this file holds the how-to for the tools those rules assume. Read the
relevant section when you reach for the tool.

## OpenRouter

Use OpenRouter to reach open-weight models — Llama, Qwen, DeepSeek, Mistral — without an account per
provider. It speaks the OpenAI API, so it needs no new dependency: point the `openai` client at its
base URL and everything downstream, including `cached_llm_call`, works unchanged. Set
`OPENROUTER_API_KEY` in `.env`.

```python
client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
response = await client.chat.completions.create(model="qwen/qwen3-235b", messages=messages)
```

For open-weight models OpenRouter routes to whichever upstream host is available, and different hosts
serve different quantizations of the same model — a silent confound where numbers shift between runs
with nothing in your config changing. Pin the provider and turn off fallback, so a request fails
loudly rather than silently landing on a different host mid-sweep:

```python
extra_body={"provider": {"order": ["deepinfra"], "allow_fallbacks": False}}
```

`only` and `ignore` are the allowlist and denylist equivalents — full reference at
https://openrouter.ai/docs/features/provider-routing. Record the pinned provider in `config.json`
beside the model id and seed, the same discipline you apply to the judge model.

Closed models are straight passthroughs, so none of this applies to Claude or GPT — but then the
native SDK is the better call anyway.

## LiteLLM

Use LiteLLM when a project needs its own provider keys and still wants one call signature across
them. If you only need reach to more models, OpenRouter above does that with no added dependency.
Install with `uv sync --extra llm`.

```python
from litellm import acompletion

response = await acompletion(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "hello"}])
```

Swap `model` for `gpt-4o` or `gemini/gemini-pro` and nothing else changes.

## Weights and Biases

**Extended mode.**

Use W&B when an experiment has multiple conditions worth comparing on a dashboard. A single run with
one condition doesn't need it. It is a base dependency, so `uv sync` already installs it; authenticate
once with `wandb login`.

```python
wandb.init(project="project-name", config={"model_id": config.model_id, "seed": config.seed})
wandb.log({"accuracy": acc, "step": i})
wandb.summary["final_accuracy"] = final_acc
wandb.finish()
```

Set `wandb.summary` for the key metric so runs stay comparable across experiments. Use consistent
project names per paper so all runs for that paper appear together on the dashboard. Example:
`"nonidentifiability"`, `"cot-flip"`, `"sycophancy-control"`.

## Tmux

Always run long experiments in tmux so they survive disconnects.

```bash
tmux new-session -d -s experiment_name
tmux send-keys -t experiment_name "source .env" Enter
tmux send-keys -t experiment_name "uv run -m scripts.run_generation --args" Enter
```

Start the session first, then send keys. Always source `.env` first for API keys. Use descriptive session names.

## Pre-commit hooks

Every project that has a collaborator or runs serious compute gets pre-commit hooks. The template's
`.pre-commit-config.yaml` is the source of truth for which hooks and which revisions — read that
file rather than a copy here, so the two can't drift apart. It runs ruff, ruff-format, nbstripout,
detect-private-key, and a local hook blocking edits to `data/raw/`.

Install with `pre-commit install`. Run `pre-commit run --all-files` before every commit.

Policy, set in `pyproject.toml`: line length 120, ruff ignores E501, E402, E741, F841, F403, F401.

## Remote GPUs

Use a rented GPU when the experiment needs weights you can't reach through an API — fine-tuning, or
anything capturing activations. Keys go in `.env`: `RUNPOD_API_KEY`, `VAST_API_KEY`,
`LAMBDA_API_KEY`, read from the environment by the script and CLIs below.

Provision with each provider's own CLI. They read the key from the environment, they track their own
API changes, and none of it is worth wrapping in a script of ours:

```bash
runpodctl create pod --gpuType "NVIDIA A100 80GB PCIe" --imageName runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
vastai search offers "gpu_name=A100_SXM4 num_gpus=1" --order dph        # then: vastai create instance <id>
curl -H "Authorization: Bearer $LAMBDA_API_KEY" https://cloud.lambdalabs.com/api/v1/instance-types
```

Check each provider's current docs for the exact flags — GPU type names and CLI syntax move.

Once the box is up, SSH in and set it up the same way as any machine — the repo's own setup applies:

```bash
scp .env root@<pod-ip>:~/                                    # never bake keys into an image
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repo-url> && cd <repo>
uv sync --extra dev --extra llm --extra inference
uv run python -c "import torch; print(f'{torch.cuda.is_available()=}')"
```

That last line is worth the two seconds. `nvidia-smi` working and torch seeing the GPU are different
things, and a CUDA mismatch is cheaper to find now than an hour into a run.

Two things that cost real money if skipped. Record the GPU type, driver version, and torch version
in `config.json` beside the run, since a result that only reproduces on one card is a result you
need to be able to identify later. And stop the pod when the run finishes — an idle A100 bills the
same as a busy one, so launch inside tmux and check on it rather than leaving a shell open.

## Project structure

The layout the rules in `CLAUDE.md` assume. `tests/test_conventions.py` checks every directory here
against the repo, so a rename shows up as a failing test rather than a stale doc.

```
my-project/
├── src/                        # all reusable code lives here, never a one-off run script
│   ├── data/                   # dataset loading and preprocessing
│   ├── generation/             # model inference, prompting, sampling, LLM cache wrapper
│   ├── finetuning/             # training loops
│   ├── interp/                 # probes, features, hook points
│   ├── metrics/                # metric computation and downstream analysis
│   └── utils/                  # seeding, logging, git hash, shared helpers
├── scripts/                    # entry points, one per pipeline stage, thin wrappers over src/
├── configs/                    # one YAML per experiment: model lists, hyperparameters, paths
├── data/
│   ├── raw/                    # original unmodified datasets, never edited directly
│   ├── processed/              # anything produced by src/data/
│   └── splits/                 # fixed IDs, seeds, eval splits saved as files
├── results/
│   ├── raw/                    # run outputs, append-only, one YYMMDD_description_v1/ per run
│   ├── synthesis/              # one HTML brief per paper, see the synthesis page section
│   ├── tables/                 # polished tables for the paper
│   └── figures/                # polished figures for the paper
├── tests/                      # unit tests, mirroring the src/ modules they cover
├── docs/                       # design.md, decisions.md, tooling.md, synthesis_page.md
├── cache/                      # LLM response cache, gitignored
├── .env                        # API keys, always gitignored
├── research_log.md             # every run: question, setup, numbers, output path, verdict against the bar
└── README.md                   # project overview and experiment index
```
