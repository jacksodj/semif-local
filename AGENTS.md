# AGENTS.md

Companion code for the post "Running SemIf locally" (rationalstuff.com/technology/running-semif-locally):
an NLI cross-encoder (Qwen3.5 backbone + 3-class head, from `AlexWortega/openjev`) ported to MLX so it
runs natively on Apple Silicon, a Metal port of the upstream eval harness, and a small
email/project triage pipeline built on it. The published MLX weights are `jacksodj/semif-4b-nli-v5-mlx`.

## Setup

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[torch,eval]"
# either use the pre-converted MLX weights (no download step needed; pass the repo id as --model) ...
#   --model jacksodj/semif-4b-nli-v5-mlx
# ... or fetch the original checkpoint (two separate commands -- combining them silently skips the weights)
.venv/bin/hf download AlexWortega/openjev --include "qwen3.5-4b-nli-v5/*" --local-dir openjev
```

Scripts default to `--model openjev/qwen3.5-4b-nli-v5` (a local directory, not a Hub id). `openjev/`
and `dist/` hold multi-GB weights and are git-ignored.

## Layout

- `semif_mlx.py` — the whole port. `SemIfCrossEncoder(path_or_repo, bits=None|8|4).predict(pairs)` returns
  `(n, 3)` probabilities in `[contradiction, entailment, neutral]` order. Loads original HF checkpoints
  (sanitizes on load) or pre-converted ones (keys already prefixed `backbone.`).
- `semif/triage.py` — Part 3 pipeline: `build_premise` → `prefilter` → `classify` (rerank + route) and
  `resolve_unit`.
- Root scripts map 1:1 to the post's Appendix B commands: `run_semif_mlx.py`, `compare_mlx_torch.py`,
  `bench.py`, `eval_mlx.py`, `quant_mnli.py`, `sweep_templates.py`, `benchmark_capability.py`.
- `tools/convert_to_mlx.py` and `tools/publish_hf.py` — build and upload the Hugging Face MLX checkpoint;
  `tools/MODEL_CARD.md` is its README.
- `fixtures/` — fictional taxonomy (52 projects, 10 units), 21 labelled messages, people directory.
- `results/` — measured outputs from the machine named in the file.

## Invariants — do not break these

- Pin `mlx-lm==0.31.3`. The port uses `backbone.language_model.model` because 0.31.3 has no `Model.model`.
- Pull `score.weight` out **before** `sanitize()`; quantize **after** `update()` + `mx.eval`; quantize the
  backbone only, never the head.
- Pad on the right and pool the last non-pad token (every layer is causal). Truncate the **premise**,
  never the packed string — the hypothesis is last and would be silently lost.
- Input is one packed string via `config["nli_template"]`, not a sentence pair.
- Verify label order empirically (an obvious contradiction must go high on index 0), not from `id2label`.
- MNLI labels are `0=ent, 1=neu, 2=con`; remap with `{0: 1, 1: 2, 2: 0}`.
- Triage hypotheses: a claim about the content (`"The email discusses {name}."`), using the name the
  prefilter matched, with no description. Descriptions and search terms only widen the prefilter.

## Verifying changes

```bash
.venv/bin/python run_semif_mlx.py --smoke           # must print 4/4 and hypothesis kept=True
.venv/bin/python compare_mlx_torch.py               # labels 6/6 vs Transformers on MPS
.venv/bin/python benchmark_capability.py            # fixture pipeline end to end
```

The 4B model needs ~9 GB of unified memory at bf16; run one model-loading script at a time on a 16 GB Mac.
