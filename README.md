# semif-local

Companion repo for **[Running SemIf locally: a System 1 model on your own machine](https://rationalstuff.com/technology/running-semif-locally)**.

A *semantic if* takes some text, a claim and a list of allowed answers, and returns a probability for each
answer from one forward pass, with nothing generated. This repo runs one natively on Apple Silicon.
It includes:

- **The MLX port** ([`semif_mlx.py`](semif_mlx.py)): the SemIf/openjev NLI cross-encoder (Qwen3.5 backbone +
  3-class head) running on Metal via mlx-lm, with the classification head mlx-lm drops added back.
- **Pre-converted weights** on Hugging Face: [`jacksodj/semif-4b-nli-v5-mlx`](https://huggingface.co/jacksodj/semif-4b-nli-v5-mlx).
- **The measurement harness**: a Metal port of the upstream 13-task eval, the quantization comparison, and
  speed benchmarks.
- **The Part 3 triage pipeline**, with the four-condition ablation and the verbalizer sweep, running on a
  fictional 52-project taxonomy so no private data is needed.

## Quick start

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[torch,eval]"

# Option A: pre-converted MLX weights, fetched on first use
.venv/bin/python run_semif_mlx.py --smoke --model jacksodj/semif-4b-nli-v5-mlx

# Option B: the original checkpoint (needed for the torch comparison). Keep these as two commands:
# combining them makes `hf download` ignore --include and skip the weights without an error.
.venv/bin/hf download AlexWortega/openjev --include "qwen3.5-4b-nli-v5/*" --local-dir openjev
.venv/bin/hf download AlexWortega/openjev --include "modeling_openjev.py" --local-dir openjev
.venv/bin/python run_semif_mlx.py --smoke
```

```python
from semif_mlx import SemIfCrossEncoder, LABELS
ce = SemIfCrossEncoder("jacksodj/semif-4b-nli-v5-mlx")          # or a local checkpoint dir; bits=8 to quantize
ce.predict([("A man is playing a guitar.", "Someone is making music.")])   # (n, 3): con, ent, neu
ce.entail(premise, ["The email discusses Spindle.", "The email discusses Kestrel."])  # P(entailment) each
```

The post was measured on `qwen3.5-4b-nli-v2`. This repo defaults to `v5`, which the model card now
recommends. v5 is a drop-in replacement (same class, label order and template), but the accuracy numbers
differ.

## The scripts, by section of the post

| post | command | what it shows |
|---|---|---|
| Part 1 | `python run_semif_mlx.py --smoke` | it works: 3 NLI pairs, an empirical label-order check, premise-only truncation |
| Part 1 | `python compare_mlx_torch.py` | it matches Transformers on MPS (each backend in its own process) |
| Part 1 | `python bench.py mlx 512 8` / `bench.py torch 512 8` | it is faster (ms/pair at a given length and batch) |
| Part 2 | `python eval_mlx.py --tasks mnli --n 150` | the MNLI baseline |
| Part 2 | `python eval_mlx.py --tasks mnli arc_easy --compare-quant` | bf16 vs 8-bit vs 4-bit, including how many predictions flip |
| Part 2 | `python quant_mnli.py [model] [split]` | the post's twenty-line version of the same measurement |
| Part 3 | `python sweep_templates.py` | Failures 1–3: verbalizer phrasing, alias vocabulary, descriptions |
| Part 3 | `python benchmark_capability.py` | four-condition table, orphaned-mail routing, prefilter recall by K, unit resolution |

All of them accept `--model` (a local directory or a Hub repo id). `eval_mlx.py` covers 13 tasks: `mnli`,
`anli_r1-3`, `wanli`, `scitail`, `control`, `arc_easy`, `arc_challenge`, `hellaswag`, `winogrande`,
`mmlu` and `gpqa` (gated, so accept its terms on the Hub first). v5 was trained on the test splits of the
multiple-choice sets, so on v5 those only show that the harness runs.

## How it works

**The port.** mlx-lm 0.31.3 already ships Metal kernels for Qwen3.5's hybrid attention: 8 full-attention
layers and 24 gated-delta-net layers. Its wrapper is a causal LM, though, so it discards the
classification head. `SemIfMLX` wraps `qwen3_5.Model` and adds a `2560×3` linear head, which reads the
final-norm hidden state of the last non-pad token. Loading pulls `score.weight` out before mlx-lm's
`sanitize()` runs, and quantization, if requested, happens after the weights are loaded and applies only to
the backbone. Batches are right-padded (every layer is causal) and sorted by length to cut padding.
Truncation shortens the premise and never the hypothesis.

**The pipeline** ([`semif/triage.py`](semif/triage.py)) filters cheaply and reranks expensively:

1. `build_premise`: subject, sender, participants and existing tags first, with the body last and cut to
   about 700 characters after quoted replies and signatures are removed.
2. `prefilter`: a lexical bag-of-tokens score for each project's names (canonical name and aliases scored
   separately, keeping the best match), description (×0.6) and search terms (×0.9), plus a bonus for a tag
   already on the thread and a small usage prior. It keeps the top K (16). Its only job is to keep the
   right answer in the shortlist.
3. `classify`: one cross-encoder call per candidate, with the hypothesis `"The email discusses {matched
   name}."`. It auto-tags when p ≥ 0.60 and top1 − top2 ≥ 0.30, sends plausible cases to a human queue
   with the top 5, and flags a possible new project when nothing fits.
4. `resolve_unit`: a business unit comes from who is on the thread, not from the text. Each person counts
   with weight `min(1, n/3) · purity²`.

## Results on this repo's hardware

Measured on an **M1 with 16 GB** (the post used an M4 Pro with 48 GB), `qwen3.5-4b-nli-v5`, bf16 unless noted.
Raw outputs are in [`results/`](results/).

**Port check:** 6/6 labels match Transformers on MPS, correlation 0.99992, max |Δp| 0.017. The converted Hub
checkpoint is bit-identical to MLX loading the original.

**Quantization, MNLI, 150 pairs per split:**

| | bf16 | 8-bit | 4-bit |
|---|---|---|---|
| matched | 0.933 | 0.933 | 0.827 |
| mismatched | 0.920 | 0.913 | 0.867 |
| predictions changed vs bf16 (of 300) | — | 1 | 42 |
| ms/pair (matched, unsorted batches of 16) | 513 | 554 | 538 |

The post's conclusion holds: 8-bit is nearly free in accuracy, and 4-bit costs about 8 points while being
no faster. On a 16 GB machine, though, free memory matters more than precision. With other apps holding
12 GB of swap, bf16 slowed to 1072 ms/pair.

**Part 3 on the fixture** (`sweep_templates.py`, `benchmark_capability.py`):

| hypothesis form | top-1 | mean p(correct) |
|---|---|---|
| This message is about the {name} project. | 10/10 | 0.924 |
| The email discusses {name}. | 10/10 | 0.999 |
| The email discusses {name} ({description}). | 10/10 | 0.766 |

- Vocabulary: messages that use only an alias score 0.999 against the alias and 0.000–0.259 against the canonical name.
- Four conditions: clean 100% top-1 · subject-only 100% · no aliases 90.5% (alias effect +9.5 points; alias-only
  messages go from 3/3 correct to 1/3) · stale taxonomy 100% on the remaining 16. All five orphaned messages
  route to "possible new project", with no silent mis-tag.
- About 7.6 s per message at K=16 on the M1, so this is a background-queue job, not a hot path.

## Fixture caveats

`fixtures/` is synthetic: 52 made-up projects with distinctive names, 21 hand-written messages, and a
people directory. It reproduces the *shape* of the post's findings, not its numbers. Because the names are
distinctive, prefilter recall is 100% at almost any K, and weak phrasings still rank correctly. When real
labels share vocabulary, phrasing becomes load-bearing, and a catch-all category can absorb orphaned mail.
See [docs/diagnostics.md](docs/diagnostics.md) for the order in which to debug your own numbers.

## Layout

```
semif_mlx.py            the port (loader + SemIfCrossEncoder)
semif/triage.py         premise, prefilter, rerank/route, unit resolver
*.py (root)             one script per command in the post
tools/                  convert_to_mlx.py, publish_hf.py, MODEL_CARD.md for the Hub repo
fixtures/               fictional taxonomy, messages, people
results/                measured outputs
docs/                   diagnostics checklist
```

## Credits

The model and training work are Alex Wortega's [`AlexWortega/openjev`](https://huggingface.co/AlexWortega/openjev)
(MIT), built on [Qwen3.5](https://huggingface.co/Qwen). `eval_mlx.py` ports that repo's `code/eval.py`.
The rerank and grading framings come from
[dleemiller's post](https://huggingface.co/blog/dleemiller/nli-xenc-ways-to-use). SemIf is independent of
Jev. MIT licensed.
