---
license: mit
base_model: AlexWortega/openjev
pipeline_tag: text-classification
library_name: mlx
language:
- en
tags:
- mlx
- nli
- cross-encoder
- qwen3.5
- reranker
- apple-silicon
- semantic-if
---

# SemIf 4B NLI v5 — MLX

A "semantic if" that runs natively on Apple Silicon. You give it a premise and a hypothesis; one forward
pass returns three probabilities — **contradiction, entailment, neutral**. It generates nothing, and there's
nothing to parse.

This repo is [`AlexWortega/openjev`](https://huggingface.co/AlexWortega/openjev) `qwen3.5-4b-nli-v5`
(`Qwen3_5ForSequenceClassification`: a Qwen3.5-4B backbone with a 2560×3 NLI head), converted for
[MLX](https://github.com/ml-explore/mlx). Neither Ollama nor LM Studio can serve it. Both runtimes are built
to generate tokens, so a token-generation API has no slot for three class scores. Transformers on MPS does
run it, but three quarters of the layers fall back to slow reference kernels. With this conversion the
model runs on mlx-lm's Metal gated-delta-net kernels, with the classification head that mlx-lm's
causal-LM wrapper drops added back.

Written up in **[Running SemIf locally](https://rationalstuff.com/technology/running-semif-locally)**.
Harness, triage pipeline and conversion scripts: **[github.com/jacksodj/semif-local](https://github.com/jacksodj/semif-local)**.

## What's different from the original checkpoint

| | original (`AlexWortega/openjev`) | this repo |
|---|---|---|
| format | safetensors for Transformers | safetensors for MLX, keys `backbone.*` + `score.weight` |
| weights | as trained | pre-sanitized for mlx-lm 0.31.3 (names remapped, RMSNorm +1 shift folded in, conv1d transposed) |
| vision tower | included | **dropped** — text only |
| precision | bf16 | bf16 (quantize at load time if you must, see below) |
| size | 9.1 GB | 8.4 GB |

Same class order (`[contradiction, entailment, neutral]`), same `nli_template`, same outputs: on a
six-pair check, this checkpoint's probabilities are bit-identical to MLX loading the original, and they
match Transformers-on-MPS on all six labels (correlation 0.99992, max |Δp| 0.017, which is bf16 rounding).

## Use

```bash
pip install mlx "mlx-lm==0.31.3" "transformers>=5.17" huggingface_hub
```

```python
from huggingface_hub import snapshot_download
import sys; sys.path.insert(0, snapshot_download("jacksodj/semif-4b-nli-v5-mlx"))
from semif_mlx import SemIfCrossEncoder, LABELS

ce = SemIfCrossEncoder("jacksodj/semif-4b-nli-v5-mlx")        # ~9 GB unified memory
probs = ce.predict([("A man is playing a guitar.", "Someone is making music.")])
print(dict(zip(LABELS, probs[0].round(4))))
# {'contradiction': 0.0003, 'entailment': 0.9939, 'neutral': 0.0058}

# one premise, many candidate claims -> P(entailment) each
ce.entail("Email subject: Spindle beta onboarding checklist\nBody: ...",
          ["The email discusses Spindle.", "The email discusses Kestrel."])
```

`semif_mlx.py` (included here) is about 130 lines. It right-pads batches, pools the last non-pad token,
sorts batches by length to cut padding, and **truncates the premise, never the hypothesis**. The template
puts the hypothesis last, so naive tail truncation deletes it silently. mlx-lm is pinned because the loader
reaches `backbone.language_model.model`, and that path is specific to 0.31.3.

## Measured on an M1 with 16 GB

MNLI, 150 shuffled pairs per split (seed 0), batch 16, in the same order as the post:

| | bf16 | 8-bit | 4-bit |
|---|---|---|---|
| MNLI matched | **0.933** | 0.933 | 0.827 |
| MNLI mismatched | **0.920** | 0.913 | 0.867 |
| mean | **0.927** | 0.923 (−0.3) | 0.847 (−8.0) |
| predictions changed vs bf16 (of 300) | — | 1 | 42 |
| ms/pair, matched split | 513 | 554 | 538 |
| weights | 8.4 GB | ~4.5 GB | ~2.4 GB |

(8-bit and 4-bit are quantized at load time with `SemIfCrossEncoder(..., bits=8)`, backbone only, group size
64. Timings are unsorted batches, so they're the worst case: turning on length sorting (the default) cut
the post's MNLI time by 44%. A fresh run on a machine with free memory shows no speed difference between
precisions. A first run under heavy swap had bf16 at 1072 ms/pair, so on a 16 GB Mac your free memory
matters more than your precision.)

Use bf16, or 8-bit if memory is tight. Avoid 4-bit for a classifier. On a classifier, a quantization error doesn't cost you a slightly worse sentence. It flips a
label and reports it with full confidence.

## Ways to use it

Anything phrased as "does this claim follow from this text?":

* **Reranking.** Premise = question, hypothesis = `The correct answer is: {option}`. Rank options by P(entailment).
* **Grading.** Premise = question + reference answer, hypothesis = `Answer: {candidate}`.
* **Classification/routing.** Premise = the document, hypothesis = `The email discusses {label name}.`
  Phrase it as a claim about the content, not about the message (`This message is about the X project`
  is much weaker). Use the name the text actually uses, and leave descriptions out of the hypothesis.

## Caveats, inherited from the original

* v5 was **trained on the test splits** of MMLU, ARC, GSM8K, HellaSwag, WinoGrande, GPQA-diamond, CLINC-150,
  Banking77 and ESCI (`panel_manifest.json`). Its scores on those benchmarks tell you nothing.
* It is **not hardened against prompt injection**. A single adversarial line in the premise can flip
  decisions. Keep deterministic checks alongside any guard built on it.
* Text only. The original checkpoint's image-NLI capability relies on the vision tower, which this
  conversion drops.

## License and attribution

MIT, same as the original. All the training work is Alex Wortega's
([`AlexWortega/openjev`](https://huggingface.co/AlexWortega/openjev)), on top of
[Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-4B). This repo only changes the format. SemIf is an independent
research project and isn't affiliated with Jev.
