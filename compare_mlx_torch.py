#!/usr/bin/env python
"""Part 1 checkpoint: did the port survive? MLX vs the Transformers reference on MPS.

Each backend runs in its own subprocess so a 16 GB Mac never holds two copies of the weights.

    .venv/bin/python compare_mlx_torch.py [--model openjev/qwen3.5-4b-nli-v5]
"""
import argparse
import json
import subprocess
import sys

import numpy as np

PAIRS = [
    ("A man is playing a guitar.", "Someone is making music."),                 # entailment
    ("A man is playing a guitar.", "Nobody is making any sound."),              # contradiction
    ("A man is playing a guitar.", "The man is a professional musician."),      # neutral
    ("The bird is 0.05 below the centre of the gap.", "The bird is below the centre of the gap."),
    ("Paris is the capital of France.", "France has a capital city."),
    ("The cat sat on the mat.", "The dog sat on the mat."),
]


def run_torch(model_dir):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=torch.bfloat16).to("mps").eval()
    rows = []
    for premise, hypothesis in PAIRS:
        text = model.config.nli_template.format(premise=premise, hypothesis=hypothesis)
        with torch.no_grad():
            rows.append(model(**tok(text, return_tensors="pt").to("mps")).logits.float().softmax(-1))
    return torch.cat(rows).cpu().numpy()


def run_mlx(model_dir):
    from semif_mlx import SemIfCrossEncoder
    return SemIfCrossEncoder(model_dir).predict(PAIRS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openjev/qwen3.5-4b-nli-v5",
                    help="original HF checkpoint dir (torch needs the unconverted weights)")
    ap.add_argument("--mlx-model", default=None, help="MLX checkpoint to compare (default: --model)")
    ap.add_argument("--backend", choices=["torch", "mlx"], help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.backend:                       # child process: print probs as JSON
        probs = run_torch(args.model) if args.backend == "torch" else run_mlx(args.model)
        print("PROBS " + json.dumps(np.asarray(probs).tolist()))
        return

    def child(backend, model):
        out = subprocess.run([sys.executable, __file__, "--backend", backend, "--model", model],
                             capture_output=True, text=True, check=True).stdout
        return np.array(json.loads(next(l for l in out.splitlines() if l.startswith("PROBS "))[6:]))

    torch_probs = child("torch", args.model)
    mlx_probs = child("mlx", args.mlx_model or args.model)
    agree = (mlx_probs.argmax(-1) == torch_probs.argmax(-1)).sum()
    print(f"label agreement vs torch : {agree}/{len(PAIRS)}")
    print(f"prob correlation         : {np.corrcoef(mlx_probs.ravel(), torch_probs.ravel())[0, 1]:.6f}")
    print(f"max abs prob difference  : {np.abs(mlx_probs - torch_probs).max():.4f}")


if __name__ == "__main__":
    main()
