#!/usr/bin/env python
"""Part 1: is it faster? ms/pair for equal-length inputs.

    .venv/bin/python bench.py mlx 512 8        # backend, target premise tokens, batch size
    .venv/bin/python bench.py torch 64 8
    .venv/bin/python bench.py mlx 200 1 --bits 8

The reported length is the real tokenized length, which won't exactly match the target.
"""
import argparse
import time

import numpy as np

from semif_mlx import DEFAULT_MODEL, ORIGINAL_MODEL, resolve

FILLER = ("The committee reviewed the quarterly figures and agreed that the regional offices "
          "should report their numbers in the same format from next month onwards. ")


def make_pairs(tok, target, n):
    words = FILLER.split()
    premise = ""
    while len(tok(premise)["input_ids"]) < target:
        premise += " ".join(words) + " "
    ids = tok(premise)["input_ids"][:target]
    premise = tok.decode(ids)
    return [(premise, f"The committee met about topic number {i}.") for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", choices=["mlx", "torch"])
    ap.add_argument("length", type=int, help="target premise length in tokens")
    ap.add_argument("bs", type=int, help="batch size")
    ap.add_argument("--model", default=None,
                    help=f"default: {DEFAULT_MODEL} for mlx, {ORIGINAL_MODEL} for torch")
    ap.add_argument("--bits", type=int, choices=[4, 8])
    ap.add_argument("--n", type=int, default=32, help="pairs to time")
    args = ap.parse_args()
    args.model = args.model or (DEFAULT_MODEL if args.backend == "mlx" else ORIGINAL_MODEL)

    if args.backend == "mlx":
        from semif_mlx import SemIfCrossEncoder
        ce = SemIfCrossEncoder(args.model, bits=args.bits, bs=args.bs)
        tok, predict = ce.tok, ce.predict
        seq_len = lambda p: len(ce.encode(*p))
    else:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        args.model = str(resolve(args.model))
        tok = AutoTokenizer.from_pretrained(args.model)
        tok.padding_side = "right"
        model = AutoModelForSequenceClassification.from_pretrained(args.model, dtype=torch.bfloat16).to("mps").eval()
        tmpl = model.config.nli_template

        @torch.no_grad()
        def predict(pairs):
            out = []
            for i in range(0, len(pairs), args.bs):
                enc = tok([tmpl.format(premise=p, hypothesis=h) for p, h in pairs[i:i + args.bs]],
                          padding=True, return_tensors="pt").to("mps")
                out.append(model(**enc).logits.float().softmax(-1).cpu().numpy())
            torch.mps.synchronize()
            return np.concatenate(out)
        seq_len = lambda p: len(tok(tmpl.format(premise=p[0], hypothesis=p[1]))["input_ids"])

    pairs = make_pairs(tok, args.length, args.n)
    predict(pairs[:args.bs])                     # warm up the kernels
    t0 = time.perf_counter()
    predict(pairs)
    dt = time.perf_counter() - t0
    prec = "bf16" if not args.bits else f"{args.bits}-bit"
    print(f"{args.backend} {prec}  seq_len={seq_len(pairs[0])} tokens  bs={args.bs}  "
          f"{dt / len(pairs) * 1000:.0f} ms/pair")


if __name__ == "__main__":
    main()
