"""What does quantization cost this classifier on MNLI?  (the post's twenty-line measurement)"""
import sys
import time

import numpy as np
from datasets import load_dataset

from semif_mlx import SemIfCrossEncoder             # the class from Part 1

CKPT = sys.argv[1] if len(sys.argv) > 1 else "openjev/qwen3.5-4b-nli-v5"
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "validation_matched"
N = 150
MNLI_TO_OURS = {0: 1, 1: 2, 2: 0}                   # MNLI order -> [con, ent, neu]

ds = load_dataset("nyu-mll/multi_nli", split=SPLIT)
ds = ds.filter(lambda x: x["label"] in (0, 1, 2)).shuffle(seed=0).select(range(N))
pairs = list(zip(ds["premise"], ds["hypothesis"]))
gold = np.array([MNLI_TO_OURS[l] for l in ds["label"]])

preds = {}
for bits in (None, 8, 4):
    # sort_by_length=False reproduces the post's shuffled-batch timings
    ce = SemIfCrossEncoder(CKPT, bits=bits, bs=16, sort_by_length=False)
    ce.predict(pairs[:2])                                    # warm up the kernels
    t0 = time.perf_counter()
    probs = ce.predict(pairs)
    dt = time.perf_counter() - t0
    preds[bits] = probs.argmax(-1)
    acc = (preds[bits] == gold).mean()
    print(f"{'bf16' if bits is None else f'{bits}-bit':>6}  acc={acc:.4f}  "
          f"{N/dt:5.2f} pairs/sec  ({dt/N*1000:.0f} ms/pair)", flush=True)
    del ce

print(f"predictions changed vs bf16: 8-bit {(preds[None] != preds[8]).sum()}, "
      f"4-bit {(preds[None] != preds[4]).sum()}")
