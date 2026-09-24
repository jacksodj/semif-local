#!/usr/bin/env python
"""Part 1 checkpoint: does it work?

    .venv/bin/python run_semif_mlx.py --smoke
    .venv/bin/python run_semif_mlx.py "premise" "hypothesis" ["hypothesis" ...]
"""
import argparse

from semif_mlx import LABELS, SemIfCrossEncoder

SMOKE = [
    ("A man is playing a guitar.", "Someone is making music.", "entailment"),
    ("A man is playing a guitar.", "Nobody is making any sound.", "contradiction"),
    ("A man is playing a guitar.", "The man is a professional musician.", "neutral"),
    # Settle the label order empirically, not from config.json.
    ("The cat is asleep.", "The cat is awake.", "contradiction"),
]


def show(premise, hypothesis, probs):
    ps = "  ".join(f"{k[:3]}={v:.4f}" for k, v in zip(LABELS, probs))
    print(f"{ps}  | {premise!r} -> {hypothesis!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("premise", nargs="?")
    ap.add_argument("hypotheses", nargs="*")
    ap.add_argument("--model", default="openjev/qwen3.5-4b-nli-v5")
    ap.add_argument("--bits", type=int, choices=[4, 8])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        ce = SemIfCrossEncoder(args.model, bits=args.bits)
        probs = ce.predict([(p, h) for p, h, _ in SMOKE])
        ok = 0
        for (p, h, want), pr in zip(SMOKE, probs):
            got = LABELS[pr.argmax()]
            ok += got == want
            show(p, h, pr)
            if pr.max() < 0.4:
                print("   ^ all ~0.33: the head did not load")
        # Truncation must cut the premise, never the hypothesis.
        short = SemIfCrossEncoder.__new__(SemIfCrossEncoder)
        short.__dict__.update(ce.__dict__, max_len=256)
        long_premise = "A man is playing a guitar. " + "The quarterly report covers routine operations. " * 60
        hyps = ["Someone is making music.", "Nobody is making any sound."]
        tprobs = short.predict([(long_premise, h) for h in hyps])
        kept = all(ce.tok.decode(short.encode(long_premise, h)).endswith(h) for h in hyps)
        print(f"truncation at max_len=256, {len(long_premise)}-char premise: hypothesis kept={kept}, "
              f"labels={[LABELS[p.argmax()] for p in tprobs]}")
        ok += kept - 1                 # a lost hypothesis fails the smoke test
        print(f"\n{ok}/{len(SMOKE)} smoke pairs labelled as expected")
        raise SystemExit(0 if ok == len(SMOKE) else 1)

    if not args.premise or not args.hypotheses:
        ap.error("give a premise and at least one hypothesis, or --smoke")
    ce = SemIfCrossEncoder(args.model, bits=args.bits)
    for h, pr in zip(args.hypotheses, ce.predict([(args.premise, h) for h in args.hypotheses])):
        show(args.premise, h, pr)


if __name__ == "__main__":
    main()
