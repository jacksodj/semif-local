#!/usr/bin/env python
"""MLX port of AlexWortega/openjev code/eval.py (which is CUDA-only).

Task semantics are kept identical: the same [con, ent, neu] label order and MNLI remap,
the same rerank / grading prompt construction, the same metrics (including
rerank_margin_acc and SciTail's two-class collapse).

    .venv/bin/python eval_mlx.py --tasks mnli --n 150
    .venv/bin/python eval_mlx.py --tasks arc_easy hellaswag gpqa --mc-n 60
    .venv/bin/python eval_mlx.py --tasks mnli arc_easy --compare-quant
"""
import argparse
import gc
import json
import time

import mlx.core as mx
import numpy as np
from datasets import load_dataset

from semif_mlx import CON, DEFAULT_MODEL, ENT, NEU, SemIfCrossEncoder

# ----------------------------------------------------------------------------- MNLI
MNLI_TO_OURS = {0: ENT, 1: NEU, 2: CON}      # MNLI order -> [con, ent, neu]


def eval_mnli(ce, n=None, seed=0):
    res = {}
    for split in ["validation_matched", "validation_mismatched"]:
        ds = load_dataset("nyu-mll/multi_nli", split=split).filter(lambda x: x["label"] in (0, 1, 2))
        if n:
            ds = ds.shuffle(seed=seed).select(range(n))
        probs = ce.predict(list(zip(ds["premise"], ds["hypothesis"])))
        gold = np.array([MNLI_TO_OURS[l] for l in ds["label"]])
        res[split] = {"acc": float((probs.argmax(-1) == gold).mean()), "n": len(ds),
                      "preds": probs.argmax(-1).tolist()}
    return res


# ----------------------------------------------------------------------------- hard NLI sets
NLI_SETS = {  # name -> (repo, config, split, premise col, hypothesis col, label col)
    "anli_r1": ("facebook/anli", None, "test_r1", "premise", "hypothesis", "label"),
    "anli_r2": ("facebook/anli", None, "test_r2", "premise", "hypothesis", "label"),
    "anli_r3": ("facebook/anli", None, "test_r3", "premise", "hypothesis", "label"),
    "wanli": ("alisawuffles/WANLI", None, "test", "premise", "hypothesis", "gold"),
    "scitail": ("allenai/scitail", "snli_format", "test", "sentence1", "sentence2", "gold_label"),
    "control": ("tasksource/ConTRoL-nli", None, "test", "premise", "hypothesis", "label"),
}
_NAME2ID = {"contradiction": CON, "entailment": ENT, "neutral": NEU, "entails": ENT, "not_entailment": NEU}


def eval_nli_set(ce, name, n=None, seed=0):
    repo, cfg, split, pc, hc, lc = NLI_SETS[name]
    ds = load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split)
    if n and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    feat = ds.features[lc]
    pairs, gold = [], []
    for ex in ds:
        v = ex[lc]
        nm = feat.names[v] if isinstance(v, int) and hasattr(feat, "names") else v
        if not isinstance(nm, str) or nm.strip().lower() not in _NAME2ID:
            continue
        pairs.append((ex[pc], ex[hc]))
        gold.append(_NAME2ID[nm.strip().lower()])
    probs = ce.predict(pairs)
    gold = np.array(gold)
    pred = probs.argmax(-1)
    two_class = len(set(gold.tolist())) == 2   # SciTail: contradiction counts as "not entailed"
    if two_class:
        pred = np.where(pred == ENT, ENT, NEU)
    return {"acc": float((pred == gold).mean()), "n": int(len(gold)), "two_class": two_class}


# ----------------------------------------------------------------------------- multiple choice
# rerank:  premise=q, hypothesis="The correct answer is: {opt}" -> argmax P(ent)
# grading: premise=q + "Reference answer: {gold}", hypothesis="Answer: {opt}" -> ent <=> gold

def _sample(ds, n, seed):
    return ds.shuffle(seed=seed).select(range(n)) if n and n < len(ds) else ds


def load_arc(cfg, n=None, seed=0):
    items = []
    for ex in _sample(load_dataset("allenai/ai2_arc", cfg, split="test"), n, seed):
        labels = ex["choices"]["label"]
        if ex["answerKey"] not in labels:
            continue
        items.append({"q": ex["question"].strip(), "opts": [t.strip() for t in ex["choices"]["text"]],
                      "gold": labels.index(ex["answerKey"])})
    return items


def load_hellaswag(n=None, seed=0):
    items = []
    for ex in _sample(load_dataset("Rowan/hellaswag", split="validation"), n, seed):
        ctx = (ex["ctx_a"] + " " + ex["ctx_b"].capitalize()).strip() if ex["ctx_b"] else ex["ctx_a"].strip()
        items.append({"q": f"{ex['activity_label']}: {ctx}", "opts": [e.strip() for e in ex["endings"]],
                      "gold": int(ex["label"]), "hyp": lambda o: o})
    return items


def load_winogrande(n=None, seed=0):
    items = []
    for ex in _sample(load_dataset("allenai/winogrande", "winogrande_xl", split="validation"), n, seed):
        sent = ex["sentence"]
        items.append({"q": sent, "opts": [ex["option1"], ex["option2"]], "gold": int(ex["answer"]) - 1,
                      "hyp": lambda o, sent=sent: sent.replace("_", o)})
    return items


def load_mmlu(n=None, seed=0):
    return [{"q": ex["question"].strip(), "opts": [c.strip() for c in ex["choices"]], "gold": int(ex["answer"])}
            for ex in _sample(load_dataset("cais/mmlu", "all", split="test"), n, seed)]


def load_gpqa(n=None, seed=0):
    # Gated dataset: accept the terms at huggingface.co/datasets/Idavidrein/gpqa first.
    items = []
    for ex in _sample(load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train"), n, seed):
        opts = [ex["Correct Answer"], ex["Incorrect Answer 1"], ex["Incorrect Answer 2"], ex["Incorrect Answer 3"]]
        items.append({"q": ex["Question"].strip(), "opts": [o.strip() for o in opts], "gold": 0})
    return items


MC_TASKS = {
    "arc_easy": lambda a: load_arc("ARC-Easy", a.mc_n),
    "arc_challenge": lambda a: load_arc("ARC-Challenge", a.mc_n),
    "hellaswag": lambda a: load_hellaswag(a.mc_n),
    "winogrande": lambda a: load_winogrande(a.mc_n),
    "mmlu": lambda a: load_mmlu(a.mc_n),
    "gpqa": lambda a: load_gpqa(a.mc_n),
}
# Note: v5 was trained on the TEST splits of MMLU, ARC, HellaSwag, WinoGrande and GPQA-diamond
# (see its panel_manifest.json). On v5, treat the MC tasks as harness smoke tests only.


def _f1(gold, pred):
    tp = int(((gold == 1) & (pred == 1)).sum())
    fp = int(((gold == 0) & (pred == 1)).sum())
    fn = int(((gold == 1) & (pred == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def eval_mc(ce, items):
    rerank_pairs, grade_pairs, grade_gold, offsets = [], [], [], []
    for it in items:
        q, opts, g = it["q"], it["opts"], it["gold"]
        hyp = it.get("hyp") or (lambda o: f"The correct answer is: {o}")
        offsets.append((len(rerank_pairs), len(opts)))
        for j, o in enumerate(opts):
            rerank_pairs.append((q, hyp(o)))
            grade_pairs.append((f"{q}\nReference answer: {opts[g]}", f"Answer: {o}"))
            grade_gold.append(1 if j == g else 0)
    pr, pg = ce.predict(rerank_pairs), ce.predict(grade_pairs)
    grade_gold = np.array(grade_gold)
    rerank, margin, rank, rand = [], [], [], []
    for (s, k), it in zip(offsets, items):
        g = it["gold"]
        rerank.append(pr[s:s + k, ENT].argmax() == g)
        margin.append((pr[s:s + k, ENT] - pr[s:s + k, CON]).argmax() == g)
        rank.append(pg[s:s + k, ENT].argmax() == g)
        rand.append(1.0 / k)
    pred_ent = (pg.argmax(-1) == ENT).astype(int)
    return {
        "n_questions": len(items),
        "random_baseline": float(np.mean(rand)),
        "rerank_acc": float(np.mean(rerank)),
        "rerank_margin_acc": float(np.mean(margin)),
        "grade_acc": float((pred_ent == grade_gold).mean()),
        "grade_f1": float(_f1(grade_gold, pred_ent)),
        "grade_rank_acc": float(np.mean(rank)),
    }


ALL_TASKS = ["mnli", *NLI_SETS, *MC_TASKS]


def run(ce, args):
    out = {}
    for task in args.tasks:
        t0 = time.perf_counter()
        try:
            if task == "mnli":
                r = eval_mnli(ce, args.n)
            elif task in NLI_SETS:
                r = eval_nli_set(ce, task, args.n)
            else:
                r = eval_mc(ce, MC_TASKS[task](args))
        except Exception as e:  # noqa: BLE001 -- one missing/gated dataset shouldn't sink the sweep
            r = {"error": f"{type(e).__name__}: {e}"}
        r["seconds"] = round(time.perf_counter() - t0, 1)
        out[task] = r
        shown = {k: v for k, v in r.items() if k != "preds"} if task != "mnli" else \
            {s: {"acc": v["acc"], "n": v["n"]} for s, v in r.items() if isinstance(v, dict)}
        print(f"{task:>14}: {json.dumps(shown)}", flush=True)
    return out


def headline(r):
    if "error" in r:
        return None
    if "validation_matched" in r:
        return r["validation_matched"]["acc"], r["validation_mismatched"]["acc"]
    return (r["acc"],) if "acc" in r else (r["rerank_acc"],)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="checkpoint dir or Hub repo id")
    ap.add_argument("--tasks", nargs="+", default=["mnli"], choices=ALL_TASKS + ["all"])
    ap.add_argument("--n", type=int, default=150, help="examples per NLI split (0 = all)")
    ap.add_argument("--mc-n", type=int, default=60, help="questions per MC task (0 = all)")
    ap.add_argument("--bits", type=int, choices=[4, 8], default=None)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--compare-quant", action="store_true", help="run at bf16, 8-bit and 4-bit")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()
    if "all" in args.tasks:
        args.tasks = ALL_TASKS
    args.n, args.mc_n = args.n or None, args.mc_n or None

    precisions = [None, 8, 4] if args.compare_quant else [args.bits]
    results = {}
    for bits in precisions:
        name = "bf16" if bits is None else f"{bits}-bit"
        print(f"== {name} ==", flush=True)
        ce = SemIfCrossEncoder(args.model, bits=bits, bs=args.bs)
        results[name] = run(ce, args)
        del ce
        gc.collect()
        mx.clear_cache()

    if args.compare_quant:
        print("\ntask                        bf16     8-bit    4-bit   Δ8-bit   Δ4-bit")
        for task in args.tasks:
            heads = [headline(results[p][task]) for p in results]
            if any(h is None for h in heads):
                continue
            labels = ["matched", "mismatched"] if task == "mnli" else [""]
            for i, lab in enumerate(labels):
                b, q8, q4 = (h[i] for h in heads)
                print(f"{(task + ' ' + lab).strip():<26} {b:7.4f}  {q8:7.4f}  {q4:7.4f}  "
                      f"{q8 - b:+7.4f}  {q4 - b:+7.4f}")
        if "mnli" in args.tasks and all("error" not in results[p]["mnli"] for p in results):
            m = {p: np.array(results[p]["mnli"]["validation_matched"]["preds"]) for p in results}
            print(f"\nMNLI matched predictions that differ from bf16: "
                  f"8-bit {(m['bf16'] != m['8-bit']).sum()}, 4-bit {(m['bf16'] != m['4-bit']).sum()}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
