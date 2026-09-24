#!/usr/bin/env python
"""Part 3: change one factor at a time and see which one moves the number.

Runs the full pipeline (premise -> prefilter -> cross-encoder -> route) on the fictional
fixture under four conditions, then checks where orphaned mail goes, how prefilter recall
depends on K, and how business units resolve from participants.

    .venv/bin/python benchmark_capability.py [--model dist/semif-4b-nli-v5-mlx] [--k 16]
"""
import argparse
import json
import time

from semif.triage import (classify, load_messages, load_people, load_taxonomy, prefilter,
                          resolve_unit, without_aliases, without_projects)
from semif_mlx import SemIfCrossEncoder


def score(ce, msgs, taxonomy, k, subject_only=False):
    rows = []
    for m in msgs:
        d = classify(ce, m, taxonomy, k=k, subject_only=subject_only)
        rows.append({"id": m["id"], "gold": m["gold"], "top": d.top, "top3": d.topk(3),
                     "route": d.route, "p": round(d.ranked[0].p, 4)})
    n = len(rows)
    auto = [r for r in rows if r["route"] == "auto"]
    return {
        "n": n,
        "top1": sum(r["top"] == r["gold"] for r in rows) / n,
        "top3": sum(r["gold"] in r["top3"] for r in rows) / n,
        "auto_rate": len(auto) / n,
        "auto_precision": (sum(r["top"] == r["gold"] for r in auto) / len(auto)) if auto else None,
        "rows": rows,
    }


def fmt(x):
    return "   —  " if x is None else f"{100 * x:5.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openjev/qwen3.5-4b-nli-v5")
    ap.add_argument("--k", type=int, default=16, help="prefilter shortlist size")
    ap.add_argument("--out", default=None, help="write per-message results JSON here")
    args = ap.parse_args()

    tax, data, people = load_taxonomy(), load_messages(), load_people()
    msgs = data["messages"]
    removed = set(data["stale_removed"])
    stale_tax = without_projects(tax, removed)
    kept = [m for m in msgs if m["gold"] not in removed]
    orphans = [m for m in msgs if m["gold"] in removed]

    ce = SemIfCrossEncoder(args.model, bs=16)
    t0 = time.perf_counter()
    conditions = {
        "full message, aliases populated": score(ce, msgs, tax, args.k),
        "subject line only": score(ce, msgs, tax, args.k, subject_only=True),
        "full message, no aliases": score(ce, msgs, without_aliases(tax), args.k),
        "full message, stale taxonomy": score(ce, kept, stale_tax, args.k),
    }
    elapsed = time.perf_counter() - t0
    n_msgs = sum(c["n"] for c in conditions.values())

    print(f"\n{'condition':<34} {'n':>3}  top-1   top-3   auto-tagged  precision when auto-tagged")
    for name, c in conditions.items():
        print(f"{name:<34} {c['n']:>3}  {fmt(c['top1'])}  {fmt(c['top3'])}  {fmt(c['auto_rate']):>11}  "
              f"{fmt(c['auto_precision']):>10}")
    base, noal = conditions["full message, aliases populated"], conditions["full message, no aliases"]
    print(f"\nalias effect on top-1: {100 * (base['top1'] - noal['top1']):+.1f} points")
    alias_only = [m["id"] for m in msgs if m.get("alias_only")]
    for name, c in [("with aliases", base), ("without aliases", noal)]:
        hits = sum(r["top"] == r["gold"] for r in c["rows"] if r["id"] in alias_only)
        print(f"  alias-only messages {name}: {hits}/{len(alias_only)} correct")

    print(f"\norphaned mail (gold project removed from the taxonomy), {len(orphans)} messages:")
    orphan_rows = []
    for m in orphans:
        d = classify(ce, m, stale_tax, k=args.k)
        hazard = d.route == "auto"
        orphan_rows.append({"id": m["id"], "gold": m["gold"], "route": d.route, "top": d.top,
                            "p": round(d.ranked[0].p, 4)})
        print(f"  {m['id']} ({m['gold']}): route={d.route:<11} top={d.top} p={d.ranked[0].p:.3f}"
              + ("   <-- confident mis-tag" + (" into the catch-all" if d.top == tax["catch_all"] else "")
                 if hazard else ""))

    print("\nprefilter recall by K (is the right answer even in the shortlist?):")
    for k in (4, 8, 12, 16, 24):
        rec = sum(m["gold"] in [c.project["id"] for c in prefilter(m, tax, k=k)] for m in msgs) / len(msgs)
        print(f"  K={k:<3} {100 * rec:5.1f}%")

    print("\nbusiness unit from participants:")
    exec_ = "marcus.bell@northwind.example"
    cases = {
        "one member who reliably maps to one unit": ["grace.kim@northwind.example"],
        "that member plus the exec who is on everything": ["grace.kim@northwind.example", exec_],
        "members from two different units": ["grace.kim@northwind.example", "amara.nwosu@northwind.example"],
        "only the exec who is on everything": [exec_],
    }
    for name, parts in cases.items():
        unit, shares = resolve_unit(parts, people)
        top = next(iter(shares.items()))
        print(f"  {name:<48} {top[0]} {top[1]:.2f} -> {unit or 'ask a human'}")
    unit_hits = sum(resolve_unit(m["participants"], people)[0] ==
                    next(p["unit"] for p in tax["projects"] if p["id"] == m["gold"]) for m in msgs)
    unit_auto = sum(resolve_unit(m["participants"], people)[0] is not None for m in msgs)
    print(f"  on the 21 fixture messages: {unit_auto} auto-assigned, {unit_hits} match the project's unit")

    print(f"\n{n_msgs} pipeline runs in {elapsed:.0f}s ({elapsed / n_msgs:.1f}s per message at K={args.k})")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"conditions": conditions, "orphans": orphan_rows,
                       "seconds_per_message": elapsed / n_msgs}, f, indent=2)


if __name__ == "__main__":
    main()
