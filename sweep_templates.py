#!/usr/bin/env python
"""Part 3, Failures 1-3 reproduced on the fictional fixture.

1. Verbalizer sweep: 10 messages, each scored against the same fixed 10-project shortlist.
2. Vocabulary: an alias-only message scored against the name it uses vs the canonical name.
3. Description: the bare name vs the name plus its description.

    .venv/bin/python sweep_templates.py [--model dist/semif-4b-nli-v5-mlx]
"""
import argparse

import numpy as np

from semif.triage import build_premise, load_messages, load_taxonomy
from semif_mlx import SemIfCrossEncoder

TEMPLATES = [
    "This message is about the {name} project.",
    "The email discusses {name}.",
    "The email mentions {name}.",
    "This email is related to {name}.",
    "The email discusses {name} ({description}).",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openjev/qwen3.5-4b-nli-v5")
    args = ap.parse_args()
    ce = SemIfCrossEncoder(args.model)
    tax, data = load_taxonomy(), load_messages()
    projects = {p["id"]: p for p in tax["projects"]}
    msgs = {m["id"]: m for m in data["messages"]}
    sweep = [msgs[i] for i in data["sweep"]["messages"]]
    shortlist = [projects[m["gold"]] for m in sweep]     # fixed: same candidates for every message

    print(f"Failure 1 -- verbalizer sweep: {len(sweep)} messages x {len(shortlist)}-project shortlist")
    print(f"{'hypothesis form':<48} top-1   mean p(correct)  mean p(others)")
    for tmpl in TEMPLATES:
        hits, p_ok, p_bad = 0, [], []
        for m in sweep:
            premise = build_premise(m)
            hyps = [tmpl.format(name=p["name"], description=p["description"]) for p in shortlist]
            probs = ce.entail(premise, hyps)
            gold = [p["id"] for p in shortlist].index(m["gold"])
            hits += int(probs.argmax() == gold)
            p_ok.append(probs[gold])
            p_bad.extend(np.delete(probs, gold))
        print(f"{tmpl:<48} {hits:>2}/{len(sweep)}   {np.mean(p_ok):>14.3f}  {np.mean(p_bad):>14.3f}")

    print("\nFailure 2 -- the model can't recognise a name you never gave it")
    for m in (m for m in data["messages"] if m.get("alias_only")):
        proj = projects[m["gold"]]
        used = next(a for a in proj["aliases"] if a.lower() in (m["subject"] + m["body"]).lower())
        probs = ce.entail(build_premise(m), [f"The email discusses {used}.",
                                              f"The email discusses {proj['name']}."])
        print(f"  {m['id']}: alias '{used}' p={probs[0]:.3f}   canonical '{proj['name']}' p={probs[1]:.3f}")

    print("\nFailure 3 -- a description is not a name")
    drops = []
    for m in sweep:
        proj = projects[m["gold"]]
        bare, glossed = ce.entail(build_premise(m), [
            f"The email discusses {proj['name']}.",
            f"The email discusses {proj['name']} ({proj['description']})."])
        drops.append(bare - glossed)
        print(f"  {m['id']}: bare {bare:.3f}   with description {glossed:.3f}   ({proj['name']})")
    print(f"  mean drop from adding the description: {np.mean(drops):+.3f}")


if __name__ == "__main__":
    main()
