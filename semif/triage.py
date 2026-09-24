"""Part 3 pipeline: premise digest -> lexical prefilter -> cross-encoder rerank -> route.

Filter cheaply, rerank expensively, and route by confidence instead of forcing a
decision. The prefilter's only job is not to lose the right answer; the
cross-encoder supplies precision.
"""
import json
import math
import pathlib
import re
from dataclasses import dataclass, field

FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "fixtures"

# The verbalizer: a claim about the CONTENT, not about the message (Failure 1),
# phrased with the name the text actually uses (Failure 2), and nothing else (Failure 3).
DEFAULT_TEMPLATE = "The email discusses {name}."
CATCH_ALL_HYPOTHESIS = "The email is general correspondence that is not about any specific project."

AUTO_P, AUTO_GAP = 0.60, 0.30      # auto-tag gates: winner's P(ent), and top1 - top2
PLAUSIBLE_P = 0.10                 # below this, nothing fits: flag as a possible new project
BODY_CHARS = 700                   # body budget in the premise; the body is what gets cut
PREFILTER_BODY_CHARS = 2000

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "and", "or", "to", "for", "in", "on", "at", "by", "with",
         "is", "are", "be", "it", "this", "that", "not", "any", "we", "our", "new"}


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


# ---------------------------------------------------------------- data

def load_taxonomy(path=FIXTURES / "taxonomy.json") -> dict:
    return json.loads(pathlib.Path(path).read_text())


def load_messages(path=FIXTURES / "messages.json") -> dict:
    return json.loads(pathlib.Path(path).read_text())


def load_people(path=FIXTURES / "people.json") -> dict:
    return json.loads(pathlib.Path(path).read_text())["people"]


def without_aliases(taxonomy: dict) -> dict:
    return {**taxonomy, "projects": [{**p, "aliases": []} for p in taxonomy["projects"]]}


def without_projects(taxonomy: dict, drop: set[str]) -> dict:
    return {**taxonomy, "projects": [p for p in taxonomy["projects"] if p["id"] not in drop]}


# ---------------------------------------------------------------- premise

_QUOTE_START = re.compile(r"^(On .+ wrote:|-----Original Message-----|From: .+)$", re.M)
_SIGNATURE = re.compile(r"\n(--|Thanks,|Best,|Regards,|Cheers,)\s*\n.*", re.S)


def clean_body(body: str) -> str:
    """Strip quoted reply history and signatures: they drag the label toward the old thread."""
    m = _QUOTE_START.search(body)
    if m:
        body = body[:m.start()]
    body = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith(">"))
    return _SIGNATURE.sub("\n", body).strip()


def build_premise(msg: dict, subject_only: bool = False) -> str:
    """Compact digest, most informative fields first and the body last."""
    lines = [f"Email subject: {msg['subject']}"]
    if subject_only:
        return lines[0]
    lines.append(f"From: {msg['from']}")
    names = [p.split("@")[0].replace(".", " ") for p in msg.get("participants", [])]
    if names:
        lines.append("Participants: " + ", ".join(names))
    if msg.get("existing_tags"):
        lines.append("Existing tags: " + ", ".join(msg["existing_tags"]))
    body = clean_body(msg.get("body", ""))
    if len(body) > BODY_CHARS:
        body = body[:BODY_CHARS].rsplit(" ", 1)[0] + "..."
    lines.append("Body: " + body.replace("\n", " "))
    return "\n".join(lines)


# ---------------------------------------------------------------- prefilter

@dataclass
class Candidate:
    project: dict
    score: float
    matched_name: str          # the form that phrases the hypothesis
    p: float = 0.0             # P(entailment) after rerank


def _form_score(form: str, bag: set[str], text: str) -> float:
    """matched_tokens / form_tokens, with half credit for a >4-char substring hit."""
    ft = tokens(form)
    if not ft:
        return 0.0
    hit = 0.0
    for t in ft:
        if t in bag:
            hit += 1.0
        elif len(t) > 4 and t in text:
            hit += 0.5            # "ledger" still fires on "LedgerModernisation"
    return hit / len(ft)


def prefilter(msg: dict, taxonomy: dict, k: int = 16, subject_only: bool = False) -> list[Candidate]:
    body = "" if subject_only else msg.get("body", "")[:PREFILTER_BODY_CHARS]
    fields = [msg["subject"], body, "" if subject_only else msg.get("from", ""),
              " ".join(msg.get("existing_tags", []))]
    text = " ".join(fields).lower()
    bag = set(tokens(text))
    tags = " ".join(msg.get("existing_tags", [])).lower()

    cands = []
    for proj in taxonomy["projects"]:
        # Names are scored separately so we know WHICH name appeared.
        best_name, name_score = proj["name"], 0.0
        for form in [proj["name"], *proj.get("aliases", [])]:
            s = _form_score(form, bag, text)
            if s > name_score:
                best_name, name_score = form, s
        # Descriptions and search terms can rescue recall but never phrase the hypothesis.
        desc = 0.6 * _form_score(proj.get("description", ""), bag, text)
        terms = max((0.9 * _form_score(t, bag, text) for t in proj.get("search_terms", [])),
                    default=0.0)
        score = max(name_score, desc, terms)
        if proj["name"].lower() in tags:
            score += 2.0                                   # already tagged on the thread
        score += 0.01 * math.log1p(proj.get("count", 0))   # tiny usage prior
        cands.append(Candidate(proj, score, best_name if name_score > 0 else proj["name"]))
    cands.sort(key=lambda c: c.score, reverse=True)
    top = cands[:k]
    # The catch-all always competes, so its hazard is measurable rather than hidden.
    catch_all = taxonomy.get("catch_all")
    if catch_all and not any(c.project["id"] == catch_all for c in top):
        top += [c for c in cands if c.project["id"] == catch_all]
    return top


# ---------------------------------------------------------------- rerank + route

def hypothesis(cand: Candidate, taxonomy: dict, template: str = DEFAULT_TEMPLATE) -> str:
    if cand.project["id"] == taxonomy.get("catch_all"):
        return CATCH_ALL_HYPOTHESIS
    return template.format(name=cand.matched_name, description=cand.project.get("description", ""))


@dataclass
class Decision:
    route: str                               # "auto" | "review" | "new_project"
    ranked: list[Candidate] = field(default_factory=list)
    premise: str = ""

    @property
    def top(self):
        return self.ranked[0].project["id"] if self.ranked else None

    def topk(self, k: int) -> list[str]:
        return [c.project["id"] for c in self.ranked[:k]]


def classify(ce, msg: dict, taxonomy: dict, k: int = 16, subject_only: bool = False,
             template: str = DEFAULT_TEMPLATE) -> Decision:
    premise = build_premise(msg, subject_only=subject_only)
    cands = prefilter(msg, taxonomy, k=k, subject_only=subject_only)
    probs = ce.entail(premise, [hypothesis(c, taxonomy, template) for c in cands])
    for c, p in zip(cands, probs):
        c.p = float(p)
    cands.sort(key=lambda c: c.p, reverse=True)
    p1 = cands[0].p
    p2 = cands[1].p if len(cands) > 1 else 0.0
    if p1 >= AUTO_P and p1 - p2 >= AUTO_GAP:
        route = "auto"
    elif p1 >= PLAUSIBLE_P:
        route = "review"                     # human queue with the top-5 shortlist
    else:
        route = "new_project"
    return Decision(route, cands, premise)


# ---------------------------------------------------------------- business unit

def resolve_unit(participants: list[str], people: dict, min_share: float = 0.65,
                 min_lead: float = 0.25):
    """A unit is a property of who is on the thread, not a topic.

    w(p) = min(1, n(p)/3) * purity(p)^2, score(u) = sum_p w(p) * dist(p)[u], normalised.
    Returns (unit or None, shares).
    """
    scores: dict[str, float] = {}
    for person in participants:
        hist = people.get(person)
        if not hist:
            continue
        n = sum(hist.values())
        dist = {u: c / n for u, c in hist.items()}
        w = min(1.0, n / 3) * max(dist.values()) ** 2
        for u, share in dist.items():
            scores[u] = scores.get(u, 0.0) + w * share
    total = sum(scores.values())
    if not total:
        return None, {}
    shares = dict(sorted(((u, s / total) for u, s in scores.items()),
                         key=lambda kv: kv[1], reverse=True))
    ranked = list(shares.values()) + [0.0]
    if ranked[0] >= min_share and ranked[0] - ranked[1] >= min_lead:
        return next(iter(shares)), shares
    return None, shares
