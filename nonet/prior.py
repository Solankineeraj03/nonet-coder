"""Doc-conditioned component costs from the MBPP-trained count table (data/prior.json)."""
import json
import math
import os
import re

_P = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "prior.json")))
ALPHA = 4.0
STOP = set("a an the to of in for and or is are be given function write that this it with by from as on at "
           "which its into each if will your you we".split())


def doc_words(doc):
    out = set()
    for w in re.findall(r"[a-z]+", doc.lower()):
        if w in STOP or len(w) < 3:
            continue
        out.add(w)
        if w.endswith("s") and len(w) > 4:
            out.add(w[:-1])
    return out


def prob(op, words, kw_hit):
    base = _P["base"].get(op, 0.02)
    p = base
    for w in words:
        n = _P["wc"].get(w)
        if not n:
            continue
        c = _P["cond"][w].get(op, 0)
        pw = (c + ALPHA * base) / (n + ALPHA)
        if pw > p:
            p = pw
    if kw_hit:
        p = max(p, 0.5)
    return min(p, 0.95)


def cost(op, words, kw_hit, floor=1):
    p = prob(op, words, kw_hit)
    return max(floor, min(7, round(-math.log2(p))))
