"""Retrieve reusable functions from a code library (MBPP solutions), verified on the examples.

Ranking is TF-IDF over description words (no embeddings, no neural nets). Library code is
NEVER executed in this process: candidates are checked in an isolated `python -I` subprocess
(nonet/sandbox_runner.py) with CPU and wall-clock limits.
"""
import ast
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter

from .prior import doc_words

_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_RUNNER = os.path.join(os.path.dirname(__file__), "sandbox_runner.py")
_LIB = None


def load_library():
    global _LIB
    if _LIB is not None:
        return _LIB
    lib, df = [], Counter()
    for line in open(os.path.join(_DIR, "mbpp.jsonl")):
        r = json.loads(line)
        code = r["code"].replace("\r", "").replace("\t", "    ")
        m = re.match(r"assert\s+(?:not\s+)?\(?\s*(?:set\()?\s*(?:math\.isclose\()?\s*([A-Za-z_]\w*)\(", r["test_list"][0])
        if not m:
            continue
        fname = m.group(1)
        try:
            tree = ast.parse(code)  # parsed only, not executed
        except SyntaxError:
            continue
        fdef = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fname]
        if not fdef or fdef[0].args.vararg or fdef[0].args.kwarg:
            continue
        code = re.sub(r"\b" + re.escape(fname) + r"\b", "lib_" + fname, code)
        ws = doc_words(r["text"])
        df.update(ws)
        lib.append({"id": r["task_id"], "text": r["text"], "code": code, "fname": "lib_" + fname,
                    "arity": len(fdef[0].args.args) - len(fdef[0].args.defaults),
                    "nargs": len(fdef[0].args.args), "words": ws})
    n = len(lib)
    idf = {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}
    _LIB = (lib, idf)
    return _LIB


def similarity(a_words, b_words, idf):
    a = {w: idf.get(w, math.log(1000)) for w in a_words}
    b = {w: idf.get(w, math.log(1000)) for w in b_words}
    dot = sum(a[w] * b[w] for w in a if w in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb + 1e-9)


def retrieve(info, examples, probes, entry, topk=150, timeout=8.0):
    """-> list of dicts {sim, code, call, crash, id} for library functions matching every example."""
    lib, idf = load_library()
    from .synth import EXLINE
    prose = "\n".join(l for l in info["doc"].split("\n") if not EXLINE.search(l) and entry not in l)
    ws = doc_words(prose) | doc_words(" ".join(re.split(r"_|(?<=[a-z])(?=[A-Z])", entry)))
    k = len(info["args"])
    scored = sorted(((similarity(it["words"], ws, idf), j) for j, it in enumerate(lib)
                     if it["arity"] <= k <= it["nargs"]), key=lambda x: -x[0])[:topk]
    scored = [(s, j) for s, j in scored if s > 0.05]
    if not scored:
        return []
    job = {"cands": [(lib[j]["code"], lib[j]["fname"], lib[j]["arity"], lib[j]["nargs"], j) for _, j in scored],
           "examples": [tuple(e) for e in examples], "probes": probes, "k": k}
    try:
        r = subprocess.run([sys.executable, "-I", _RUNNER], input=repr(job), capture_output=True,
                           text=True, timeout=timeout, cwd=os.path.dirname(_RUNNER))
        res = ast.literal_eval(r.stdout) if r.stdout.strip() else []
    except (subprocess.TimeoutExpired, ValueError, SyntaxError):
        res = []
    sims = dict((j, s) for s, j in scored)
    out = []
    for j, perm, crash in res:
        it = lib[j]
        call = f"{it['fname']}({', '.join(info['args'][p] for p in perm)})"
        out.append({"sim": sims[j], "code": it["code"], "call": call, "crash": crash, "id": it["id"]})
    return out
