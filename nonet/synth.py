"""NoNet Coder synthesizer: bottom-up enumerative search over Python expressions.

Input: the prompt text only (signature + docstring). Output: Python source.
No neural networks, no learned weights, no network access.
"""
import math
import re
import time

from .extract import extract_examples
from .lib import LIB_SRC
from .prior import cost as prior_cost, doc_words

L = {}
exec(LIB_SRC, L)

MAXLEN = 2000


class _Err:
    __slots__ = ()

    def __repr__(self):
        return "<ERR>"


ERR = _Err()


def tag(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, tuple):
        return "tuple"
    if v is None:
        return "none"
    if isinstance(v, dict):
        return "dict"
    return "other"


def tagset(vals):
    return frozenset(tag(v) for v in vals if v is not ERR)


NUM = {"int", "float", "bool"}
GROUPS = {
    "num": NUM, "int": {"int", "bool"}, "str": {"str"}, "list": {"list"},
    "seq": {"str", "list", "tuple"}, "lst": {"list", "tuple"}, "bool": {"bool"},
    "any": None, "nn": {"int", "float", "bool", "str", "list", "tuple", "dict"},
    "ar": {"int", "float"}, "ix": {"int"}, "addable": {"int", "float", "str", "list", "tuple"},
    "strflt": {"str", "float"}, "strint": {"str", "int"}, "obj": {"int", "float", "str", "list", "tuple", "dict"},
}


def ok_tags(ts, g):
    allowed = GROUPS[g]
    return allowed is None or ts <= allowed


def guard(v):
    if isinstance(v, (str, list, tuple)) and len(v) > MAXLEN:
        raise ValueError
    if isinstance(v, int) and not isinstance(v, bool) and abs(v) > 10**15:
        raise ValueError
    if isinstance(v, float) and (v != v or abs(v) > 1e15):
        raise ValueError
    if isinstance(v, (map, filter, zip, range)) or callable(v):
        raise ValueError
    return v


def _pow(a, b):
    if abs(b) > 64 or abs(a) > 10**6:
        raise ValueError
    return a ** b


def _mul(a, b):
    if isinstance(a, (str, list)) or isinstance(b, (str, list)):
        n = b if isinstance(b, int) else a
        if not isinstance(n, int) or n > 200:
            raise ValueError
    return a * b


def _rng(a):
    if a > 10000:
        raise ValueError
    return list(range(a))


def _rng2(a, b):
    if b - a > 10000:
        raise ValueError
    return list(range(a, b))


# (name, arggroups, fn, code-format, keywords, base-cost)
# keywords: if any appears in the docstring, cost drops to 1.
UN = [
    ("len", ["seq"], len, "len({0})", (), 1),
    ("neg", ["ar"], lambda a: -a, "-{0}", ("negative",), 1),
    ("abs", ["ar"], abs, "abs({0})", ("absolute", "abs", "distance"), 2),
    ("not", ["bool"], lambda a: not a, "not {0}", (), 1),
    ("sum", ["lst"], sum, "sum({0})", ("sum", "total", "add"), 2),
    ("max", ["seq"], max, "max({0})", ("max", "largest", "biggest", "greatest", "highest"), 2),
    ("min", ["seq"], min, "min({0})", ("min", "smallest", "lowest", "least"), 2),
    ("sorted", ["seq"], sorted, "sorted({0})", ("sort", "order", "ascending"), 2),
    ("sorted_desc", ["seq"], lambda a: sorted(a, reverse=True), "sorted({0}, reverse=True)", ("descending", "reverse"), 3),
    ("rev", ["seq"], lambda a: a[::-1], "{0}[::-1]", ("reverse", "backward", "palindrome"), 2),
    ("first", ["seq"], lambda a: a[0], "{0}[0]", ("first",), 2),
    ("last", ["seq"], lambda a: a[-1], "{0}[-1]", ("last",), 2),
    ("tail", ["seq"], lambda a: a[1:], "{0}[1:]", (), 2),
    ("init", ["seq"], lambda a: a[:-1], "{0}[:-1]", (), 2),
    ("evens_idx", ["seq"], lambda a: a[::2], "{0}[::2]", ("even", "index", "indices"), 3),
    ("odds_idx", ["seq"], lambda a: a[1::2], "{0}[1::2]", ("odd", "index", "indices"), 3),
    ("set", ["seq"], lambda a: sorted(set(a)), "sorted(set({0}))", ("unique", "distinct", "duplicate"), 2),
    ("nuniq", ["seq"], lambda a: len(set(a)), "len(set({0}))", ("unique", "distinct"), 2),
    ("list", ["seq"], list, "list({0})", ("list", "character"), 2),
    ("str", ["obj"], str, "str({0})", ("string", "digit"), 2),
    ("int", ["strflt"], int, "int({0})", ("integer", "int", "round"), 2),
    ("float", ["strint"], float, "float({0})", ("float",), 3),
        ("round", ["ar"], round, "round({0})", ("round", "nearest", "closest"), 2),
    ("ceil", ["ar"], math.ceil, "math.ceil({0})", ("ceil", "upper", "round up"), 3),
    ("floor", ["ar"], math.floor, "math.floor({0})", ("floor", "lower", "round down"), 3),
    ("sqrt", ["ar"], math.sqrt, "math.sqrt({0})", ("square root", "sqrt"), 3),
    ("sq", ["ar"], lambda a: a * a, "{0} ** 2", ("square",), 2),
    ("cube", ["ar"], lambda a: a * a * a, "{0} ** 3", ("cube",), 3),
    ("half", ["ar"], lambda a: a / 2, "{0} / 2", ("half", "mean", "average"), 3),
    ("even", ["ix"], lambda a: a % 2 == 0, "{0} % 2 == 0", ("even",), 2),
    ("odd", ["ix"], lambda a: a % 2 == 1, "{0} % 2 == 1", ("odd",), 2),
    ("lower", ["str"], str.lower, "{0}.lower()", ("lower", "case", "insensitive"), 2),
    ("upper", ["str"], str.upper, "{0}.upper()", ("upper", "capital", "case"), 2),
    ("swapcase", ["str"], str.swapcase, "{0}.swapcase()", ("swap", "flip", "case"), 3),
    ("strip", ["str"], str.strip, "{0}.strip()", ("strip", "whitespace", "trim"), 3),
    ("split", ["str"], str.split, "{0}.split()", ("word", "split", "space", "separated"), 2),
    ("isupper", ["str"], str.isupper, "{0}.isupper()", ("upper", "capital"), 3),
    ("islower", ["str"], str.islower, "{0}.islower()", ("lower",), 3),
    ("isdigit", ["str"], str.isdigit, "{0}.isdigit()", ("digit", "number"), 3),
    ("isalpha", ["str"], str.isalpha, "{0}.isalpha()", ("letter", "alpha"), 3),
    ("isspace", ["str"], str.isspace, "{0}.isspace()", ("space", "whitespace"), 3),
    ("ord", ["str"], ord, "ord({0})", ("ascii", "ord", "alphabet", "shift"), 3),
    ("chr", ["ix"], chr, "chr({0})", ("ascii", "chr", "alphabet", "shift"), 3),
    ("bin", ["ix"], lambda a: bin(a)[2:], "bin({0})[2:]", ("binary",), 3),
    ("range", ["ix"], _rng, "list(range({0}))", ("range", "up to", "from 0"), 2),
    ("range1", ["ix"], lambda a: _rng2(1, a + 1), "list(range(1, {0} + 1))", ("range", "up to", "from 1", "1 to"), 2),
    ("running", ["lst"], L["running"], "running({0})", ("running", "cumulative", "balance", "prefix"), 3),
    ("is_prime", ["ix"], L["is_prime"], "is_prime({0})", ("prime",), 4),
    ("digits", ["ar"], L["digits"], "digits({0})", ("digit",), 4),
    ("fib", ["ix"], L["fib"], "fib({0})", ("fib", "fibonacci"), 5),
    ("factorial", ["ix"], L["factorial"], "factorial({0})", ("factorial",), 5),
    ("divisors", ["ix"], L["divisors"], "divisors({0})", ("divisor", "divide"), 5),
    ("prime_factors", ["ix"], L["prime_factors"], "prime_factors({0})", ("prime factor", "factor"), 5),
    ("is_pal", ["seq"], L["is_palindrome"], "is_palindrome({0})", ("palindrom",), 4),
    ("any", ["lst"], any, "any({0})", ("any", "exist", "there is", "there are", "at least one"), 2),
    ("all", ["lst"], all, "all({0})", ("all", "every", "each"), 2),
    ("join", ["lst"], lambda a: "".join(a), "''.join({0})", ("concatenat", "join", "string"), 2),
    ("joinsp", ["lst"], lambda a: " ".join(a), "' '.join({0})", ("word", "space", "sentence"), 3),
    ("tuple", ["lst"], tuple, "tuple({0})", ("tuple",), 3),
]

BIN = [
    ("add", ["addable", "addable"], lambda a, b: a + b, "({0} + {1})", (), 1),
    ("sub", ["ar", "ar"], lambda a, b: a - b, "({0} - {1})", (), 1),
    ("mul", ["addable", "ar"], _mul, "{0} * {1}", (), 1),
    ("div", ["ar", "ar"], lambda a, b: a / b, "{0} / {1}", ("divide", "average", "mean", "ratio"), 2),
    ("fdiv", ["ar", "ar"], lambda a, b: a // b, "{0} // {1}", ("divide", "integer"), 2),
    ("mod", ["ar", "ar"], lambda a, b: a % b, "{0} % {1}", ("divisible", "modulo", "remainder", "mod"), 2),
    ("pow", ["ar", "ar"], _pow, "{0} ** {1}", ("power",), 2),
    ("eq", ["obj", "obj"], lambda a, b: a == b, "{0} == {1}", (), 1),
    ("ne", ["obj", "obj"], lambda a, b: a != b, "{0} != {1}", (), 2),
    ("lt", ["ar", "ar"], lambda a, b: a < b, "{0} < {1}", (), 1),
    ("le", ["ar", "ar"], lambda a, b: a <= b, "{0} <= {1}", (), 2),
    ("and", ["bool", "bool"], lambda a, b: a and b, "({0} and {1})", ("and", "both"), 2),
    ("or", ["bool", "bool"], lambda a, b: a or b, "({0} or {1})", ("or",), 2),
    ("in", ["obj", "obj"], lambda a, b: a in b, "{0} in {1}", ("contain", "in"), 2),
    ("idx", ["seq", "ix"], lambda a, b: a[b], "{0}[{1}]", ("index", "position", "th"), 2),
    ("take", ["seq", "ix"], lambda a, b: a[:b], "{0}[:{1}]", ("first",), 2),
    ("drop", ["seq", "ix"], lambda a, b: a[b:], "{0}[{1}:]", (), 2),
    ("count", ["seq", "any"], lambda a, b: a.count(b), "{0}.count({1})", ("count", "number of", "how many", "times", "occurr"), 2),
    ("startswith", ["str", "str"], str.startswith, "{0}.startswith({1})", ("start", "begin", "prefix"), 2),
    ("endswith", ["str", "str"], str.endswith, "{0}.endswith({1})", ("end", "suffix"), 2),
    ("splitby", ["str", "str"], lambda a, b: a.split(b), "{0}.split({1})", ("split", "separat"), 2),
    ("joinby", ["str", "lst"], lambda a, b: a.join(b), "{0}.join({1})", ("join", "separat", "delimit", "concat"), 2),
    ("find", ["seq", "any"], lambda a, b: a.index(b), "{0}.index({1})", ("index", "position", "find"), 3),
    ("max2", ["ar", "ar"], max, "max({0}, {1})", ("max", "larg", "great"), 2),
    ("min2", ["ar", "ar"], min, "min({0}, {1})", ("min", "small"), 2),
    ("round2", ["ar", "ix"], round, "round({0}, {1})", ("round", "decimal"), 2),
    ("gcd", ["ix", "ix"], math.gcd, "math.gcd({0}, {1})", ("divisor", "gcd", "common"), 3),
    ("range2", ["ix", "ix"], _rng2, "list(range({0}, {1}))", ("range", "between", "interval"), 2),
    ("rangestep", ["seq", "ix"], lambda a, b: a[::b], "{0}[::{1}]", (), 3),
]

TER = [
    ("ite", ["any", "any", "any"], lambda c, a, b: a if c else b, "({1} if {0} else {2})", (), 2),
    ("slice", ["seq", "ix", "ix"], lambda s, a, b: s[a:b], "{0}[{1}:{2}]", (), 3),
    ("replace", ["str", "str", "str"], lambda s, a, b: s.replace(a, b), "{0}.replace({1}, {2})", ("replace",), 3),
]


import ast as _ast
_ATOM = (_ast.Name, _ast.Constant, _ast.Call, _ast.Subscript, _ast.Attribute, _ast.List, _ast.ListComp,
         _ast.Dict, _ast.Set, _ast.GeneratorExp)
_pc = {}


def par(code):
    r = _pc.get(code)
    if r is None:
        try:
            node = _ast.parse(code, mode="eval").body
            r = code if isinstance(node, _ATOM) and not (isinstance(node, _ast.Constant) and
                                                         isinstance(node.value, (int, float)) and node.value < 0) else f"({code})"
        except SyntaxError:
            r = f"({code})"
        _pc[code] = r
    return r


_fc = {}


def render(fmt, codes):
    spec = _fc.get(fmt)
    if spec is None:
        spec = []
        for i in range(len(codes)):
            ph = "{%d}" % i
            k = fmt.find(ph)
            before = fmt[k - 1] if k > 0 else ""
            before2 = fmt[k - 2:k]
            after = fmt[k + len(ph)] if k + len(ph) < len(fmt) else ""
            spec.append((before == "(" or before2 == ", ") and after in "),")
        _fc[fmt] = spec
    return fmt.format(*[c if raw else par(c) for c, raw in zip(codes, spec)])


class Term:
    __slots__ = ("code", "cost", "vals", "tags")

    def __init__(self, code, cost, vals):
        self.code = code
        self.cost = cost
        self.vals = vals
        self.tags = tagset(vals)


def op_cost(name, base, kws, doc, words):
    hit = bool(kws) and any(k in doc for k in kws)
    return prior_cost(name, words, hit)


def same(v, out):
    try:
        if isinstance(v, bool) != isinstance(out, bool):
            return False
        if isinstance(out, float) or isinstance(v, float):
            return isinstance(v, (int, float)) and isinstance(out, (int, float)) and abs(v - out) < 1e-6
        if out is None or v is None:
            return v is out
        return type(v) == type(out) and v == out
    except Exception:
        return False


class Bank:
    """Cost-indexed bank of terms over a fixed set of rows, with OE pruning."""

    def __init__(self, nrows, doc, deadline, max_terms=200000, real=None, banned=()):
        self.n = nrows
        self.real = real if real is not None else [True] * nrows
        self.doc = doc
        self.by_cost = {}
        self.seen = set()
        self.deadline = deadline
        self.max_terms = max_terms
        self.count = 0
        words = doc_words(doc)
        ok = lambda f: not any(re.search(r"\b" + re.escape(bn) + r"\(", f) for bn in banned)
        self.un = [(n, g, f, c, op_cost(n, b, k, doc, words)) for n, g, f, c, k, b in UN if ok(c)]
        self.bi = [(n, g, f, c, op_cost(n, b, k, doc, words)) for n, g, f, c, k, b in BIN if ok(c)]
        self.te = [(n, g, f, c, op_cost(n, b, k, doc, words)) for n, g, f, c, k, b in TER if ok(c)]
        self.target_hook = None
        self.extra = {}  # cost -> list of pre-built terms to inject (e.g. comprehensions)

    def add(self, code, cost, vals):
        try:
            key = repr(vals)
        except Exception:
            return None
        if key in self.seen:
            return None
        self.seen.add(key)
        t = Term(code, cost, vals)
        self.by_cost.setdefault(cost, []).append(t)
        self.count += 1
        return t

    def apply(self, fn, args):
        out = []
        real = self.real
        for r in range(self.n):
            xs = [a.vals[r] for a in args]
            try:
                if any(x is ERR for x in xs):
                    raise ValueError
                v = guard(fn(*xs))
            except Exception:
                if real[r]:
                    raise
                v = ERR
            out.append(v)
        return tuple(out)

    def grow(self, cost, on_new=None):
        """Generate all terms of exactly `cost`. on_new(term) -> True to stop."""
        new = []

        def emit(code, vals):
            if self.target_hook is not None:
                self.target_hook(code, cost, vals)
            t = self.add(code, cost, vals)
            if t is not None:
                new.append(t)
                if on_new and on_new(t):
                    return True
            return False

        for t in self.extra.get(cost, []):
            if emit(t.code, t.vals):
                return True
        for name, gs, fn, fmt, oc in self.un:
            for a in self.by_cost.get(cost - oc, []):
                if not ok_tags(a.tags, gs[0]):
                    continue
                try:
                    v = self.apply(fn, [a])
                except Exception:
                    continue
                if emit(render(fmt, [a.code]), v):
                    return True
            if time.time() > self.deadline or self.count > self.max_terms:
                return False
        for name, gs, fn, fmt, oc in self.bi:
            rem = cost - oc
            commut = name in ("add", "mul", "eq", "ne", "and", "or", "max2", "min2", "gcd")
            for c1 in range(1, rem):
                c2 = rem - c1
                if commut and c1 > c2:
                    continue
                A = [a for a in self.by_cost.get(c1, []) if ok_tags(a.tags, gs[0])]
                B = [b for b in self.by_cost.get(c2, []) if ok_tags(b.tags, gs[1])]
                for a in A:
                    for b in B:
                        if commut and c1 == c2 and a.code > b.code:
                            continue
                        if name == "add" and not (a.tags == b.tags or (a.tags | b.tags) <= {"int", "float"}):
                            continue
                        try:
                            v = self.apply(fn, [a, b])
                        except Exception:
                            continue
                        if emit(render(fmt, [a.code, b.code]), v):
                            return True
                    if time.time() > self.deadline or self.count > self.max_terms:
                        return False
        for name, gs, fn, fmt, oc in self.te:
            rem = cost - oc
            for c1 in range(1, rem - 1):
                for c2 in range(1, rem - c1):
                    c3 = rem - c1 - c2
                    A = [a for a in self.by_cost.get(c1, []) if ok_tags(a.tags, gs[0])]
                    if name == "ite":
                        A = [a for a in A if a.tags <= {"bool"}]
                    B = [b for b in self.by_cost.get(c2, []) if ok_tags(b.tags, gs[1])]
                    C = [c for c in self.by_cost.get(c3, []) if ok_tags(c.tags, gs[2])]
                    for a in A:
                        for b in B:
                            for c in C:
                                if name == "ite" and not (b.tags == c.tags or "none" in (b.tags | c.tags)
                                                          or (b.tags | c.tags) <= {"int", "float"}):
                                    continue
                                try:
                                    v = self.apply(fn, [a, b, c])
                                except Exception:
                                    continue
                                if emit(render(fmt, [a.code, b.code, c.code]), v):
                                    return True
                            if time.time() > self.deadline or self.count > self.max_terms:
                                return False
        return False


EXLINE = re.compile(r">>>|==>|=>|➞|->|output|input|for example|e\.g\.|should return|\w+\(.*\)\s*(==|=|returns?)", re.I)


def doc_constants(doc, examples, entry):
    """Constants from the prose only (never from example lines) to avoid copying answers."""
    prose = "\n".join(l for l in doc.split("\n") if not EXLINE.search(l) and entry not in l)
    nums, strs = set(), set()
    for m in re.finditer(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", prose):
        try:
            v = int(m.group()) if "." not in m.group() else float(m.group())
            if abs(v) <= 1000:
                nums.add(v)
        except ValueError:
            pass
    for m in re.finditer(r"'([^'\n]{0,20})'|\"([^\"\n]{0,20})\"", prose):
        strs.add(m.group(1) if m.group(1) is not None else m.group(2))
    outvals = {repr(o) for _, o in examples}
    nums = {v for v in nums if repr(v) not in outvals and repr(float(v)) not in outvals and v not in (0, 1, 2)}
    strs = {x for x in strs if repr(x) not in outvals}
    souts = [o for _, o in examples if isinstance(o, str)]
    if souts and len(set(souts)) <= 2 and all(len(o) <= 8 for o in souts):
        strs |= set(souts)  # categorical answers like 'YES'/'NO'
    return nums, strs


BASE_CONSTS = [(0, 1), (1, 1), (2, 1), (-1, 2), (10, 2), (3, 2), ("", 2), (" ", 2), (True, 2), (False, 2), (None, 2),
               ("aeiouAEIOU", 3), ("abcdefghijklmnopqrstuvwxyz", 3)]


def seed(bank, env, nums, strs):
    for name, vals in env:
        bank.add(name, 1, tuple(vals))
    consts = list(BASE_CONSTS) + [(v, 2) for v in nums] + [(x, 2) for x in strs]
    for v, c in consts:
        bank.add(repr(v), c, tuple([v] * bank.n))


def mutate(v, k):
    """Deterministic input perturbations used as probe inputs (no outputs known)."""
    if isinstance(v, bool):
        return [not v, v][k % 2]
    if isinstance(v, int):
        return [v + 1, v * 2 + 3, max(0, v - 1), v + 7, abs(v) + 13, v * 3][k % 6]
    if isinstance(v, float):
        return [v + 0.5, v * 2 + 0.25, v - 1.5, v + 3.75, abs(v) + 0.1, v * 3][k % 6]
    if isinstance(v, str):
        if not v:
            return ["ab", "Hello world", "x", "", "aba", "Z"][k % 6]
        return [v[::-1], v[1:] + v[:1], v + v[:2], v[: max(1, len(v) // 2)], v.swapcase(), v[1:]][k % 6]
    if isinstance(v, list):
        if not v:
            return [[1, 2], [], [3], [5, 1, 4], [], [0]][k % 6]
        return [v[::-1], v[1:] + v[:1], v + v[:2], v[: max(1, len(v) // 2)], sorted(v, key=repr), v[1:]][k % 6]
    if isinstance(v, tuple):
        return tuple(mutate(list(v), k))
    return v


class Synth:
    def __init__(self, prompt, entry, budget=10.0):
        self.prompt = prompt
        self.entry = entry
        self.info, self.ex = extract_examples(prompt, entry)
        self.doc = self.info["doc"].lower()
        self.budget = budget
        self.banned = [entry] + self.info["helpers"]
        self.last_key = None

    def probes(self, k=6):
        seen = {repr(e[0]) for e in self.ex}
        out = []
        for j in range(k * 3):
            base = self.ex[j % len(self.ex)][0]
            p = [mutate(v, j + i) for i, v in enumerate(base)]
            if repr(p) not in seen:
                seen.add(repr(p))
                out.append(p)
            if len(out) >= k:
                break
        return out

    def run(self):
        t0 = time.time()
        deadline = t0 + self.budget
        args = self.info["args"]
        ex = self.ex
        if not ex:
            return None, "no-examples"
        outs = [o for _, o in ex]
        n = len(ex)
        prb = self.probes()
        inputs = [e[0] for e in ex] + prb
        N = len(inputs)
        real = [True] * n + [False] * (N - n)
        self.N, self.real = N, real
        nums, strs = doc_constants(self.info["doc"], ex, self.entry)
        out_tags = {tag(o) for o in outs}

        top = Bank(N, self.doc, deadline, real=real, banned=self.banned)
        env = [(a, [inp[i] for inp in inputs]) for i, a in enumerate(args)]
        cands = {}

        def hook(code, cost, vals):
            if code not in cands and all(same(vals[r], outs[r]) for r in range(n)):
                crash = sum(1 for v in vals[n:] if v is ERR)
                badtype = sum(1 for v in vals[n:] if v is not ERR and tag(v) not in out_tags and
                              not (v is None or "none" in out_tags))
                cands[code] = (cost, crash, badtype)

        top.target_hook = hook
        seed(top, env, nums, strs)
        for c in sorted(top.by_cost):
            for t in top.by_cost[c]:
                hook(t.code, t.cost, t.vals)
        first = [None]

        def done(c):
            good = [k for k, v in cands.items() if self.uses_all(k) and v[1] == 0]
            if good and first[0] is None:
                first[0] = c
            return first[0] is not None and c >= first[0] + 2

        for c in range(2, 4):
            top.grow(c)
            if done(c):
                return self.pick(cands), "phase1"
        sources = []
        for c in range(1, 5):
            for t in top.by_cost.get(c, []):
                vs = [v for v in t.vals[:n]]
                if t.tags <= {"list", "str", "tuple"} and all(v is not ERR for v in vs) and \
                        all(len(v) <= 60 for v in t.vals if v is not ERR) and any(len(v) > 0 for v in vs):
                    sources.append(t)
        sources = sources[:12]
        lam_deadline = time.time() + self.budget * 0.35
        hof_terms = self.build_hof(sources, env, nums, strs, lam_deadline)
        for t in hof_terms:
            hook(t.code, t.cost, t.vals)
            top.extra.setdefault(max(4, t.cost), []).append(t)
        c = 4
        while time.time() < deadline and c < 60:
            top.grow(c)
            if done(c):
                break
            c += 1
        code = self.pick(cands)
        return code, ("search" if code else "timeout")

    def uses_all(self, code):
        return all(re.search(r"\b" + re.escape(a) + r"\b", code) for a in self.info["args"])

    def rank_key(self, k, v):
        cost, crash, badtype = v
        return (not self.uses_all(k), crash > 0, badtype > 0, cost + 2 * crash + 2 * badtype, len(k))

    def pick(self, cands):
        if not cands:
            return None
        best = min(cands, key=lambda k: self.rank_key(k, cands[k]))
        self.last_key = self.rank_key(best, cands[best])
        return best

    def build_hof(self, sources, env, nums, strs, lam_deadline):
        """Comprehension / key-function terms over each source with a per-element body search."""
        out = []
        N, real = self.N, self.real
        W = doc_words(self.doc)
        fc = prior_cost("M_filter", W, False)
        kc = prior_cost("M_key", W, any(w in self.doc for w in ("key", "sort", "order", "largest", "smallest")))
        mx = prior_cost("M_x", W, False)
        per_src = max(0.2, (lam_deadline - time.time()) / max(1, len(sources)))
        for src in sources:
            if time.time() > lam_deadline:
                break
            sd = min(lam_deadline, time.time() + per_src)
            S = par(src.code)
            for mode in ("x", "ix", "pair", "adj"):
                rows = []
                bad = False
                for e in range(N):
                    sv = src.vals[e]
                    if sv is ERR:
                        continue
                    seq = list(sv)
                    if mode == "x":
                        rows += [(e, (x,)) for x in seq]
                    elif mode == "ix":
                        rows += [(e, (i, x)) for i, x in enumerate(seq)]
                    elif mode == "pair":
                        if len(seq) > 25:
                            bad = True
                            break
                        rows += [(e, (x, y)) for i, x in enumerate(seq) for y in seq[i + 1:]]
                    else:
                        rows += [(e, (x, y)) for x, y in zip(seq, seq[1:])]
                if bad or not rows or len(rows) > 900:
                    continue
                names = {"x": ["x"], "ix": ["i", "x"], "pair": ["x", "y"], "adj": ["x", "y"]}[mode]
                b = Bank(len(rows), self.doc, sd, max_terms=5000, real=[real[e] for e, _ in rows], banned=self.banned)
                lenv = [(nm, [r[1][k] for r in rows]) for k, nm in enumerate(names)]
                for name, vals in env:
                    if name != src.code:
                        lenv.append((name, [vals[r[0]] for r in rows]))
                seed(b, lenv, nums, strs)
                for c in range(2, 16):
                    b.grow(c)
                    if time.time() > sd or b.count > 5000:
                        break
                gen = {"x": f"for x in {src.code}", "ix": f"for i, x in enumerate({src.code})",
                       "pair": f"for i, x in enumerate({src.code}) for y in {S}[i + 1:]",
                       "adj": f"for x, y in zip({src.code}, {S}[1:])"}[mode]
                mc = {"x": 0, "ix": prior_cost("M_ix", W, False) - 1, "pair": prior_cost("M_pair", W, False) - 1,
                      "adj": prior_cost("M_adj", W, False) - 1}[mode]
                base = src.cost + mx + max(0, mc)
                bodies = [t for c in sorted(b.by_cost) for t in b.by_cost[c]]
                preds = [t for t in bodies if t.tags <= {"bool"}]

                def group(t):
                    g = [[] if src.vals[e] is not ERR else ERR for e in range(N)]
                    for r, (e, _) in enumerate(rows):
                        v = t.vals[r]
                        if g[e] is ERR:
                            continue
                        if v is ERR:
                            g[e] = ERR
                        else:
                            g[e].append(v)
                    return g

                gb = {id(t): group(t) for t in bodies}
                elems = [[] if src.vals[e] is not ERR else ERR for e in range(N)]
                for e, el in rows:
                    elems[e].append(el[-1] if mode in ("x", "ix") else el)
                for t in bodies:
                    if t.code in names:
                        continue
                    out.append(Term(f"[{t.code} {gen}]", base + t.cost, tuple(gb[id(t)])))
                for p in preds:
                    gp = gb[id(p)]
                    if mode in ("x", "ix"):
                        vals = tuple(ERR if gp[e] is ERR or elems[e] is ERR else
                                     [el for el, k in zip(elems[e], gp[e]) if k] for e in range(N))
                        out.append(Term(f"[x {gen} if {p.code}]", base + p.cost + fc, vals))
                    for t in bodies[:300]:
                        if t is p or t.code in names and mode in ("x", "ix") and t.code == "x":
                            continue
                        gt = gb[id(t)]
                        vals = tuple(ERR if gp[e] is ERR or gt[e] is ERR else
                                     [v for v, k in zip(gt[e], gp[e]) if k] for e in range(N))
                        out.append(Term(f"[{t.code} {gen} if {p.code}]", base + p.cost + t.cost + fc, vals))
                if mode == "x":
                    for t in bodies:
                        if t.code == "x":
                            continue
                        vs, vmx, vmn = [], [], []
                        okk = True
                        for e in range(N):
                            ge = gb[id(t)][e]
                            if ge is ERR or elems[e] is ERR:
                                if real[e]:
                                    okk = False
                                    break
                                vs.append(ERR); vmx.append(ERR); vmn.append(ERR)
                                continue
                            pairs = list(zip(ge, elems[e]))
                            try:
                                vs.append([x for _, x in sorted(pairs, key=lambda q: q[0])])
                                vmx.append(max(pairs, key=lambda q: q[0])[1] if pairs else ERR)
                                vmn.append(min(pairs, key=lambda q: q[0])[1] if pairs else ERR)
                            except Exception:
                                okk = False
                                break
                        if not okk:
                            continue
                        out.append(Term(f"sorted({src.code}, key=lambda x: {t.code})", base + t.cost + kc, tuple(vs)))
                        if all(v is not ERR for v in vmx[:len(self.ex)]):
                            out.append(Term(f"max({src.code}, key=lambda x: {t.code})", base + t.cost + kc, tuple(vmx)))
                            out.append(Term(f"min({src.code}, key=lambda x: {t.code})", base + t.cost + kc, tuple(vmn)))
        return out


def choose(enum_code, enum_key, rets):
    """Pick between the enumerative program and verified library functions."""
    clean = sorted([r for r in rets if r["crash"] == 0], key=lambda r: -r["sim"])
    if not clean:
        return ("enum", None) if enum_code else (None, None)
    best = clean[0]
    if enum_code is None:
        return "ret", best
    simple = enum_key is not None and not enum_key[0] and not enum_key[1] and not enum_key[2] and enum_key[3] <= SIMPLE
    if simple and best["sim"] < STRONG:
        return "enum", None
    return "ret", best


SIMPLE = 6
STRONG = 0.45


def synthesize(prompt, entry, budget=10.0, use_retrieval=True):
    from .retrieve import retrieve
    s = Synth(prompt, entry, budget)
    rets = []
    if use_retrieval and s.ex:
        try:
            rets = retrieve(s.info, s.ex, s.probes(), entry)
        except Exception:
            rets = []
    try:
        code, how = s.run()
    except Exception as e:  # never crash the harness
        code, how = None, f"error:{type(e).__name__}"
    key = s.last_key if code else None
    which, r = choose(code, key, rets)
    s.debug = {"enum": code, "enum_key": key, "rets": [(x["sim"], x["call"], x["crash"], x["id"]) for x in rets]}
    if which == "ret":
        return LIB_SRC + "\n" + r["code"] + "\n\n" + prompt + f"    return {r['call']}\n", r["call"], "retrieval", len(s.ex)
    body = "    return None\n" if code is None else f"    return {code}\n"
    return LIB_SRC + "\n" + prompt + body, code, how, len(s.ex)
