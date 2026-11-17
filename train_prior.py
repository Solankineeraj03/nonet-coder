"""Learn a word -> component prior from MBPP solutions (NOT HumanEval).

Pure counting. Output: data/prior.json = {"base": {op: p}, "cond": {word: {op: count}}, "wc": {word: n}}
"""
import ast
import json
import re
from collections import Counter, defaultdict

CALLS = {"len": "len", "abs": "abs", "sum": "sum", "list": "list", "str": "str", "int": "int",
         "float": "float", "bool": "bool", "ord": "ord", "chr": "chr", "bin": "bin", "any": "any",
         "all": "all", "tuple": "tuple", "set": "set", "enumerate": "M_ix", "zip": "M_adj",
         "accumulate": "running", "ceil": "ceil", "floor": "floor", "sqrt": "sqrt", "gcd": "gcd",
         "factorial": "factorial"}
METHODS = {"lower": "lower", "upper": "upper", "swapcase": "swapcase", "strip": "strip",
           "isupper": "isupper", "islower": "islower", "isdigit": "isdigit", "isalpha": "isalpha",
           "isspace": "isspace", "count": "count", "startswith": "startswith", "endswith": "endswith",
           "index": "find", "find": "find", "replace": "replace"}
BINOPS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.FloorDiv: "fdiv",
          ast.Mod: "mod", ast.Pow: "pow"}
CMPS = {ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.Gt: "lt", ast.LtE: "le", ast.GtE: "le",
        ast.In: "in", ast.NotIn: "in"}


def const(n):
    return isinstance(n, ast.Constant) and not isinstance(n.value, bool) and isinstance(n.value, int)


def ops_of(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    ops = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            kws = {k.arg for k in n.keywords}
            if name in ("max", "min"):
                ops.add(name + ("2" if len(n.args) >= 2 else ""))
                if "key" in kws:
                    ops.add("M_key")
            elif name == "sorted" or name == "sort":
                ops.add("sorted_desc" if "reverse" in kws else "sorted")
                if "key" in kws:
                    ops.add("M_key")
            elif name == "round":
                ops.add("round2" if len(n.args) == 2 else "round")
            elif name == "range":
                ops.add({1: "range", 2: "range2", 3: "range2"}.get(len(n.args), "range"))
                if len(n.args) >= 2 and const(n.args[0]) and n.args[0].value == 1:
                    ops.add("range1")
            elif name == "split" and isinstance(f, ast.Attribute):
                ops.add("splitby" if n.args else "split")
            elif name == "join" and isinstance(f, ast.Attribute):
                v = f.value
                if isinstance(v, ast.Constant) and v.value == "":
                    ops.add("join")
                elif isinstance(v, ast.Constant) and v.value == " ":
                    ops.add("joinsp")
                else:
                    ops.add("joinby")
            elif name == "reversed":
                ops.add("rev")
            elif isinstance(f, ast.Attribute) and name in METHODS:
                ops.add(METHODS[name])
            elif name in CALLS:
                ops.add(CALLS[name])
            if name and "prime" in name.lower():
                ops.add("is_prime")
        elif isinstance(n, ast.BinOp):
            o = BINOPS.get(type(n.op))
            if o == "pow" and const(n.right) and n.right.value in (2, 3):
                o = {2: "sq", 3: "cube"}[n.right.value]
            if o == "div" and const(n.right) and n.right.value == 2:
                ops.add("half")
            if o:
                ops.add(o)
        elif isinstance(n, ast.Compare):
            for op, rhs in zip(n.ops, n.comparators):
                o = CMPS.get(type(op))
                if o:
                    ops.add(o)
            l = n.left
            if isinstance(l, ast.BinOp) and isinstance(l.op, ast.Mod) and const(l.right) and l.right.value == 2:
                ops.add("even")
                ops.add("odd")
        elif isinstance(n, ast.UnaryOp):
            if isinstance(n.op, ast.USub) and not isinstance(n.operand, ast.Constant):
                ops.add("neg")
            if isinstance(n.op, ast.Not):
                ops.add("not")
        elif isinstance(n, ast.BoolOp):
            ops.add("and" if isinstance(n.op, ast.And) else "or")
        elif isinstance(n, ast.IfExp) or isinstance(n, ast.If):
            ops.add("ite")
        elif isinstance(n, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
            ops.add("M_x")
            if any(g.ifs for g in n.generators):
                ops.add("M_filter")
            if len(n.generators) >= 2:
                ops.add("M_pair")
        elif isinstance(n, (ast.For, ast.While)):
            ops.add("M_x")
            ops.add("loop")
        elif isinstance(n, ast.Subscript):
            s = n.slice
            if isinstance(s, ast.Slice):
                st = s.step
                if st is not None:
                    if isinstance(st, ast.UnaryOp) or (const(st) and st.value < 0):
                        ops.add("rev")
                    else:
                        ops.add("rangestep")
                        ops.add("evens_idx")
                        ops.add("odds_idx")
                elif s.lower is not None and s.upper is not None:
                    ops.add("slice")
                elif s.lower is not None:
                    ops.add("tail" if const(s.lower) and s.lower.value == 1 else "drop")
                elif s.upper is not None:
                    ops.add("take")
                    if isinstance(s.upper, ast.UnaryOp):
                        ops.add("init")
            else:
                if isinstance(s, ast.Constant) and s.value == 0:
                    ops.add("first")
                elif isinstance(s, ast.UnaryOp) and isinstance(s.operand, ast.Constant) and s.operand.value == 1:
                    ops.add("last")
                else:
                    ops.add("idx")
    return ops


STOP = set("a an the to of in for and or is are be given function write that this it with by from as on at "
           "which its into each if will your you we".split())


def words(text):
    ws = re.findall(r"[a-z]+", text.lower())
    out = set()
    for w in ws:
        if w in STOP or len(w) < 3:
            continue
        out.add(w)
        if w.endswith("s") and len(w) > 4:
            out.add(w[:-1])
    return out


def main():
    rows = [json.loads(l) for l in open("data/mbpp.jsonl")]
    base = Counter()
    cond = defaultdict(Counter)
    wc = Counter()
    n = 0
    for r in rows:
        ops = ops_of(r["code"].replace("\r", ""))
        if ops is None:
            continue
        n += 1
        ws = words(r["text"])
        for o in ops:
            base[o] += 1
        for w in ws:
            wc[w] += 1
            for o in ops:
                cond[w][o] += 1
    prior = {"n": n, "base": {o: c / n for o, c in base.items()},
             "cond": {w: dict(c) for w, c in cond.items() if wc[w] >= 3},
             "wc": {w: c for w, c in wc.items() if c >= 3}}
    json.dump(prior, open("data/prior.json", "w"))
    print("problems", n, "ops", len(base), "words", len(prior["wc"]))
    for o, p in sorted(prior["base"].items(), key=lambda x: -x[1])[:15]:
        print(f"  {o:12} {p:.2f}")


if __name__ == "__main__":
    main()
