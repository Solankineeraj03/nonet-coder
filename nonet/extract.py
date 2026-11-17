"""Parse a HumanEval-style prompt: signature, docstring, and I/O examples.

Only the prompt text is used. Never the hidden tests or reference solution.
"""
import ast
import re

ARROWS = ["==>", "=>", "➞", "->", "==", "should return", "returns", "return", "=", "is", ":"]


def _balanced(s, i):
    """s[i] == '(' -> index just past matching ')' (string-literal aware)."""
    depth, q = 0, None
    j = i
    while j < len(s):
        c = s[j]
        if q:
            if c == "\\":
                j += 2
                continue
            if c == q:
                q = None
        elif c in "'\"":
            q = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return -1


def _literal_prefix(s):
    """Longest prefix of s (stripped) that is a Python literal."""
    s = s.strip()
    if not s:
        return None, False
    # try progressively shorter prefixes ending at line end / punctuation
    cands = [s.split("\n")[0].strip()]
    line = cands[0]
    for m in re.finditer(r"[\s,.;)]", line):
        cands.append(line[: m.start()])
    cands.append(line.rstrip(".,; "))
    best = None
    for c in sorted(set(cands), key=len, reverse=True):
        c = c.strip()
        if not c:
            continue
        try:
            v = ast.literal_eval(c)
            return v, True
        except Exception:
            pass
        lc = c.lower()
        if lc in ("true", "false"):
            return lc == "true", True
        if lc == "none":
            return None, True
    return best, False


def parse_prompt(prompt, entry):
    tree = ast.parse(prompt)
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry:
            fn = node
    args = [a.arg for a in fn.args.args]
    ann = [ast.unparse(a.annotation) if a.annotation else None for a in fn.args.args]
    ret = ast.unparse(fn.returns) if fn.returns else None
    doc = ast.get_docstring(fn) or ""
    return {"args": args, "ann": ann, "ret": ret, "doc": doc, "helpers": [
        n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != entry]}


def extract_examples(prompt, entry):
    info = parse_prompt(prompt, entry)
    exs = _extract(info, entry)
    if not exs:  # docstring may call the function by another name
        names = set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", info["doc"]))
        for nm in sorted(names):
            exs = _extract(info, nm)
            if exs:
                break
    return info, exs


def _extract(info, entry):
    doc = info["doc"]
    exs = []
    seen = set()
    for m in re.finditer(r"\b" + re.escape(entry) + r"\s*\(", doc):
        start = m.end() - 1
        end = _balanced(doc, start)
        if end < 0:
            continue
        argsrc = doc[start + 1: end - 1]
        try:
            call = ast.parse(f"f({argsrc})", mode="eval").body
            inputs = [ast.literal_eval(a) for a in call.args]
            if call.keywords:
                continue
        except Exception:
            continue
        rest = doc[end:]
        out, ok = None, False
        # doctest style: result on next non-empty line
        stripped = rest.lstrip(" \t")
        if stripped.startswith("\n"):
            nxt = stripped[1:]
            first = nxt.split("\n")[0].strip()
            if first.startswith(">>>") or first == "":
                # e.g. `>>> longest([])` followed by blank -> None
                if first == "" and m.start() > 3 and ">>>" in doc[max(0, m.start() - 6): m.start()]:
                    out, ok = None, True
            else:
                out, ok = _literal_prefix(first)
        else:
            r = stripped
            for a in ARROWS:
                if r.lower().startswith(a):
                    rhs = r[len(a):].split("\n")[0]
                    # "-> 19 - 5 - 6 = 8" / "== 1 + 9 + 0 = 10": take the final value
                    if re.match(r"^\s*[-\d.]+\s*[-+*/]\s*[\d.].*=", rhs):
                        rhs = rhs.rsplit("=", 1)[1]
                    out, ok = _literal_prefix(rhs)
                    break
        if ok:
            key = repr((inputs, out))
            if key not in seen:
                seen.add(key)
                exs.append((inputs, out))
    # "Input: ... Output: ..." style blocks
    for m in re.finditer(r"Input:\s*(.+?)\n\s*Output:\s*(.+?)(\n|$)", doc):
        try:
            src = m.group(1).strip()
            if "=" in src:
                parts = re.split(r",\s*(?=\w+\s*=)", src)
                inputs = [ast.literal_eval(p.split("=", 1)[1].strip()) for p in parts]
            else:
                inputs = [ast.literal_eval(src)]
            out, ok = _literal_prefix(m.group(2))
            if ok and len(inputs) == len(info["args"]):
                key = repr((inputs, out))
                if key not in seen:
                    seen.add(key)
                    exs.append((inputs, out))
        except Exception:
            pass
    # "For lst = [..] the output should be 10"
    for m in re.finditer(r"[Ff]or\s+(.+?)\s*,?\s+the output should be\s+(.+)", doc):
        try:
            parts = re.split(r",\s*(?=\w+\s*=)", m.group(1).strip())
            inputs = [ast.literal_eval(p.split("=", 1)[1].strip()) for p in parts]
            out, ok = _literal_prefix(m.group(2))
            if ok:
                key = repr((inputs, out))
                if key not in seen:
                    seen.add(key)
                    exs.append((inputs, out))
        except Exception:
            pass
    # multi-line "Input:\n name : value\n name : value\n Output: v"
    for m in re.finditer(r"Input:\s*\n((?:\s*\w+\s*[:=].+\n)+)\s*Output:\s*(.+)", doc):
        try:
            lines = [l for l in m.group(1).split("\n") if l.strip()]
            inputs = [ast.literal_eval(re.split(r"[:=]", l, 1)[1].strip()) for l in lines]
            out, ok = _literal_prefix(m.group(2))
            if ok:
                key = repr((inputs, out))
                if key not in seen:
                    seen.add(key)
                    exs.append((inputs, out))
        except Exception:
            pass
    exs = [e for e in exs if len(e[0]) == len(info["args"])]
    return exs
