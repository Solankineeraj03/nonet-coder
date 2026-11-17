"""Parse a HumanEval-style prompt: signature, docstring, and I/O examples.

Only the prompt text is used. Never the hidden tests or reference solution.
"""
import ast
import re

# A line that looks like an example (call, arrow, doctest, Input/Output header, ...).
EXLINE = re.compile(r">>>|==>|=>|➞|->|output|input|for example|e\.g\.|should return|\w+\(.*\)\s*(==|=|returns?)", re.I)
# A line that opens an example span: following data-like lines are example data too. Weaker EXLINE hits
# (e.g. "...can output a list...") only exclude their own line, so a spec table after them stays prose.
EXSTART = re.compile(r">>>|==>|=>|➞|^\s*(examples?|explanation|input|output|for example)\b", re.I)


def _looks_like_data(s):
    """Inside an example span: is this line example data (kept out of prose) rather than a sentence?
    Conservative: anything that is not clearly a sentence counts as data."""
    s = s.strip()
    if "=" in s or re.match(r"^[\w\s]{1,30}:", s):  # name = value, name: value, "Note: ..."
        return True
    try:
        ast.literal_eval(s.rstrip(",;"))
        return True
    except Exception:
        pass
    return len(re.findall(r"[A-Za-z]{2,}", s)) < 4


def classify_doc(doc, entry, ex_lines=()):
    """Label each docstring line 'prose', 'example' or 'blank'.

    Example lines: lines the example extractor consumed (`ex_lines`), lines matching EXLINE/EXSTART or
    naming the function, and data-like continuation lines that follow them until a blank line or a sentence.
    """
    kinds, in_ex = [], False
    ent = re.compile(r"\b" + re.escape(entry) + r"\b") if entry else None
    for i, l in enumerate(doc.split("\n")):
        if not l.strip():
            kinds.append("blank")
            in_ex = False
        elif i in ex_lines or EXSTART.search(l) or (ent and ent.search(l)):
            kinds.append("example")
            in_ex = True
        elif EXLINE.search(l):
            kinds.append("example")
        elif in_ex and _looks_like_data(l):
            kinds.append("example")
        else:
            kinds.append("prose")
            in_ex = False
    return kinds


def prose_text(doc, entry, ex_lines=()):
    return "\n".join(l for l, k in zip(doc.split("\n"), classify_doc(doc, entry, ex_lines)) if k == "prose")


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
    spans = []
    exs = _extract(info, entry, spans)
    if not exs:  # docstring may call the function by another name
        names = set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", info["doc"]))
        for nm in sorted(names):
            exs = _extract(info, nm, spans)
            if exs:
                break
    # doc line indices of every example-like match (accepted or not); these are never prose
    doc = info["doc"]
    info["ex_lines"] = {ln for a, b in spans for ln in range(doc.count("\n", 0, a), doc.count("\n", 0, max(a, b - 1)) + 1)}
    return info, exs


def _line_end(doc, i):
    j = doc.find("\n", i)
    return len(doc) if j < 0 else j


def _extract(info, entry, spans=None):
    doc = info["doc"]
    exs = []
    seen = set()
    spans = [] if spans is None else spans
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
        nl = _line_end(doc, end)
        spans.append((m.start(), _line_end(doc, nl + 1) if doc[end:nl].strip() == "" else nl))
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
        spans.append((m.start(), m.end()))
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
        spans.append((m.start(), m.end()))
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
        spans.append((m.start(), m.end()))
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
