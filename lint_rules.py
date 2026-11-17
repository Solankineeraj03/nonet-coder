"""Configured static checks for the AGENTS.md hard rules.

These are narrow string/AST checks. Passing them does NOT prove constant provenance, data-access
boundaries, or isolation of MBPP code; those are covered (partially) by behavioral tests in tests/,
and task-specific knowledge still needs human review.
"""
import ast, json, re, sys, pathlib

P = [json.loads(l) for l in open("data/HumanEval.jsonl")]
files = {p.name: p.read_text() for p in pathlib.Path("nonet").glob("*.py")}
src = "\n".join(files.values())
bad = []

# Rule 3: no HumanEval entry-point names (generic helper names in lib.py are allowed)
for p in P:
    e = p["entry_point"]
    if "_" in e and re.search(r"\b" + re.escape(e) + r"\b", src) and e not in ("is_palindrome", "digits", "is_prime"):
        bad.append(f"entry point name in source: {e}")

# Rules 1/2: forbidden references in text
for dep in ("canonical_solution", "\"test\"", "'test'", "HumanEval.jsonl"):
    if dep in src:
        bad.append(f"forbidden reference: {dep}")

FORBIDDEN_MODS = {"torch", "tensorflow", "transformers", "jax", "onnx", "onnxruntime", "sklearn", "keras",
                  "numpy", "requests", "urllib", "urllib3", "http", "socket", "ftplib", "httpx", "aiohttp"}
# Counts call sites only; it does not establish WHAT those calls execute.
EXEC_ALLOWED = {"synth.py": 1, "sandbox_runner.py": 2}  # exec(LIB_SRC); runner's candidate + scoring exec

for name, text in files.items():
    tree = ast.parse(text, name)
    execs = 0
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
            if name == "sandbox_runner.py" and (node.level or (node.module or "").startswith("nonet")):
                bad.append("sandbox_runner.py imports project code")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ("exec", "eval"):
                execs += 1
            if node.func.id == "__import__":
                bad.append(f"{name}: dynamic __import__")
        for m in mods:
            if m.split(".")[0] in FORBIDDEN_MODS:
                bad.append(f"{name}: forbidden import {m}")
            if name == "sandbox_runner.py" and m.split(".")[0] == "nonet":
                bad.append("sandbox_runner.py imports project code")
    if execs > EXEC_ALLOWED.get(name, 0):
        bad.append(f"{name}: {execs} exec/eval call site(s), configured maximum {EXEC_ALLOWED.get(name, 0)}")

# Scoring path (rule 5) and constant provenance (rule 6) are checked behaviorally in tests/test_regressions.py.

print("\n".join(bad) or "lint OK: configured static checks passed (not a proof of the hard rules; see tests/ and review)")
sys.exit(1 if bad else 0)
