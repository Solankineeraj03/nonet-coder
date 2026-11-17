"""Fail if the synthesizer source contains HumanEval-specific knowledge or neural deps."""
import json, re, sys, pathlib
P = [json.loads(l) for l in open("data/HumanEval.jsonl")]
src = "\n".join(p.read_text() for p in pathlib.Path("nonet").glob("*.py"))
bad = []
for p in P:
    e = p["entry_point"]
    if "_" in e and re.search(r"\b" + re.escape(e) + r"\b", src) and e not in ("is_palindrome", "digits", "is_prime"):
        bad.append(f"entry point name in source: {e}")
for dep in ("torch", "tensorflow", "transformers", "jax", "onnx", "sklearn", "requests", "urllib", "canonical_solution", "\"test\""):
    if re.search(r"\b" + re.escape(dep) + r"\b", src):
        bad.append(f"forbidden reference: {dep}")
print("\n".join(bad) or "lint OK: no HumanEval names, no neural/network deps")
sys.exit(1 if bad else 0)
