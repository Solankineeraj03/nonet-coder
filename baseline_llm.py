"""Baseline: small LLM on HumanEval, greedy pass@1, standard completion format."""
import json, os, subprocess, sys, tempfile, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-360M"
P = [json.loads(l) for l in open("data/HumanEval.jsonl")]
S = json.load(open("data/split.json"))
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev).eval()
STOPS = ["\ndef ", "\nclass ", "\nif __name__", "\n#", "\nprint(", "\nassert "]
res = []
for i, p in enumerate(P):
    pr = p["prompt"].rstrip()  # as in bigcode-evaluation-harness
    ids = tok(pr, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=384, do_sample=False, pad_token_id=tok.eos_token_id)
    comp = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    cut = min([comp.find(s) for s in STOPS if s in comp] + [len(comp)])
    comp = comp[:cut]
    src = pr + comp + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
    try:
        ok = subprocess.run([sys.executable, "-I", f.name], capture_output=True, timeout=15).returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
    os.unlink(f.name)
    res.append({"id": i, "passed": ok, "completion": comp})
    print(i, ok, flush=True)
name = MODEL.split("/")[-1]
json.dump(res, open(f"results_llm_{name}.json", "w"), indent=1)
for sp in ("dev", "test"):
    k = sum(res[i]["passed"] for i in S[sp])
    print(f"{name} {sp}: {k}/{len(S[sp])} = {100*k/len(S[sp]):.1f}%")
k = sum(r["passed"] for r in res)
print(f"{name} all: {k}/164 = {100*k/164:.1f}%")
