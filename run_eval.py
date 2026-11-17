"""Run NoNet Coder on a split and score with HumanEval's hidden tests (in subprocesses)."""
import argparse, json, os, subprocess, sys, tempfile, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from nonet.synth import synthesize

P = [json.loads(l) for l in open("data/HumanEval.jsonl")]


class HardTimeout(BaseException):
    pass


def _alarm(*_):
    raise HardTimeout


def work(args):
    import signal
    from nonet.lib import LIB_SRC
    i, budget = args
    p = P[i]
    t0 = time.time()
    signal.signal(signal.SIGVTALRM, _alarm)
    signal.setitimer(signal.ITIMER_VIRTUAL, budget * 2 + 5)
    # synthesizer gets ONLY the prompt text and entry-point name
    try:
        program, code, how, nex = synthesize(p["prompt"], p["entry_point"], budget)
    except HardTimeout:
        program, code, how, nex = LIB_SRC + "\n" + p["prompt"] + "    return None\n", None, "HARDTIMEOUT", -1
    signal.setitimer(signal.ITIMER_VIRTUAL, 0)
    dt = time.time() - t0
    test_src = program + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(test_src)
    try:
        r = subprocess.run([sys.executable, "-I", f.name], capture_output=True, timeout=15)
        passed = r.returncode == 0
    except subprocess.TimeoutExpired:
        passed = False
    os.unlink(f.name)
    return {"debug": None, "id": i, "entry": p["entry_point"], "code": code, "how": how, "nex": nex,
            "passed": passed, "synth_found": code is not None, "secs": round(dt, 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--budget", type=float, default=10.0)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    S = json.load(open("data/split.json"))
    ids = [int(x) for x in a.only.split(",")] if a.only else (S[a.split] if a.split != "all" else list(range(164)))
    with Pool(7) as pool:
        res = pool.map(work, [(i, a.budget) for i in ids], chunksize=1)
    res.sort(key=lambda r: r["id"])
    for r in res:
        mark = "PASS" if r["passed"] else ("WRONG" if r["synth_found"] else "----")
        print(f"{mark:5} {r['id']:3} {r['entry'][:28]:28} {r['how']:10} {str(r['code'])[:90]}")
    np_ = sum(r["passed"] for r in res)
    nf = sum(r["synth_found"] for r in res)
    print(f"\n{a.split}: pass@1 = {np_}/{len(res)} = {100*np_/len(res):.1f}%   (found-consistent {nf}, overfit {nf-np_})")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
