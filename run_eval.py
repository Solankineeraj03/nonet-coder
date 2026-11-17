"""Run NoNet Coder on a split and score with HumanEval's hidden tests (in subprocesses)."""
import argparse, json, os, platform, resource, signal as _signal, subprocess, sys, tempfile, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from nonet.synth import synthesize

P = [json.loads(l) for l in open("data/HumanEval.jsonl")]
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nonet", "sandbox_runner.py")


class HardTimeout(BaseException):
    pass


def _alarm(*_):
    raise HardTimeout


def score(src, scorer, cpu=None):
    """Run program+tests; return a reason: pass | assert | exception | cpu-limit | wall-timeout | runner-error | killed:N.
    scorer='runner' goes through nonet/sandbox_runner.py --script (rule 5); 'legacy' is the pre-v5 plain
    `python -I file` path, kept only for the A/B compatibility check."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
    argv = [sys.executable, "-I", RUNNER, "--script", f.name] + ([str(cpu)] if cpu else []) if scorer == "runner" else [sys.executable, "-I", f.name]
    try:
        r = subprocess.run(argv, capture_output=True, timeout=15)
        rc = r.returncode
        if rc == 0:
            return "pass"
        if rc < 0:
            return "cpu-limit" if -rc == _signal.SIGXCPU else f"killed:{-rc}"
        if scorer == "runner":
            return {1: "assert", 2: "exception", 3: "runner-error"}.get(rc, f"exit:{rc}")
        return "assert" if b"AssertionError" in r.stderr else "exception"
    except subprocess.TimeoutExpired:
        return "wall-timeout"
    finally:
        os.unlink(f.name)


def work(args):
    import signal
    from nonet.lib import LIB_SRC
    i, budget, ab = args
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
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
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    synth_cpu = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)  # excludes retrieval subprocess
    test_src = program + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})\n"
    reason = score(test_src, "runner")
    out = {"debug": None, "id": i, "entry": p["entry_point"], "code": code, "how": how, "nex": nex,
           "passed": reason == "pass", "reason": reason, "synth_found": code is not None,
           "secs": round(dt, 2), "synth_cpu": round(synth_cpu, 2)}
    if ab:
        out["reason_legacy"] = score(test_src, "legacy")
    return out


def meta(a, n):
    def git(*c):
        try:
            return subprocess.run(["git", *c], capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    import hashlib
    h = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "workers": WORKERS, "budget": a.budget, "split": a.split, "n": n,
            "scorer": "sandbox_runner --script (RLIMIT_CPU 15s, wall 15s; wall usually binds first)", "ab_scorer": a.ab_scorer,
            "data_sha256_16": {f: h(f) for f in ("data/HumanEval.jsonl", "data/mbpp.jsonl", "data/prior.json", "data/split.json")},
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


WORKERS = 7


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--budget", type=float, default=10.0)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--ab-scorer", action="store_true", help="also score each program with the pre-v5 plain subprocess")
    a = ap.parse_args()
    S = json.load(open("data/split.json"))
    ids = [int(x) for x in a.only.split(",")] if a.only else (S[a.split] if a.split != "all" else list(range(164)))
    M = meta(a, len(ids))
    t0 = time.time()
    with Pool(WORKERS) as pool:
        res = pool.map(work, [(i, a.budget, a.ab_scorer) for i in ids], chunksize=1)
    M["wall_secs"] = round(time.time() - t0, 1)
    res.sort(key=lambda r: r["id"])
    for r in res:
        mark = "PASS" if r["passed"] else ("WRONG" if r["synth_found"] else "----")
        print(f"{mark:5} {r['id']:3} {r['entry'][:28]:28} {r['how']:10} {str(r['code'])[:90]}")
    np_ = sum(r["passed"] for r in res)
    nf = sum(r["synth_found"] for r in res)
    print(f"\n{a.split}: pass@1 = {np_}/{len(res)} = {100*np_/len(res):.1f}%   (found-consistent {nf}, overfit {nf-np_})")
    from collections import Counter
    print("reasons:", dict(Counter(r["reason"] for r in res)))
    if a.ab_scorer:
        diff = [(r["id"], r["reason_legacy"], r["reason"]) for r in res if r["reason_legacy"] != r["reason"]]
        print(f"scorer A/B: legacy pass {sum(r['reason_legacy'] == 'pass' for r in res)}, runner pass {np_}, "
              f"differences (id, legacy, runner): {diff}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        json.dump(M, open(a.out.rsplit(".json", 1)[0] + ".meta.json", "w"), indent=1)
