"""Isolated runner for library (MBPP) candidates. Run as: python -I sandbox_runner.py < job

stdin: repr() of {"cands": [(code, fname, arity, nargs, tag)], "examples": [(inputs, out)], "probes": [inputs], "k": nargs}
stdout: repr() of [(tag, perm, crash_count)] for candidates that reproduce every example.
Stdlib only; never imports project code.
"""
import ast
import copy
import itertools
import resource
import signal
import sys


class TO(BaseException):
    pass


def alarm(*_):
    raise TO


def same(v, out):
    try:
        if isinstance(v, bool) != isinstance(out, bool):
            return False
        if isinstance(out, float) or isinstance(v, float):
            return isinstance(v, (int, float)) and isinstance(out, (int, float)) and abs(v - out) < 1e-6
        if out is None or v is None:
            return v is out
        return type(v) == type(out) and v == out
    except BaseException:
        return False


def timed(fn, t):
    signal.setitimer(signal.ITIMER_REAL, t)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    signal.signal(signal.SIGALRM, alarm)
    job = ast.literal_eval(sys.stdin.read())
    sys.stdin = None
    real_stdout = sys.stdout
    sys.stdout = open("/dev/null", "w")  # library code may print
    k = job["k"]
    res = []
    for code, fname, arity, nargs, tag in job["cands"]:
        if not (arity <= k <= nargs):
            continue
        g = {"__name__": "lib"}
        try:
            timed(lambda: exec(code, g), 0.5)
        except BaseException:
            continue
        f = g.get(fname)
        if not callable(f):
            continue
        for perm in itertools.permutations(range(k)):
            ok = True
            for inp, out in job["examples"]:
                try:
                    v = timed(lambda: f(*[copy.deepcopy(inp[p]) for p in perm]), 0.3)
                except BaseException:
                    ok = False
                    break
                if not same(v, out):
                    ok = False
                    break
            if not ok:
                continue
            crash = 0
            for inp in job["probes"]:
                try:
                    timed(lambda: f(*[copy.deepcopy(inp[p]) for p in perm]), 0.3)
                except BaseException:
                    crash += 1
            res.append((tag, perm, crash))
            break
    real_stdout.write(repr(res))


if __name__ == "__main__":
    main()
