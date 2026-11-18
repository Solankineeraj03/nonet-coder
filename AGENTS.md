# AGENTS.md: NoNet Coder

Context for coding agents working in this repo. Read it fully before changing anything.

## 1. What this project is

NoNet Coder writes the body of a Python function from its signature and docstring
(HumanEval format) **without any neural network**. It combines:

1. **Enumerative program search.** It builds Python expressions bottom-up, from cheapest to most
   expensive, and keeps the ones that reproduce every input/output example in the docstring.
2. **A learned cost prior.** The prior is plain word/component co-occurrence counts from MBPP
   solutions. It decides which components are cheap for a given docstring.
3. **Retrieval.** It ranks MBPP's 974 solved functions by TF-IDF similarity to the docstring and
   runs the best ones in a separate resource-limited process on the examples.

Goal: beat a 360M-parameter LLM (SmolLM2-360M) on HumanEval pass@1 using one CPU core.

### Current results (greedy pass@1, HumanEval hidden tests)

| System | Dev (82) | Test (82, held out) | All 164 |
|---|---|---|---|
| SmolLM2-360M base (`baseline_llm.py`) | 13 (15.9%) | 8 (9.8%) | 21 (12.8%) |
| NoNet Coder, checkpoint 1 (commit `3c816d1`) | 19 (23.2%) | 14 (17.1%) | 33 (20.1%) |
| NoNet Coder, milestone 1 (commit `0940738`) | 18 (22.0%)* | not run | – |
| NoNet Coder, CPU-budget experiment (previously uncommitted) | 20 (24.4%)* | not run | – |
| NoNet Coder, clean commit `6a0164f` (3 dev runs) | 20 (24.4%) each | not run | – |

\* Two identical runs gave 18/82 with the same solved set (`results_dev_v5*.json`, with metadata in
`results_dev_v5.meta.json`). The search budget is **wall-clock** and 7 workers oversubscribe the machine:
the only v4→v5 loss (58) and the provisional-run win (101) both time out under the pool but pass 3/3 in
isolation. Treat ±1–2 dev tasks as within measurement noise. Subsequent runs use a different,
CPU-time enumeration budget and should be compared as a separate evaluation protocol.
The old and new scorers agreed on all 82 programs (`--ab-scorer`).
The CPU-budget experiment used 10 CPU seconds for enumeration and four workers
(`results_dev_cpu.json` and `.meta.json`); dev 58 and 101 passed. In a separate repeated
check, 58 passed twice with one worker but only once in two attempts with four workers.
CPU deadlines reduce scheduling sensitivity; they do not make the search deterministic.
Three sequential dev runs of clean commit `6a0164f` solved the same 20 tasks; their result and
metadata files are `results_dev_clean_1..3.json` and corresponding `.meta.json` files.

Failure breakdown on test: 32 overfit (the program matches the examples but fails the hidden tests),
25 timed out, 9 had no examples, 2 crashed with a KeyError.

## 2. Hard rules (do not break these)

1. **No neural networks** in `nonet/`: no weights, embeddings, model calls, or network access.
   `lint_rules.py` enforces part of this (it rejects torch/transformers/sklearn/requests/urllib).
   The LLM is used only in `baseline_llm.py`, as the opponent.
2. **The synthesizer only sees the prompt text and the entry-point name.** It must never read
   `canonical_solution` or `test` from `data/HumanEval.jsonl`. Only `run_eval.py` uses `test`,
   and only to score.
3. **No problem-specific code.** Never hardcode HumanEval function names, task IDs or solutions.
   `lint_rules.py` fails if any HumanEval entry-point name containing `_` appears in `nonet/*.py`.
   Generic English words (`unique`, `longest`) are allowed.
4. **Split discipline.** Develop and tune only on the **dev** split (`data/split.json`, `"dev"` key).
   The **test** split has been evaluated once (checkpoint 1). Any further test run must be
   reported as a re-run, and you must never tune against test failures.
5. **MBPP code is untrusted third-party code.** Run it only through `nonet/sandbox_runner.py`
   (a separate `python -I` subprocess with CPU and wall-clock limits). Never `exec` MBPP code in
   the main process. The repo owner explicitly approved sandboxed execution only.
6. Constants used in synthesis must come from the docstring's **prose**, never from example lines
   or example outputs (this prevents copying answers). See `doc_constants`.

## 3. Repository layout

```
nonet-coder/
├── AGENTS.md                 this file
├── README.md                 short public summary and results table
├── nonet/                    THE SYSTEM (the only code that counts as "the model")
│   ├── extract.py            parse the prompt: signature, docstring, I/O examples
│   ├── __main__.py           command-line interface for generating a program from a prompt
│   ├── synth.py              the enumerative synthesizer, ranking, final choice
│   ├── prior.py              doc-conditioned component costs from data/prior.json
│   ├── lib.py                LIB_SRC: concept-library helpers, prepended to every program
│   ├── retrieve.py           TF-IDF retrieval over MBPP + call into the runner
│   └── sandbox_runner.py     resource-limited runner for MBPP candidates (stdlib only)
├── train_prior.py            builds data/prior.json from data/mbpp.jsonl (AST counting)
├── run_eval.py               runs the synthesizer on a split, scores with hidden tests
├── baseline_llm.py           SmolLM2-360M HumanEval baseline (runs in .venv-llm)
├── lint_rules.py             rule checker (see §2)
├── data/
│   ├── HumanEval.jsonl       164 problems (prompt, entry_point, canonical_solution, test)
│   ├── mbpp.jsonl            974 MBPP problems (text, code, test_list); training/library data
│   ├── split.json            {"dev": [82 ids], "test": [82 ids]}, random.Random(0) shuffle
│   └── prior.json            output of train_prior.py
├── results_dev_v0..v4.json   per-problem dev results by version (v0 and v4 = 23.2%)
├── results_test_checkpoint1.json   the single test evaluation
├── results_llm_SmolLM2-360M.json   baseline per-problem completions and pass flags
└── .venv-llm/                isolated venv for the baseline only (gitignored)
```

## 4. Pipeline, end to end

`run_eval.py:work()` → `nonet.synth.synthesize(prompt, entry, budget)` → returns
`(program_source, chosen_expr, how, n_examples)`. Then `run_eval.py` appends the HumanEval `test`
code and `check(entry)`, and runs it via `python -I nonet/sandbox_runner.py --script <file>` (15 s CPU limit,
120 s wall watchdog), because a retrieved program embeds MBPP code (rule 5).
Each result records a `reason` (pass/assert/exception/cpu-limit/wall-timeout/runner-error), and
`--out X.json` also writes `X.meta.json` (commit, Python, data hashes, budget, workers, wall time).
`--ab-scorer` additionally scores with the pre-v5 plain subprocess for comparison. These are resource
limits, **not** an OS security sandbox: MBPP code can still reach the filesystem and network.
Generated files may embed MBPP code; running them separately does not inherit the runner's limits.

### 4.1 `nonet/extract.py`: examples from the docstring
- `parse_prompt(prompt, entry)` (line 64) → `{"args", "ann", "ret", "doc", "helpers"}`, using
  `ast`. `helpers` lists the other functions defined in the prompt.
- `extract_examples(prompt, entry)` (line 78) → `(info, [(inputs_list, output), ...])`.
  `_extract` (line 90) supports these formats:
  - doctest: `>>> f(x)` with the result on the next line
  - arrows: `f(x) ==> y`, `=>`, `➞`, `->`, `==`, `should return`, `returns`, `=`
  - arithmetic chains (`-> 19 - 5 - 6 = 8`): the final value is taken
  - `For lst = [...] the output should be 10`
  - `Input: ... Output: ...`, on one line or across multiple lines
  - a docstring that calls the function by a different name (falls back to any `name(` with the
    right arity)
- `parse_prompt` also recognizes a descriptive string following leading imports inside a function.
- Known gaps: 3 dev problems yield no examples (32, 50, 83). HumanEval has its own docstring
  errors (for example `median` shows 15.0).

### 4.2 `nonet/synth.py`: the synthesizer (the core, about 720 lines)
- **Values and types:** `tag()` and `tagset()` (lines 30, 50). `ERR` (line 27) marks a crash on a
  *probe* row. `GROUPS` (line 55) holds the type groups used for well-typed composition (`ar` = int|float,
  `ix` = int, `addable`, `obj`, ...). `guard()` (line 69) rejects huge values.
- **Component library (the grammar):** `UN` (line 109), `BIN` (line 171) and `TER` (line 203) are
  lists of `(name, arg_groups, python_fn, code_format, keywords, base_cost)`. The `name` must match
  the op names `train_prior.py` emits, so the prior can price it.
- **Code rendering:** `par()` and `render()` (lines 216, 232) add parentheses where precedence
  needs them.
- **`Bank`** (line 275) is a cost-indexed term bank with **observational-equivalence pruning**: two
  terms with identical value vectors on all rows are merged, keeping the cheaper. Rows are the real
  examples plus probes (`real` flags). An exception on a real row rejects the term; on a probe row
  it becomes `ERR`. `grow(cost)` (line 324) generates every unary, binary and ternary term at that
  exact cost. `target_hook` sees **every** candidate, including OE duplicates, so semantically
  different programs that happen to agree on the examples are all recorded as candidates.
- **Constants:** `doc_constants()` reads prose spans only. The prose/example split is
  `extract.classify_doc()`, which is shared with extraction: every doc line the extractor consumed
  (`info["ex_lines"]`) is an example line, and strong markers (`>>>`, arrows, `Input:`/`Output:`/
  `Example`/`Explanation` headers, the function name) open a span that continues over data-like lines
  until a blank line or a sentence. Example outputs are never added as constants. Known gap: the number
  regex skips a number directly followed by `.` (e.g. "base 7."). `BASE_CONSTS` is at line 428 and `seed()` at line 432.
- **Probes:** `mutate()` (line 440) perturbs the example inputs deterministically.
  `Synth.probes()` (line 471) returns up to 6 probes. Probes have no known outputs; they are used
  only to penalize candidates that crash or return the wrong type.
- **`Synth.run()`** (line 484):
  1. Seed the arguments and constants, then grow costs 2–3 (phase 1).
  2. Choose iteration *sources*: list or str terms of cost ≤ 4, at most 12.
  3. `build_hof()` (line 565): for each source and each mode `x` / `ix` (enumerate) / `pair`
     (i<j combinations) / `adj` (`zip(L, L[1:])`), run a separate per-element `Bank` for the
     comprehension body. It then emits `[body for ...]`, `[x for ... if pred]`,
     `[body for ... if pred]`, and `sorted`/`max`/`min(..., key=lambda x: body)` as top-level terms.
  4. Keep growing until the process CPU-time deadline. The search stops 2 cost levels after the first clean candidate
     (one that uses all args and doesn't crash).
- **Ranking:** `rank_key()` (line 554) is
  `(not uses_all_args, crashes_on_probes>0, wrong_type_on_probes>0, cost + 2*crash + 2*badtype, len)`.
- **Final choice:** `choose()` (line 684) picks the retrieval candidate (crash-free, highest
  similarity) unless the enumerative answer is clean and simple (`cost <= SIMPLE=6`) and retrieval
  similarity is `< STRONG=0.45`.
- **`synthesize()`** (line 702) runs retrieval first, then enumeration, then `choose()`. Output is
  `LIB_SRC + [MBPP function source] + prompt + "    return <expr>\n"`.
- The ops in `banned` (the entry name plus prompt helpers) are removed, so `lib.py` helpers named
  `is_palindrome` or `digits` don't shadow the target function and recurse.

### 4.3 `nonet/prior.py` and `train_prior.py`: the learned prior
- `train_prior.py:ops_of(code)` (line 29) maps an MBPP solution's AST to the same op names used in
  `synth.py` (`len`, `add`, `M_x` = comprehension/loop, `M_filter`, `M_key`, `M_pair`, `M_ix`,
  `M_adj`, `ite`, `even`, ...). `main()` writes the counts to `data/prior.json` as
  `{"n", "base": {op: P(op)}, "cond": {word: {op: count}}, "wc": {word: count}}`.
- `prior.prob(op, words, kw_hit)` (line 24) returns the max over docstring words of the
  smoothed P(op | word) (with `ALPHA = 4`), with a floor at the base rate. A hand-written keyword
  hit raises it to 0.5. `prior.cost()` (line 40) returns `round(-log2 p)`, clamped to [1, 7].
- After editing op names in `synth.py` or `train_prior.py`, rerun `python3 train_prior.py`.

### 4.4 `nonet/retrieve.py` and `nonet/sandbox_runner.py`: library retrieval
- `load_library()` (line 23) parses MBPP **without executing it**. It finds each function's name
  from `test_list[0]`, renames it `lib_<name>` to avoid clashes, and records its arity.
- `retrieve()` (line 63) takes TF-IDF cosine similarity (`similarity`, line 54) between the
  docstring's prose words plus the split entry name, and each MBPP description. It sends the
  top 150 (sim > 0.05) to the sandbox in a single subprocess call.
- `sandbox_runner.py` reads a `repr()` job from stdin and `ast.literal_eval`s it (JSON would turn
  tuples into lists). For each candidate it tries every argument permutation on the examples and
  counts crashes on probes. It limits each call to 0.3 s CPU (`ITIMER_PROF`), the whole process to
  20 s of CPU (`RLIMIT_CPU`), and sends stdout to /dev/null. The caller applies a 120 s wall watchdog.
  Enumeration's CPU budget excludes this subprocess. It's **stdlib only** and must never
  import project code.
- Retrieval contributed 3 dev passes and 5 test passes.

### 4.5 `nonet/lib.py`: concept library
`LIB_SRC` is a source string that is both `exec`'d by `synth.py` (to evaluate ops) and prepended
to every generated program. Its helpers: `is_prime`, `digits`, `fib`, `factorial`, `divisors`,
`prime_factors`, `is_palindrome`, `running` (accumulate), `safe_range`. Each guards against huge
inputs. Keep it generic.

## 5. Commands

```bash
cd ~/nonet-coder
python3 lint_rules.py                                   # static checks only; must print "lint OK"
python3 -m unittest discover tests                      # regression tests (stdlib only)
python3 train_prior.py                                  # rebuild data/prior.json
python3 run_eval.py --split dev --budget 10 --workers 4 --out results_dev_vN.json
python3 run_eval.py --only 12,51 --trace-candidates --out dev_trace.json  # dev tasks only
python3 run_eval.py --only 0,12,51 --budget 10          # specific problem ids
.venv-llm/bin/python baseline_llm.py HuggingFaceTB/SmolLM2-360M   # about 40 min on CPU/MPS
```
- `run_eval.py` defaults to one worker; `--workers` controls parallel problems. `--budget` is the
  enumeration process CPU time in seconds, excluding retrieval and scoring. CPU timing reduces
  scheduler sensitivity but does not guarantee repeatability; wall watchdogs, hardware, CPU
  frequency, and decisions close to deadlines can still change results. Output is
  one line per problem, `PASS|WRONG|----  id  entry  how  expr`, ending with a summary line
  giving pass@1, found-consistent and overfit counts.
- Use the global `python3` (3.10) for `nonet/`, which needs only the stdlib. The global numpy and
  transformers are binary-incompatible, so the LLM baseline **must** use `.venv-llm`.

## 6. Gotchas learned the hard way

- **Timers:** `run_eval.py` sets the hard per-problem timeout with `ITIMER_VIRTUAL`/`SIGVTALRM`,
  because `ITIMER_REAL`/`SIGALRM` is shared with `signal.alarm` and anything else that uses it.
  Don't switch it back.
- **Baseline prompt:** HumanEval prompts must be `.rstrip()`'d before generation (as
  bigcode-evaluation-harness does). Without it, SmolLM2 scores 0/164 because it starts a new `def`.
- **Overfitting is the main failure mode.** Two or three examples admit many spurious programs,
  for example `txt == 'apple pi e'` or `arr[3] - arr.count(2)`. The existing defenses are prose-only
  constants, type discipline, uses-all-args, and probe crash/type penalties. A lower cost does not
  mean the program is more likely to be correct.
- The search's code strings are formatted, not built as ASTs. Every new op `code_format` must be
  safe under `render()`'s parenthesization rules (a placeholder directly inside `(`/`, `...`)`/`,` is
  left raw; anywhere else it is wrapped in parentheses).
- Value equality is strict (`same()`, line 262): bool ≠ int, list ≠ tuple, and floats compare
  within 1e-6.

## 7. Known issues and next tasks, by priority

1. **Retrieval adaptation (highest expected gain).** Allow `g(lib_f(h(args)))`, where `g` and `h`
   are small searched transforms such as `len`, `sorted`, `'YES' if _ else 'NO'`, or argument
   slicing. Today only exact MBPP matches count.
2. **Overfitting defenses.** Check probe outputs against constraints stated in the docstring
   (return annotation such as `-> bool` or `List[int]`, "sorted", "non-negative") and penalize
   violations in `rank_key`.
3. ~~**The KeyError.**~~ Fixed in v5: `prior.prob` assumed every `wc` word had a `cond` entry
   (5 words don't). Reproduced on dev 101/152; regression test in `tests/`.
4. **Timeouts (25 on test).** Profile `Bank.grow` and try `--budget 30`. Speed ideas: index terms by
   tag, and cache `repr` keys.
5. **No-example problems (9 on test).** Add more extraction formats, or fall back to retrieval
   ranked by similarity alone.
6. Clean up the repo: `nonet/__pycache__/*.pyc` and the `*.log` files were committed before
   `.gitignore` existed. Remove them with `git rm --cached`.

Always report dev numbers next to the previous version's, and keep `lint_rules.py` passing.
