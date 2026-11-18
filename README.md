# NoNet Coder

**Program synthesis without neural networks.** NoNet Coder writes the body of a Python function
from its signature and docstring using classical search, counting statistics and retrieval: no
weights, no embeddings, no model calls, and only the Python standard library.

On HumanEval it **outperforms a 360M-parameter language model (SmolLM2-360M)**:
**17.1% vs 9.8% pass@1** on a held-out split, running on a single CPU core per problem.

## Highlights

- **Enumerative program search:** bottom-up, cost-ordered, type-directed construction of Python
  expressions, with observational-equivalence pruning (programs that behave identically on all inputs
  are merged) and synthesized comprehensions, filters and `key=` lambdas.
- **Learned cost prior:** docstring-word / code-component co-occurrence counts from 974 MBPP
  solutions decide which components are cheap to try for a given docstring.
- **Retrieval:** TF-IDF ranking of the MBPP library; candidates are verified on the docstring's
  examples in a separate, resource-limited Python process.
- **Overfitting defenses:** probe inputs penalize candidates that crash or change type, constants come
  only from the docstring's prose (never from example answers), and candidates that use every
  argument are ranked first.
- **Evaluation discipline:** fixed dev/test split with a single held-out evaluation, static and
  regression checks against answer leakage, and a CPU-time search budget that reproduced identical
  results across repeated runs.

## How it works

```
prompt ──► extract signature, docstring, I/O examples
             │
             ├─► retrieval: TF-IDF over MBPP ──► verify on examples (subprocess)
             │
             └─► enumerative search, guided by the prior ──► candidates matching all examples
                                                              │
                         rank (probe crashes, types, cost) ◄──┘
                                       │
                                       ▼
                     choose retrieval vs. search ──► generated function
```

## Results (HumanEval, greedy pass@1, hidden tests)

| System | Dev (82) | Test (82, held out) | All 164 |
|---|---|---|---|
| SmolLM2-360M (base, bigcode-style prompt) | 13 (15.9%) | 8 (9.8%) | 21 (12.8%) |
| **NoNet Coder** (checkpoint 1) | 19 (23.2%) | **14 (17.1%)** | **33 (20.1%)** |
| NoNet Coder (current, CPU-time budget) | 20 (24.4%) | not re-run | – |

The test split was evaluated once, after development on dev only; the current version has not been
re-evaluated on it. The current dev score was identical across three clean runs. Of the 62 unsolved
dev problems, 27 produce a program that matches the docstring's examples but fails the hidden tests,
32 find no matching program within the budget, and 3 have no parseable examples.

Large LLMs solve most of HumanEval; the point here is how far search plus simple statistics can go
against a small neural model, not to compete with them.

## Data

- [HumanEval](https://github.com/openai/human-eval) (OpenAI, MIT License): evaluation problems in
  `data/HumanEval.jsonl`. Only prompts are given to the synthesizer; hidden tests are used for scoring.
- [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) (Google Research,
  CC BY 4.0): training data for the prior and the retrieval library, in `data/mbpp.jsonl`.

## Rules
- Synthesizer sees only the prompt; never `canonical_solution` or `test`.
- No weights, embeddings, or model calls. `lint_rules.py` checks for HumanEval names and NN/network deps.
- During synthesis and evaluation, library (MBPP) code runs through a separate `python -I` process
  with CPU and wall-clock limits.

## Generate a function

Save a Python function signature and docstring with input/output examples in `prompt.py`:

```python
def double(x):
    """Return twice x.

    >>> double(2)
    4
    >>> double(3)
    6
    """
```

Then run:

```bash
python3 -m nonet prompt.py --budget 10 -o generated.py
```

Omit `prompt.py` to read from standard input, or omit `-o` to write the generated program to
standard output. Use `--entry NAME` when the prompt contains multiple top-level functions.
The target function must be the last top-level statement so the generated body can be appended to it.
`--budget` sets the enumeration CPU budget in seconds; retrieval has a separate limit. Run
`python3 -m nonet --help` for options. If no candidate is found, the command exits with an error.

A candidate matches the examples the parser extracted; it may still be wrong. Review and test it
before use. Retrieval executes third-party library code in a separate resource-limited Python
process, but this is **not an OS security sandbox** and does not block filesystem or network access.
Use `--no-retrieval` to disable it.
If retrieval is used, the generated file can contain third-party MBPP code. Running that file later
executes the code with the privileges and resource limits of your own Python process; the retrieval
runner's limits do not follow the generated file. Inspect it before running it.

## Development

```bash
python3 lint_rules.py
python3 -m unittest discover -s tests
python3 train_prior.py                     # counts from MBPP -> data/prior.json
python3 run_eval.py --split dev --workers 4 --budget 10 --out results_dev.json
.venv-llm/bin/python baseline_llm.py       # LLM baseline
```

The held-out test split has already been evaluated once; see `AGENTS.md` before running it again.
To inspect search alternatives on dev tasks, add `--trace-candidates` with `--out` to the
evaluation command. Each result then includes the example-matching enumerated candidates, their
costs, probe failures, and rank keys, plus matching retrieval candidates. By default the log keeps
the best 1,000 enumerated candidates per task; `--candidate-limit 0` records all of them. Tracing
is restricted to dev tasks and can add time and produce large files.
