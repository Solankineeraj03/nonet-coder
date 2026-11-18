# NoNet Coder

Writes Python from a docstring with **no neural network**: enumerative program search +
an MBPP-counted prior + TF-IDF retrieval from a code library, verified on the docstring's examples.
The synthesizer uses one CPU core per problem; search time depends on the CPU budget.

## Results (HumanEval, greedy pass@1, hidden tests)

| System | Dev (82) | Test (82, held out) | All 164 |
|---|---|---|---|
| SmolLM2-360M (base, bigcode-style prompt) | 13 (15.9%) | 8 (9.8%) | 21 (12.8%) |
| **NoNet Coder** (checkpoint 1) | 19 (23.2%) | **14 (17.1%)** | **33 (20.1%)** |

Test split was evaluated once, after development on dev only.

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
