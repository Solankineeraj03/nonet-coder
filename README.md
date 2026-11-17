# NoNet Coder

Writes Python from a docstring with **no neural network**: enumerative program search +
an MBPP-counted prior + TF-IDF retrieval from a code library, verified on the docstring's examples.
One CPU core, ~10-18 s per problem, ~300 KB of code + data.

## Results (HumanEval, greedy pass@1, hidden tests)

| System | Dev (82) | Test (82, held out) | All 164 |
|---|---|---|---|
| SmolLM2-360M (base, bigcode-style prompt) | 13 (15.9%) | 8 (9.8%) | 21 (12.8%) |
| **NoNet Coder** (checkpoint 1) | 19 (23.2%) | **14 (17.1%)** | **33 (20.1%)** |

Test split was evaluated once, after development on dev only.

## Rules
- Synthesizer sees only the prompt; never `canonical_solution` or `test`.
- No weights, embeddings, or model calls. `lint_rules.py` checks for HumanEval names and NN/network deps.
- Library (MBPP) code is executed only in an isolated `python -I` subprocess with CPU/time limits.

## Run
    python3 train_prior.py                     # counts from MBPP -> data/prior.json
    python3 run_eval.py --split dev            # or --split test / all
    .venv-llm/bin/python baseline_llm.py       # LLM baseline
