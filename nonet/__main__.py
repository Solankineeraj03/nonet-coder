"""Generate a function body from a Python prompt, without executing the result."""
import argparse
import ast
import sys
from pathlib import Path

from .synth import synthesize


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate Python from a function signature and docstring.")
    parser.add_argument("prompt", nargs="?", help="prompt file; reads standard input if omitted")
    parser.add_argument("--entry", help="target function name; inferred when the prompt has one function")
    parser.add_argument("--budget", type=float, default=10.0, help="enumeration CPU seconds (default: 10)")
    parser.add_argument("--no-retrieval", action="store_true", help="disable retrieval from the code library")
    parser.add_argument("-o", "--output", help="write generated Python to this file instead of stdout")
    args = parser.parse_args(argv)
    if not 0 < args.budget < float("inf"):
        parser.error("--budget must be a positive finite number")
    try:
        prompt = Path(args.prompt).read_text() if args.prompt else sys.stdin.read()
        prompt = prompt.rstrip() + "\n"
        tree = ast.parse(prompt)
        functions = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        if args.entry:
            if args.entry not in functions:
                parser.error("--entry must name a top-level function in the prompt")
            entry = args.entry
        elif len(functions) == 1:
            entry = functions[0]
        else:
            parser.error("provide --entry when the prompt does not contain exactly one top-level function")
        if not tree.body or not isinstance(tree.body[-1], ast.FunctionDef) or tree.body[-1].name != entry:
            parser.error("the target function must be the last top-level statement in the prompt")
        program, code, how, examples = synthesize(prompt, entry, args.budget, not args.no_retrieval)
        if code is None:
            print(f"No program found ({how}; {examples} extracted examples).", file=sys.stderr)
            return 1
        if args.output:
            Path(args.output).write_text(program)
        else:
            sys.stdout.write(program)
        print(f"Generated using {how}; matched {examples} extracted examples. Review and test the result.",
              file=sys.stderr)
        return 0
    except (OSError, SyntaxError, ValueError) as exc:
        print(f"Cannot process prompt: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
