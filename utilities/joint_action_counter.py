#!/usr/bin/env python3
"""
Compute the average rate of joint actions with exactly one non-nop action.

Definition:
- A "joint action line" is any line whose (stripped) text starts with "(operators:".
- A line "has exactly one non-nop action" if after "(operators:" there are parentheses-delimited
  actions and exactly one of them is not a nop. We treat "(nop)", "(nop )", and whitespace/case
  variants as nop.

Output:
- Prints a single line with the overall average across all files:
    average = total_lines_with_exactly_one_non_nop / total_joint_action_lines
- Use --verbose to also see per-file counts.
"""

import argparse
import re
from pathlib import Path

# Regex to capture every (...) action after "(operators:"
PARENS_RE = re.compile(r"\([^\)]*\)")


def is_nop_token(tok: str) -> bool:
    # Normalize: remove outer parens, collapse spaces, lowercase
    inner = tok.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    # Collapse internal whitespace
    inner = " ".join(inner.split()).lower()
    # Accept common nop spellings
    return inner in {"nop", "nop)", "(nop", "no-op", "noop"} or inner.startswith("nop ")


def analyze_file(path: Path):
    joint_lines = 0
    exactly_one_non_nop = 0

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith("(operators:"):
                continue
            joint_lines += 1

            # Get all (...) groups on the line (these are the per-agent actions)
            tokens = PARENS_RE.findall(line)
            # If the line includes the opening "(operators:" itself matching as a token,
            # filter it out by checking the inner text doesn't start with "operators:"
            filtered = []
            for t in tokens:
                inner = t[1:-1].strip().lower()
                if inner.startswith("operators:"):
                    # Remove "operators:" prefix if someone wrote "(operators:(a) (b) ...)"
                    # and recurse into any (...) after it
                    rest = line[line.lower().find("operators:") + len("operators:") :]
                    filtered = PARENS_RE.findall(rest)
                    break
                else:
                    filtered.append(t)

            if not filtered:
                continue  # nothing to evaluate on this line

            non_nop_count = sum(0 if is_nop_token(t) else 1 for t in filtered)
            if non_nop_count == 1:
                exactly_one_non_nop += 1

    return joint_lines, exactly_one_non_nop


def main():
    ap = argparse.ArgumentParser(description="Average of joint actions with exactly one non-nop.")
    ap.add_argument("directory", type=Path, help="Directory containing the input files")
    ap.add_argument(
        "--pattern",
        default="*",
        help="Glob pattern for files to include (default: '*', e.g., '*.trajectory')",
    )
    ap.add_argument("--verbose", action="store_true", help="Print per-file stats")
    args = ap.parse_args()

    total_joint = 0
    total_exactly_one = 0
    files = sorted(args.directory.glob(args.pattern))

    if not files:
        print("0.0")
        return

    for p in files:
        if p.is_dir():
            continue
        jl, one = analyze_file(p)
        total_joint += jl
        total_exactly_one += one
        if args.verbose:
            avg = (one / jl) if jl else 0.0
            print(f"{p}: joint_lines={jl}, exactly_one_non_nop={one}, fraction={avg:.6f}")

    overall = (total_exactly_one / total_joint) if total_joint else 0.0
    # Final required output:
    print(f"{overall:.6f}")


if __name__ == "__main__":
    main()
