#!/usr/bin/env python3
"""Run the ATM-Bench experiment (paper Tables 1 and 3)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scrapmem.cli import build_memory, judge_for
from scrapmem.experiments import run_atm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/paper.json")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--policy", default="no_forget")
    parser.add_argument("--as-of")
    parser.add_argument("--judge-config")
    args = parser.parse_args()

    memory = build_memory(args.store, args.config)
    report = run_atm(
        memory,
        args.data_root,
        args.output,
        k=args.k,
        policy=args.policy,
        as_of=args.as_of,
        judge=judge_for(args.judge_config),
    )
    print(report)


if __name__ == "__main__":
    main()
