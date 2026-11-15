#!/usr/bin/env python3
"""Run an external baseline under the common top-k protocol."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scrapmem.baselines import run_external


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--command-json")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    command = (
        json.loads(Path(args.command_json).read_text())
        if args.command_json
        else None
    )
    print(
        run_external(
            args.name,
            args.upstream,
            command=command,
            data_root=args.data_root,
            k=args.k,
        )
    )


if __name__ == "__main__":
    main()
