#!/usr/bin/env python3
"""Run the controlled experiments (paper Appendix B)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scrapmem.adapters import read_rows
from scrapmem.backends import Endpoint
from scrapmem.cli import build_memory
from scrapmem.controlled import ControlledJudge, run_controlled
from scrapmem.engine import ScrapMem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--variant", choices=["full", "no-forget", "no-em-graph"], default="full"
    )
    parser.add_argument("--policy", default="timed_gentle")
    parser.add_argument("--as-of")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--judge-config")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    shared = {}

    def factory(path, use_graph):
        if not shared:
            memory = build_memory(path, args.config)
            shared.update(encoder=memory.encoder, backend=memory.backend)
        options = dict(cfg.get("memory", {}), use_graph=use_graph)
        return ScrapMem(path, shared["encoder"], shared["backend"], **options)

    judge = ControlledJudge(
        Endpoint(**json.loads(Path(args.judge_config).read_text()))
        if args.judge_config
        else None
    )
    print(
        run_controlled(
            read_rows(args.records),
            read_rows(args.questions),
            args.output,
            factory,
            judge,
            variant=args.variant,
            policy=args.policy,
            as_of=args.as_of,
            k=args.k,
        )
    )


if __name__ == "__main__":
    main()
