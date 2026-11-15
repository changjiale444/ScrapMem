"""ScrapMem ingestion, two-stage retrieval, benchmark experiments, and evaluation."""

import argparse
import json
from pathlib import Path
import sys

from .adapters import atm_records, evaluate_pages, read_rows
from .backends import Endpoint, SentenceEncoder, VLMBackend, ChatClient
from .engine import ScrapMem
from .evaluation import OpenEndedJudge, evaluate
from .experiments import batch, run_atm
from .forgetting import SCHEDULES
from .measurement import measure_storage
from .storage import atomic_json


def build_memory(store, config_path):
    config = json.loads(Path(config_path).read_text())
    encoder = SentenceEncoder(**config.get("embedding", {}))
    mode = config.get("backend", "native")
    if mode == "native":
        from .qwen import NativeQwenBackend

        backend = NativeQwenBackend(config["perception"], config.get("answer"))
    elif mode == "controlled-http":
        from .backends import ControlledBackend

        backend = ControlledBackend(
            Endpoint(**config["perception"]), Endpoint(**config["answer"])
        )
    elif mode == "http":
        backend = VLMBackend(
            Endpoint(**config["perception"]),
            answer=Endpoint(**config["answer"]) if config.get("answer") else None,
        )
    else:
        raise ValueError("backend must be native, http, or controlled-http")
    return ScrapMem(store, encoder, backend, **config.get("memory", {}))


def judge_for(path):
    return (
        OpenEndedJudge(Endpoint(**json.loads(Path(path).read_text())))
        if path
        else OpenEndedJudge()
    )


def write_new(path, data):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Choose a new output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, data)


def parser():
    p = argparse.ArgumentParser(
        description="ScrapMem: daily graph retrieval → fine-grained rerank → evidence-grounded QA"
    )
    subs = p.add_subparsers(dest="command", required=True)
    for name in [
        "ingest",
        "import-atm",
        "retrieve",
        "retrieve-paths",
        "answer",
        "forget",
        "batch",
        "run-atm",
    ]:
        s = subs.add_parser(name)
        s.add_argument("--store", type=Path, required=True)
        s.add_argument("--config", type=Path, required=True)
        if name == "ingest":
            s.add_argument("--records", type=Path, required=True)
        if name == "import-atm":
            for option in [
                "email-file",
                "image-file",
                "video-file",
                "image-root",
                "video-root",
            ]:
                s.add_argument("--" + option, type=Path)
        if name in {"retrieve", "answer", "batch", "run-atm"}:
            s.add_argument("--k", type=int, default=10)
        if name in {"retrieve", "retrieve-paths", "answer"}:
            s.add_argument("--question", required=True)
        if name in {"forget", "run-atm"}:
            s.add_argument("--as-of")
            s.add_argument(
                "--policy",
                choices=list(SCHEDULES),
                default="timed_gentle" if name == "forget" else "no_forget",
            )
        if name == "batch":
            s.add_argument("--questions", type=Path, required=True)
            s.add_argument("--output", type=Path, required=True)
            s.add_argument("--generate-answers", action="store_true")
        if name == "run-atm":
            s.add_argument("--data-root", type=Path, required=True)
            s.add_argument("--qa-file", type=Path)
            s.add_argument("--output", type=Path, required=True)
            s.add_argument("--judge-config", type=Path)
    for name in ["evaluate", "evaluate-pages", "controlled-evaluate"]:
        s = subs.add_parser(name)
        s.add_argument("--predictions", type=Path, required=True)
        s.add_argument("--gold", type=Path, required=True)
        s.add_argument("--output", type=Path)
        if name != "controlled-evaluate":
            s.add_argument("--k", type=int, default=10)
        if name == "evaluate":
            s.add_argument("--item-days", type=Path, required=True)
            s.add_argument("--judge-config", type=Path)
            s.add_argument("--retrieval-details", type=Path)
        if name == "controlled-evaluate":
            s.add_argument("--judge-config", type=Path)
    s = subs.add_parser("measure-storage")
    s.add_argument("--store", type=Path, required=True)
    s.add_argument("--raw-corpus", type=Path)
    s.add_argument("--output", type=Path)
    s = subs.add_parser("run-baseline")
    s.add_argument("--name", required=True)
    s.add_argument("--upstream", type=Path, required=True)
    s.add_argument("--command-json", type=Path)
    s.add_argument("--data-root", type=Path)
    s.add_argument("--k", type=int, default=10)
    s = subs.add_parser("controlled-generate")
    s.add_argument("--config", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s = subs.add_parser("controlled-run")
    s.add_argument("--config", type=Path, required=True)
    s.add_argument("--records", type=Path, required=True)
    s.add_argument("--questions", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--as-of")
    s.add_argument("--policy", choices=list(SCHEDULES), default="timed_gentle")
    s.add_argument(
        "--variant", choices=["full", "no-forget", "no-em-graph"], default="full"
    )
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--judge-config", type=Path)
    s = subs.add_parser("fidelity")
    s.add_argument("--config", type=Path, required=True)
    s.add_argument("--store", type=Path, required=True)
    s.add_argument("--annotations", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    return p


def execute(args):
    cmd = args.command
    if cmd in {"evaluate", "evaluate-pages", "controlled-evaluate"}:
        predictions, gold = read_rows(args.predictions), read_rows(args.gold)
        if cmd == "evaluate-pages":
            result = evaluate_pages(predictions, gold, args.k)
        elif cmd == "evaluate":
            if args.retrieval_details:
                from .baselines import merge_outputs

                predictions = merge_outputs(
                    predictions, read_rows(args.retrieval_details)
                )
            result = evaluate(
                predictions,
                gold,
                item_days=json.loads(args.item_days.read_text()),
                k=args.k,
                judge=judge_for(args.judge_config),
            )
        else:
            from .controlled import ControlledJudge, evaluate_controlled

            endpoint = (
                Endpoint(**json.loads(args.judge_config.read_text()))
                if args.judge_config
                else None
            )
            result = evaluate_controlled(predictions, gold, ControlledJudge(endpoint))
        if args.output:
            write_new(args.output, result)
        return result
    if cmd == "measure-storage":
        result = measure_storage(args.store, args.raw_corpus)
        if args.output:
            write_new(args.output, result)
        return result
    if cmd == "run-baseline":
        from .baselines import run_external

        return run_external(
            args.name,
            args.upstream,
            command=(
                json.loads(args.command_json.read_text()) if args.command_json else None
            ),
            data_root=args.data_root,
            k=args.k,
        )
    if cmd == "controlled-generate":
        from .controlled import generate_dataset, QwenImageGenerator

        cfg = json.loads(args.config.read_text())
        return generate_dataset(
            args.output,
            ChatClient(Endpoint(**cfg["generation"])),
            QwenImageGenerator(**cfg["image_generation"]),
            **cfg.get("dataset", {}),
        )
    if cmd == "controlled-run":
        from .controlled import run_controlled, ControlledJudge

        cfg = json.loads(args.config.read_text())
        shared = {}

        def factory(path, use_graph):
            if not shared:
                memory = build_memory(path, args.config)
                shared.update(encoder=memory.encoder, backend=memory.backend)
            options = dict(cfg.get("memory", {}), use_graph=use_graph)
            return ScrapMem(path, shared["encoder"], shared["backend"], **options)

        judge = ControlledJudge(
            Endpoint(**json.loads(args.judge_config.read_text()))
            if args.judge_config
            else None
        )
        return run_controlled(
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
    memory = build_memory(args.store, args.config)
    if cmd == "ingest":
        from .types import Record

        records = []
        for row in read_rows(args.records):
            row = dict(row)
            row.pop("user_id", None)
            if row.get("media_path") and not Path(row["media_path"]).is_absolute():
                row["media_path"] = str(args.records.parent / row["media_path"])
            records.append(Record(**row))
        return memory.ingest(records)
    if cmd == "import-atm":
        return memory.ingest(
            atm_records(
                **{
                    k: getattr(args, k)
                    for k in [
                        "email_file",
                        "image_file",
                        "video_file",
                        "image_root",
                        "video_root",
                    ]
                }
            )
        )
    if cmd in {"retrieve", "answer"}:
        return getattr(memory, cmd)(args.question, args.k)
    if cmd == "retrieve-paths":
        return memory.retrieve_paths(args.question)
    if cmd == "forget":
        if not args.as_of:
            raise ValueError("--as-of is required")
        return memory.forget(args.as_of, args.policy)
    if cmd == "batch":
        if args.output.exists():
            raise FileExistsError("Choose a new output path")
        predictions = batch(
            memory, read_rows(args.questions), args.k, args.generate_answers
        )
        write_new(args.output, predictions)
        write_new(
            args.output.with_name(args.output.stem + "-item-days.json"),
            {i: r["timestamp"][:10] for i, r in memory.items.items()},
        )
        return {
            "queries": len(predictions),
            "output": str(args.output),
            "retrieval_unit": "record ID",
            "k": args.k,
        }
    if cmd == "run-atm":
        return run_atm(
            memory,
            args.data_root,
            args.output,
            qa_file=args.qa_file,
            k=args.k,
            policy=args.policy,
            as_of=args.as_of,
            judge=judge_for(args.judge_config),
        )
    from .controlled import fidelity_sweep

    return fidelity_sweep(read_rows(args.annotations), memory.backend, args.output)


def main():
    try:
        result = execute(parser().parse_args())
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"scrapmem: {exc}", file=sys.stderr)
        return 1
    return 0
