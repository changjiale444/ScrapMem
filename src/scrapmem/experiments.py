"""Run the ATM-Bench import, two-stage retrieval, answering, and scoring pipeline."""

from dataclasses import asdict
from pathlib import Path
import json

from .adapters import atm_records, read_rows
from .evaluation import evaluate
from .measurement import measure_storage
from .storage import atomic_json
from .types import Record


def batch(memory, questions, k=10, generate_answers=True):
    ids = [str(q["id"]) for q in questions]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Question IDs must be nonempty and unique")
    predictions = []
    for row in questions:
        before = (
            memory.backend.token_usage()
            if hasattr(memory.backend, "token_usage")
            else None
        )
        result = (
            memory.answer(row["question"], k)
            if generate_answers
            else {"evidence": memory.retrieve(row["question"], k)}
        )
        evidence = result["evidence"]
        current = memory.backend.token_usage() if before is not None else None
        usage = (
            {
                role: {
                    key: current[role][key] - before[role][key] for key in current[role]
                }
                for role in current
            }
            if before is not None and current is not None
            else None
        )
        predictions.append(
            {
                "id": str(row["id"]),
                "question": row["question"],
                "prediction": result.get("answer"),
                "answer": result.get("answer"),
                "retrieval_ids": [e["item_id"] for e in evidence],
                "stage_a_day_ids": evidence[0]["stage_a_day_ids"] if evidence else [],
                "retrieved_path_ids": [e["path_id"] for e in evidence],
                "retrieved_page_ids": list(
                    dict.fromkeys(e["day_id"] for e in evidence)
                ),
                "item_day_ids": {e["item_id"]: e["day_id"] for e in evidence},
                "retrieval_scores": [e["score"] for e in evidence],
                "usage": usage,
            }
        )
    return predictions


def run_atm(
    memory,
    data_root,
    output,
    *,
    qa_file=None,
    k=10,
    policy="no_forget",
    as_of=None,
    judge=None,
):
    root = Path(data_root)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new output directory")
    qa_file = Path(qa_file) if qa_file else root / "atm-bench" / "atm-bench.json"
    questions = read_rows(qa_file)
    records = atm_records(
        email_file=root / "raw_memory/email/emails.json",
        image_file=root / "processed_memory/image_batch_results.json",
        video_file=root / "processed_memory/video_batch_results.json",
        image_root=root / "raw_memory/image",
        video_root=root / "raw_memory/video",
    )
    valid_ids = {r.id for r in records}
    unknown = {str(i) for q in questions for i in q["evidence_ids"]} - valid_ids
    if unknown:
        raise ValueError(f"QA references {len(unknown)} unknown evidence IDs")
    if policy != "no_forget" and not as_of:
        raise ValueError("Forgetting experiments require --as-of")
    # Commit manageable groups of complete days so long corpus builds can resume.
    import sys
    from collections import defaultdict

    groups = defaultdict(list)
    for record in records:
        groups[record.day].append(record)
    for day, page in memory.pages.items():
        expected = {r.id: asdict(r) for r in groups.get(day, [])}
        if set(expected) != set(page["source_ids"]) or any(
            memory.items[i] != row for i, row in expected.items()
        ):
            raise ValueError("Existing store does not match the input corpus")
    pending = [day for day in sorted(groups) if day not in memory.pages]
    for offset in range(0, len(pending), 30):
        days = pending[offset : offset + 30]
        memory.ingest([r for day in days for r in groups[day]])
        print(f"Ingested {len(memory.pages)}/{len(groups)} days", file=sys.stderr)
    if policy != "no_forget":
        memory.forget(as_of, policy)
    predictions = batch(memory, questions, k, True)
    days = {r.id: r.day for r in records}
    report = evaluate(predictions, questions, item_days=days, k=k, judge=judge)
    output.mkdir(parents=True)
    atomic_json(output / "predictions.json", predictions)
    atomic_json(output / "metrics.json", report)
    atomic_json(output / "item-days.json", days)
    atomic_json(
        output / "storage.json", measure_storage(memory.root, root / "raw_memory")
    )
    atomic_json(
        output / "run.json",
        {
            "k": k,
            "day_top_n": memory.day_top_n,
            "policy": policy,
            "as_of": as_of,
            "records": len(records),
            "questions": len(questions),
            "signature": memory.signature,
        },
    )
    return report
