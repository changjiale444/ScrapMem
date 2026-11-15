"""Common top-k evaluation contract for external baselines and evidence bounds."""

from dataclasses import asdict
from pathlib import Path
import json
import os
import subprocess

from .evaluation import evaluate
from .storage import atomic_json

UPSTREAM_SCRIPTS = {
    "A-Mem": "scripts/QA_Agent/A-Mem/run.sh",
    "Mem0": "scripts/QA_Agent/mem0/run.sh",
    "HippoRAG2": "scripts/QA_Agent/HippoRag2/run.sh",
    "ATM-RAG": "scripts/QA_Agent/MMRAG/run.sh",
    "Oracle": "scripts/QA_Agent/Oracle/run_oracle_qwen3vl8b_raw.sh",
    "No-Evidence": "scripts/QA_Agent/Oracle/run_oracle_no_evidence_qwen3vl8b.sh",
}


def run_external(name, upstream, *, command=None, data_root=None, k=10):
    root = Path(upstream).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if command is None:
        if name not in UPSTREAM_SCRIPTS:
            raise ValueError(
                "Supply a command JSON list for this official baseline implementation"
            )
        script = root / UPSTREAM_SCRIPTS[name]
        if not script.is_file():
            raise FileNotFoundError(script)
        command = ["bash", str(script)]
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(x, str) for x in command)
    ):
        raise ValueError("Baseline command must be a nonempty JSON string list")
    env = dict(os.environ, RETRIEVAL_TOP_K=str(k), TOP_K=str(k))
    if data_root:
        env["RAW_BASE"] = str(Path(data_root).resolve())
    subprocess.run(command, cwd=root, env=env, check=True)
    return {"baseline": name, "status": "completed", "k": k}


def evaluate_baseline(predictions, gold, *, item_days, k=10, judge=None):
    for row in predictions:
        if len(row["retrieval_ids"]) > k:
            raise ValueError("Baseline exceeded the common top-k evidence budget")
    return evaluate(predictions, gold, item_days=item_days, k=k, judge=judge)


def bound_predictions(memory, questions, *, oracle=False, k=10):
    output = []
    for row in questions:
        ids = (
            list(dict.fromkeys(str(i) for i in row["evidence_ids"]))[:k]
            if oracle
            else []
        )
        evidence = []
        for iid in ids:
            record = memory.items[iid]
            day = record["timestamp"][:10]
            evidence.append(dict(record, item_id=iid, day_id=day, page_id=day))
        output.append(
            {
                "id": str(row["id"]),
                "prediction": memory.backend.answer(row["question"], evidence),
                "retrieval_ids": ids,
                "stage_a_day_ids": list(dict.fromkeys(e["day_id"] for e in evidence)),
            }
        )
    return output


def merge_outputs(answers, retrieval_details):
    answer_map = {str(r["id"]): r for r in answers}
    retrieval_map = {str(r["id"]): r for r in retrieval_details}
    if (
        len(answer_map) != len(answers)
        or len(retrieval_map) != len(retrieval_details)
        or set(answer_map) != set(retrieval_map)
    ):
        raise ValueError("Answer/retrieval output IDs must match uniquely")
    output = []
    for iid, row in answer_map.items():
        details = retrieval_map[iid]
        output.append(
            dict(
                row,
                prediction=row.get("prediction", row.get("answer")),
                retrieval_ids=details["retrieval_ids"],
                **(
                    {"stage_a_day_ids": details["stage_a_day_ids"]}
                    if "stage_a_day_ids" in details
                    else {}
                )
            )
        )
    return output
