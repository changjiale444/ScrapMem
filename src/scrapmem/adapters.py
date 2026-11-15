"""ATM-Bench corpus import and page-level evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import re

from .types import Record


def read_rows(path) -> list[dict]:
    source = Path(path)
    raw = (
        [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if source.suffix == ".jsonl"
        else json.loads(source.read_text(encoding="utf-8"))
    )
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        for key in ("results", "items", "data", "emails", "questions", "qa_pairs"):
            if isinstance(raw.get(key), list):
                rows = raw[key]
                break
        else:
            raise ValueError("Expected a list or a results/items/data/emails list")
    else:
        raise ValueError("Expected JSON records")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Every record must be an object")
    return rows


def resolve_media(raw: str, root: Path | None) -> Path:
    path = Path(raw)
    candidates = [path] if path.is_absolute() else []
    if root:
        candidates.extend(
            [root / raw, root / path.name]
            if not path.is_absolute()
            else [root / path.name]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Media path is unavailable; supply the original media root")


def atm_records(
    *,
    email_file=None,
    image_file=None,
    video_file=None,
    image_root=None,
    video_root=None,
) -> list[Record]:
    records = []
    if email_file:
        for row in read_rows(email_file):
            text = "\n".join(
                str(row[k])
                for k in ("subject", "short_summary", "detail")
                if row.get(k)
            )
            records.append(Record(str(row["id"]), row["timestamp"], "email", text))
    for modality, source, root in [
        ("image", image_file, image_root),
        ("video", video_file, video_root),
    ]:
        if not source:
            continue
        for row in read_rows(source):
            raw = row[modality + "_path"]
            timestamp = row.get("timestamp")
            if not timestamp:
                match = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", Path(raw).stem)
                if not match:
                    raise ValueError("Media record has no timestamp/date in filename")
                timestamp = "-".join(match.groups())
            text = "\n".join(
                str(row[k])
                for k in ("short_caption", "caption", "ocr_text", "location_name")
                if row.get(k)
            )
            records.append(
                Record(
                    Path(raw).stem,
                    timestamp,
                    modality,
                    text,
                    str(resolve_media(raw, Path(root) if root else None)),
                )
            )
    if not records:
        raise ValueError("No memory records were provided")
    return records


def evaluate_pages(predictions: list[dict], gold: list[dict], k=10) -> dict:
    """Compute page-level macro recall from predictions and gold page IDs."""
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    lookup = {str(p["id"]): p for p in predictions}
    ids = [str(row["id"]) for row in gold]
    if not ids or len(set(ids)) != len(ids) or len(lookup) != len(predictions):
        raise ValueError("Evaluation ids must be nonempty and unique")
    if set(lookup) != set(ids):
        raise ValueError("Prediction and gold ids must match exactly")
    scores = []
    for row in gold:
        gold_ids = row["gold_page_ids"]
        predicted_ids = lookup[str(row["id"])]["retrieved_page_ids"]
        if any(
            not isinstance(values, list) or any(not isinstance(x, str) for x in values)
            for values in (gold_ids, predicted_ids)
        ):
            raise ValueError("Page ids must be lists of strings")
        targets = set(gold_ids)
        if not targets:
            raise ValueError("Each gold item requires at least one page id")
        ranked = list(dict.fromkeys(predicted_ids))[:k]
        scores.append(len(targets & set(ranked)) / len(targets))
    return {f"page_macro_recall@{k}": sum(scores) / len(scores), "queries": len(scores)}
