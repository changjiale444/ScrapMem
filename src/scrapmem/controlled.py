"""Controlled-history construction, four-axis judging, and fidelity measurements."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
import json
import os
import random
import re
import urllib.request

from PIL import Image

from .backends import ChatClient, Endpoint
from .pages import render_day
from .storage import atomic_json
from .types import Record

TASKS = [
    "Physiological Profiling",
    "Causal Reasoning",
    "Socio-psychological Evolution",
    "Preference Adaptation",
]
AXES = ["PI", "TL", "CA", "FI"]


class QwenImageGenerator:
    def __init__(
        self,
        model="qwen-image-max",
        url_env="DASHSCOPE_IMAGE_URL",
        key_env="DASHSCOPE_API_KEY",
    ):
        self.model, self.url_env, self.key_env = model, url_env, key_env

    def __call__(self, prompt, destination, seed):
        url = os.environ.get(self.url_env)
        key = os.environ.get(self.key_env)
        if not url or not key:
            raise RuntimeError(
                "Set the image generation URL and API-key environment variables"
            )
        payload = {
            "model": self.model,
            "input": {"messages": [{"role": "user", "content": [{"text": prompt}]}]},
            "parameters": {
                "size": "1328*1328",
                "watermark": False,
                "prompt_extend": False,
                "seed": seed,
            },
        }
        request = urllib.request.Request(
            url,
            json.dumps(payload).encode(),
            {"Content-Type": "application/json", "Authorization": "Bearer " + key},
        )
        with urllib.request.urlopen(request, timeout=240) as response:
            result = json.load(response)
        content = result["output"]["choices"][0]["message"]["content"]
        image_url = next(part["image"] for part in content if "image" in part)
        with urllib.request.urlopen(image_url, timeout=120) as response:
            data = response.read()
        destination.write_bytes(data)
        with Image.open(destination) as image:
            image.verify()


def generate_dataset(
    output,
    generator,
    image_generator,
    *,
    users=10,
    events_per_user=110,
    queries_per_task=10,
    seed=7,
):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new controlled-data directory")
    if not 1 <= events_per_user <= 184 or users < 1 or queries_per_task < 1:
        raise ValueError("Invalid controlled dataset dimensions")
    output.mkdir(parents=True)
    (output / "media").mkdir()
    rng = random.Random(seed)
    records = []
    queries = []
    profiles = []
    annotations = []
    for u in range(users):
        user = f"user-{u+1:02d}"
        profile = generator.json(
            [
                {
                    "role": "system",
                    "content": "Create a fictional health/lifestyle profile, age 25--45, with occupation, chronic conditions, "
                    "family context, positive/negative preferences and five Big Five traits at Low/Medium/High. "
                    "Return one JSON object including an integer age.",
                },
                {
                    "role": "user",
                    "content": f"Fictional identifier: {user}; seed: {seed+u}",
                },
            ],
            max_tokens=2048,
        )
        if type(profile.get("age")) is not int or not 25 <= profile["age"] <= 45:
            raise ValueError("Invalid profile age")
        profiles.append(dict(user_id=user, profile=profile))
        timeline = []
        days = sorted(rng.sample(range(184), events_per_user))
        for start in range(0, len(days), 10):
            dates = [
                (date(2025, 6, 1) + timedelta(days=d)).isoformat()
                for d in days[start : start + 10]
            ]
            payload = generator.json(
                [
                    {
                        "role": "system",
                        "content": "Generate coherent fictional longitudinal health/lifestyle events for exactly the supplied dates. "
                        "Include state changes, preferences, causal dependencies and obsolete/current facts. "
                        'Return {"events":[{"date":string,"text":string,"kind":"text|app|medical|photo",'
                        '"medical_entities":[string],"photo_prompt":string}]}. Preserve causal/temporal consistency.',
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "profile": profile,
                                "dates": dates,
                                "previous_events": timeline,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                max_tokens=8192,
            )
            events = payload["events"]
            if not isinstance(events, list) or [e["date"] for e in events] != dates:
                raise ValueError("Generated event dates must match exactly")
            for event in events:
                number = len(timeline) + 1
                iid = f"{user}-event-{number:03d}"
                timestamp = event["date"] + "T12:00:00"
                records.append(
                    dict(
                        user_id=user,
                        **asdict(Record(iid, timestamp, "text", event["text"])),
                    )
                )
                kind = event["kind"]
                media = None
                if kind in {"app", "medical", "photo"}:
                    media = output / "media" / f"{iid}.png"
                    if kind == "photo":
                        image_generator(
                            event["photo_prompt"], media, seed + u * 1000 + number
                        )
                    else:
                        # Programmatic screenshots/documents preserve the generated event text.
                        rendered = media.with_suffix(".jpg")
                        render_day(
                            [
                                Record(
                                    iid,
                                    timestamp,
                                    "text",
                                    kind.upper() + "\n" + event["text"],
                                )
                            ],
                            rendered,
                        )
                        with Image.open(rendered) as image:
                            image.save(media, format="PNG")
                        rendered.unlink()
                    records.append(
                        dict(
                            user_id=user,
                            **asdict(
                                Record(
                                    iid + "-image",
                                    timestamp,
                                    "image",
                                    "",
                                    str(media.resolve()),
                                )
                            ),
                        )
                    )
                elif kind != "text":
                    raise ValueError("Unknown synthetic event modality")
                event = dict(event, id=iid, user_id=user)
                timeline.append(event)
                if media and event.get("medical_entities"):
                    annotations.append(
                        {
                            "image": str(media.resolve()),
                            "entities": event["medical_entities"],
                            "id": iid,
                            "text": event["text"],
                        }
                    )
        for task in TASKS:
            payload = generator.json(
                [
                    {
                        "role": "system",
                        "content": "Create exactly the requested number of questions for the task from the supplied fictional timeline. "
                        "Questions must probe changing states and relevant historical context. Answer using the timeline. "
                        'Return {"queries":[{"question":string,"answer":string,"evidence_ids":[event_id],'
                        '"rubric_context":string}]}. Use only supplied event ids.',
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "profile": profile,
                                "timeline": timeline,
                                "task": task,
                                "count": queries_per_task,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                max_tokens=8192,
            )
            rows = payload["queries"]
            allowed = {e["id"] for e in timeline}
            if len(rows) != queries_per_task:
                raise ValueError("Incorrect controlled query count")
            for n, row in enumerate(rows):
                if not row["evidence_ids"] or not set(row["evidence_ids"]) <= allowed:
                    raise ValueError("Invalid controlled evidence ids")
                queries.append(
                    dict(
                        row,
                        id=f"{user}-{TASKS.index(task)}-{n+1}",
                        user_id=user,
                        task=task,
                        qtype="open_end",
                    )
                )
        atomic_json(output / f"{user}-timeline.json", timeline)
        atomic_json(output / "profiles.json", profiles)
        atomic_json(output / "records.json", records)
        atomic_json(output / "questions.json", queries)
        atomic_json(output / "fidelity-annotations.json", annotations)
    result = {
        "seed": seed,
        "users": users,
        "events_per_user": events_per_user,
        "queries_per_task": queries_per_task,
        "records": len(records),
        "daily_pages": users * events_per_user,
        "queries": len(queries),
    }
    atomic_json(output / "manifest.json", result)
    return result


class ControlledJudge:
    def __init__(self, endpoint=None):
        self.client = ChatClient(
            endpoint
            or Endpoint(
                "gpt-4o", "https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"
            )
        )

    def __call__(self, row, prediction):
        result = self.client.json(
            [
                {
                    "role": "system",
                    "content": "Score a controlled personal-memory answer from 1 to 5 on PI (personalization), TL (temporal logic), "
                    "CA (causal accuracy), FI (selective forgetting). Reward user-specific evidence, historical/current "
                    "distinctions and supported causal links. Prioritize recent evidence, retaining relevant early context; "
                    "penalize obsolete-detail dumping and invented relations. Return JSON with PI,TL,CA,FI and brief rationale.",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": row["task"],
                            "question": row["question"],
                            "reference": row["answer"],
                            "rubric_context": row.get("rubric_context", ""),
                            "prediction": prediction,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            max_tokens=1024,
        )
        if any(type(result.get(k)) is not int or not 1 <= result[k] <= 5 for k in AXES):
            raise ValueError("Controlled scores must be integers 1..5")
        return {k: result[k] for k in AXES}


def evaluate_controlled(predictions, questions, judge):
    pred = {str(p["id"]): p for p in predictions}
    ids = [str(q["id"]) for q in questions]
    if (
        not ids
        or len(pred) != len(predictions)
        or len(set(ids)) != len(ids)
        or set(pred) != set(ids)
    ):
        raise ValueError("Controlled prediction and question IDs must match exactly")
    details = []
    for row in questions:
        prediction = pred[str(row["id"])].get(
            "prediction", pred[str(row["id"])].get("answer")
        )
        if not isinstance(prediction, str):
            raise ValueError("Missing controlled answer")
        scores = judge(row, prediction)
        if any(
            type(scores.get(k)) not in {int, float} or not 1 <= scores[k] <= 5
            for k in AXES
        ):
            raise ValueError("Invalid rubric scores")
        details.append(
            dict(
                id=row["id"],
                task=row["task"],
                **scores,
                Overall=sum(scores[k] for k in AXES) / 4,
            )
        )
    by_task = {}
    for task in TASKS:
        group = [r for r in details if r["task"] == task]
        if group:
            by_task[task] = {
                k: sum(r[k] for r in group) / len(group) for k in AXES + ["Overall"]
            }
    return {"count": len(details), "by_task": by_task, "details": details}


def recognized_nrr(text, entities, backend):
    payload = backend.perceiver.json(
        [
            {
                "role": "system",
                "content": "Identify recognized medical entities and check them against annotations. Return "
                '{"recognized_entities":[string],"valid":[boolean]}; one boolean per entity.',
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"recognized_text": text, "annotations": entities}
                ),
            },
        ],
        max_tokens=1024,
    )
    recognized, valid = payload["recognized_entities"], payload["valid"]
    if (
        not isinstance(recognized, list)
        or any(not isinstance(e, str) for e in recognized)
        or len(recognized) != len(valid)
        or any(type(v) is not bool for v in valid)
    ):
        raise ValueError("Invalid NRR judgment")
    return sum(valid) / len(valid) if valid else 0.0


def fidelity_sweep(
    annotations, backend, output, *, resolutions=(1024, 640, 512, 256, 128, 64)
):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new fidelity output directory")
    output.mkdir(parents=True)
    rows = []
    for entry in annotations:
        original = Path(entry["image"])
        entities = entry["entities"]
        if not entities:
            raise ValueError("Medical entity annotations cannot be empty")
        if entry.get("text") is not None:
            text = entry["text"]
            rows.append(
                {
                    "id": entry["id"],
                    "resolution": "text",
                    "disk_bytes": len(text.encode("utf-8")),
                    "text_tokens": backend.text_token_count(text),
                    "ME-Recall": sum(
                        str(e).casefold() in text.casefold() for e in entities
                    )
                    / len(entities),
                    "NRR": recognized_nrr(text, entities, backend),
                }
            )
        for side in resolutions:
            path = output / f'{entry["id"]}-{side}.jpg'
            with Image.open(original) as image:
                image = image.convert("RGB")
                image.thumbnail((side, side), Image.Resampling.LANCZOS)
                image.save(path, format="JPEG", quality=90)
            analysis = backend.perceive(path, "2000-01-01")
            blob = (analysis.ocr_text + " " + analysis.visual_summary).casefold()
            recall = sum(str(e).casefold() in blob for e in entities) / len(entities)
            tokens = backend.image_token_count(path)
            nrr = recognized_nrr(analysis.ocr_text, entities, backend)
            rows.append(
                {
                    "id": entry["id"],
                    "resolution": side,
                    "disk_bytes": path.stat().st_size,
                    "vision_tokens": tokens,
                    "ME-Recall": recall,
                    "NRR": nrr,
                }
            )
    atomic_json(output / "measurements.json", rows)
    return rows


def run_controlled(
    records,
    questions,
    output,
    memory_factory,
    judge,
    *,
    variant="full",
    policy="timed_gentle",
    as_of=None,
    k=10,
):
    """Separate user stores; shared backend/encoder; full/no-forget/no-graph runs."""
    from .experiments import batch
    from .measurement import measure_storage

    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new controlled experiment directory")
    if variant not in {"full", "no-forget", "no-em-graph"}:
        raise ValueError("Unknown ablation")
    users = sorted({r["user_id"] for r in records})
    if not users or any(q["user_id"] not in users for q in questions):
        raise ValueError("Invalid user partition")
    output.mkdir(parents=True)
    predictions = []
    footprints = {}
    for user in users:
        memory = memory_factory(output / "stores" / user, variant != "no-em-graph")
        selected = []
        for row in records:
            if row["user_id"] == user:
                data = dict(row)
                data.pop("user_id")
                selected.append(Record(**data))
        memory.ingest(selected)
        if variant != "no-forget" and policy != "no_forget":
            if not as_of:
                raise ValueError(
                    "Controlled forgetting requires an explicit as_of date"
                )
            memory.forget(as_of, policy)
        predictions.extend(
            batch(memory, [q for q in questions if q["user_id"] == user], k, True)
        )
        footprints[user] = measure_storage(memory.root)
        atomic_json(output / "predictions.json", predictions)
    scores = evaluate_controlled(predictions, questions, judge)
    usage = {}
    for row in predictions:
        if row.get("usage"):
            for role, counts in row["usage"].items():
                for key, value in counts.items():
                    usage[f"{role}_{key}"] = usage.get(f"{role}_{key}", 0) + value
    atomic_json(output / "scores.json", scores)
    atomic_json(output / "storage.json", footprints)
    atomic_json(output / "tokens.json", {"queries": len(predictions), "totals": usage})
    return {
        "variant": variant,
        "users": len(users),
        "queries": len(predictions),
        "scores": scores["by_task"],
    }
