"""Daily EM-Graph retrieval followed by fine-grained evidence reranking."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from datetime import date
from pathlib import Path
import json
import uuid

from .forgetting import profile, transform
from .graph import EMGraph, prepared, unit
from .pages import render_day
from .storage import atomic_json, writer_lock
from .types import Analysis, Record


class ScrapMem:
    def __init__(
        self,
        store,
        encoder,
        backend,
        *,
        canonical_threshold=0.65,
        page_width=1400,
        font_path=None,
        max_page_pixels=80_000_000,
        video_max_frames=8,
        day_top_n=20,
        use_graph=True,
    ):
        self.root = Path(store)
        self.root.mkdir(parents=True, exist_ok=True)
        for folder in ("pages", "sources"):
            (self.root / folder).mkdir(exist_ok=True)
        self.encoder, self.backend = encoder, backend
        self.threshold = canonical_threshold
        EMGraph(canonical_threshold)
        if type(day_top_n) is not int or day_top_n < 1:
            raise ValueError("day_top_n must be a positive integer")
        self.day_top_n = day_top_n
        self.use_graph = bool(use_graph)
        self.renderer = dict(
            width=page_width,
            font_path=font_path,
            max_pixels=max_page_pixels,
            video_max_frames=video_max_frames,
        )
        self.signature = {
            "text_encoder": encoder.name,
            "canonical_threshold": canonical_threshold,
            "node_representation": "original-text+visual-perception-text",
            "use_graph": self.use_graph,
            "backend": type(backend).__name__,
            "perception_model": getattr(
                getattr(backend, "perceiver", None), "name", None
            )
            or getattr(
                getattr(getattr(backend, "perceiver", None), "endpoint", None),
                "model",
                None,
            ),
            "answer_model": getattr(getattr(backend, "answerer", None), "name", None)
            or getattr(
                getattr(getattr(backend, "answerer", None), "endpoint", None),
                "model",
                None,
            ),
        }
        self._load()

    def _load(self):
        path = self.root / "memory.json"
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("schema_version") != 2:
                raise ValueError(
                    "Use a fresh store for the text-grounded two-stage pipeline"
                )
            if state["signature"] != self.signature:
                raise ValueError(
                    "Store encoder/threshold differs; use its original configuration"
                )
            self.pages = state["pages"]
            self.maintenance_date = state.get("maintenance_date")
            self.graph = EMGraph(self.threshold, state["graph"])
            self.source_index = state["source_index"]
            self.items = json.loads(
                (self.root / self.source_index).read_text(encoding="utf-8")
            )
        else:
            self.pages, self.items = {}, {}
            self.maintenance_date, self.source_index = None, None
            self.graph = EMGraph(self.threshold)

    def _save(self):
        atomic_json(
            self.root / "memory.json",
            {
                "schema_version": 2,
                "signature": self.signature,
                "pages": self.pages,
                "graph": self.graph.to_dict(),
                "maintenance_date": self.maintenance_date,
                "source_index": self.source_index,
            },
        )

    def _perceive(self, path, day):
        analysis = self.backend.perceive(path, day)
        analysis.validate()
        analysis.visual_nodes = list(analysis.semantic_nodes)
        return analysis

    def _construct(self, records, visual):
        original_nodes, source_nodes = [], {}
        for record in records:
            if record.modality in {"text", "email"} and record.text.strip():
                nodes = self.backend.extract_record_nodes(record)
                if (
                    not isinstance(nodes, list)
                    or len(nodes) > 8
                    or any(not isinstance(n, str) or not n.strip() for n in nodes)
                ):
                    raise ValueError("Invalid record-side nodes")
                source_nodes[record.id] = nodes
                original_nodes.extend(nodes)
        result = self.backend.consolidate(
            records[0].day, list(dict.fromkeys(original_nodes)), visual
        )
        result.validate()
        supplied = set(original_nodes) | set(visual.semantic_nodes)
        if (
            len(result.semantic_nodes) > 10
            or not set(result.semantic_nodes) <= supplied
        ):
            raise ValueError(
                "Daily paths must use at most ten supplied text/visual-text nodes"
            )
        result.text_nodes = [n for n in result.semantic_nodes if n in original_nodes]
        result.visual_nodes = [
            n for n in result.semantic_nodes if n in visual.semantic_nodes
        ]
        result.source_nodes = {
            k: [n for n in v if n in result.semantic_nodes]
            for k, v in source_nodes.items()
        }
        return result

    def _specs(self, analysis):
        specs = prepared(
            analysis.semantic_nodes, self.encoder.encode(analysis.semantic_nodes)
        )
        for spec in specs:
            spec["origins"] = [
                origin
                for origin, labels in [
                    ("text", analysis.text_nodes),
                    ("visual-text", analysis.visual_nodes),
                ]
                if spec["label"] in labels
            ]
        return specs

    def ingest(self, records):
        ids = [r.id for r in records]
        if not records or len(set(ids)) != len(ids):
            raise ValueError("Supply nonempty records with unique IDs")
        groups = defaultdict(list)
        for record in records:
            groups[record.day].append(record)
        staged = []
        with writer_lock(self.root):
            self._load()
            if set(groups) & self.pages.keys() or set(ids) & self.items.keys():
                raise ValueError(
                    "Day or record ID already indexed; supply complete days in a fresh store"
                )
            old_source = self.root / self.source_index if self.source_index else None
            try:
                for day in sorted(groups):
                    path = self.root / "pages" / f"{day}-{uuid.uuid4().hex}.jpg"
                    staged.append(path)
                    size = render_day(groups[day], path, **self.renderer)
                    analysis = self._construct(groups[day], self._perceive(path, day))
                    positions = {n: i for i, n in enumerate(analysis.semantic_nodes)}
                    if self.use_graph:
                        self.graph.add_page(
                            day,
                            self._specs(analysis),
                            [[positions[n] for n in p] for p in analysis.paths],
                        )
                    self.pages[day] = {
                        "image": path.relative_to(self.root).as_posix(),
                        "original_size": list(size),
                        "source_ids": [r.id for r in groups[day]],
                        "quality": 100,
                        "scale": 1.0,
                        "policy": "no_forget",
                        "as_of": day,
                        "analysis": asdict(analysis),
                    }
                self.items.update({r.id: asdict(r) for r in records})
                source = self.root / "sources" / f"{uuid.uuid4().hex}.json"
                staged.append(source)
                atomic_json(source, self.items)
                self.source_index = source.relative_to(self.root).as_posix()
                self._save()
            except Exception:
                for path in staged:
                    path.unlink(missing_ok=True)
                self._load()
                raise
            if old_source:
                old_source.unlink()
        return {"pages_added": len(groups), "records_added": len(records)}

    def _retrieve_paths(self, question):
        if not self.use_graph:
            return [
                dict(
                    path_id=f"{day}:page",
                    page_id=day,
                    nodes=page["analysis"]["semantic_nodes"],
                    image=str(self.root / page["image"]),
                    source_ids=page["source_ids"],
                    ocr_text=page["analysis"]["ocr_text"],
                    visual_summary=page["analysis"]["visual_summary"],
                )
                for day, page in self.pages.items()
            ]
        nodes = self.backend.extract_query(question)
        if not isinstance(nodes, list) or any(
            not isinstance(n, str) or not n.strip() for n in nodes
        ):
            raise ValueError("Invalid query nodes")
        vectors = self.encoder.encode(nodes)
        if len(vectors) != len(nodes):
            raise ValueError("Encoder returned wrong query vector count")
        ranked = self.graph.retrieve(vectors, max(1, len(self.graph.paths)))
        selected, days = [], set()
        for path in ranked:
            day = path["page_id"]
            if day not in days and len(days) >= self.day_top_n:
                continue
            days.add(day)
            page = self.pages[day]
            path.update(
                image=str(self.root / page["image"]),
                source_ids=page["source_ids"],
                ocr_text=page["analysis"]["ocr_text"],
                visual_summary=page["analysis"]["visual_summary"],
            )
            selected.append(path)
        return selected

    def _retrieve(self, question, k):
        if type(k) is not int or k < 1:
            raise ValueError("k must be positive")
        paths = self._retrieve_paths(question)
        by_day = {}
        for path in paths:
            by_day.setdefault(path["page_id"], path)
        # Only items belonging to Stage-A candidate days enter Stage B.
        candidate_ids = list(
            dict.fromkeys(i for day in by_day for i in self.pages[day]["source_ids"])
        )
        if not candidate_ids:
            return []
        query = unit(self.encoder.encode([question])[0])
        texts = [
            self.items[i]["timestamp"] + "\n" + self.items[i]["text"]
            for i in candidate_ids
        ]
        vectors = self.encoder.encode(texts)
        if len(vectors) != len(candidate_ids):
            raise ValueError("Encoder returned wrong candidate vector count")
        scores = {i: float(unit(v) @ query) for i, v in zip(candidate_ids, vectors)}
        ranked = sorted(candidate_ids, key=lambda i: (-scores[i], i))[:k]
        evidence = []
        for iid in ranked:
            record = self.items[iid]
            day = Record(**record).day
            item = dict(by_day[day])
            item.update(
                item_id=iid,
                day_id=day,
                text=record["text"],
                timestamp=record["timestamp"],
                modality=record["modality"],
                media_path=record["media_path"],
                score=scores[iid],
                fine_path_id=f"{day}:record:{iid}",
                stage_a_day_ids=list(by_day),
            )
            evidence.append(item)
        return evidence

    def retrieve_paths(self, question):
        if not question.strip():
            raise ValueError("Question cannot be empty")
        with writer_lock(self.root):
            self._load()
            return self._retrieve_paths(question)

    def retrieve(self, question, k=10):
        if not question.strip():
            raise ValueError("Question cannot be empty")
        with writer_lock(self.root):
            self._load()
            return self._retrieve(question, k)

    def answer(self, question, k=10):
        if not question.strip():
            raise ValueError("Question cannot be empty")
        with writer_lock(self.root):
            self._load()
            evidence = self._retrieve(question, k)
            if not evidence:
                return {"answer": "Insufficient memory evidence.", "evidence": []}
            return {
                "answer": self.backend.answer(question, evidence),
                "evidence": evidence,
            }

    def forget(self, as_of, policy="timed_gentle"):
        date.fromisoformat(as_of)
        staged, obsolete, updates = [], [], []
        with writer_lock(self.root):
            self._load()
            if self.maintenance_date and as_of < self.maintenance_date:
                raise ValueError("Maintenance clock cannot move backwards")
            before_pages, before_graph, before_date = (
                deepcopy(self.pages),
                self.graph.to_dict(),
                self.maintenance_date,
            )
            try:
                for day, page in self.pages.items():
                    target = profile(policy, day, as_of)
                    if as_of < page["as_of"]:
                        raise ValueError("Maintenance clock cannot move backwards")
                    if target.scale > page["scale"] or target.quality > page["quality"]:
                        raise ValueError("Cannot restore removed page information")
                    if (target.scale, target.quality) == (
                        page["scale"],
                        page["quality"],
                    ):
                        continue
                    old = self.root / page["image"]
                    new = self.root / "pages" / f"{day}-{uuid.uuid4().hex}.jpg"
                    staged.append(new)
                    transform(old, new, target, tuple(page["original_size"]))
                    current = self._perceive(new, day)
                    change = self.graph.prune_page(
                        day,
                        self._specs(current),
                        lambda labels: self.backend.coherent(labels, current),
                    )
                    page.update(
                        image=new.relative_to(self.root).as_posix(),
                        quality=target.quality,
                        scale=target.scale,
                        policy=policy,
                        as_of=as_of,
                        analysis=asdict(current),
                    )
                    obsolete.append(old)
                    updates.append(dict(page_id=day, **change))
                self.maintenance_date = as_of
                self._save()
            except Exception:
                self.pages, self.maintenance_date = before_pages, before_date
                self.graph = EMGraph(self.threshold, before_graph)
                for path in staged:
                    path.unlink(missing_ok=True)
                raise
            for path in obsolete:
                path.unlink()
        return {"pages_changed": len(updates), "updates": updates}
