"""Data contracts shared across the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Protocol, Sequence


def record_day(timestamp: str) -> str:
    """Return the timestamp's calendar day."""
    if not isinstance(timestamp, str) or not timestamp.strip():
        raise ValueError("A timestamp is required")
    value = timestamp.strip()
    if len(value) == 10:
        return date.fromisoformat(value).isoformat()
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()


@dataclass(frozen=True)
class Record:
    id: str
    timestamp: str
    modality: str
    text: str = ""
    media_path: str | None = None

    def __post_init__(self):
        record_day(self.timestamp)
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Record id must be a nonempty string")
        if self.modality not in {"text", "email", "image", "video"}:
            raise ValueError(f"Unsupported modality: {self.modality}")
        if self.modality in {"image", "video"} and not self.media_path:
            raise ValueError("Image/video records need media_path")
        if not isinstance(self.text, str):
            raise ValueError("Record text must be a string")

    @property
    def day(self) -> str:
        return record_day(self.timestamp)


@dataclass
class Analysis:
    """Perception result for one page.

    ``paths`` holds ordered sequences of ``semantic_nodes``. Visual nodes are
    phrases from optical-perception text; both kinds use the semantic encoder.
    """

    ocr_text: str
    visual_summary: str
    semantic_nodes: list[str]
    paths: list[list[str]]
    text_nodes: list[str] = field(default_factory=list)
    visual_nodes: list[str] = field(default_factory=list)
    source_nodes: dict[str, list[str]] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.ocr_text, str) or not isinstance(
            self.visual_summary, str
        ):
            raise ValueError("Perception text must be strings")
        if not isinstance(self.semantic_nodes, list) or any(
            not isinstance(x, str) or not x.strip() for x in self.semantic_nodes
        ):
            raise ValueError("semantic_nodes must contain nonempty strings")
        if len(set(self.semantic_nodes)) != len(self.semantic_nodes):
            raise ValueError("semantic_nodes must be unique")
        if not isinstance(self.paths, list):
            raise ValueError("paths must be a list")
        referenced = set()
        for path in self.paths:
            if (
                not isinstance(path, list)
                or not path
                or any(
                    not isinstance(n, str) or n not in self.semantic_nodes for n in path
                )
            ):
                raise ValueError(
                    "Each path must be a nonempty ordered subset of semantic_nodes"
                )
            if len(set(path)) != len(path):
                raise ValueError("An EM-Path cannot repeat a semantic node")
            referenced.update(path)
        if referenced != set(self.semantic_nodes):
            raise ValueError("Every semantic node must belong to at least one EM-Path")


class TextEncoder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class Backend(Protocol):
    def perceive(self, image: Path, day: str) -> Analysis: ...
    def extract_record_nodes(self, record: Record) -> list[str]: ...
    def consolidate(
        self, day: str, text_nodes: list[str], visual: Analysis
    ) -> Analysis: ...
    def extract_query(self, question: str) -> list[str]: ...
    def coherent(self, labels: list[str], current: Analysis) -> bool: ...
    def answer(self, question: str, evidence: list[dict]) -> str: ...
