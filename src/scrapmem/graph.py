"""Canonical merging, sparse binary incidence retrieval, and page-local pruning."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Callable, Sequence

import numpy as np


def unit(vector) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    if arr.ndim != 1 or not len(arr) or not np.isfinite(arr).all():
        raise ValueError("Embedding must be a finite nonempty vector")
    norm = np.linalg.norm(arr)
    if norm <= 1e-12:
        raise ValueError("Embedding must be nonzero")
    return arr / norm


def prepared(
    labels: list[str], vectors: Sequence[Sequence[float]], kind="semantic"
) -> list[dict]:
    if len(labels) != len(vectors):
        raise ValueError("Encoder returned the wrong number of vectors")
    return [
        {"label": label, "kind": kind, "vector": unit(v).tolist()}
        for label, v in zip(labels, vectors)
    ]


class EMGraph:
    """Sparse canonical-node graph with binary node-overlap retrieval.

    Occurrences record which page supports each shared node; Q is kept as an
    inverted index from node to paths.
    """

    def __init__(self, threshold: float = 0.65, state: dict | None = None):
        if not np.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("canonical threshold must be in (0, 1]")
        self.threshold = threshold
        self.nodes: dict[str, dict] = {}
        self.occurrences: dict[str, dict] = {}
        self.paths: dict[str, dict] = {}
        self.next_node = 0
        if state is not None:
            self.nodes = deepcopy(state["nodes"])
            self.occurrences = deepcopy(state["occurrences"])
            self.paths = deepcopy(state["paths"])
            self.next_node = state["next_node"]
        self._reindex()

    def to_dict(self) -> dict:
        return deepcopy(
            {
                "nodes": self.nodes,
                "occurrences": self.occurrences,
                "paths": self.paths,
                "next_node": self.next_node,
            }
        )

    def _reindex(self):
        self.node_paths: dict[str, set[str]] = defaultdict(set)
        for pid, path in self.paths.items():
            for oid in path["occurrences"]:
                self.node_paths[self.occurrences[oid]["node_id"]].add(pid)

    def _refresh_centers(self):
        supported: dict[str, list[dict]] = defaultdict(list)
        for occurrence in self.occurrences.values():
            supported[occurrence["node_id"]].append(occurrence)
        for nid in list(self.nodes):
            group = supported.get(nid)
            if not group:
                del self.nodes[nid]
                continue
            # Recompute centroid/label from surviving observations only.
            center = unit(np.sum([o["vector"] for o in group], axis=0))
            self.nodes[nid] = {
                "kind": group[0]["kind"],
                "label": group[0]["label"],
                "vector": center.tolist(),
            }
        self._reindex()

    def add_page(self, page_id: str, specs: list[dict], ordered_paths: list[list[int]]):
        if any(p["page_id"] == page_id for p in self.paths.values()):
            raise ValueError(f"Page already indexed: {page_id}")
        # Validate before changing state.
        for spec in specs:
            unit(spec["vector"])
            if spec["kind"] != "semantic":
                raise ValueError("Unknown node kind")
        for path in ordered_paths:
            if not path or any(
                type(i) is not int or not 0 <= i < len(specs) for i in path
            ):
                raise ValueError("Invalid path node index")
        groups = defaultdict(list)
        for occurrence in self.occurrences.values():
            groups[occurrence["node_id"]].append(occurrence)
        sums = {
            nid: np.sum([o["vector"] for o in group], axis=0)
            for nid, group in groups.items()
        }
        ids = []
        for index, spec in enumerate(specs):
            vector = unit(spec["vector"])
            candidates = [
                (nid, n) for nid, n in self.nodes.items() if n["kind"] == spec["kind"]
            ]
            if candidates:
                centers = np.asarray([n["vector"] for _, n in candidates])
                if centers.shape[1] != len(vector):
                    raise ValueError("Embedding dimensions changed")
                similarities = centers @ vector
                best = int(np.argmax(similarities))
                nid = (
                    candidates[best][0]
                    if similarities[best] >= self.threshold
                    else None
                )
            else:
                nid = None
            if nid is None:
                nid = f"n{self.next_node}"
                self.next_node += 1
                self.nodes[nid] = {
                    "label": spec["label"],
                    "kind": spec["kind"],
                    "vector": vector.tolist(),
                }
            oid = f"{page_id}:{index}"
            self.occurrences[oid] = dict(
                spec, page_id=page_id, node_id=nid, vector=vector.tolist()
            )
            ids.append(oid)
            # Update just the affected centroid, not the entire graph.
            sums[nid] = sums.get(nid, np.zeros_like(vector)) + vector
            self.nodes[nid]["vector"] = unit(sums[nid]).tolist()
        for index, path in enumerate(ordered_paths):
            self.paths[f"{page_id}:p{index}"] = {
                "page_id": page_id,
                "occurrences": list(dict.fromkeys(ids[i] for i in path)),
            }
        self._reindex()

    def match_nodes(
        self, query_vectors: Sequence[Sequence[float]], kind="semantic"
    ) -> set[str]:
        candidates = [(nid, n) for nid, n in self.nodes.items() if n["kind"] == kind]
        if not candidates or not len(query_vectors):
            return set()
        centers = np.asarray([n["vector"] for _, n in candidates])
        result = set()
        for q in query_vectors:
            sims = centers @ unit(q)
            best = int(np.argmax(sims))
            if sims[best] >= self.threshold:
                result.add(candidates[best][0])
        return result

    def retrieve(self, semantic_vectors, k: int = 10) -> list[dict]:
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        matches = self.match_nodes(semantic_vectors)
        counts: dict[str, int] = defaultdict(int)
        for nid in matches:
            for pid in self.node_paths.get(nid, ()):
                counts[pid] += 1
        # Ranking is by matched-node overlap only.
        ranking = sorted(counts, key=lambda pid: (-counts[pid], pid))[:k]
        return [
            {
                "path_id": pid,
                "page_id": self.paths[pid]["page_id"],
                "score": counts[pid],
                "nodes": [
                    self.occurrences[o]["label"]
                    for o in self.paths[pid]["occurrences"]
                    if self.occurrences[o]["kind"] == "semantic"
                ],
            }
            for pid in ranking
        ]

    def prune_page(
        self,
        page_id: str,
        current_specs: list[dict],
        coherent: Callable[[list[str]], bool],
    ) -> dict:
        """Update one page's support after re-perception.

        Re-perceived features can only support existing occurrences; no new
        nodes are added during forgetting.
        """
        old = {oid: o for oid, o in self.occurrences.items() if o["page_id"] == page_id}
        updates = {}
        for oid, occurrence in old.items():
            candidates = [s for s in current_specs if s["kind"] == occurrence["kind"]]
            if not candidates:
                continue
            scores = [
                float(unit(s["vector"]) @ unit(occurrence["vector"]))
                for s in candidates
            ]
            best = int(np.argmax(scores))
            if scores[best] >= self.threshold:
                updates[oid] = dict(
                    occurrence,
                    label=candidates[best]["label"],
                    vector=unit(candidates[best]["vector"]).tolist(),
                    origins=candidates[best].get(
                        "origins", occurrence.get("origins", [])
                    ),
                )
        replacements = {}
        removed_paths = 0
        for pid, path in self.paths.items():
            if path["page_id"] != page_id:
                continue
            remaining = [oid for oid in path["occurrences"] if oid in updates]
            labels = [
                updates[o]["label"]
                for o in remaining
                if updates[o]["kind"] == "semantic"
            ]
            valid = bool(remaining)
            if valid and len(remaining) != len(path["occurrences"]):
                valid = coherent(labels)
                if type(valid) is not bool:
                    raise ValueError("Coherence callback must return bool")
            replacements[pid] = dict(path, occurrences=remaining) if valid else None
        # Model failures above leave the graph untouched.
        for pid, replacement in replacements.items():
            if replacement is None:
                del self.paths[pid]
                removed_paths += 1
            else:
                self.paths[pid] = replacement
        used = {oid for p in self.paths.values() for oid in p["occurrences"]}
        for oid in old:
            if oid in used and oid in updates:
                self.occurrences[oid] = updates[oid]
            else:
                self.occurrences.pop(oid, None)
        self._refresh_centers()
        return {
            "removed_occurrences": len(old) - sum(o in self.occurrences for o in old),
            "removed_paths": removed_paths,
        }
