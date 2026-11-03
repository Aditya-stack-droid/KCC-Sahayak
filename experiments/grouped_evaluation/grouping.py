"""Question-identity grouping primitives (exact and near-duplicate).

The unit of evaluation is a *question group*, not a row. Two logged questions
belong to the same group if they are identical after normalisation, or if
their sentence-embedding cosine similarity is at least a configurable
threshold (default 0.95). Groups are built with union-find, so near-duplicate
links are transitive.
"""
from __future__ import annotations

import re

import numpy as np

_WS = re.compile(r"\s+")


class UnionFind:
    """Small path-compressed union-find with union by rank."""

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)
        self.n_components = n

    def find(self, x: int) -> int:
        parent = self.parent
        root = int(x)
        while parent[root] != root:
            root = int(parent[root])
        # path compression
        while parent[x] != root:
            nxt = int(parent[x])
            parent[x] = root
            x = nxt
        return root

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.n_components -= 1
        return True

    def roots(self) -> np.ndarray:
        return np.array([self.find(i) for i in range(len(self.parent))], dtype=np.int64)


def normalize_question(text) -> str:
    return _WS.sub(" ", str(text).strip().lower())


def group_exact(questions) -> UnionFind:
    """Union all rows whose normalised question strings are identical."""
    uf = UnionFind(len(questions))
    first: dict[str, int] = {}
    for i, q in enumerate(questions):
        key = normalize_question(q)
        if key in first:
            uf.union(first[key], i)
        else:
            first[key] = i
    return uf


def apply_edges(uf: UnionFind, edges) -> int:
    """Merge all (i, j) edges; returns the number of successful merges."""
    merged = 0
    for i, j in edges:
        if uf.union(int(i), int(j)):
            merged += 1
    return merged


def canonical_group_ids(uf: UnionFind) -> np.ndarray:
    """Map each row to the smallest row index in its component (deterministic)."""
    roots = uf.roots()
    smallest: dict[int, int] = {}
    for i, r in enumerate(roots.tolist()):
        if r not in smallest or i < smallest[r]:
            smallest[r] = i
    return np.array([smallest[r] for r in roots.tolist()], dtype=np.int64)


def group_sizes(group_ids: np.ndarray) -> np.ndarray:
    _, counts = np.unique(group_ids, return_counts=True)
    return counts


def group_leakage(group_ids: np.ndarray, split: np.ndarray) -> dict:
    """How many evaluation queries share a group with any training query."""
    split = np.asarray(split)
    if not set(np.unique(split)) >= {"train", "test"}:
        raise ValueError("split must contain at least 'train' and 'test'")
    train_groups = set(group_ids[split == "train"].tolist())
    test_mask = split == "test"
    test_groups = group_ids[test_mask]
    leaking = np.array([g in train_groups for g in test_groups.tolist()])
    return {
        "n_test": int(test_mask.sum()),
        "n_test_group_leaking": int(leaking.sum()),
        "share_test_group_leaking": round(float(leaking.mean()), 6) if len(leaking) else None,
        "n_train_groups": len(train_groups),
        "n_test_groups": int(len(set(test_groups.tolist()))),
    }
