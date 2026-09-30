"""FAISS IndexFlatIP. Vector da L2-normalize nen inner product = cosine."""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class VectorIndex:
    def __init__(self, index: faiss.Index):
        self.index = index
        self.dim = index.d

    @classmethod
    def build(cls, vectors: np.ndarray) -> "VectorIndex":
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors, dtype="float32"))
        return cls(index)

    def search(self, queries: np.ndarray, k: int):
        q = np.ascontiguousarray(np.atleast_2d(queries), dtype="float32")
        return self.index.search(q, min(k, self.index.ntotal))

    def save(self, path: Path) -> None:
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        return cls(faiss.read_index(str(path)))
