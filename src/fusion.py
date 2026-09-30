"""Gop danh sach xep hang (RRF) va gop vector truy van chu + anh."""
from __future__ import annotations

import numpy as np


def rrf(rank_lists: list[np.ndarray], k: int = 60):
    """Reciprocal Rank Fusion. Moi phan tu la mang id theo thu tu hang.
    Tra ve (ids, scores) sap giam dan."""
    acc: dict[int, float] = {}
    for ids in rank_lists:
        for r, i in enumerate(ids):
            acc[int(i)] = acc.get(int(i), 0.0) + 1.0 / (k + r + 1)
    ids = sorted(acc, key=lambda i: -acc[i])
    return np.asarray(ids, "int64"), np.asarray([acc[i] for i in ids], "float32")


def fuse_query(v_img: np.ndarray | None, v_txt: np.ndarray | None, alpha: float = 0.5) -> np.ndarray:
    """q = norm(alpha*v_anh + (1-alpha)*v_chu). Chi co mot ben thi tra ve ben do."""
    if v_img is None:
        return v_txt
    if v_txt is None:
        return v_img
    q = alpha * v_img + (1.0 - alpha) * v_txt
    return q / np.linalg.norm(q, axis=-1, keepdims=True)
