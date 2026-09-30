"""BM25 tren caption (tang 1, nhanh sparse). Diem cua mot anh = max diem 5 caption cua no."""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

STOP = set("a an the of in on at to and or is are was were with for his her its their "
           "this that these those he she it they there be been being as by from into "
           "while over up down out off very some".split())


def tokenize(text: str, drop_stop: bool = True) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in toks if t not in STOP] if drop_stop else toks


class BM25Index:
    def __init__(self, captions: list[str], caption_image_idx: list[int], n_images: int,
                 drop_stop: bool = True):
        self.drop_stop = drop_stop
        self.owner = np.asarray(caption_image_idx)
        self.n_images = n_images
        self.bm25 = BM25Okapi([tokenize(c, drop_stop) for c in captions])

    def caption_scores(self, query: str) -> np.ndarray:
        toks = tokenize(query, self.drop_stop)
        if not toks:
            return np.zeros(len(self.owner), "float32")
        return np.asarray(self.bm25.get_scores(toks), "float32")

    def image_scores(self, query: str, exclude_caption: int | None = None) -> np.ndarray:
        s = self.caption_scores(query)
        if exclude_caption is not None:
            s[exclude_caption] = -np.inf   # leave-one-out khi danh gia
        out = np.full(self.n_images, -np.inf, "float32")
        np.maximum.at(out, self.owner, s)
        out[out == -np.inf] = 0.0
        return out

    def search(self, query: str, k: int, exclude_caption: int | None = None):
        """Tra ve (idxs, scores) cac anh co diem > 0, toi da k."""
        s = self.image_scores(query, exclude_caption)
        order = np.argsort(-s, kind="stable")[:k]
        order = order[s[order] > 0]
        return order, s[order]
