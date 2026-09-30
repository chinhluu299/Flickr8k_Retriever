"""SearchEngine: nap artifact da build va chay pipeline hai tang.

Tang 1: dense (CLIP B/32 + FAISS) va BM25 tren caption, gop bang RRF.
Tang 2: cham lai top-N bang CLIP L/14.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

import configs
from src.dataset import Corpus
from src.encoders import load_encoder
from src.fusion import fuse_query, rrf
from src.index import VectorIndex
from src.sparse import BM25Index

STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    rank: int
    score: float
    image_path: str
    image_idx: int
    captions: list[str] = field(default_factory=list)


class SearchEngine:
    def __init__(self, model_key: str = configs.RETRIEVAL_MODEL):
        d = configs.ARTIFACTS / model_key
        if not (d / "image.faiss").exists():
            raise FileNotFoundError(f"chua build {model_key}, chay: python3 build_index.py --model {model_key}")
        self.corpus = Corpus.load(d / "corpus.json")
        self.encoder = load_encoder(model_key)
        self.image_index = VectorIndex.load(d / "image.faiss")
        self.bm25 = BM25Index(self.corpus.captions, self.corpus.caption_image_idx, self.corpus.n_images)
        self._reranker = None

    @property
    def reranker(self):
        # nap L/14 lan dau can den
        if self._reranker is None:
            from src.rerank import ClipReranker
            self._reranker = ClipReranker()
        return self._reranker

    def encode_query(self, text: str | None = None, image=None, alpha: float = 0.5) -> np.ndarray:
        v_txt = self.encoder.encode_texts([text]) if text else None
        v_img = self.encoder.encode_images([image]) if image is not None else None
        return fuse_query(v_img, v_txt, alpha)

    def search(self, text: str | None = None, image=None, k: int = configs.DEFAULT_TOP_K,
               alpha: float = 0.5, strategy: str = "hybrid_rerank",
               n_cand: int = configs.N_CAND, rerank_n: int = configs.RERANK_N,
               rrf_k: int = configs.RRF_K):
        """Nhan chu va/hoac anh, tra ve (hits, timing ms tung buoc).

        strategy:
          dense          CLIP + FAISS
          bm25           BM25 tren caption (can chu)
          hybrid         RRF(dense top n_cand, bm25 top n_cand)
          hybrid_rerank  hybrid roi cham lai top rerank_n bang L/14
        """
        text = text.strip() if text and text.strip() else None
        if text is None and image is None:
            raise ValueError("can cau chu hoac anh")
        if strategy == "bm25" and text is None:
            raise ValueError("BM25 chi tim duoc bang cau chu")
        timing: dict[str, float] = {}
        t_all = time.perf_counter()

        dense, sparse = [], []
        if strategy != "bm25":
            t0 = time.perf_counter()
            q = self.encode_query(text, image, alpha)
            timing["encode"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            scores, idxs = self.image_index.search(q, k if strategy == "dense" else n_cand)
            dense = list(zip(idxs[0].tolist(), scores[0].tolist()))
            timing["dense"] = (time.perf_counter() - t0) * 1000
        if strategy != "dense" and text is not None:
            t0 = time.perf_counter()
            ids, sc = self.bm25.search(text, k if strategy == "bm25" else n_cand)
            sparse = list(zip(ids.tolist(), sc.tolist()))
            timing["bm25"] = (time.perf_counter() - t0) * 1000

        if strategy == "dense":
            ranked = dense
        elif strategy == "bm25":
            ranked = sparse
        else:
            t0 = time.perf_counter()
            lists = [np.asarray([i for i, _ in branch], "int64") for branch in (dense, sparse) if branch]
            ids, sc = rrf(lists, k=rrf_k)
            ranked = list(zip(ids.tolist(), sc.tolist()))
            timing["rrf"] = (time.perf_counter() - t0) * 1000
            if strategy == "hybrid_rerank":
                t0 = time.perf_counter()
                cand = ids[:rerank_n]
                rs = self.reranker.rerank(cand, text=text, image=image, alpha=alpha)
                order = np.argsort(-rs, kind="stable")
                ranked = [(int(cand[j]), float(rs[j])) for j in order] + ranked[rerank_n:]
                timing["rerank"] = (time.perf_counter() - t0) * 1000

        hits = [Hit(rank=r + 1, score=float(s), image_path=self.corpus.image_paths[i], image_idx=int(i),
                    captions=self.corpus.captions_of(int(i))[:2])
                for r, (i, s) in enumerate(ranked[:k])]
        timing["total"] = (time.perf_counter() - t_all) * 1000
        return hits, {name: round(v, 1) for name, v in timing.items()}

    def info(self) -> dict:
        return {
            "label": self.encoder.label,
            "dim": self.image_index.dim,
            "device": self.encoder.device,
            "n_images": self.corpus.n_images,
            "n_captions": self.corpus.n_captions,
        }
