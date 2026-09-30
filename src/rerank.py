"""Tang 2: cham lai top-N ung vien bang CLIP L/14.

Embedding anh L/14 da build san, luc chay chi encode truy van roi nhan voi
embedding cua N ung vien.
"""
from __future__ import annotations

import numpy as np

import configs
from src.encoders import load_encoder
from src.fusion import fuse_query


class ClipReranker:
    def __init__(self, model_key: str = configs.RERANK_MODEL):
        path = configs.ARTIFACTS / model_key / "image_emb.npy"
        if not path.exists():
            raise FileNotFoundError(f"chua build {model_key}, chay: python3 build_index.py --model {model_key}")
        self.encoder = load_encoder(model_key)
        self.image_emb = np.load(path)

    def rerank(self, cand: np.ndarray, text: str | None = None, image=None,
               alpha: float = 0.5) -> np.ndarray:
        v_txt = self.encoder.encode_texts([text]) if text else None
        v_img = self.encoder.encode_images([image]) if image is not None else None
        q = fuse_query(v_img, v_txt, alpha)
        return self.image_emb[cand] @ q[0]
