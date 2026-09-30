#!/usr/bin/env python3
"""Danh gia truy van anh va chu + anh. Flickr8k khong co nhan cho kieu truy van nay nen
anh truy van la mot vung cat ngau nhien (10-30% dien tich, co the lat ngang) cua anh
dich; phan chu la caption cua anh dich (BM25 van leave-one-out). Ghi results/multimodal.md.

  python3 evaluation/multimodal.py
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (RESULTS, bm25_sim, fmt, load_embeddings, raw_embeddings, rerank_sim,
                      retrieval_metrics, rrf_sim)
import configs
from src.dataset import Corpus, load_images

ALPHAS = (0.25, 0.5, 0.75)
CROP_SCALE = (0.1, 0.3)


def random_crops(images, seed: int = 0):
    """Moi anh mot vung cat: dien tich CROP_SCALE, ti le canh 3/4-4/3, lat ngang 50%."""
    from PIL import ImageOps
    rng = np.random.default_rng(seed)
    out = []
    for im in images:
        w, h = im.size
        area = rng.uniform(*CROP_SCALE) * w * h
        ratio = np.exp(rng.uniform(np.log(3 / 4), np.log(4 / 3)))
        cw, ch = min(w, int(round(np.sqrt(area * ratio)))), min(h, int(round(np.sqrt(area / ratio))))
        x, y = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
        crop = im.crop((x, y, x + cw, y + ch))
        out.append(ImageOps.mirror(crop) if rng.random() < 0.5 else crop)
    return out


def crop_embeddings(corpus: Corpus) -> dict[str, np.ndarray]:
    """Embedding cua anh cat, cache trong artifacts/<model>/crop_emb*.npy."""
    keys = {"b32": (configs.RETRIEVAL_MODEL, "crop_emb.npy", True),
            "b32_raw": (configs.RETRIEVAL_MODEL, "crop_emb_raw.npy", False),
            "l14": (configs.RERANK_MODEL, "crop_emb.npy", True)}
    paths = {k: configs.ARTIFACTS / m / f for k, (m, f, _) in keys.items()}
    if not all(p.exists() for p in paths.values()):
        from src.encoders import load_encoder
        crops = random_crops(load_images(corpus))
        for model in (configs.RETRIEVAL_MODEL, configs.RERANK_MODEL):
            enc = load_encoder(model)
            for k, (m, _, norm) in keys.items():
                if m == model:
                    np.save(paths[k], enc.encode_images(crops, progress=True, normalize=norm))
    return {k: np.load(p) for k, p in paths.items()}


def fuse(v_img: np.ndarray, v_txt: np.ndarray, alpha: float) -> np.ndarray:
    q = alpha * v_img + (1.0 - alpha) * v_txt
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def main() -> None:
    corpus = Corpus.load(configs.ARTIFACTS / configs.RETRIEVAL_MODEL / "corpus.json")
    gt = np.asarray(corpus.caption_image_idx)
    img_b, cap_b = load_embeddings(configs.RETRIEVAL_MODEL, corpus)
    img_l, cap_l = load_embeddings(configs.RERANK_MODEL, corpus)
    crop = {k: v[gt] for k, v in crop_embeddings(corpus).items()}  # anh truy van cua tung caption
    sim_bm = bm25_sim(corpus)
    N, K, C = configs.RERANK_N, configs.RRF_K, configs.N_CAND

    def full(q_b, q_l, use_bm25=True):
        d = q_b @ img_b.T
        base = rrf_sim(d, sim_bm, k=K, n_cand=C) if use_bm25 else d
        return d, rerank_sim(base, q_l @ img_l.T, N)

    dense, pipe = {}, {}
    d, p = full(cap_b, cap_l)
    dense["chỉ chữ"], pipe["chỉ chữ"] = retrieval_metrics(d, gt), retrieval_metrics(p, gt)
    for a in ALPHAS:
        d, p = full(fuse(crop["b32"], cap_b, a), fuse(crop["l14"], cap_l, a))
        dense[f"alpha={a}"], pipe[f"alpha={a}"] = retrieval_metrics(d, gt), retrieval_metrics(p, gt)
    d, p = full(crop["b32"], crop["l14"], use_bm25=False)
    dense["chỉ ảnh"], pipe["chỉ ảnh"] = retrieval_metrics(d, gt), retrieval_metrics(p, gt)

    # tron vector tho (khong L2-normalize tung ben truoc khi cong)
    _, cap_raw = raw_embeddings(configs.RETRIEVAL_MODEL, corpus)
    raw_rows = {"alpha=0.5, chuẩn hoá từng bên": dense["alpha=0.5"],
                "alpha=0.5, trộn vector thô": retrieval_metrics(fuse(crop["b32_raw"], cap_raw, 0.5) @ img_b.T, gt)}
    n_txt, n_img = np.linalg.norm(cap_raw, axis=1).mean(), np.linalg.norm(crop["b32_raw"], axis=1).mean()

    report = [
        "# Truy vấn ảnh và chữ + ảnh", "",
        f"Ảnh truy vấn: vùng cắt ngẫu nhiên {int(CROP_SCALE[0]*100)}-{int(CROP_SCALE[1]*100)}% diện tích của ảnh đích "
        f"(lật ngang 50%, seed 0). Chữ: caption của ảnh đích, BM25 leave-one-out. {len(gt)} truy vấn; "
        f"dòng 'chỉ ảnh' ứng với {corpus.n_images} ảnh cắt, mỗi ảnh lặp 5 lần.", "",
        fmt("Dense CLIP B/32 (q = norm(alpha·v_ảnh + (1−alpha)·v_chữ))", dense),
        fmt(f"Pipeline đầy đủ (hybrid RRF + rerank L/14, N={N}); 'chỉ ảnh' không có BM25", pipe),
        fmt("Trộn vector khi chưa chuẩn hoá từng bên, dense B/32", raw_rows),
        f"Độ dài trung bình vector thô: chữ {n_txt:.2f}, ảnh cắt {n_img:.2f}.\n",
    ]
    out = RESULTS / "multimodal.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
