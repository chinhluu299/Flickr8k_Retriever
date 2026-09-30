#!/usr/bin/env python3
"""Danh chi muc offline: tai corpus, encode anh bang CLIP, luu embedding va FAISS index.

  python3 build_index.py --model clip-vit-b32   # tang 1
  python3 build_index.py --model clip-vit-l14   # tang 2 (rerank)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import configs
from src.dataset import build_corpus, load_images
from src.encoders import load_encoder
from src.index import VectorIndex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=configs.RETRIEVAL_MODEL, choices=list(configs.MODELS))
    ap.add_argument("--split", default=configs.SPLIT)
    ap.add_argument("--limit", type=int, default=None, help="chi lay n anh dau de chay thu")
    args = ap.parse_args()

    out = configs.ARTIFACTS / args.model
    out.mkdir(parents=True, exist_ok=True)

    corpus = build_corpus(args.split, limit=args.limit)
    corpus.save(out / "corpus.json")
    print(f"{corpus.n_images} anh, {corpus.n_captions} caption")

    enc = load_encoder(args.model)
    t0 = time.time()
    img_emb = enc.encode_images(load_images(corpus), progress=True)
    t_img = time.time() - t0
    np.save(out / "image_emb.npy", img_emb)
    VectorIndex.build(img_emb).save(out / "image.faiss")

    meta = {
        "model": args.model, "hf_id": configs.MODELS[args.model]["hf_id"], "dim": enc.dim,
        "split": args.split, "n_images": corpus.n_images, "n_captions": corpus.n_captions,
        "device": enc.device, "encode_seconds": {"images": round(t_img, 2)},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{img_emb.shape} trong {t_img:.1f}s -> {out}")


if __name__ == "__main__":
    main()
