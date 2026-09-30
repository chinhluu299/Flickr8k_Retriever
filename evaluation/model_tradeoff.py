#!/usr/bin/env python3
"""Bang trade-off model: dung luong, thoi gian build, do tre encode truy van, R@1.
Ghi evaluation/results/model_tradeoff.md."""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import configs
from src.dataset import Corpus
from src.encoders import load_encoder, pick_device

RESULTS = Path(__file__).parent / "results"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
CE_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY = "two dogs playing in the snow"


def dir_mb(hf_id: str) -> float | None:
    d = HF_CACHE / ("models--" + hf_id.replace("/", "--"))
    if not d.exists():
        return None
    # cache co the chua ca pytorch_model.bin lan model.safetensors cua cung trong so: chi dem mot ban
    for name in ("model.safetensors", "pytorch_model.bin"):
        files = list(d.glob(f"snapshots/*/{name}"))
        if files:
            return files[0].resolve().stat().st_size / 1e6
    return None


def r_at_1(model_key: str) -> float | None:
    d = configs.ARTIFACTS / model_key
    if not (d / "caption_emb.npy").exists():
        return None  # chay evaluate.py truoc de cache embedding truy van
    img, cap = np.load(d / "image_emb.npy"), np.load(d / "caption_emb.npy")
    gt = np.asarray(Corpus.load(d / "corpus.json").caption_image_idx)
    return float((np.argmax(cap @ img.T, axis=1) == gt).mean() * 100)


def query_ms(fn) -> float:
    fn()
    ts = []
    for _ in range(10):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def main():
    roles = {configs.RETRIEVAL_MODEL: "tầng 1 (dense)", configs.RERANK_MODEL: "tầng 2 (rerank)"}
    rows = []
    for key, cfg in configs.MODELS.items():
        meta = json.loads((configs.ARTIFACTS / key / "meta.json").read_text())
        enc = load_encoder(key)
        ms_img = meta["encode_seconds"]["images"] / meta["n_images"] * 1000
        rows.append((cfg["label"], cfg["dim"], dir_mb(cfg["hf_id"]), ms_img,
                     query_ms(lambda: enc.encode_texts([QUERY])), r_at_1(key), roles.get(key, "")))

    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(CE_ID, device=pick_device())
    pairs = [(QUERY, "A dog runs in the snow .")] * 50
    rows.append(("cross-encoder ms-marco-MiniLM-L-6", None, dir_mb(CE_ID), None,
                 query_ms(lambda: ce.predict(pairs, show_progress_bar=False)), None,
                 "đối chứng tầng 2, text-text, 50 cặp"))

    f = lambda v, spec: "-" if v is None else format(v, spec)
    lines = ["# Trade-off giữa các model", "",
             "| model | dim | dung lượng (MB) | build ms/ảnh | encode truy vấn (ms) | R@1 text→image | vai trò |",
             "|---|---|---|---|---|---|---|"]
    for lab, dim, mb, ms_img, ms_q, r1, role in rows:
        lines.append(f"| {lab} | {f(dim, 'd')} | {f(mb, '.0f')} | {f(ms_img, '.1f')} | {f(ms_q, '.1f')} | {f(r1, '.2f')} | {role} |")
    lines += ["", f"Độ trễ đo với batch 1 trên {pick_device()}, trung vị 10 lần sau warm-up. "
              "Dung lượng là kích thước file trọng số của model.", ""]
    (RESULTS / "model_tradeoff.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {RESULTS / 'model_tradeoff.md'}")


if __name__ == "__main__":
    main()
