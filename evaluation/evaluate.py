#!/usr/bin/env python3
"""Danh gia text->image tren Flickr8k test: moi caption la mot truy van, dung neu
tim ra anh goc cua no. Ghi evaluation/results/ablation.md.

  python3 evaluation/evaluate.py
  python3 evaluation/evaluate.py --ablations --latency
"""
from __future__ import annotations

import os
# torch va faiss moi ben mang mot libomp rieng tren macOS; gioi han thread truoc khi
# import de tranh OMP Error #15. Metric tinh bang numpy nen o day khong import faiss.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import configs
from src.dataset import Corpus
from src.sparse import BM25Index

RESULTS = Path(__file__).parent / "results"
KS = (1, 5, 10)


def retrieval_metrics(sim: np.ndarray, gt: np.ndarray) -> dict:
    """sim[q, i]: diem anh i cho truy van q; gt[q]: anh dung.
    Diem bang nhau thi anh co chi so nho xep truoc (nhu argsort stable trong SearchEngine),
    khong mac dinh cho anh dung dung dau."""
    ranks = ranks_of(sim)[np.arange(len(gt)), gt]
    out = {f"R@{k}": float((ranks <= k).mean() * 100) for k in KS}
    out["MRR"] = float((1.0 / ranks).mean())
    out["nDCG@10"] = float(np.where(ranks <= 10, 1.0 / np.log2(ranks + 1), 0.0).mean())
    out["MedR"] = float(np.median(ranks))
    return out


def load_embeddings(model_key: str, corpus: Corpus):
    """(image_emb, query_emb). Truy van la cac caption, encode 1 lan roi cache."""
    d = configs.ARTIFACTS / model_key
    img = np.load(d / "image_emb.npy")
    cache = d / "caption_emb.npy"
    if not cache.exists():
        from src.encoders import load_encoder
        np.save(cache, load_encoder(model_key).encode_texts(corpus.captions, progress=True))
    return img, np.load(cache)


def raw_embeddings(model_key: str, corpus: Corpus):
    """(image, caption) chua L2-normalize, cho ablation E. Cache rieng."""
    d = configs.ARTIFACTS / model_key
    fi, fc = d / "image_emb_raw.npy", d / "caption_emb_raw.npy"
    if not (fi.exists() and fc.exists()):
        from src.dataset import load_images
        from src.encoders import load_encoder
        enc = load_encoder(model_key)
        np.save(fi, enc.encode_images(load_images(corpus), progress=True, normalize=False))
        np.save(fc, enc.encode_texts(corpus.captions, progress=True, normalize=False))
    return np.load(fi), np.load(fc)


def ranks_of(sim: np.ndarray) -> np.ndarray:
    """rank (1-based) cua moi anh trong tung hang."""
    order = np.argsort(-sim, axis=1, kind="stable")
    r = np.empty_like(order)
    r[np.arange(sim.shape[0])[:, None], order] = np.arange(sim.shape[1])[None, :] + 1
    return r


def bm25_sim(corpus: Corpus, drop_stop: bool = True, leave_one_out: bool = True) -> np.ndarray:
    """sim[q, i] cua BM25. leave-one-out: bo caption q khoi tai lieu khi chay truy van q."""
    bm = BM25Index(corpus.captions, corpus.caption_image_idx, corpus.n_images, drop_stop=drop_stop)
    sim = np.zeros((corpus.n_captions, corpus.n_images), "float32")
    for qi, cap in enumerate(corpus.captions):
        sim[qi] = bm.image_scores(cap, exclude_caption=qi if leave_one_out else None)
    return sim


def rrf_sim(dense: np.ndarray, bm25: np.ndarray, k: int = 60, n_cand: int = 100) -> np.ndarray:
    """RRF tren top n_cand cua moi nhanh. Nhanh BM25 chi tinh anh co diem > 0.
    Hoa diem thi xu ly nhu src.fusion.rrf: anh trong list dense truoc (theo hang dense),
    sau do anh chi co o BM25 (theo hang BM25)."""
    r_d, r_b = ranks_of(dense), ranks_of(bm25)
    in_d, in_b = r_d <= n_cand, (r_b <= n_cand) & (bm25 > 0)
    total = np.where(in_d, 1.0 / (k + r_d), 0.0) + np.where(in_b, 1.0 / (k + r_b), 0.0)
    tie = np.where(in_d, r_d, np.where(in_b, n_cand + r_b, 3 * n_cand))
    return total - 1e-9 * tie


def rerank_sim(base: np.ndarray, scores: np.ndarray, n: int) -> np.ndarray:
    """Top n theo base duoc xep lai theo scores (dat len tren), phan con lai giu base."""
    return np.where(ranks_of(base) <= n, 10.0 + scores, base)


def crossencoder_rerank_metrics(corpus: Corpus, base: np.ndarray, n: int, sample: int, seed: int = 0):
    """Doi chung: cross-encoder text-text cham (query, caption) cho top n ung vien,
    diem anh = max tren caption cua anh, tren `sample` truy van."""
    from sentence_transformers import CrossEncoder
    from src.encoders import pick_device
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=pick_device())
    owner = np.asarray(corpus.caption_image_idx)
    qidx = np.random.default_rng(seed).choice(corpus.n_captions, size=min(sample, corpus.n_captions), replace=False)
    r = ranks_of(base[qidx])
    out = base[qidx].copy()
    t0 = time.time()
    for row, qi in enumerate(qidx):
        cand = np.flatnonzero(r[row] <= n)
        pairs, owners = [], []
        for pos, img_i in enumerate(cand):
            for ci in np.flatnonzero(owner == img_i):
                if ci != qi:  # leave-one-out
                    pairs.append((corpus.captions[qi], corpus.captions[ci])); owners.append(pos)
        sc = np.asarray(ce.predict(pairs, batch_size=128, show_progress_bar=False), "float32")
        agg = np.full(len(cand), -np.inf, "float32")
        np.maximum.at(agg, np.asarray(owners), sc)
        out[row, cand] = 100.0 + agg
    m = retrieval_metrics(out, owner[qidx])
    m["ms/query (offline)"] = round((time.time() - t0) / len(qidx) * 1000, 1)
    return m, qidx


def measure_latency() -> dict:
    """Chay latency.py o process rieng (can ca torch lan faiss)."""
    out = subprocess.run([sys.executable, str(Path(__file__).parent / "latency.py")],
                         capture_output=True, text=True, timeout=1800)
    if out.returncode != 0:
        print("latency.py loi:", out.stderr.strip()[-400:])
        return {}
    return json.loads(out.stdout.strip().splitlines()[-1])


def fmt(title: str, rows: dict) -> str:
    cols = list(dict.fromkeys(c for r in rows.values() for c in r))
    w = max(len(k) for k in rows)
    lines = [f"| {'Cau hinh'.ljust(w)} | " + " | ".join(c.rjust(9) for c in cols) + " |",
             f"|{'-' * (w + 2)}|" + "|".join("-" * 11 for _ in cols) + "|"]
    for name, r in rows.items():
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(("-" if v is None else (f"{v:.4f}" if abs(v) < 1 and v != 0 else f"{v:.2f}")).rjust(9))
        lines.append(f"| {name.ljust(w)} | " + " | ".join(cells) + " |")
    return f"\n### {title}\n\n" + "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--ce-sample", type=int, default=1000, help="so truy van cho cross-encoder, 0 = bo")
    ap.add_argument("--latency", action="store_true", help="do ms/query online qua SearchEngine")
    ap.add_argument("--out", default=str(RESULTS / "ablation.md"))
    args = ap.parse_args()

    corpus = Corpus.load(configs.ARTIFACTS / configs.RETRIEVAL_MODEL / "corpus.json")
    gt = np.asarray(corpus.caption_image_idx)
    img_b, cap_b = load_embeddings(configs.RETRIEVAL_MODEL, corpus)
    img_l, cap_l = load_embeddings(configs.RERANK_MODEL, corpus)

    t0 = time.time()
    sim_bm = bm25_sim(corpus)
    print(f"BM25 leave-one-out: {time.time() - t0:.1f}s")
    sim_dense = cap_b @ img_b.T
    sim_l14 = cap_l @ img_l.T
    sim_h = rrf_sim(sim_dense, sim_bm, k=configs.RRF_K, n_cand=configs.N_CAND)

    rows = {
        "BM25 tren caption (LOO)": retrieval_metrics(sim_bm, gt),
        "dense CLIP ViT-B/32": retrieval_metrics(sim_dense, gt),
        "hybrid RRF (dense + BM25)": retrieval_metrics(sim_h, gt),
        "dense CLIP ViT-L/14": retrieval_metrics(sim_l14, gt),
        f"hybrid + rerank L/14, N={configs.RERANK_N}": retrieval_metrics(rerank_sim(sim_h, sim_l14, configs.RERANK_N), gt),
    }
    if args.ce_sample > 0:
        print(f"cross-encoder tren {args.ce_sample} truy van ...")
        ce, qidx = crossencoder_rerank_metrics(corpus, sim_h, 20, args.ce_sample)
        tag = f"({len(qidx)} truy van)"
        rows[f"hybrid RRF {tag}"] = retrieval_metrics(sim_h[qidx], gt[qidx])
        rows[f"hybrid + rerank L/14, N=20 {tag}"] = retrieval_metrics(rerank_sim(sim_h, sim_l14, 20)[qidx], gt[qidx])
        rows[f"hybrid + rerank cross-encoder, N=20 {tag}"] = ce
    if args.latency:
        for name, ms in measure_latency().items():
            if name in rows:
                rows[name]["ms/query"] = ms
    for name, m in rows.items():
        print(f"{name}: R@1={m['R@1']:.2f} nDCG@10={m['nDCG@10']:.4f}")

    report = [
        "# Ket qua danh gia", "",
        f"Text->image tren split {corpus.split} cua Flickr8k ({corpus.n_images} anh, {corpus.n_captions} caption). "
        "Moi caption la mot truy van, dung neu anh goc nam trong top-K. R@K theo %, nDCG@10 voi "
        "mot tai lieu dung, MedR la thu hang trung vi cua anh dung.", "",
        fmt("Bang chinh", rows),
        f"BM25 leave-one-out: bo caption dang lam truy van khoi tai lieu. Hybrid gop top-{configs.N_CAND} "
        f"cua dense va BM25 bang RRF (k={configs.RRF_K}). Rerank xep lai top-N cua hybrid bang CLIP L/14. "
        "ms/query do online qua SearchEngine, trung vi 50 truy van sau warm-up.",
    ]

    if args.ablations:
        report.append(fmt("A. So ung vien dua vao rerank L/14", {
            f"N={n}": retrieval_metrics(rerank_sim(sim_h, sim_l14, n), gt) for n in (10, 20, 50, 100, 1000)}))
        report.append(fmt("B. Hang so k cua RRF", {
            f"k={k}": retrieval_metrics(rrf_sim(sim_dense, sim_bm, k=k, n_cand=configs.N_CAND), gt)
            for k in (10, 30, 60, 100)}))
        report.append(fmt("C. So ung vien moi nhanh truoc khi gop", {
            f"n_cand={n}": retrieval_metrics(rrf_sim(sim_dense, sim_bm, k=configs.RRF_K, n_cand=n), gt)
            for n in (20, 50, 100, 300)}))
        report.append(fmt("D. BM25 co / khong bo stopword", {
            "bo stopword": retrieval_metrics(sim_bm, gt),
            "giu stopword": retrieval_metrics(bm25_sim(corpus, drop_stop=False), gt)}))
        img_raw, cap_raw = raw_embeddings(configs.RETRIEVAL_MODEL, corpus)
        n_img = np.linalg.norm(img_raw, axis=1)
        report.append(fmt("E. L2-normalize (cosine) vs dot product tren vector CLIP tho, nhanh dense", {
            "co L2-norm (cosine)": retrieval_metrics(sim_dense, gt),
            "khong norm (dot product tho)": retrieval_metrics(cap_raw @ img_raw.T, gt)}))
        report.append(f"Vector tho la output get_image_features / get_text_features truoc F.normalize. "
                      f"Do dai vector anh: trung binh {n_img.mean():.2f}, min {n_img.min():.2f}, max {n_img.max():.2f}.\n")

    Path(args.out).write_text("\n".join(report), encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
