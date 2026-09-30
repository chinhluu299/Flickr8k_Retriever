#!/usr/bin/env python3
"""Do do tre online qua SearchEngine (encode + tim + rerank): trung vi tren 50 truy
van, bo lan dau (warm-up). In JSON {ten cau hinh: ms} ra stdout cho evaluate.py doc."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import configs
from src.search import SearchEngine

# ten trung voi ten dong trong bang chinh cua evaluate.py
CONFIGS = {
    "BM25 trên caption (LOO)": dict(strategy="bm25"),
    "dense CLIP ViT-B/32": dict(strategy="dense"),
    "hybrid RRF (dense + BM25)": dict(strategy="hybrid"),
    f"hybrid + rerank L/14, N={configs.RERANK_N}": dict(strategy="hybrid_rerank"),
}


def median_ms(engine: SearchEngine, queries: list[str], **kw) -> float:
    engine.search(text=queries[0], k=10, **kw)
    return round(float(np.median([engine.search(text=q, k=10, **kw)[1]["total"] for q in queries])), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    eng = SearchEngine(configs.RETRIEVAL_MODEL)
    rng = np.random.default_rng(0)
    qs = [eng.corpus.captions[i] for i in rng.choice(eng.corpus.n_captions, args.n, replace=False)]
    out = {name: median_ms(eng, qs, **kw) for name, kw in CONFIGS.items()}
    out["dense CLIP ViT-L/14"] = median_ms(SearchEngine(configs.RERANK_MODEL), qs, strategy="dense")
    for name, ms in out.items():
        print(f"{name}: {ms} ms", file=sys.stderr)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
