#!/usr/bin/env python3
"""Thong ke dataset va cac case dinh tinh. Ghi vao evaluation/results/:
dataset_stats.md, caption_length.png, qualitative.md, good_cases.png, bad_cases.png,
strategy_cases.md, strategy_cases.png
"""
from __future__ import annotations

import argparse, re, sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import configs
from src.dataset import Corpus
from src.sparse import STOP

RESULTS = Path(__file__).parent / "results"

# 3 cau de + 5 cau kho (dem so, quan he khong gian, phu dinh, chu trong anh)
PROBE_QUERIES = [
    ("two dogs playing in the snow",            "dễ, chủ đề phổ biến"),
    ("a surfer riding a big wave",              "dễ, đối tượng và cảnh rõ"),
    ("a man in a red shirt climbing a rock",    "trung bình, màu + hành động"),
    ("exactly three people sitting on a bench", "khó, đếm số"),
    ("a dog to the LEFT of a child",            "khó, quan hệ không gian"),
    ("a street with no cars at all",            "khó, phủ định"),
    ("a sign that reads STOP",                  "khó, chữ trong ảnh"),
    ("a person NOT wearing a helmet",           "khó, phủ định"),
]

# truy van de thay khac biet BM25 vs dense: tu hiem, dien dat khac tu
STRATEGY_QUERIES = [
    "two Dobermans playing on a lawn",
    "a dog catching a frisbee",
    "someone riding a board with wheels down the street",
    "kids splashing around in a pool",
    "a street with no cars at all",
    "a man in a red shirt climbing a rock",
]
STRATEGIES = {"BM25": "bm25", "dense": "dense", "hybrid": "hybrid", "hybrid+rerank": "hybrid_rerank"}


def first_caption(hit) -> str:
    return (hit.captions[0] if hit.captions else "").replace("|", "\\|")


def dataset_stats(corpus: Corpus) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lens = [len(c.split()) for c in corpus.captions]
    words = [w for c in corpus.captions for w in re.findall(r"[a-z']+", c.lower())
             if w not in STOP and len(w) > 2]
    lines = [
        "# Dataset Flickr8k", "",
        f"- nguồn: `{configs.DATASET_NAME}` (HuggingFace), split `{corpus.split}`",
        f"- số ảnh: {corpus.n_images:,}",
        f"- số caption: {corpus.n_captions:,} ({corpus.n_captions / corpus.n_images:.1f}/ảnh)",
        "", "## Độ dài caption (số từ)", "", "| | |", "|---|---|",
        f"| trung bình | {np.mean(lens):.1f} |",
        f"| trung vị | {np.median(lens):.0f} |",
        f"| độ lệch chuẩn | {np.std(lens):.1f} |",
        f"| nhỏ nhất | {min(lens)} |",
        f"| lớn nhất | {max(lens)} |",
        f"| p90 | {np.percentile(lens, 90):.0f} |",
        "", "## 25 từ phổ biến nhất", "", "| từ | tần số |", "|---|---|",
        *[f"| {w} | {c} |" for w, c in Counter(words).most_common(25)], "",
    ]
    (RESULTS / "dataset_stats.md").write_text("\n".join(lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(lens, bins=range(0, max(lens) + 2), color="#6c8cff", edgecolor="white", linewidth=.4)
    ax.axvline(np.mean(lens), color="#e0525f", ls="--", lw=1.6, label=f"trung binh {np.mean(lens):.1f} tu")
    ax.set_xlabel("so tu"); ax.set_ylabel("so caption"); ax.set_title("Do dai caption Flickr8k")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(RESULTS / "caption_length.png", dpi=150); plt.close(fig)


def image_grid(rows, fname: str, title: str, col_titles=None) -> None:
    """rows: [(nhan hang, [duong dan anh | None, ...])]"""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    n_cols = max(len(paths) for _, paths in rows)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(11, 2.5 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (label, paths) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]; ax.axis("off")
            if c < len(paths) and paths[c]:
                ax.imshow(Image.open(configs.ARTIFACTS / paths[c]))
            if col_titles and r == 0:
                ax.set_title(col_titles[c], fontsize=9)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(label[:38], fontsize=7.5, rotation=0, ha="right", va="center", labelpad=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(); fig.savefig(RESULTS / fname, dpi=130, bbox_inches="tight"); plt.close(fig)


def qualitative(engine, k: int = 5) -> None:
    lines = ["# Case tốt / case xấu (nhánh dense)", "",
             f"Model: {engine.encoder.label}, strategy=dense.", ""]
    grid = []
    for q, note in PROBE_QUERIES:
        hits, timing = engine.search(text=q, k=k, strategy="dense")
        lines += [f"## \"{q}\"", "", f"({note}; {timing['total']:.0f} ms)", "",
                  "| # | điểm | caption của ảnh trả về |", "|---|---|---|",
                  *[f"| {h.rank} | {h.score:.4f} | {first_caption(h)[:95]} |" for h in hits], ""]
        grid.append((q, [h.image_path for h in hits[:4]]))
    (RESULTS / "qualitative.md").write_text("\n".join(lines), encoding="utf-8")
    image_grid(grid[:3], "good_cases.png", "Case tot")
    image_grid(grid[3:7], "bad_cases.png", "Case xau")


def strategy_compare(engine, k: int = 3) -> None:
    engine.search(text="warm up", k=1, strategy="hybrid_rerank")  # nap L/14 truoc khi do ms
    lines = ["# So sánh 4 cấu hình trên cùng truy vấn", "",
             f"Rerank: CLIP ViT-L/14, N={configs.RERANK_N}. Mỗi ô là caption đầu tiên của ảnh top-1..3.", ""]
    grid = []
    for q in STRATEGY_QUERIES:
        lines += [f"## \"{q}\"", "", "| cấu hình | #1 | #2 | #3 | ms |", "|---|---|---|---|---|"]
        top1 = []
        for name, strategy in STRATEGIES.items():
            hits, timing = engine.search(text=q, k=k, strategy=strategy)
            cells = [first_caption(h)[:55] for h in hits] + [""] * (k - len(hits))
            lines.append(f"| {name} | " + " | ".join(cells) + f" | {timing['total']:.0f} |")
            top1.append(hits[0].image_path if hits else None)
        lines.append("")
        grid.append((q, top1))
    (RESULTS / "strategy_cases.md").write_text("\n".join(lines), encoding="utf-8")
    image_grid(grid, "strategy_cases.png", "Top-1 cua tung cau hinh", col_titles=list(STRATEGIES))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-strategy", action="store_true")
    args = ap.parse_args()

    from src.search import SearchEngine
    engine = SearchEngine()
    if not args.only_strategy:
        dataset_stats(engine.corpus)
        qualitative(engine)
    strategy_compare(engine)
    print(f"-> {RESULTS}")


if __name__ == "__main__":
    main()
