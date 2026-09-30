"""FastAPI cho SearchEngine. Chay: python3 -m uvicorn app.api:app --port 8000"""
from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image

import configs
from src.search import STRATEGIES, SearchEngine

app = FastAPI(title="Tim kiem anh da phuong thuc")
INDEX_HTML = Path(__file__).parent / "static" / "index.html"
_engine: SearchEngine | None = None


def get_engine() -> SearchEngine:
    global _engine
    if _engine is None:
        try:
            _engine = SearchEngine()
        except FileNotFoundError as e:
            raise HTTPException(503, str(e))
    return _engine


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/api/info")
def info():
    return get_engine().info()


@app.post("/api/search")
async def search(
    query: str | None = Form(None),
    file: UploadFile | None = File(None),
    k: int = Form(configs.DEFAULT_TOP_K),
    alpha: float = Form(0.5),
    strategy: str = Form("hybrid_rerank"),
    rerank_n: int = Form(configs.RERANK_N),
):
    """Nhan chu, anh, hoac ca hai; tra ve top-k anh. alpha = trong so anh khi co ca hai."""
    if strategy not in STRATEGIES:
        raise HTTPException(400, f"strategy phai la mot trong {STRATEGIES}")
    text = (query or "").strip() or None
    img = None
    if file is not None:
        raw = await file.read()
        if raw:
            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                raise HTTPException(400, "khong doc duoc anh, dung JPG/PNG")
    if text is None and img is None:
        raise HTTPException(400, "nhap cau truy van hoac chon anh")

    try:
        hits, timing = get_engine().search(
            text=text, image=img, k=min(max(k, 1), configs.MAX_TOP_K),
            alpha=min(max(alpha, 0.0), 1.0), strategy=strategy, rerank_n=max(1, rerank_n),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    mode = "text+image" if text and img else ("image" if img else "text")
    parts = ([f"“{text}”"] if text else []) + ([file.filename or "anh tai len"] if img else [])
    return {
        "query": " + ".join(parts),
        "mode": mode,
        "strategy": strategy,
        "alpha": alpha if mode == "text+image" else None,
        "elapsed_ms": timing["total"],
        "timing": timing,
        "count": len(hits),
        "results": [
            {"rank": h.rank, "score": round(h.score, 4), "image_url": f"/images/{h.image_path}",
             "captions": h.captions, "image_idx": h.image_idx}
            for h in hits
        ],
    }


@app.get("/images/{path:path}")
def serve_image(path: str):
    full = (configs.ARTIFACTS / path).resolve()
    # chan path traversal
    if not full.is_relative_to(configs.ARTIFACTS.resolve()) or not full.is_file():
        raise HTTPException(404, "khong tim thay anh")
    return FileResponse(full)
