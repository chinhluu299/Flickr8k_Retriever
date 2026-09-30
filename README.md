# Tìm kiếm ảnh đa phương thức hai tầng (Flickr8k)

Tìm ảnh bằng câu chữ, bằng một tấm ảnh, hoặc cả hai.

![Phương pháp](figures/proposed_method.png)

- **Tầng 1:** nhánh dense (CLIP ViT-B/32 + FAISS, so truy vấn với ảnh) và nhánh
  sparse (BM25 trên caption của ảnh, chỉ khi truy vấn có chữ). Mỗi nhánh lấy
  top-100, gộp bằng RRF (k = 60).
- **Tầng 2:** chấm lại top-50 bằng CLIP ViT-L/14. Embedding ảnh L/14 được tính sẵn
  lúc đánh chỉ mục, nên lúc chạy chỉ cần encode truy vấn.
- Truy vấn chữ + ảnh: `q = norm(α·v_ảnh + (1−α)·v_chữ)`.

## Cài đặt và chạy

```bash
pip install -r requirements.txt

# đánh chỉ mục (tải Flickr8k test, encode 1000 ảnh)
python3 build_index.py --model clip-vit-b32   # tầng 1
python3 build_index.py --model clip-vit-l14   # tầng 2

# chạy demo tại http://127.0.0.1:8000
python3 -m uvicorn app.api:app --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/api/search -F "query=a dog catching a frisbee" -F "k=5"
curl -X POST http://127.0.0.1:8000/api/search -F "file=@anh.jpg" -F "query=in the snow" -F "alpha=0.6"
```

Tham số `strategy` (`dense`, `bm25`, `hybrid`, `hybrid_rerank`) cho phép chạy từng
thành phần riêng để so sánh; mặc định là `hybrid_rerank`.

## Cấu trúc

```
configs.py         đường dẫn, model, siêu tham số pipeline
build_index.py     đánh chỉ mục offline
src/
  dataset.py       tải Flickr8k, lưu ảnh và caption
  encoders.py      CLIP image/text encoder, L2-normalize
  index.py         FAISS IndexFlatIP
  sparse.py        BM25 trên caption, điểm ảnh = max trên 5 caption
  fusion.py        RRF, gộp vector chữ + ảnh
  rerank.py        tầng 2: CLIP L/14
  search.py        SearchEngine: pipeline hai tầng
app/               FastAPI + giao diện web
figures/           sơ đồ phương pháp
evaluation/        đánh giá, ablation, phân tích case (xem bên dưới)
```

## Đánh giá

Giao thức text→image trên Flickr8k test (1000 ảnh, 5000 caption): mỗi caption là
một truy vấn, đúng nếu tìm ra ảnh gốc. BM25 chạy leave-one-out, tức là caption
đang làm truy vấn bị loại khỏi chỉ mục. Điểm bằng nhau được xếp như trong hệ thống, không ưu tiên ảnh đúng.

```bash
pip install -r evaluation/requirements.txt
python3 evaluation/evaluate.py --ablations --latency   # results/ablation.md
python3 evaluation/model_tradeoff.py                   # results/model_tradeoff.md
python3 evaluation/multimodal.py                       # results/multimodal.md: truy vấn ảnh, chữ + ảnh
python3 evaluation/analyze_cases.py                    # thống kê dataset, case định tính
```

| Cấu hình | R@1 | R@5 | R@10 | MRR | nDCG@10 | ms/query |
|---|---|---|---|---|---|---|
| BM25 trên caption | 43.90 | 69.78 | 77.92 | 0.557 | 0.604 | 3.6 |
| dense CLIP ViT-B/32 | 55.16 | 81.68 | 89.46 | 0.670 | 0.720 | 8.6 |
| hybrid RRF | 59.54 | 83.40 | 90.58 | 0.703 | 0.749 | 11.7 |
| dense CLIP ViT-L/14 | 62.14 | 87.16 | 93.38 | 0.731 | 0.777 | 11.0 |
| hybrid + rerank L/14, N=50 | **62.66** | **87.84** | **94.18** | **0.737** | **0.784** | 22.1 |

Bảng đầy đủ và các ablation (N, k của RRF, số ứng viên, stopword, L2-normalize)
nằm trong [evaluation/results/ablation.md](evaluation/results/ablation.md). Độ trễ
đo trên Apple Silicon (MPS), trung vị 50 truy vấn.

Dataset: mirror `jxie/flickr8k` trên HuggingFace. Bản gốc: Hodosh, Young,
Hockenmaier, "Framing Image Description as a Ranking Task", JAIR 2013.
