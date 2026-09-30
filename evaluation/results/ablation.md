# Ket qua danh gia

Text->image tren split test cua Flickr8k (1000 anh, 5000 caption). Moi caption la mot truy van, dung neu anh goc nam trong top-K. R@K theo %, nDCG@10 voi mot tai lieu dung, MedR la thu hang trung vi cua anh dung.


### Bang chinh

| Cau hinh                                            |       R@1 |       R@5 |      R@10 |       MRR |   nDCG@10 |      MedR |  ms/query | ms/query (offline) |
|-----------------------------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| BM25 tren caption (LOO)                             |     43.90 |     69.78 |     77.92 |    0.5566 |    0.6039 |      2.00 |      3.60 |         - |
| dense CLIP ViT-B/32                                 |     55.16 |     81.68 |     89.46 |    0.6701 |    0.7203 |      1.00 |      8.60 |         - |
| hybrid RRF (dense + BM25)                           |     59.54 |     83.40 |     90.58 |    0.7033 |    0.7487 |      1.00 |     11.70 |         - |
| dense CLIP ViT-L/14                                 |     62.14 |     87.16 |     93.38 |    0.7305 |    0.7772 |      1.00 |     11.00 |         - |
| hybrid + rerank L/14, N=50                          |     62.66 |     87.84 |     94.18 |    0.7370 |    0.7844 |      1.00 |     22.10 |         - |
| hybrid RRF (1000 truy van)                          |     61.20 |     84.10 |     91.20 |    0.7168 |    0.7605 |      1.00 |         - |         - |
| hybrid + rerank L/14, N=20 (1000 truy van)          |     64.30 |     87.70 |     93.40 |    0.7439 |    0.7876 |      1.00 |         - |         - |
| hybrid + rerank cross-encoder, N=20 (1000 truy van) |     58.80 |     82.30 |     90.80 |    0.6977 |    0.7446 |      1.00 |         - |     79.10 |

BM25 leave-one-out: bo caption dang lam truy van khoi tai lieu. Hybrid gop top-100 cua dense va BM25 bang RRF (k=60). Rerank xep lai top-N cua hybrid bang CLIP L/14. ms/query do online qua SearchEngine, trung vi 50 truy van sau warm-up.

### A. So ung vien dua vao rerank L/14

| Cau hinh |       R@1 |       R@5 |      R@10 |       MRR |   nDCG@10 |      MedR |
|--------|-----------|-----------|-----------|-----------|-----------|-----------|
| N=10   |     64.52 |     86.98 |     90.58 |    0.7445 |    0.7807 |      1.00 |
| N=20   |     63.84 |     87.98 |     93.08 |    0.7439 |    0.7872 |      1.00 |
| N=50   |     62.66 |     87.84 |     94.18 |    0.7370 |    0.7844 |      1.00 |
| N=100  |     62.24 |     87.34 |     93.72 |    0.7325 |    0.7796 |      1.00 |
| N=1000 |     62.14 |     87.16 |     93.38 |    0.7305 |    0.7772 |      1.00 |


### B. Hang so k cua RRF

| Cau hinh |       R@1 |       R@5 |      R@10 |       MRR |   nDCG@10 |      MedR |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|
| k=10  |     60.60 |     86.86 |     93.04 |    0.7197 |    0.7680 |      1.00 |
| k=30  |     59.78 |     84.06 |     91.34 |    0.7077 |    0.7538 |      1.00 |
| k=60  |     59.54 |     83.40 |     90.58 |    0.7033 |    0.7487 |      1.00 |
| k=100 |     59.48 |     83.14 |     90.14 |    0.7019 |    0.7465 |      1.00 |


### C. So ung vien moi nhanh truoc khi gop

| Cau hinh   |       R@1 |       R@5 |      R@10 |       MRR |   nDCG@10 |      MedR |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|
| n_cand=20  |     59.56 |     84.36 |     92.64 |    0.7069 |    0.7572 |      1.00 |
| n_cand=50  |     59.52 |     83.20 |     90.02 |    0.7029 |    0.7465 |      1.00 |
| n_cand=100 |     59.54 |     83.40 |     90.58 |    0.7033 |    0.7487 |      1.00 |
| n_cand=300 |     59.52 |     83.38 |     90.68 |    0.7035 |    0.7489 |      1.00 |


### D. BM25 co / khong bo stopword

| Cau hinh     |       R@1 |       R@5 |      R@10 |       MRR |   nDCG@10 |      MedR |
|--------------|-----------|-----------|-----------|-----------|-----------|-----------|
| bo stopword  |     43.90 |     69.78 |     77.92 |    0.5566 |    0.6039 |      2.00 |
| giu stopword |     42.58 |     67.48 |     75.72 |    0.5402 |    0.5857 |      2.00 |


### E. L2-normalize (cosine) vs dot product tren vector CLIP tho, nhanh dense

| Cau hinh                     |       R@1 |       R@5 |      R@10 |       MRR |   nDCG@10 |      MedR |
|------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|
| co L2-norm (cosine)          |     55.16 |     81.68 |     89.46 |    0.6701 |    0.7203 |      1.00 |
| khong norm (dot product tho) |     47.14 |     75.74 |     85.64 |    0.6008 |    0.6569 |      2.00 |

Vector tho la output get_image_features / get_text_features truoc F.normalize. Do dai vector anh: trung binh 10.50, min 8.94, max 12.24.
