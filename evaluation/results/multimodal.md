# Truy van anh va chu + anh

Anh truy van: vung cat ngau nhien 10-30% dien tich cua anh dich (lat ngang 50%, seed 0). Chu: caption cua anh dich, BM25 leave-one-out. 5000 truy van; dong 'chi anh' tuong ung 1000 anh cat, moi anh lap 5 lan.


### Dense CLIP B/32 (fuse q = norm(alpha*v_anh + (1-alpha)*v_chu))

| Cau hinh   |       R@1 |       R@5 |      R@10 |       MRR |   nDCG@10 |      MedR |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|
| chi chu    |     55.16 |     81.68 |     89.46 |    0.6701 |    0.7203 |      1.00 |
| alpha=0.25 |     83.98 |     96.32 |     98.42 |    0.8942 |    0.9157 |      1.00 |
| alpha=0.5  |     84.68 |     95.44 |     97.94 |    0.8932 |    0.9134 |      1.00 |
| alpha=0.75 |     77.96 |     90.60 |     94.72 |    0.8378 |    0.8626 |      1.00 |
| chi anh    |     71.50 |     85.40 |     89.90 |    0.7790 |    0.8050 |      1.00 |


### Pipeline day du (hybrid RRF + rerank L/14, N=50); 'chi anh' khong co BM25

| Cau hinh   |       R@1 |       R@5 |      R@10 |       MRR |   nDCG@10 |      MedR |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|
| chi chu    |     62.66 |     87.84 |     94.18 |    0.7370 |    0.7844 |      1.00 |
| alpha=0.25 |     85.30 |     97.28 |     98.74 |    0.9062 |    0.9259 |      1.00 |
| alpha=0.5  |     88.26 |     96.96 |     98.36 |    0.9206 |    0.9355 |      1.00 |
| alpha=0.75 |     83.02 |     94.42 |     96.44 |    0.8803 |    0.8997 |      1.00 |
| chi anh    |     74.90 |     88.20 |     91.40 |    0.8074 |    0.8311 |      1.00 |


### Tron vector khi chua chuan hoa tung ben, dense B/32

| Cau hinh                      |       R@1 |       R@5 |      R@10 |       MRR |   nDCG@10 |      MedR |
|-------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|
| alpha=0.5, chuan hoa tung ben |     84.68 |     95.44 |     97.94 |    0.8932 |    0.9134 |      1.00 |
| alpha=0.5, tron vector tho    |     82.24 |     93.72 |     96.76 |    0.8729 |    0.8947 |      1.00 |

Do dai trung binh vector tho: chu 7.90, anh cat 11.36.
