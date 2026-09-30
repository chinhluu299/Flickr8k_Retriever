# Trade-off giữa các model

| model | dim | dung lượng (MB) | build ms/ảnh | encode truy vấn (ms) | R@1 text→image | vai trò |
|---|---|---|---|---|---|---|
| CLIP ViT-B/32 | 512 | 605 | 6.0 | 7.8 | 55.16 | tầng 1 (dense) |
| CLIP ViT-L/14 | 768 | 1711 | 55.2 | 11.7 | 62.14 | tầng 2 (rerank) |
| cross-encoder ms-marco-MiniLM-L-6 | - | 92 | - | 31.4 | - | đối chứng tầng 2, text-text, 50 cặp |

Độ trễ đo với batch 1 trên mps, trung vị 10 lần sau warm-up. Dung lượng là kích thước file trọng số của model.
