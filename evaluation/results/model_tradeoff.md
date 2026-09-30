# Trade-off giua cac model

| model | dim | dung luong (MB) | build ms/anh | encode truy van (ms) | R@1 text->image | vai tro |
|---|---|---|---|---|---|---|
| CLIP ViT-B/32 | 512 | 605 | 6.0 | 7.8 | 55.16 | tang 1 (dense) |
| CLIP ViT-L/14 | 768 | 1711 | 55.2 | 11.7 | 62.14 | tang 2 (rerank) |
| cross-encoder ms-marco-MiniLM-L-6 | - | 92 | - | 31.4 | - | doi chung tang 2, text-text, 50 cap |

Do tre do voi batch 1 tren mps, trung vi 10 lan sau warm-up. Dung luong la thu muc model trong cache HuggingFace.
