from pathlib import Path

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
IMAGE_DIR = ARTIFACTS / "images"

DATASET_NAME = "jxie/flickr8k"
SPLIT = "test"
CAPTIONS_PER_IMAGE = 5

MODELS = {
    "clip-vit-b32": {"hf_id": "openai/clip-vit-base-patch32", "dim": 512, "label": "CLIP ViT-B/32"},
    "clip-vit-l14": {"hf_id": "openai/clip-vit-large-patch14", "dim": 768, "label": "CLIP ViT-L/14"},
}
RETRIEVAL_MODEL = "clip-vit-b32"   # tang 1
RERANK_MODEL = "clip-vit-l14"      # tang 2

IMAGE_BATCH_SIZE = 64
TEXT_BATCH_SIZE = 256

# pipeline hai tang
N_CAND = 100     # so ung vien moi nhanh truoc khi gop
RRF_K = 60
RERANK_N = 50    # so ung vien dua vao tang 2

DEFAULT_TOP_K = 12
MAX_TOP_K = 50
