"""CLIP: nhung anh va text vao cung mot khong gian, L2-normalize."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import configs


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _features(out) -> torch.Tensor:
    # transformers 5.x tra ve ModelOutput, 4.x tra ve Tensor
    return out if isinstance(out, torch.Tensor) else out.pooler_output


class ClipEncoder:
    def __init__(self, key: str):
        from transformers import CLIPModel, CLIPProcessor

        self.key = key
        self.label = configs.MODELS[key]["label"]
        self.device = pick_device()
        hf_id = configs.MODELS[key]["hf_id"]
        self.model = CLIPModel.from_pretrained(hf_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(hf_id)
        self.dim = int(self.model.config.projection_dim)

    def _to_numpy(self, feats: torch.Tensor, normalize: bool = True) -> np.ndarray:
        # normalize=False chi dung cho ablation L2
        return (F.normalize(feats, dim=-1) if normalize else feats).cpu().float().numpy()

    @torch.no_grad()
    def encode_images(self, images, progress: bool = False, normalize: bool = True) -> np.ndarray:
        bs = configs.IMAGE_BATCH_SIZE
        chunks = []
        for i in tqdm(range(0, len(images), bs), desc=f"{self.key} anh", disable=not progress):
            inputs = self.processor(images=images[i:i + bs], return_tensors="pt").to(self.device)
            chunks.append(self._to_numpy(_features(self.model.get_image_features(**inputs)), normalize))
        return np.vstack(chunks)

    @torch.no_grad()
    def encode_texts(self, texts, progress: bool = False, normalize: bool = True) -> np.ndarray:
        bs = configs.TEXT_BATCH_SIZE
        chunks = []
        for i in tqdm(range(0, len(texts), bs), desc=f"{self.key} text", disable=not progress):
            inputs = self.processor(text=texts[i:i + bs], return_tensors="pt", padding=True,
                                    truncation=True, max_length=77).to(self.device)
            chunks.append(self._to_numpy(_features(self.model.get_text_features(**inputs)), normalize))
        return np.vstack(chunks)


def load_encoder(key: str) -> ClipEncoder:
    return ClipEncoder(key)
