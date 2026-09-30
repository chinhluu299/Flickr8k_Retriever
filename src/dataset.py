"""Flickr8k tu HuggingFace: luu anh ra dia, giu caption dang phang."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import configs


@dataclass
class Corpus:
    """captions[j] thuoc anh caption_image_idx[j]."""
    image_paths: list[str]
    captions: list[str]
    caption_image_idx: list[int]
    split: str

    @property
    def n_images(self) -> int:
        return len(self.image_paths)

    @property
    def n_captions(self) -> int:
        return len(self.captions)

    def captions_of(self, image_idx: int) -> list[str]:
        return [c for c, m in zip(self.captions, self.caption_image_idx) if m == image_idx]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Corpus":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def build_corpus(split: str = configs.SPLIT, limit: int | None = None) -> Corpus:
    from datasets import load_dataset
    from tqdm import tqdm

    image_dir = configs.IMAGE_DIR / split
    image_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(configs.DATASET_NAME, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    image_paths, captions, owner = [], [], []
    for idx, row in enumerate(tqdm(ds, desc="luu anh", unit="img")):
        dest = image_dir / f"{split}_{idx:05d}.jpg"
        if not dest.exists():
            row["image"].convert("RGB").save(dest, "JPEG", quality=90)
        image_paths.append(str(dest.relative_to(configs.ARTIFACTS)))
        for i in range(configs.CAPTIONS_PER_IMAGE):
            captions.append(" ".join(row[f"caption_{i}"].split()))
            owner.append(idx)
    return Corpus(image_paths, captions, owner, split)


def load_images(corpus: Corpus):
    from PIL import Image
    return [Image.open(configs.ARTIFACTS / p).convert("RGB") for p in corpus.image_paths]
