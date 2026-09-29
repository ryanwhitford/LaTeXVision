"""PyTorch Dataset over the combined expression manifest
(src/data/build_expression_dataset.py), plus the single image->tensor path
shared with serving.

`image_to_tensor` is what both this Dataset and the API's transformer path
call -- raw image in, normalized model tensor out -- so training and serving
can't diverge (see experiments/audit/AUDIT_REPORT.md, Finding 1, for what
that costs when they do).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from src.data.expression_images import normalize_expression_image
from src.data.latex_tokenizer import PAD, TOKEN_TO_ID, encode
from src.data.preprocessing import HASY_MEAN, HASY_STD
from src.models.im2latex import TRUNK_STRIDE

BACKGROUND = (1.0 - HASY_MEAN) / HASY_STD  # normalized value of a white pixel


def normalized_array(image: Image.Image) -> np.ndarray | None:
    """Raw expression image -> normalized uint8 array (H, W), or None if blank."""
    normalized = normalize_expression_image(image)
    return None if normalized is None else np.asarray(normalized, dtype=np.uint8)


def array_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """uint8 (H, W) -> (1, H, W) float tensor with the classifier's normalization."""
    return torch.from_numpy((arr.astype(np.float32) / 255.0 - HASY_MEAN) / HASY_STD).unsqueeze(0)


def image_to_tensor(image: Image.Image) -> torch.Tensor | None:
    """Raw expression image -> (1, H, W) normalized float tensor, or None if blank."""
    arr = normalized_array(image)
    return None if arr is None else array_to_tensor(arr)


def load_manifest(path: Path, splits: list[str] | None = None, sources: list[str] | None = None) -> list[dict]:
    records = [json.loads(line) for line in open(path)]
    return [
        r for r in records
        if (splits is None or r["split"] in splits) and (sources is None or r["source"] in sources)
    ]


class ExpressionDataset(Dataset):
    def __init__(self, records: list[dict], root: Path, augment: bool = False, cache: bool = True) -> None:
        self.records = records
        self.root = root
        self.augment = augment
        # uint8 normalized arrays, not float tensors: ~4x less memory for a
        # 50k-image training set, and the same two shared functions
        # (normalized_array -> array_to_tensor) that serving composes.
        self._cache: dict[int, np.ndarray] = {} if cache else None

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, idx: int) -> torch.Tensor:
        arr = self._cache.get(idx) if self._cache is not None else None
        if arr is None:
            arr = normalized_array(Image.open(self.root / self.records[idx]["image"]))
            if arr is None:
                raise ValueError(f"blank image: {self.records[idx]['image']}")
            if self._cache is not None:
                self._cache[idx] = arr
        return array_to_tensor(arr)

    def __getitem__(self, idx: int):
        image = self._load(idx)
        if self.augment:
            image = _augment(image)
        return image, torch.tensor(encode(self.records[idx]["tokens"]), dtype=torch.long)

    def sizes(self, log_every: int = 5000) -> list[tuple[int, int]]:
        """(H, W) of every normalized image -- preloads the cache."""
        import logging

        out = []
        for i in range(len(self)):
            out.append(tuple(self._load(i).shape[1:]))
            if log_every and (i + 1) % log_every == 0:
                logging.getLogger(__name__).info("  preloaded %d / %d", i + 1, len(self))
        return out


class BucketBatchSampler(Sampler):
    """Batches of similarly-sized images. Random batches pad every image up
    to the largest in the batch -- measured at 4.2x the real pixel count on
    this dataset (1.7x when bucketed), and cross-attention cost scales with
    that padded memory length. Shuffles within large chunks, sorts each
    chunk by size, then shuffles batch order, so batches stay random across
    epochs while their members stay size-matched."""

    def __init__(
        self,
        sizes: list[tuple[int, int]],
        batch_size: int,
        max_pixels: int = 500_000,
        chunk_batches: int = 50,
        seed: int = 0,
    ) -> None:
        self.sizes = sizes
        self.batch_size = batch_size
        # Cap on padded pixels per batch (batch x max_H x max_W). A fixed
        # batch of 32 of the largest images (136x520) measured 9-12 GB of
        # MPS memory -- which on Apple Silicon is system RAM -- and drove the
        # whole machine into swap. Large images get proportionally smaller
        # batches instead; typical ones (68x204) still fill all 32 slots.
        self.max_pixels = max_pixels
        self.chunk = batch_size * chunk_batches
        self.seed = seed
        self.epoch = 0
        self._n_batches = len(self._build(random.Random(seed)))

    def _build(self, rng: random.Random) -> list[list[int]]:
        order = list(range(len(self.sizes)))
        rng.shuffle(order)
        batches = []
        for start in range(0, len(order), self.chunk):
            chunk = sorted(order[start : start + self.chunk], key=lambda i: (self.sizes[i][1], self.sizes[i][0]))
            batch, max_h, max_w = [], 0, 0
            for i in chunk:
                h, w = self.sizes[i]
                nh, nw = max(max_h, h), max(max_w, w)
                if batch and (len(batch) >= self.batch_size or (len(batch) + 1) * nh * nw > self.max_pixels):
                    batches.append(batch)
                    batch, nh, nw = [], h, w
                batch.append(i)
                max_h, max_w = nh, nw
            if batch:
                batches.append(batch)
        rng.shuffle(batches)
        return batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        return iter(self._build(rng))

    def __len__(self) -> int:
        return self._n_batches


def _augment(image: torch.Tensor) -> torch.Tensor:
    """Mild scale jitter only (+-12%). Deliberately no rotation/shear: the
    decoder's job is reading vertical offsets (superscript vs subscript), and
    tilting the whole expression would blur exactly that signal."""
    scale = random.uniform(0.88, 1.12)
    _, h, w = image.shape
    new_h, new_w = max(4, round(h * scale)), max(4, round(w * scale))
    return torch.nn.functional.interpolate(image[None], size=(new_h, new_w), mode="bilinear", align_corners=False)[0]


# Padded batch dimensions are rounded up to these buckets. PyTorch's MPS
# backend caches buffers and compiled graphs per tensor shape; with every
# batch a unique (H, W, T), process footprint grew ~5 GB per 100 steps
# (measured: 5.2 -> 10 GB over 200 steps) and a full epoch pushed the Mac
# into swap. Bucketed shapes (+ periodic empty_cache in the training loop)
# held it flat at ~1.5 GB. Height/width buckets are multiples of the trunk
# stride, so the encoder grid stays exact.
H_BUCKET, W_BUCKET, T_BUCKET = 16, 32, 8


def _round_up(n: int, k: int) -> int:
    return n + (-n) % k


def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    """Pad images (with the background value) to a common bucketed size and
    build the valid-pixel mask; pad token sequences to a bucketed length.
    Returns images (B,1,H,W), mask (B,H,W), tgt_in (B,T), tgt_out (B,T)."""
    images, seqs = zip(*batch)
    max_h = _round_up(max(img.shape[1] for img in images), H_BUCKET)
    max_w = _round_up(max(img.shape[2] for img in images), W_BUCKET)
    assert max_h % TRUNK_STRIDE == 0 and max_w % TRUNK_STRIDE == 0
    out = torch.full((len(images), 1, max_h, max_w), BACKGROUND)
    mask = torch.zeros((len(images), max_h, max_w), dtype=torch.bool)
    for i, img in enumerate(images):
        _, h, w = img.shape
        out[i, :, :h, :w] = img
        mask[i, :h, :w] = True

    # +1 so that after the tgt_in/tgt_out shift the length is a bucket multiple.
    max_t = _round_up(max(len(s) for s in seqs) - 1, T_BUCKET) + 1
    ids = torch.full((len(seqs), max_t), TOKEN_TO_ID[PAD], dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = s
    return out, mask, ids[:, :-1], ids[:, 1:]
