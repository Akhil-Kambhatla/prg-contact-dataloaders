"""Base class for hand-object contact datasets."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ContactDatasetBase(Dataset):
    """Shared logic for VISOR/HOI4D contact datasets."""

    def __init__(
        self,
        image_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.image_size = image_size
        self.transform = transform
        self.records = []

    def __len__(self) -> int:
        return len(self.records)

    def _resize_triple(
        self,
        image: Image.Image,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
    ) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
        if self.image_size is None:
            return image, left_mask, right_mask
        target_h, target_w = self.image_size
        image = image.resize((target_w, target_h), Image.BILINEAR)
        left_mask = np.array(
            Image.fromarray(left_mask).resize((target_w, target_h), Image.NEAREST)
        )
        right_mask = np.array(
            Image.fromarray(right_mask).resize((target_w, target_h), Image.NEAREST)
        )
        return image, left_mask, right_mask

    @staticmethod
    def _normalize_image(image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(arr.transpose(2, 0, 1).copy())
