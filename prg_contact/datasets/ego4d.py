"""Ego4D STA contact dataset.

Loads preprocessed frames and SAM2 contact masks. Each sample is one
(frame, object) pair. Unlike VISOR/HOI4D, returns a single mask per
sample (shape [1, H, W]) since STA does not attribute contact to a
specific hand.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from prg_contact.parsing.ego4d_parser import Ego4DSampleRecord, parse_ego4d_split

from .base import ContactDatasetBase

DEFAULT_PROCESSED_ROOT = "/fs/nexus-scratch/kakhil/ego4d_pipeline/processed"


def _default_bbox_csv_path() -> str:
    """Locate the shipped STA bbox manifest."""
    from importlib import resources
    with resources.path("prg_contact.splits", "ego4d_sta_object_bboxes.csv") as p:
        return str(p)


class Ego4DContactDataset(ContactDatasetBase):
    """Ego4D STA hand-object contact dataset.

    Each sample returns:
        image: [3, H, W] float, ImageNet-normalized
        contact_state: scalar LongTensor (always 1 for STA)
        contact_mask: [1, H, W] float, binary
        meta: dict
    """

    def __init__(
        self,
        split: str = "train",
        processed_root: str = DEFAULT_PROCESSED_ROOT,
        bbox_csv_path: Optional[str] = None,
        mask_version: str = "v2",
        image_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
        skip_missing: bool = True,
    ):
        super().__init__(image_size=image_size, transform=transform)
        self.split = split
        self.mask_version = mask_version
        if bbox_csv_path is None:
            bbox_csv_path = _default_bbox_csv_path()
        self.records = parse_ego4d_split(
            bbox_csv_path=bbox_csv_path,
            processed_root=processed_root,
            split=split,
            mask_version=mask_version,
            skip_missing=skip_missing,
        )

    def _load_image_and_mask(self, record: Ego4DSampleRecord):
        image = Image.open(record.image_path).convert("RGB")
        img_w, img_h = image.size

        mask_pil = Image.open(record.mask_path).convert("L")
        mask = np.asarray(mask_pil)
        if mask.shape != (img_h, img_w):
            mask = np.array(
                Image.fromarray(mask).resize((img_w, img_h), Image.NEAREST)
            )
        return image, mask

    def _resize_pair(self, image: Image.Image, mask: np.ndarray):
        if self.image_size is None:
            return image, mask
        target_h, target_w = self.image_size
        image = image.resize((target_w, target_h), Image.BILINEAR)
        mask = np.array(
            Image.fromarray(mask).resize((target_w, target_h), Image.NEAREST)
        )
        return image, mask

    def _record_meta(self, record: Ego4DSampleRecord) -> Dict:
        return {
            "dataset": "ego4d",
            "clip_uid": record.clip_uid,
            "clip_frame": record.clip_frame,
            "object_index": record.object_index,
            "noun": record.noun,
            "verb": record.verb,
            "time_to_contact": record.time_to_contact,
            "bbox": list(record.bbox),
            "frame_path": record.image_path,
            "mask_path": record.mask_path,
        }

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image, mask = self._load_image_and_mask(record)
        image, mask = self._resize_pair(image, mask)
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        image_tensor = self._normalize_image(image)
        mask_tensor = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        state_tensor = torch.tensor(1, dtype=torch.long)

        return {
            "image": image_tensor,
            "contact_state": state_tensor,
            "contact_mask": mask_tensor,
            "meta": self._record_meta(record),
        }
