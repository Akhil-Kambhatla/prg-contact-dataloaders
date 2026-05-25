"""Ego4D fho_scod contact dataset.

Loads frames and per-hand binary contact labels derived from fho_scod v1
annotations. The contact label for each hand is 1 if both the hand bbox and
the object_of_change bbox are present in that keyframe, else 0.

No segmentation masks are provided for Ego4D (fho_scod has no segmentations).
The dataset returns an empty (zeros) mask tensor to keep the output shape
consistent with VISOR and HOI4D.
"""

from __future__ import annotations

import csv
import os
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .base import ContactDatasetBase

DEFAULT_MANIFEST_DIR = "/fs/nexus-scratch/kakhil/ego4d_pipeline/fho_hands_processed"
DEFAULT_FRAMES_DIR = os.path.join(DEFAULT_MANIFEST_DIR, "frames")


class Ego4DContactDataset(ContactDatasetBase):
    """Ego4D fho_scod per-hand contact dataset.

    Each sample returns:
        image: [3, H, W] float, ImageNet-normalized
        contact_state: [2] LongTensor, [contact_left, contact_right]
        contact_mask: [2, H, W] float, all zeros (no seg available)
        meta: dict
    """

    def __init__(
        self,
        split: str = "train",
        manifest_dir: str = DEFAULT_MANIFEST_DIR,
        frames_dir: str = DEFAULT_FRAMES_DIR,
        image_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
        skip_missing: bool = True,
    ):
        super().__init__(image_size=image_size, transform=transform)
        self.split = split
        self.frames_dir = frames_dir

        csv_path = os.path.join(manifest_dir, f"fho_scod_{split}_available.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(manifest_dir, f"fho_scod_{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No manifest CSV at {csv_path}")

        records = []
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                clip_uid = r["clip_uid"]
                frame_num = int(r["frame_number"])
                img_path = os.path.join(frames_dir, clip_uid, f"{frame_num:08d}.jpg")
                if skip_missing and not os.path.exists(img_path):
                    continue
                records.append({
                    "clip_uid": clip_uid,
                    "record_idx": int(r["record_idx"]),
                    "keyframe_type": r["keyframe_type"],
                    "frame_number": frame_num,
                    "width": int(r["width"]),
                    "height": int(r["height"]),
                    "contact_left": int(r["contact_left"]),
                    "contact_right": int(r["contact_right"]),
                    "image_path": img_path,
                })
        self.records = records

    def __len__(self):
        return len(self.records)

    def _load_image(self, record):
        return Image.open(record["image_path"]).convert("RGB")

    def _resize(self, image: Image.Image):
        if self.image_size is None:
            return image
        target_h, target_w = self.image_size
        return image.resize((target_w, target_h), Image.BILINEAR)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image = self._load_image(record)
        image = self._resize(image)

        if self.transform is not None:
            image, _ = self.transform(image, None)

        image_tensor = self._normalize_image(image)

        # Empty mask placeholder, shape [2, H, W] to match per-hand convention
        H, W = image_tensor.shape[-2], image_tensor.shape[-1]
        mask_tensor = torch.zeros((2, H, W), dtype=torch.float32)

        state_tensor = torch.tensor(
            [record["contact_left"], record["contact_right"]],
            dtype=torch.long,
        )

        meta = {
            "dataset": "ego4d",
            "clip_uid": record["clip_uid"],
            "record_idx": record["record_idx"],
            "keyframe_type": record["keyframe_type"],
            "frame_number": record["frame_number"],
            "orig_width": record["width"],
            "orig_height": record["height"],
            "frame_path": record["image_path"],
        }

        return {
            "image": image_tensor,
            "contact_state": state_tensor,
            "contact_mask": mask_tensor,
            "meta": meta,
        }
