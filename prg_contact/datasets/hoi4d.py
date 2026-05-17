"""HOI4D contact dataset."""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from prg_contact.parsing.hoi4d_parser import (
    HOI4DSampleRecord,
    PRIMARY_OBJECT_LABEL,
    parse_hoi4d_split,
)

from .base import ContactDatasetBase

DEFAULT_HOI4D_ROOT = "/fs/vulcan-datasets/HOI4D"

# Pascal VOC colormap: RGB triple -> label index.
# HOI4D 2Dseg masks are saved with this colormap; PIL reads them as RGB.
# We decode back to integer labels here. First 8 entries cover the
# typical HOI4D label range (background, primary object, hand, parts).
_VOC_COLOR_TO_LABEL = {
    (0, 0, 0): 0,
    (128, 0, 0): 1,
    (0, 128, 0): 2,
    (128, 128, 0): 3,
    (0, 0, 128): 4,
    (128, 0, 128): 5,
    (0, 128, 128): 6,
    (128, 128, 128): 7,
    (64, 0, 0): 8,
    (192, 0, 0): 9,
    (64, 128, 0): 10,
    (192, 128, 0): 11,
}


def _decode_rgb_to_labels(rgb: np.ndarray) -> np.ndarray:
    """Convert an (H, W, 3) Pascal VOC colormap image to an (H, W) label map."""
    h, w, _ = rgb.shape
    labels = np.zeros((h, w), dtype=np.uint8)
    for color, idx in _VOC_COLOR_TO_LABEL.items():
        matches = np.all(rgb == np.array(color, dtype=np.uint8), axis=-1)
        labels[matches] = idx
    return labels


def _split_file_for(split: str) -> str:
    """Locate the shipped split file."""
    from importlib import resources
    fname = "hoi4d_train.txt" if split == "train" else "hoi4d_val.txt"
    with resources.path("prg_contact.splits", fname) as p:
        return str(p)


class HOI4DContactDataset(ContactDatasetBase):
    """HOI4D hand-object contact dataset."""

    def __init__(
        self,
        split: str = "train",
        hoi4d_root: str = DEFAULT_HOI4D_ROOT,
        split_list_path: Optional[str] = None,
        image_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
        require_mask: bool = True,
    ):
        super().__init__(image_size=image_size, transform=transform)
        self.split = split
        if split_list_path is None:
            split_list_path = _split_file_for(split)
        self.records = parse_hoi4d_split(
            hoi4d_root=hoi4d_root,
            split_list_path=split_list_path,
            require_mask=require_mask,
        )
        self._video_cache: Dict[str, cv2.VideoCapture] = {}

    def _get_capture(self, video_path: str) -> cv2.VideoCapture:
        cap = self._video_cache.get(video_path)
        if cap is None:
            cap = cv2.VideoCapture(video_path)
            self._video_cache[video_path] = cap
        return cap

    def _load_image_and_masks(self, record: HOI4DSampleRecord):
        cap = self._get_capture(record.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, record.frame_index)
        ret, bgr = cap.read()
        if not ret or bgr is None:
            cap.release()
            cap = cv2.VideoCapture(record.video_path)
            self._video_cache[record.video_path] = cap
            cap.set(cv2.CAP_PROP_POS_FRAMES, record.frame_index)
            ret, bgr = cap.read()
            if not ret or bgr is None:
                raise RuntimeError(
                    f"Failed to read frame {record.frame_index} from {record.video_path}"
                )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        img_w, img_h = image.size

        if record.mask_path is not None and os.path.exists(record.mask_path):
            seg_pil = Image.open(record.mask_path)
            seg_rgb = np.asarray(seg_pil.convert("RGB"))
            labels = _decode_rgb_to_labels(seg_rgb)
            obj_mask = np.where(labels == PRIMARY_OBJECT_LABEL, 255, 0).astype(np.uint8)
            if obj_mask.shape != (img_h, img_w):
                obj_mask = np.array(
                    Image.fromarray(obj_mask).resize((img_w, img_h), Image.NEAREST)
                )
        else:
            obj_mask = np.zeros((img_h, img_w), dtype=np.uint8)

        left_mask = obj_mask if record.left_contact else np.zeros_like(obj_mask)
        right_mask = obj_mask if record.right_contact else np.zeros_like(obj_mask)
        return image, left_mask, right_mask

    def _record_meta(self, record: HOI4DSampleRecord) -> Dict:
        return {
            "dataset": "hoi4d",
            "recording_path": record.recording_path,
            "frame_index": record.frame_index,
            "event_name": record.event_name,
        }

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image, left_mask, right_mask = self._load_image_and_masks(record)
        image, left_mask, right_mask = self._resize_triple(image, left_mask, right_mask)
        if self.transform is not None:
            image, left_mask, right_mask = self.transform(image, left_mask, right_mask)

        image_tensor = self._normalize_image(image)
        masks_tensor = torch.stack([
            torch.from_numpy((left_mask > 127).astype(np.float32)),
            torch.from_numpy((right_mask > 127).astype(np.float32)),
        ])
        state_tensor = torch.tensor(
            [int(record.left_contact), int(record.right_contact)],
            dtype=torch.long,
        )
        return {
            "image": image_tensor,
            "contact_state": state_tensor,
            "contact_mask": masks_tensor,
            "meta": self._record_meta(record),
        }

    def __del__(self):
        for cap in self._video_cache.values():
            try:
                cap.release()
            except Exception:
                pass
