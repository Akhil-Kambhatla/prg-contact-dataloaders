"""VISOR contact dataset."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from prg_contact.masks.polygon import polygons_to_mask
from prg_contact.parsing.visor_parser import VISORSampleRecord, parse_visor_split

from .base import ContactDatasetBase

DEFAULT_VISOR_ANNOTATIONS = "/fs/vulcan-datasets/VISOR/GroundTruth-SparseAnnotations/annotations"
DEFAULT_EPIC_ROOT = "/fs/vulcan-datasets/EPIC-Kitchens-2020"


class VISORContactDataset(ContactDatasetBase):
    """EPIC-VISOR hand-object contact dataset."""

    def __init__(
        self,
        split: str = "train",
        visor_annotations_root: str = DEFAULT_VISOR_ANNOTATIONS,
        epic_root: str = DEFAULT_EPIC_ROOT,
        image_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
        filter_ambiguous_contact: bool = False,
    ):
        super().__init__(image_size=image_size, transform=transform)
        self.split = split
        self.records = parse_visor_split(
            visor_annotations_root=visor_annotations_root,
            epic_root=epic_root,
            split=split,
            filter_ambiguous_contact=filter_ambiguous_contact,
        )

    def _load_image_and_masks(self, record: VISORSampleRecord):
        image = Image.open(record.image_path).convert("RGB")
        img_w, img_h = image.size

        poly_h, poly_w = record.native_size
        scale_x = img_w / poly_w
        scale_y = img_h / poly_h

        def _mask_for_hand(hand) -> np.ndarray:
            if not hand.in_contact or not hand.polygons:
                return np.zeros((img_h, img_w), dtype=np.uint8)
            scaled_polys = [
                [[p[0] * scale_x, p[1] * scale_y] for p in poly]
                for poly in hand.polygons
            ]
            return polygons_to_mask(scaled_polys, image_size=(img_h, img_w))

        left_mask = _mask_for_hand(record.left)
        right_mask = _mask_for_hand(record.right)
        return image, left_mask, right_mask

    def _record_meta(self, record: VISORSampleRecord) -> Dict:
        return {
            "dataset": "visor",
            "video_id": record.video_id,
            "frame_name": record.frame_name,
            "image_path": record.image_path,
            "left_contact_object": record.left.contact_object_name,
            "right_contact_object": record.right.contact_object_name,
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
            [int(record.left.in_contact), int(record.right.in_contact)],
            dtype=torch.long,
        )
        return {
            "image": image_tensor,
            "contact_state": state_tensor,
            "contact_mask": masks_tensor,
            "meta": self._record_meta(record),
        }
