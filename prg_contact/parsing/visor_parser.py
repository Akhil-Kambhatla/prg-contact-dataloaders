"""Parse VISOR HOS annotations into per-frame sample records.

VISOR ships sparse annotations: about 80K annotated frames across many EPIC
videos. Each frame's annotations may contain hand entries whose
`in_contact_object` field is the id of another annotation in the same frame.

This parser walks all annotation JSONs for a split and emits one record per
annotated frame. The record carries enough info for the dataset class to
load the image and rasterize contact masks at runtime.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# Values of `in_contact_object` that mean "not actually in contact"
NOT_IN_CONTACT_VALUES = {
    "hand-not-in-contact",
    "none-of-the-above",
    "inconclusive",
}

# Ambiguous values where the hand IS in contact but we have no usable mask
AMBIGUOUS_VALUES = {
    "none-of-the-above",
    "inconclusive",
}


@dataclass
class HandRecord:
    in_contact: bool
    polygons: Optional[List[Sequence[Sequence[float]]]] = None
    contact_object_name: Optional[str] = None


@dataclass
class VISORSampleRecord:
    image_path: str
    video_id: str
    frame_name: str
    native_size: tuple
    left: HandRecord
    right: HandRecord
    meta: Dict = field(default_factory=dict)


def _epic_frame_path(epic_root: str, video_id: str, frame_name: str) -> Optional[str]:
    """Map a VISOR frame name to its EPIC-Kitchens jpg path.

    VISOR: "P01_01_frame_0000000298.jpg"
    EPIC:  "<epic_root>/P01_01/frame_0000000298.jpg"
    """
    if not frame_name.startswith(f"{video_id}_"):
        return None
    short = frame_name[len(video_id) + 1:]
    candidate = os.path.join(epic_root, video_id, short)
    return candidate if os.path.exists(candidate) else None


def _classify_hand(ann: dict, frame_index_by_id: Dict[str, dict]) -> HandRecord:
    """Build a HandRecord from a hand annotation."""
    contact_field = ann.get("in_contact_object")
    if contact_field is None or contact_field in NOT_IN_CONTACT_VALUES:
        return HandRecord(in_contact=False)

    target = frame_index_by_id.get(str(contact_field))
    if target is None:
        return HandRecord(in_contact=False)

    segments = target.get("segments")
    if not segments:
        return HandRecord(in_contact=False)

    return HandRecord(
        in_contact=True,
        polygons=segments,
        contact_object_name=target.get("name"),
    )


def parse_visor_split(
    visor_annotations_root: str,
    epic_root: str,
    split: str,
    filter_ambiguous_contact: bool = False,
    skip_missing_frame: bool = True,
) -> List[VISORSampleRecord]:
    """Walk VISOR <split> JSONs and emit one record per annotated frame."""
    split_dir = os.path.join(visor_annotations_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"VISOR split dir not found: {split_dir}")

    json_files = sorted(f for f in os.listdir(split_dir) if f.endswith(".json"))
    records: List[VISORSampleRecord] = []

    for fname in json_files:
        with open(os.path.join(split_dir, fname), "r") as f:
            data = json.load(f)

        for entry in data.get("video_annotations", []):
            image_info = entry.get("image", {})
            annotations = entry.get("annotations", [])

            frame_name = image_info.get("name", "")
            video_id = image_info.get("video", "")
            if not frame_name or not video_id:
                continue

            if filter_ambiguous_contact:
                if any(
                    "hand" in a.get("name", "").lower()
                    and a.get("in_contact_object") in AMBIGUOUS_VALUES
                    for a in annotations
                ):
                    continue

            image_path = _epic_frame_path(epic_root, video_id, frame_name)
            if image_path is None:
                if skip_missing_frame:
                    continue
                image_path = os.path.join(epic_root, video_id, frame_name[len(video_id) + 1:])

            index_by_id = {str(a.get("id")): a for a in annotations if a.get("id") is not None}

            left = HandRecord(in_contact=False)
            right = HandRecord(in_contact=False)
            for ann in annotations:
                name = ann.get("name", "").lower().strip()
                if name in ("left hand", "left:hand"):
                    left = _classify_hand(ann, index_by_id)
                elif name in ("right hand", "right:hand"):
                    right = _classify_hand(ann, index_by_id)

            records.append(VISORSampleRecord(
                image_path=image_path,
                video_id=video_id,
                frame_name=frame_name,
                native_size=(1080, 1920),
                left=left,
                right=right,
                meta={"source_json": fname},
            ))

    return records
