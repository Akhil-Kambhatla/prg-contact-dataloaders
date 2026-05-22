"""Parse Ego4D STA bbox manifest into per-sample records.

Each row in sta_object_bboxes.csv is one (frame, object) pair: a frame
identified by clip_uid + clip_frame, plus a single object bounding box,
noun/verb, and time_to_contact. We unroll multi-object frames so each
sample is exactly one (frame, object) pair.

STA annotates pre-contact keyframes. The `time_to_contact` field (seconds)
indicates how long until contact happens. This parser supports filtering
to a ttc range so callers can choose how close-to-contact they want
samples to be.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Ego4DSampleRecord:
    image_path: str
    mask_path: str
    clip_uid: str
    clip_frame: int
    object_index: int
    bbox: tuple              # (x1, y1, x2, y2)
    noun: str
    verb: str
    time_to_contact: Optional[float]
    split: str
    meta: Dict = field(default_factory=dict)


def parse_ego4d_split(
    bbox_csv_path: str,
    processed_root: str,
    split: str,
    mask_version: str = "v2",
    skip_missing: bool = True,
    max_time_to_contact: Optional[float] = 0.3,
    min_time_to_contact: float = 0.0,
) -> List[Ego4DSampleRecord]:
    """Walk the STA bbox manifest and emit one record per (frame, object) pair.

    Args:
        bbox_csv_path: path to sta_object_bboxes.csv
        processed_root: root of preprocessed Ego4D output, e.g.
            "/fs/nexus-scratch/kakhil/ego4d_pipeline/processed"
        split: "train" or "val"
        mask_version: "v1" or "v2". v2 is recommended (box + center prompt +
            multimask, picks highest-scoring mask).
        skip_missing: if True, skip records whose JPG or mask PNG doesn't
            exist on disk.
        max_time_to_contact: seconds. Only keep records with ttc <= this.
            Default 0.3s keeps frames where contact is imminent (~9 video
            frames before contact at 30fps). Set to None to disable the
            upper bound and include all records.
        min_time_to_contact: seconds. Only keep records with ttc >= this.
            Default 0.0.

    Returns:
        List of Ego4DSampleRecord.
    """
    if mask_version not in ("v1", "v2"):
        raise ValueError(f"mask_version must be 'v1' or 'v2', got {mask_version!r}")

    records: List[Ego4DSampleRecord] = []

    with open(bbox_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != split:
                continue

            try:
                ttc = float(row["time_to_contact"]) if row["time_to_contact"] else None
            except (ValueError, KeyError):
                ttc = None

            # Filter by ttc range. Records with missing ttc are skipped when
            # a filter is active.
            if max_time_to_contact is not None or min_time_to_contact > 0.0:
                if ttc is None:
                    continue
                if ttc < min_time_to_contact:
                    continue
                if max_time_to_contact is not None and ttc > max_time_to_contact:
                    continue

            clip_uid = row["clip_uid"]
            clip_frame = int(row["clip_frame"])
            object_index = int(row["object_index"])

            frame_dir = os.path.join(processed_root, split, clip_uid)
            image_path = os.path.join(frame_dir, f"clipframe_{clip_frame:06d}.jpg")
            if mask_version == "v1":
                mask_name = f"clipframe_{clip_frame:06d}_obj{object_index}.png"
            else:
                mask_name = f"clipframe_{clip_frame:06d}_obj{object_index}_v2.png"
            mask_path = os.path.join(frame_dir, mask_name)

            if skip_missing:
                if not os.path.exists(image_path) or not os.path.exists(mask_path):
                    continue

            records.append(Ego4DSampleRecord(
                image_path=image_path,
                mask_path=mask_path,
                clip_uid=clip_uid,
                clip_frame=clip_frame,
                object_index=object_index,
                bbox=(
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                ),
                noun=row.get("noun", ""),
                verb=row.get("verb", ""),
                time_to_contact=ttc,
                split=split,
                meta={"annotation_uid": row.get("annotation_uid", "")},
            ))

    return records
