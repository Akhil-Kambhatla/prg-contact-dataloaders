"""Parse HOI4D recordings into per-frame sample records."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

CONTACT_EVENTS = {
    "Grasp", "Pickup", "carry", "putdown",
    "Carrywithbothhands",
    "open", "close",
    "push", "pull",
    "dump", "cut", "paper-cut",
    "Press", "binding",
    "turn&on&the&switch",
}

NO_CONTACT_EVENTS = {
    "rest", "Reachout", "Stop",
    "go", "Lookaround",
}

BOTH_HANDS_EVENTS = {
    "Carrywithbothhands",
}

PRIMARY_OBJECT_LABEL = 1
FRAMES_PER_RECORDING = 300

# Subdirectory names under 2Dseg/ that contain per-frame mask PNGs.
# Tried in order; first one that exists wins.
MASK_SUBDIR_CANDIDATES = ("shift_mask", "mask")


@dataclass
class HOI4DSampleRecord:
    video_path: str
    frame_index: int
    mask_path: Optional[str]
    left_contact: bool
    right_contact: bool
    recording_path: str
    event_name: str
    meta: Dict = field(default_factory=dict)


def _get_category(recording_path: str) -> Optional[str]:
    """Extract 'C<n>' from a path like 'ZY.../H4/C8/N14/S71/s03/T2'."""
    for part in recording_path.split("/"):
        if part.startswith("C") and part[1:].isdigit():
            return part
    return None


def _resolve_mask_dir(annotations_root: str, recording_path: str) -> Optional[str]:
    """Return the absolute path to the 2Dseg mask dir for this recording, or None."""
    seg_dir = os.path.join(annotations_root, recording_path, "2Dseg")
    if not os.path.isdir(seg_dir):
        return None
    for candidate in MASK_SUBDIR_CANDIDATES:
        candidate_path = os.path.join(seg_dir, candidate)
        if os.path.isdir(candidate_path):
            return candidate_path
    return None


def _load_action_events(action_json_path: str) -> Tuple[List[Tuple[int, int, str]], float]:
    """Read color.json and return (frame_events, fps)."""
    with open(action_json_path, "r") as f:
        data = json.load(f)

    info = data.get("info", {})
    duration = info.get("duration") or info.get("Duration")
    if duration is None:
        raise ValueError(f"No duration in {action_json_path}")

    fps = FRAMES_PER_RECORDING / float(duration)

    if "events" in data:
        raw_events = data["events"]
        ts_key, te_key = "startTime", "endTime"
    elif "markResult" in data:
        raw_events = data["markResult"]["marks"]
        ts_key, te_key = "hdTimeStart", "hdTimeEnd"
    else:
        raise ValueError(f"No events in {action_json_path}")

    out = []
    for ev in raw_events:
        start = int(ev[ts_key] * fps)
        end = int(ev[te_key] * fps)
        start = max(0, min(start, FRAMES_PER_RECORDING - 1))
        end = max(0, min(end, FRAMES_PER_RECORDING - 1))
        out.append((start, end, ev.get("event", "")))
    return out, fps


def _event_for_frame(
    frame_idx: int, frame_events: List[Tuple[int, int, str]]
) -> Optional[str]:
    for start, end, name in frame_events:
        if start <= frame_idx <= end:
            return name
    return None


def _classify_event(event_name: str) -> Optional[Tuple[bool, bool]]:
    """Return (left_contact, right_contact) for an event, or None to skip."""
    if event_name in CONTACT_EVENTS:
        right = True
        left = event_name in BOTH_HANDS_EVENTS
        return (left, right)
    if event_name in NO_CONTACT_EVENTS:
        return (False, False)
    return None


def parse_hoi4d_split(
    hoi4d_root: str,
    split_list_path: str,
    require_mask: bool = True,
) -> List[HOI4DSampleRecord]:
    """Walk an HOI4D split's recording list and emit per-frame records.

    Args:
        hoi4d_root: e.g. "/fs/vulcan-datasets/HOI4D"
        split_list_path: text file of recording paths, one per line.
        require_mask: if True (default), skip recordings without a 2Dseg
            mask directory (shift_mask or mask). Set False to include all
            classifiable frames; mask-less frames will have mask_path=None
            and the dataset will return zero contact masks for them.
    """
    videos_root = os.path.join(hoi4d_root, "HOI4D_release")
    annotations_root = os.path.join(hoi4d_root, "HOI4D_annotations")

    with open(split_list_path, "r") as f:
        recording_paths = [line.strip() for line in f if line.strip()]

    records: List[HOI4DSampleRecord] = []

    for rec_path in recording_paths:
        video_path = os.path.join(videos_root, rec_path, "align_rgb", "image.mp4")
        action_path = os.path.join(annotations_root, rec_path, "action", "color.json")

        if not os.path.exists(video_path) or not os.path.exists(action_path):
            continue

        mask_dir = _resolve_mask_dir(annotations_root, rec_path)
        if require_mask and mask_dir is None:
            continue

        try:
            frame_events, _fps = _load_action_events(action_path)
        except (ValueError, KeyError, json.JSONDecodeError):
            continue

        category = _get_category(rec_path)

        for frame_idx in range(FRAMES_PER_RECORDING):
            event = _event_for_frame(frame_idx, frame_events)
            if event is None:
                continue
            classification = _classify_event(event)
            if classification is None:
                continue
            left_contact, right_contact = classification

            mask_path = None
            if mask_dir is not None:
                candidate = os.path.join(mask_dir, f"{frame_idx:05d}.png")
                if os.path.exists(candidate):
                    mask_path = candidate

            records.append(HOI4DSampleRecord(
                video_path=video_path,
                frame_index=frame_idx,
                mask_path=mask_path,
                left_contact=left_contact,
                right_contact=right_contact,
                recording_path=rec_path,
                event_name=event,
                meta={"category": category},
            ))

    return records
