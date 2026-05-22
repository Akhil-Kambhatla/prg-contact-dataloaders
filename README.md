# prg-contact-dataloaders

PyTorch dataloaders for hand-object contact detection on EPIC-VISOR and HOI4D.

## What this provides

`VISORContactDataset` and `HOI4DContactDataset` — PyTorch `Dataset` classes that load raw data directly from `/fs/vulcan-datasets/` on the UMIACS Nexus cluster.

Each sample returns:

```python
{
    "image":         FloatTensor [3, H, W],   # ImageNet-normalized RGB
    "contact_state": LongTensor [2],          # (left, right), 0 or 1
    "contact_mask":  FloatTensor [2, H, W],   # (left, right), binary 0/1
    "meta":          dict,                    # debugging info, not for training
}
```

## Installation

```bash
git clone https://github.com/Akhil-Kambhatla/prg-contact-dataloaders.git
cd prg-contact-dataloaders
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick usage

```python
from prg_contact.datasets import VISORContactDataset, HOI4DContactDataset
from torch.utils.data import DataLoader

ds = VISORContactDataset(split="train", image_size=(384, 384))
loader = DataLoader(ds, batch_size=8, num_workers=4, shuffle=True)
for batch in loader:
    images = batch["image"]            # [B, 3, 384, 384]
    states = batch["contact_state"]    # [B, 2]
    masks  = batch["contact_mask"]     # [B, 2, 384, 384]
    break
```

## Verifying the dataloaders work

After installation on the cluster:

```bash
python scripts/verify_visor.py --split val --n 10 --output_dir verify_outputs/visor --prefer_contact
python scripts/verify_hoi4d.py --split val --n 10 --output_dir verify_outputs/hoi4d --prefer_contact
```

Each script samples N frames, saves side-by-side visualizations of the image, left-hand contact mask overlay (green), and right-hand contact mask overlay (red).

## VISOR contact semantics

Hand annotations have an `in_contact_object` field. Values are treated as:

- A 32-char hex id → in contact, mask comes from the referenced annotation's polygons.
- `hand-not-in-contact`, `none-of-the-above`, `inconclusive`, missing → not in contact, mask is zeros.

Constructor argument `filter_ambiguous_contact=True` (default `False`) skips frames where any hand is `none-of-the-above` or `inconclusive`.

## HOI4D contact semantics

Frame-level contact is derived from action events in `color.json`. Events from `CONTACT_EVENTS` map to contact=1 (right hand only by default, both hands for `Carrywithbothhands`). Events from `NO_CONTACT_EVENTS` map to contact=0. Frames in unknown events or with no covering event are skipped.

The contacted-object mask is the primary-object label (always label index 1) from the corresponding 2Dseg mask PNG.

## Splits

- VISOR: native `train/` and `val/` JSON directories under `GroundTruth-SparseAnnotations/annotations/`.
- HOI4D: `release.txt` (train) and `testset.txt` (val) from the official [HOI4D-Instructions](https://github.com/hoi4d/HOI4D-Instructions) repo. Copies shipped in `prg_contact/splits/`.

## HOI4D coverage caveat

HOI4D's 2D segmentation annotations are concentrated on the train split:

| Split | Recordings | With 2Dseg mask | Coverage |
|-------|-----------:|----------------:|---------:|
| train | 2971       | 2970            | 99.97%   |
| val   | 933        | 32              | 3.4%     |

The mask subdirectory is either `2Dseg/shift_mask` (typical for newer recordings) or `2Dseg/mask` (older). The dataloader checks for `shift_mask` first, falls back to `mask`. Recordings with neither are skipped by default (`require_mask=True`); pass `require_mask=False` to include all classifiable frames with zero masks for the mask-less ones.

Because val has only 3.4% mask coverage, it is not suitable as-is for mask-supervised validation. Options for the consuming project:

1. Use HOI4D train only and validate on VISOR.
2. Re-split HOI4D combining all recordings, filtering by mask presence first, then 80/20.
3. Accept the small (32-recording) HOI4D val and report that limitation explicitly.

Mask values use the Pascal VOC colormap. The dataloader decodes RGB triples back to integer label indices (label 1 = primary object = the contacted object's mask).

## Ego4D STA dataset

`Ego4DContactDataset` loads preprocessed Ego4D STA (Short-Term Anticipation) frames and SAM2-generated contact masks.

### Important differences from VISOR and HOI4D

Ego4D STA does not attribute contact to a specific hand. Each annotation is "this is the next object the wearer will contact" without left/right labels. As a result:

- `contact_state` is a scalar `LongTensor` (always `1`), not `[2]`.
- `contact_mask` has shape `[1, H, W]`, not `[2, H, W]`.

Additionally, **STA is a forecasting task.** The annotated keyframe is BEFORE contact happens; `time_to_contact` (in `meta`) tells how many seconds until contact. The annotated object may not be currently in the hand's path.

### Preprocessing

Ego4D videos are MP4s on the cluster at `/fs/vulcan-projects/Force_Learning/Ego4D/v2/clips/`. We extract keyframes and run SAM2 (bbox-prompted) to produce binary contact masks. Output structure:

```
processed/<split>/<clip_uid>/
├── clipframe_000984.jpg              # extracted RGB frame
├── clipframe_000984_obj0.png         # v1 SAM2 mask (box prompt only)
└── clipframe_000984_obj0_v2.png      # v2 SAM2 mask (box + center point + multimask)
```

The dataloader defaults to v2 masks. Pass `mask_version="v1"` to use v1.

### Mask quality caveat

Ego4D STA bounding boxes are looser than VISOR/HOI4D ground-truth annotations because STA was designed for forecasting, not segmentation. SAM2 masks reflect the bbox quality: when the bbox is tight, the mask is good; when the bbox is loose or partly occluded, the mask is noisier. Roughly 60-70% of v2 masks look correct on visual inspection; the remaining 30-40% have varying quality, mostly traceable to STA bbox imprecision.

If you need to filter, mask area is a useful proxy: very small masks (under ~1000 pixels) often indicate a poor bbox-mask match.

### Usage

```python
from prg_contact.datasets import Ego4DContactDataset
from torch.utils.data import DataLoader

ds = Ego4DContactDataset(split="train", image_size=(384, 384))
loader = DataLoader(ds, batch_size=8, num_workers=4, shuffle=True)
for batch in loader:
    images = batch["image"]            # [B, 3, 384, 384]
    states = batch["contact_state"]    # [B]
    masks  = batch["contact_mask"]     # [B, 1, 384, 384]
    break
```
