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
