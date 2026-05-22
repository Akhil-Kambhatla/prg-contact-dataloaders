"""Sanity check the Ego4D dataloader."""

import argparse
import os

import numpy as np
from PIL import Image

from prg_contact.datasets import Ego4DContactDataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize(image_tensor):
    arr = image_tensor.numpy().transpose(1, 2, 0)
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def overlay(image, mask, color):
    out = image.copy().astype(np.float32)
    color = np.array(color, dtype=np.float32)
    m = mask[:, :, None]
    out = out * (1 - 0.5 * m) + color * 0.5 * m
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--image_size", type=int, nargs=2, default=None)
    parser.add_argument("--output_dir", default="verify_outputs/ego4d")
    parser.add_argument("--mask_version", default="v2", choices=["v1", "v2"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    image_size = tuple(args.image_size) if args.image_size else None
    ds = Ego4DContactDataset(
        split=args.split,
        mask_version=args.mask_version,
        image_size=image_size,
    )
    print(f"Dataset size ({args.split}, mask_version={args.mask_version}): {len(ds)}")
    if len(ds) == 0:
        print("No records. Exiting.")
        return

    import random
    random.seed(42)
    indices = random.sample(range(len(ds)), min(args.n, len(ds)))

    for k, idx in enumerate(indices):
        sample = ds[idx]
        img = denormalize(sample["image"])
        mask = sample["contact_mask"][0].numpy()
        over = overlay(img, mask, color=(0, 200, 255))
        panel = np.concatenate([img, over], axis=1)
        meta = sample["meta"]
        clean_clip = meta["clip_uid"][:8]
        fname = (
            f"{k:02d}_{clean_clip}_f{meta['clip_frame']:06d}"
            f"_obj{meta['object_index']}_{meta['verb']}.jpg"
        )
        Image.fromarray(panel).save(os.path.join(args.output_dir, fname))
        print(f"  {fname}  noun={meta['noun'][:30]}  ttc={meta['time_to_contact']}s")


if __name__ == "__main__":
    main()
