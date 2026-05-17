"""Sanity check the HOI4D dataloader."""

import argparse
import os

import numpy as np
from PIL import Image

from prg_contact.datasets import HOI4DContactDataset

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
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--image_size", type=int, nargs=2, default=None)
    parser.add_argument("--output_dir", default="verify_outputs/hoi4d")
    parser.add_argument("--prefer_contact", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    image_size = tuple(args.image_size) if args.image_size else None
    ds = HOI4DContactDataset(split=args.split, image_size=image_size)
    print(f"Dataset size: {len(ds)}")
    if len(ds) == 0:
        print("No records. Exiting.")
        return

    indices = []
    if args.prefer_contact:
        for i, rec in enumerate(ds.records):
            if rec.left_contact or rec.right_contact:
                indices.append(i)
                if len(indices) >= args.n:
                    break
        if len(indices) < args.n:
            extra = [i for i in range(len(ds)) if i not in indices][: args.n - len(indices)]
            indices += extra
    else:
        step = max(1, len(ds) // args.n)
        indices = list(range(0, len(ds), step))[: args.n]

    for k, idx in enumerate(indices):
        sample = ds[idx]
        img = denormalize(sample["image"])
        left_mask = sample["contact_mask"][0].numpy()
        right_mask = sample["contact_mask"][1].numpy()
        left_over = overlay(img, left_mask, color=(0, 255, 0))
        right_over = overlay(img, right_mask, color=(255, 0, 0))
        panel = np.concatenate([img, left_over, right_over], axis=1)
        state = sample["contact_state"].tolist()
        meta = sample["meta"]
        clean_path = meta["recording_path"].replace("/", "_")
        fname = f"{k:02d}_{clean_path}_f{meta['frame_index']:05d}_L{state[0]}_R{state[1]}_{meta['event_name']}.jpg"
        Image.fromarray(panel).save(os.path.join(args.output_dir, fname))
        print(f"  {fname}")


if __name__ == "__main__":
    main()
