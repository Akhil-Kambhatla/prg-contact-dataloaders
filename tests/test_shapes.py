"""Tiny smoke test: verify output tensor shapes."""

import os

import pytest
import torch

from prg_contact.datasets import VISORContactDataset

VISOR_AVAILABLE = os.path.exists(
    "/fs/vulcan-datasets/VISOR/GroundTruth-SparseAnnotations/annotations/train"
)


@pytest.mark.skipif(not VISOR_AVAILABLE, reason="VISOR not on this filesystem")
def test_visor_shapes():
    ds = VISORContactDataset(split="val", image_size=(384, 384))
    assert len(ds) > 0
    sample = ds[0]
    assert sample["image"].shape == (3, 384, 384)
    assert sample["image"].dtype == torch.float32
    assert sample["contact_state"].shape == (2,)
    assert sample["contact_state"].dtype == torch.long
    assert sample["contact_mask"].shape == (2, 384, 384)
    assert sample["contact_mask"].dtype == torch.float32
