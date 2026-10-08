"""
Unified dataloader for medical image restoration datasets.

Supports 5 modalities:
  - speckle in OCT
  - speckle in ultra sound
  - noise in cryo-EM image
  - noise in low dose CT
  - Gaussian noise in MRI

Data layout (per modality):
  <root>/<modality>/train/LQ/*.raw
  <root>/<modality>/train/HQ/*.raw
  <root>/<modality>/val/LQ/*.raw
  <root>/<modality>/val/HQ/*.raw
  <root>/<modality>/test/LQ/*.raw
  <root>/<modality>/test/HQ/*.raw

Image format: raw float32, shape (224, 224).
"""

import os
import glob
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ── I/O ────────────────────────────────────────────────────────────────
def _load_raw(path, dtype=np.float32, shape=(224, 224)):
    with open(path, "rb") as f:
        data = np.fromfile(f, dtype=dtype)
    return data.reshape(shape)


def _parse_path(path):
    """Extract (modality, filename_stem) from a path like .../modality/split/LQ/xxx.raw."""
    parts = path.replace("\\", "/").split("/")
    fname = os.path.splitext(parts[-1])[0]
    modality = parts[-4]
    return modality, fname


# ── Label maps ─────────────────────────────────────────────────────────
MODALITY_LIST = [
    "speckle in OCT",
    "speckle in ultra sound",
    "noise in cryo-EM image",
    "noise in low dose CT",
    "Gaussian noise in MRI",
]

MODALITY_TO_ID = {m: i for i, m in enumerate(MODALITY_LIST)}


# ── Dataset ────────────────────────────────────────────────────────────
class MedicalRestorationDataset(Dataset):
    """Dataset for train / val / test splits.

    Parameters
    ----------
    root_dir : str
        Root data path.
    modality_list : list of str
        Modalities to include.
    split : 'train' | 'val' | 'test'
    use_num : int or None
        Max number of samples per modality (None = all).
    patch_size : int or None
        If set, random-crop train samples to this size.
    """

    def __init__(self, root_dir, modality_list=None, split="train",
                 use_num=None, patch_size=None):
        self.split = split
        self.patch_size = patch_size

        if modality_list is None:
            modality_list = MODALITY_LIST

        self.lq_paths = []
        self.hq_paths = []

        for modality in modality_list:
            pattern = os.path.join(root_dir, modality, split, "LQ", "*.raw")
            paths = sorted(glob.glob(pattern))
            n = len(paths) if use_num is None else min(use_num, len(paths))
            for p in paths[:n]:
                self.lq_paths.append(p)
                self.hq_paths.append(p.replace("LQ", "HQ"))

    def __len__(self):
        return len(self.lq_paths)

    def __getitem__(self, idx):
        lq = _load_raw(self.lq_paths[idx])
        hq = _load_raw(self.hq_paths[idx])
        modality, fname = _parse_path(self.lq_paths[idx])

        # min-max normalise to [0, 1]
        lq = _normalize(lq, modality)
        hq = _normalize(hq, modality)

        # channel dim
        lq = torch.from_numpy(lq).unsqueeze(0).float()
        hq = torch.from_numpy(hq).unsqueeze(0).float()

        # random crop (train only)
        if self.patch_size is not None and self.split == "train":
            _, H, W = lq.shape
            ph, pw = self.patch_size, self.patch_size
            if H > ph and W > pw:
                top = np.random.randint(0, H - ph + 1)
                left = np.random.randint(0, W - pw + 1)
                lq = lq[:, top:top + ph, left:left + pw]
                hq = hq[:, top:top + ph, left:left + pw]

        modality_id = MODALITY_TO_ID.get(modality, 0)
        return lq, hq, modality, fname, modality_id


# ── Normalisation ──────────────────────────────────────────────────────
def _normalize(img, modality):
    """Per-image min-max normalise to [0, 1] (matches original training)."""
    if modality == "noise in cryo-EM image":
        img = img / 255.0
    else:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img.astype(np.float32)


# ── DataLoader builders ────────────────────────────────────────────────
def build_train_loader(root_dir, modality_list=None, batch_size=4,
                       use_num=None, patch_size=None, num_workers=4):
    """Build per-modality dataloaders for balanced training.

    Each modality gets its own Dataset + DataLoader so that the training
    loop can sample one modality at a time.  This way every modality
    contributes roughly equally to each optimization step regardless of
    how many samples it contains.

    drop_last is disabled because the balanced sampler re-iterates each
    modality's loader independently; with drop_last=True the last partial
    batch of a small modality would never be seen.
    """
    if modality_list is None:
        modality_list = MODALITY_LIST

    loaders = []
    n_train_total = 0
    for modality in modality_list:
        dataset = MedicalRestorationDataset(
            root_dir, [modality], split="train",
            use_num=use_num, patch_size=patch_size,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=False, num_workers=num_workers,
                            pin_memory=True)
        loaders.append(loader)
        n_train_total += len(dataset)

    return loaders, n_train_total


def build_val_loader(root_dir, modality_list=None, use_num=None,
                     num_workers=2):
    dataset = MedicalRestorationDataset(
        root_dir, modality_list, split="val",
        use_num=use_num,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return loader, len(dataset)


def build_test_loader(root_dir, modality_list=None, use_num=None,
                      num_workers=2):
    dataset = MedicalRestorationDataset(
        root_dir, modality_list, split="test",
        use_num=use_num,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=num_workers)
    return loader, len(dataset)


class BalancedModalitySampler:
    """Infinite stream of batches sampled uniformly across modalities.

    At every step one modality is chosen at random with equal probability,
    then a full batch is drawn from that modality's DataLoader.  This
    ensures each modality receives roughly the same number of gradient
    updates over the course of training, independent of its dataset size.

    Each yielded batch comes entirely from a single modality, which keeps
    the SGM text-conditioning consistent within a batch.

    Empty modalities (0 samples) are skipped silently so that a missing
    modality does not crash training.
    """

    def __init__(self, dataloaders, seed=None):
        # keep only non-empty loaders so we never pick a modality with
        # zero batches available
        self.dataloaders = [
            dl for dl in dataloaders if len(dl.dataset) > 0
        ]
        if not self.dataloaders:
            raise RuntimeError(
                "BalancedModalitySampler: no non-empty modalities found")
        self._iters = [iter(dl) for dl in self.dataloaders]
        self._rng = np.random.RandomState(seed)

    def __iter__(self):
        return self

    def __next__(self):
        # pick one modality uniformly at random
        idx = self._rng.randint(0, len(self._iters))
        try:
            return next(self._iters[idx])
        except StopIteration:
            # exhausted this modality's loader — reshuffle and retry
            self._iters[idx] = iter(self.dataloaders[idx])
            return next(self._iters[idx])

    # ── state save / restore (for checkpoint resumption) ──────────
    def state_dict(self):
        return {"rng": self._rng.get_state()}

    def load_state_dict(self, state):
        if "rng" in state:
            self._rng.set_state(state["rng"])
