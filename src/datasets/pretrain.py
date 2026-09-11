import time

import h5py
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset

class PretrainCache(Dataset):
    """
    Read precomputed 2D tensors from an HDF5 cache for pretraining.
    Expects dataset shaped (N, C, H, W).
    """

    def __init__(self, h5_path, key="spectrogram"):
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 cache not found: {self.h5_path}")

        self._h5 = None
        self._data = None
        with h5py.File(self.h5_path, "r") as f:
            if key not in f:
                raise KeyError(f"Expected dataset '{key}' in {self.h5_path}")
            self.N = f[key].shape[0]
        self.key = key

    def __len__(self):
        return self.N

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(
                self.h5_path,
                "r",
                libver="latest",
                swmr=True,
                rdcc_nbytes=64 * 1024 * 1024,
                rdcc_nslots=100003,
                rdcc_w0=0.25,
            )
            self._data = self._h5[self.key]

    def __getitem__(self, idx):
        self._ensure_open()
        x = self._data[idx]
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        x = torch.from_numpy(x)
        return x

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass