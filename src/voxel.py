# voxel.py
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd

from .config import GridSpec


class Voxeliser:
    """
    Accumulates dwell time per (voxel, region) and can return

    - a discrete probability distribution in every voxel  ('probs')
    - the dominant-region map ('most_occupied')
    """

    # ------------------------------------------------------------------
    def __init__(self, grid: GridSpec):
        self.grid = grid
        self._vox = defaultdict(lambda: np.zeros(grid.shape, dtype=np.float32))

    # ------------------------------------------------------------------
    def accumulate(self, df: pd.DataFrame, dt_minutes: float | None = None) -> None:
        """
        Add `dt_minutes` dwell time for every sample in *df*.

        If dt_minutes is None it is inferred from the first two timestamps.
        """

        if dt_minutes is None:
            ts = pd.to_datetime(df.time).values
            dt_minutes = ((ts[1] - ts[0]) / 60e9) if len(ts) > 1 else 1.0

        e = self.grid.edges
        nx, ny, nz = self.grid.shape

        ix = np.digitize(df["x"], e) - 1
        iy = np.digitize(df["y"], e) - 1
        iz = np.digitize(df["z"], e) - 1
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)

        for i, j, k, reg in zip(ix[ok], iy[ok], iz[ok], df["region"][ok]):
            if pd.isna(reg):
                continue
            self._vox[str(reg)][i, j, k] += dt_minutes  # ensure key is hashable

    @property
    def vox(self) -> dict[str, np.ndarray]:
        """Read-only access to the (region -> 3-D time array) map."""
        return self._vox

    def probs(self) -> tuple[list[str], np.ndarray]:
        """
        Return (regions, P) where **P** is a 4-D array
            shape = (nx, ny, nz, R)

        P[i,j,k,r] is the **probability** (0-1) of being in region *r*
        given the spacecraft is in voxel (i,j,k).  Voxels never visited
        have probability 0 for every region.
        """
        if not self._vox:
            raise RuntimeError("No data accumulated")

        regions = sorted(self._vox.keys())              # stable order
        stack = np.stack([self._vox[r] for r in regions], axis=-1)  # (nx,ny,nz,R)
        total = stack.sum(axis=-1, keepdims=True)                     # (nx,ny,nz,1)
        with np.errstate(divide="ignore", invalid="ignore"):
            probs = np.where(total > 0, stack / total, 0.0)
        return regions, probs

    def most_occupied(self) -> np.ndarray:
        """
        Return a 3-D array of the **mode region label** per voxel.

        Voxels never visited get the empty string ''.
        """
        regions, prob_cube = self.probs()               # (nx,ny,nz,R)
        winner_idx = np.argmax(prob_cube, axis=-1)      # (nx,ny,nz)
        total = prob_cube.sum(axis=-1)                  # (nx,ny,nz)

        label_cube = np.empty(self.grid.shape, dtype=object)
        label_cube[:] = ''
        for idx, reg in enumerate(regions):
            label_cube[winner_idx == idx] = reg
        label_cube[total == 0] = ''                     # never visited
        return label_cube
