from collections import defaultdict
import numpy as np
import pandas as pd

from .config import GridSpec

class Voxeliser:
    """
    Accumulates time per (voxel, region) and can return the
    dominant region in each voxel.
    """
    def __init__(self, grid: GridSpec):
        self.grid = grid
        self._vox = defaultdict(lambda: np.zeros(grid.shape, dtype=np.float32))

    def accumulate(self, df: pd.DataFrame, dt_minutes: float = 1.0) -> None:
        e = self.grid.edges
        nx, ny, nz = self.grid.shape

        ix = np.digitize(df["x"], e) - 1
        iy = np.digitize(df["y"], e) - 1
        iz = np.digitize(df["z"], e) - 1
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)

        for i, j, k, reg in zip(ix[ok], iy[ok], iz[ok], df["region"][ok]):
            if pd.isna(reg):
                continue
            self._vox[reg][i, j, k] += dt_minutes

    @property
    def vox(self) -> dict[str, np.ndarray]:
        """Read-only access to the (region -> 3-D time array) map."""
        return self._vox

    def most_occupied(self) -> np.ndarray:
        """
        Return a 3-D array of strings: the region that has the largest
        total time in each voxel.  Empty voxels get ''.
        """
        regions = list(self._vox.keys())
        if not regions:
            raise RuntimeError("No time data accumulated")

        stack = np.stack([self._vox[r] for r in regions], axis=0)   # (R, nx, ny, nz)
        winner_idx = np.argmax(stack, axis=0)                       # (nx, ny, nz)
        total = stack.sum(axis=0)

        label_cube = np.empty(self.grid.shape, dtype=object)
        label_cube[:] = ''                                          # default = empty
        for idx, reg in enumerate(regions):
            label_cube[winner_idx == idx] = reg
        label_cube[total == 0] = ''                                 # never visited
        return label_cube
