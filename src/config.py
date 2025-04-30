from dataclasses import dataclass
import numpy as np

@dataclass(slots=True)
class GridSpec:
    r_min: float = -20          # in RE
    r_max: float = 20
    voxel: float = 0.5          # edge length in RE
    re_km: float = 6371.0       # convert km -> RE

    @property
    def edges(self) -> np.ndarray:
        return np.arange(self.r_min, self.r_max + self.voxel, self.voxel)

    @property
    def shape(self) -> tuple[int, int, int]:
        n = len(self.edges) - 1
        return n, n, n
