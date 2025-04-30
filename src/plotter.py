from collections import defaultdict
import numpy as np
import plotly.graph_objects as go

from .config import GridSpec

class VoxelPlotter:
    _group_colors = {"IN": "blue", "OUT": "red", "OTHER": "grey"}

    def __init__(self, grid: GridSpec):
        self.grid = grid

    def plot_regions_inout(self, label_cube: np.ndarray) -> None:
        edges, s = self.grid.edges, self.grid.voxel
        nx, ny, nz = self.grid.shape
        xs, ys, zs = np.indices((nx, ny, nz))

        # find cente of each voxel
        flat_lab = label_cube.ravel()
        flat_x = edges[0] + (xs + 0.5) * s
        flat_y = edges[0] + (ys + 0.5) * s
        flat_z = edges[0] + (zs + 0.5) * s

        traces = []
        for group in ("IN", "OUT", "OTHER"):
            mask = np.array([
                lab.startswith("IN/") if group == "IN"
                else lab.startswith("OUT/") if group == "OUT"
                else (lab != '' and not lab.startswith(("IN/", "OUT/")))  # not empty, but also not IN or OUT
                for lab in flat_lab
            ])

            if not np.any(mask):
                continue

            trace = go.Scatter3d(
                x=flat_x.ravel()[mask],
                y=flat_y.ravel()[mask],
                z=flat_z.ravel()[mask],
                mode="markers",
                marker=dict(size=3, color=self._group_colors[group]),
                name=f"{group} (mode)"
            )
            traces.append(trace)

        fig = go.Figure(data=traces)
        fig.update_layout(
            scene=dict(
                xaxis_title='X [RE]', yaxis_title='Y [RE]', zaxis_title='Z [RE]',
                aspectmode='data'
            ),
            title="Voxel Averaged Regions (IN / OUT / OTHER)"
        )
        fig.show()