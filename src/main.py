from pathlib import Path
import argparse
import pandas as pd

from .config import GridSpec
from .grmb import GRMBIntervals
from .cluster import ClusterPositions
from .voxel import Voxeliser
from .plotter import VoxelPlotter

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cef", required=True, type=Path)
    p.add_argument("--pos", required=True, type=Path)
    p.add_argument("--year", type=int, default=2001)
    p.add_argument("--vox", type=float, default=0.5)
    args = p.parse_args()

    grid = GridSpec(voxel=args.vox)
    grmb = GRMBIntervals(args.cef, stop_before=pd.Timestamp(f"{args.year+1}-01-01T00:00:00Z"))
    pos = ClusterPositions(args.pos)
    pos.to_re(grid)
    pos.df["region"] = grmb.label_series(pos.df["time"])

    pos.df.to_csv('pos.csv', index=False)

    vox = Voxeliser(grid)
    vox.accumulate(pos.df)

    plot = VoxelPlotter(grid)
    plot.plot_regions_inout(vox.most_occupied())

if __name__ == "__main__":
    main()
