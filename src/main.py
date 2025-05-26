from pathlib import Path
import argparse
import pandas as pd

from .config import GridSpec
from .voxel import Voxeliser
from .plotter import VoxelPlotter


def main() -> None:
    p = argparse.ArgumentParser(
        description="Voxelise and plot Cluster + GRMB Parquet dataset"
    )
    p.add_argument("--parquet", required=True, type=Path,
                   help="File produced by match_gsm_grmb.py")
    p.add_argument("--vox", type=float, default=0.5,
                   help="Voxel edge length in RE (default 0.5)")
    p.add_argument("--dt", type=float, default=1.0,
                   help="Time step per sample in minutes (default 1)")
    args = p.parse_args()

    # -------------------------------------------------- load + normalise
    print(f"Reading {args.parquet}")
    df = pd.read_parquet(args.parquet)

    grid = GridSpec(voxel=args.vox)
    for ax in ("x", "y", "z"):
        df[ax] /= grid.re_km        # km -> RE

    # -------------------------------------------------- voxelise
    vox = Voxeliser(grid)
    vox.accumulate(df, dt_minutes=args.dt)

    # -------------------------------------------------- visualise
    plot = VoxelPlotter(grid)
    plot.plot_regions_inout(vox.most_occupied())


if __name__ == "__main__":
    main()
