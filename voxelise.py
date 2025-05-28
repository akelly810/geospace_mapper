#!/usr/bin/env python3
"""
Voxelise a Parquet table into a regular 3-D grid.

* Probabilities (region labels)  → 4-D cube  (nx,ny,nz,R)
* Scalar field (e.g. plasma β)   → 3-D cube  (nx,ny,nz)

The script **streams** the Parquet file row-group by row-group using PyArrow
so it works even for multi-GB cubes.

Example
-------
# probability mode (region labels)
python voxelise.py --parquet data/C3_beta_cube_gsm_2005.parquet \
                   --out     data/C3_beta_voxel_2005.npz \
                   --vox 0.5 --dt 1

# scalar mode (voxelised mean β)
python voxelise.py --parquet data/C3_beta_cube_gsm_2005.parquet \
                   --out     data/C3_beta_voxel_2005.npz \
                   --vox 0.5 --scalar beta
"""
from __future__ import annotations
import argparse, numpy as np, pathlib as pl, pyarrow.parquet as pq
import pandas as pd

from src.config import GridSpec
from src.voxel  import Voxeliser          # unchanged, for probability mode

# --------------------------------------------------------------------- CLI --
ap = argparse.ArgumentParser(description="Voxelise Parquet → npz cube")
ap.add_argument("--parquet", required=True, type=pl.Path,
                help="Input Parquet file (GSM coordinates)")
ap.add_argument("--out",     required=True, type=pl.Path,
                help="Output *.npz (compressed)")
ap.add_argument("--vox",     required=True, type=float,
                help="Voxel edge length in R_E")
ap.add_argument("--dt",      type=float, default=1.0,
                help="[probability mode] Δt in minutes (default 1)")
ap.add_argument("--scalar",  metavar="COLUMN",
                help="Name of numeric column to voxelise (mean per cell). "
                     "If omitted, the script expects a 'region' column "
                     "and generates a probability cube.")
args = ap.parse_args()

# ------------------------------------------------------------------- GRID --
grid = GridSpec(voxel=args.vox)
nx = ny = nz = len(grid.edges) - 1        # cube is always cubic in GridSpec

# ----------------------------------------------------------------- HELPERS --
def to_indices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert x,y,z columns [R_E] → integer voxel indices (ix,iy,iz).
    Out-of-bounds points are masked with -1.
    """
    ix = np.floor((df["x"] / grid.re_km - grid.edges[0]) / args.vox).astype(int)
    iy = np.floor((df["y"] / grid.re_km - grid.edges[0]) / args.vox).astype(int)
    iz = np.floor((df["z"] / grid.re_km - grid.edges[0]) / args.vox).astype(int)

    ok = (0 <= ix) & (ix < nx) & (0 <= iy) & (iy < ny) & (0 <= iz) & (iz < nz)
    ix[~ok] = iy[~ok] = iz[~ok] = -1      # mark OOB
    return ix, iy, iz

# =============================================================== SCALAR MODE
if args.scalar:
    print(f"[voxelise] scalar-mode on '{args.scalar}'  → mean per voxel")
    # two auxiliary cubes: sum and count
    sum_cube   = np.zeros((nx, ny, nz), dtype=np.float64)
    count_cube = np.zeros((nx, ny, nz), dtype=np.int32)

    pq_file = pq.ParquetFile(args.parquet)
    for batch in pq_file.iter_batches():
        df = batch.to_pandas()
        if args.scalar not in df.columns:
            raise KeyError(f"Column '{args.scalar}' not in {args.parquet}")

        ix, iy, iz = to_indices(df)

        vals = pd.to_numeric(df[args.scalar], errors="coerce").to_numpy()
        for i, j, k, v in zip(ix, iy, iz, vals, strict=False):
            if i < 0:            # out-of-bounds – skip
                continue
            if np.isfinite(v):
                sum_cube[i, j, k]   += v
                count_cube[i, j, k] += 1

    # compute mean, leaving NaN where no samples
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_cube = sum_cube / count_cube
    mean_cube = mean_cube.astype(np.float32)

    np.savez_compressed(args.out,
                        P=mean_cube,
                        voxel=args.vox,
                        edges=grid.edges)
    print("✓ scalar cube saved ->", args.out)
    raise SystemExit(0)

# ========================================================= PROBABILITY MODE
print("[voxelise] probability-mode (region labels)")
pq_file = pq.ParquetFile(args.parquet)
vox     = Voxeliser(grid)

for batch in pq_file.iter_batches():
    df = batch.to_pandas()

    # ensure lowercase column is present
    if "region" not in df.columns:
        if "Region" in df.columns:
            df = df.rename(columns={"Region": "region"})
        else:
            raise KeyError("No 'region' column found in Parquet file")

    # convert distance columns to R_E
    for ax in ("x", "y", "z"):
        df[ax] /= grid.re_km

    vox.accumulate(df, dt_minutes=args.dt)

regions, P = vox.probs()
np.savez_compressed(args.out,
                    regions=np.array(regions, dtype=object),
                    P=P.astype(np.float32),
                    voxel=args.vox,
                    edges=grid.edges)
print("✓ probability cube saved ->", args.out)
