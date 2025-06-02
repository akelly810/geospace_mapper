#!/usr/bin/env python
"""
Voxelise either               - categorical Regions  → probability P(x,y,z,r)
or a scalar column (e.g. β) → mean value  S(x,y,z)

Usage examples
--------------
# original behaviour (region probabilities)
python voxelise.py --parquet data/cluster_2001-2005.parquet \
                   --out data/cluster_voxel.npz \
                   --vox 0.5 --dt 1

# mean β per voxel
python voxelise.py --parquet data/C3_beta_cube_gsm_2005.parquet \
                   --out     data/C3_beta_voxel_2005.npz \
                   --vox 0.5 --dt 1 --scalar beta
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import GridSpec           # defines uniform cubic grid in R_E
from tqdm import tqdm

# ----------------------------------------------------------------------
# CLI ------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--parquet", required=True,
                help="input Parquet (may be large → streamed chunk-by-chunk)")
ap.add_argument("--out",     required=True,
                help="output .npz (compressed)")
ap.add_argument("--vox",     type=float, required=True,
                help="voxel edge length in R_E")
ap.add_argument("--dt", type=float, default=1,
                help="[probability mode only] time step in minutes")
ap.add_argument("--scalar",  default=None,
                help="name of a scalar column to average instead of region probs")
args = ap.parse_args()

# ----------------------------------------------------------------------
# Grid initialisation --------------------------------------------------
grid = GridSpec(voxel=args.vox)           # provides .edges (n+1) and .re_km
NX = len(grid.edges)-1                    # cube size along one axis

# STORAGE
if args.scalar is None:
    # --- Region probability case (original) ---------------------------
    from src.voxel import Voxeliser
    vox = Voxeliser(grid)
else:
    # --- Scalar mean --------------------------------------------------
    sum_cube  = np.zeros((NX, NX, NX), dtype=np.float64)
    count_cube= np.zeros_like(sum_cube,        dtype=np.int32)

# ----------------------------------------------------------------------
# Stream through the Parquet file chunk-by-chunk -----------------------
reader = pd.read_parquet                   # pandas 2.2+ stream-reader
try:
    ds_iter = reader(
        args.parquet, chunksize=1_000_000, engine="pyarrow")   # type: ignore
except TypeError:
    # Older pandas/pyarrow versions - fall back to whole file
    ds_iter = [pd.read_parquet(args.parquet)]

for df in tqdm(ds_iter, desc="Voxelising", unit="rows"):
    # harmonise column case
    cols = {c.lower(): c for c in df.columns}
    if args.scalar is None and "region" not in cols:
        raise KeyError(f"'region' column not found in {args.parquet}")

    # positions to R_E
    for ax in ("x", "y", "z"):
        df[cols[ax]] /= grid.re_km

    # filter finite positions
    m = np.isfinite(df[cols["x"]]) & np.isfinite(df[cols["y"]]) & np.isfinite(df[cols["z"]])
    df = df.loc[m]

    # locate voxel indices
    ix = np.floor((df[cols["x"]] - grid.edges[0]) / args.vox).astype(int)
    iy = np.floor((df[cols["y"]] - grid.edges[0]) / args.vox).astype(int)
    iz = np.floor((df[cols["z"]] - grid.edges[0]) / args.vox).astype(int)

    in_bounds = (ix>=0)&(ix<NX)&(iy>=0)&(iy<NX)&(iz>=0)&(iz<NX)
    ix,iy,iz = ix[in_bounds], iy[in_bounds], iz[in_bounds]
    df       = df.iloc[in_bounds.values]        # keep rows in-bounds

    if args.scalar is None:
        # ========== probability ======================================
        vox.accumulate(df[["time", "region", cols["x"], cols["y"], cols["z"]]],
                       dt_minutes=args.dt)
    else:
        # ========== scalar mean ======================================
        values = pd.to_numeric(df[cols[args.scalar]], errors="coerce").values
        mask   = np.isfinite(values)
        ix,iy,iz,values = ix[mask],iy[mask],iz[mask],values[mask]
        # accumulate sums & counts
        np.add.at(sum_cube,   (ix,iy,iz), values)
        np.add.at(count_cube, (ix,iy,iz), 1)

# ----------------------------------------------------------------------
# Write output ---------------------------------------------------------
if args.scalar is None:
    regions, P = vox.probs()                       # (nx,ny,nz,R)
    np.savez_compressed(
        args.out, regions=np.array(regions),
        P=P.astype(np.float32),
        voxel=args.vox, edges=grid.edges)
else:
    mean_cube = np.full_like(sum_cube, np.nan, dtype=np.float32)
    m = count_cube > 0
    mean_cube[m] = (sum_cube[m] / count_cube[m]).astype(np.float32)

    np.savez_compressed(
        args.out, P=mean_cube,
        voxel=args.vox, edges=grid.edges)

print("voxel file written →", Path(args.out).resolve())
