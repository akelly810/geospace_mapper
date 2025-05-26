#!/usr/bin/env python
import argparse, numpy as np, pandas as pd, pathlib as pl
from src.config import GridSpec
from src.voxel  import Voxeliser

p = argparse.ArgumentParser()
p.add_argument("--parquet", required=True)
p.add_argument("--out", required=True)
p.add_argument("--vox",  type=float, required=True)
p.add_argument("--dt",   type=float, required=True)
args = p.parse_args()

grid = GridSpec(voxel=args.vox)
df   = pd.read_parquet(args.parquet)
for ax in ("x","y","z"):
    df[ax] /= grid.re_km

vox = Voxeliser(grid)
vox.accumulate(df, dt_minutes=args.dt)

regions, P = vox.probs()
np.savez_compressed(args.out, regions=np.array(regions), P=P,
                    voxel=args.vox, edges=grid.edges)
print("cube saved ->", args.out)
