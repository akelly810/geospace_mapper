#!/usr/bin/env python3
"""
Stream-convert the plasma-β data set from **GSE → GSM**.

INPUT   : C{N}_HIA_FGM_labelled_{YEAR}.csv  (output of hia_fgm_pipeline.py)
OUTPUT  : C{N}_beta_cube_gsm_{YEAR}.parquet (x,y,z,beta,Region • GSM, km)

Why km?  voxelise.py expects coordinates in km and normalises them
internally (df[ax] /= grid.re_km) before passing them to Voxeliser :contentReference[oaicite:0]{index=0}.
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from spacepy import coordinates as coord
from spacepy.time import Ticktock

coord.DEFAULTS.set_values(use_irbem=True, itol=10)      # matches convert_gse_to_gsm.py :contentReference[oaicite:1]{index=1}


# ---------------------------------------------------------------- helpers ----
def gse2gsm_block(df: pd.DataFrame) -> np.ndarray:
    """Return (n,3) GSM coords (km) for the rows in *df*."""
    time_col = df.columns[0]                    # robust: first col is the tag
    tt = pd.to_datetime(df[time_col], utc=True, format='mixed', errors='coerce')
    ticks = Ticktock(tt.to_list(), "UTC")
    xyz_km = df[["Position in GSE_0", "Position in GSE_1",
                 "Position in GSE_2"]].astype(float).to_numpy()
    return coord.Coords(xyz_km, "GSE", "car", ticks=ticks).convert("GSM", "car").data



def stream_convert(csv_path: Path, out_parquet: Path, chunksize: int = 500_000) -> None:
    keep = ["Center time", "Position in GSE_0", "Position in GSE_1",
            "Position in GSE_2", "beta", "Region"]

    writer: pq.ParquetWriter | None = None
    for chunk in pd.read_csv(csv_path, usecols=keep, chunksize=chunksize):
        gsm = gse2gsm_block(chunk)                        # km
        chunk[["x", "y", "z"]] = gsm
        slab = chunk[["x", "y", "z", "beta", "Region"]]   # minimise file size

        table = pa.Table.from_pandas(slab, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_parquet, table.schema,
                                      compression="zstd")
        writer.write_table(table)

    if writer is None:
        print("No rows found - nothing written.", file=sys.stderr)
    else:
        writer.close()
        print(f"✓ GSM β-cube written → {out_parquet.resolve()}")


# ----------------------------------------------------------------- main ------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert plasma-β cube from GSE to GSM (streaming)")
    p.add_argument("--csv",  type=Path, required=True,
                   help="C{N}_HIA_FGM_labelled_{YEAR}.csv")
    p.add_argument("--out",  type=Path, required=True,
                   help="Output Parquet path (GSM, km)")
    p.add_argument("--chunksize", type=int, default=500_000,
                   help="Rows per batch (default 500k)")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stream_convert(args.csv, args.out, args.chunksize)


if __name__ == "__main__":
    main()
