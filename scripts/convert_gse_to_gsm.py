"""
Convert the Cluster positon data from GSE to GSM.
YYYY-MM-DDThh:mm:ssZ   X_GSM_km   Y_GSM_km   Z_GSM_km
"""

import argparse, glob, os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from spacepy import coordinates as coord
from spacepy.time import Ticktock

coord.DEFAULTS.set_values(use_irbem=True, itol=10)  # itol too small takes ages...

def convert_block(df):
    """Return an (n,3) array with GSM coordinates for the rows in df."""
    ticks = Ticktock(pd.to_datetime(df["time"], utc=True).to_list(), "UTC")
    xyz   = df[["x_gse", "y_gse", "z_gse"]].to_numpy(dtype=np.float64)

    c = coord.Coords(xyz, "GSE", "car", ticks=ticks)
    gsm = c.convert("GSM", "car").data      # in km
    return gsm


def process_one_file(fname, chunk_lines):
    base = Path(fname).stem
    out  = Path(fname).with_name(base + "_gsm.txt")
    cols = ["time", "x_gse", "y_gse", "z_gse"]

    write_hdr = True
    with pd.read_csv(fname, sep=r"\s+", names=cols,
                     comment="#", chunksize=chunk_lines) as reader, \
         out.open("w", buffering=1024 * 1024) as fh:

        if write_hdr:
            fh.write(f"# {base}  converted to GSM\n")
            fh.write("# UTC  X_km  Y_km  Z_km\n")

        for df in reader:
            gsm = convert_block(df)
            np.savetxt(fh,
                       np.column_stack((df["time"], gsm)),
                       fmt=["%s", "%.3f", "%.3f", "%.3f"],
                       delimiter=" ")

    print(f"Done: {out}")


def main():
    ap = argparse.ArgumentParser(
        description="Convert Cluster SSCWeb position data (GSE) to GSM.")
    ap.add_argument("files", nargs="*", default=glob.glob("cluster?.txt"),  # ? -> followed by 1 char
                    help="Input text files (default: cluster?.txt)")
    ap.add_argument("--chunk", type=int, default=500_000,
                    help="Lines per chunk (lower to reduce memory)")
    args = ap.parse_args()

    if not args.files:
        print("No input files found.", file=sys.stderr)
        sys.exit(1)

    for f in args.files:
        process_one_file(f, args.chunk)


if __name__ == "__main__":
    main()
