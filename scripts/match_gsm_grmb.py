#!/usr/bin/env python3
"""
match_gsm_grmb.py  –  memory-efficient version

Reads large Cluster *_gsm.txt files in chunks, attaches GRMB region labels,
and appends each chunk to a Parquet file.  Handles multi-year ranges without
blowing up RAM.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ------------------------------------------------------------------ helpers ---
class GRMBIntervals:
    """Parse a GRMB CEF file and expose an interval-lookup helper."""

    _pat = re.compile(
        r"(\d{4}-\d{2}-\d{2}T[0-9:]{8}Z)/"
        r"(\d{4}-\d{2}-\d{2}T[0-9:]{8}Z),\s*\"([^\"]+)\""
    )

    def __init__(self, cef_path: Path, stop_before: pd.Timestamp):
        self.table = self._parse(Path(cef_path), stop_before)

    @staticmethod
    def _parse(cef_path: Path, stop_before: pd.Timestamp) -> pd.DataFrame:
        recs: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = []
        with cef_path.open() as fh:
            for line in fh:
                m = GRMBIntervals._pat.match(line)
                if not m:
                    continue
                t0, t1, label = m.groups()
                t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)
                if t0 >= stop_before:
                    break      # CEF is time-ordered → safe to stop
                recs.append((t0, t1, label))
        return pd.DataFrame(recs, columns=["start", "stop", "region"])

    # fast, reusable lookup
    def make_labeller(self):
        idx = pd.IntervalIndex.from_arrays(self.table.start,
                                           self.table.stop,
                                           closed="left")

        def _lab(ts):
            try:
                return self.table.region.iloc[idx.get_loc(ts)]
            except KeyError:
                return None

        return _lab


# ----------------------------------------------------------------- utilities -
def parse_years(text: str) -> Tuple[int, int]:
    """Return (start_year, end_year) inclusive."""
    if "-" in text:
        a, b = text.split("-", 1)
        return int(a), int(b)
    y = int(text)
    return y, y


def write_chunk(
    df: pd.DataFrame,
    writer: Optional[pq.ParquetWriter],
    out_path: Path,
) -> pq.ParquetWriter:
    """Append chunk to Parquet, open writer if first time."""
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(out_path, tbl.schema, compression="zstd")
    writer.write_table(tbl)
    return writer


# -------------------------------------------------------------------- main ---
def main() -> None:
    ap = argparse.ArgumentParser(description="Merge Cluster GSM positions with GRMB labels (streaming)")
    ap.add_argument("--pos", type=Path, required=True, help="Directory with cluster?_gsm.txt files")
    ap.add_argument("--cef", type=Path, required=True, help="Directory with GRMB CEF files")
    ap.add_argument("--years", required=True, metavar="Y or Y1-Y2",
                    help="Year or inclusive range (e.g. 2001 or 2003-2007)")
    ap.add_argument("--out", type=Path, default=Path("matched_positions.parquet"),
                    help="Output Parquet path")
    ap.add_argument("--chunksize", type=int, default=200_000,
                    help="Rows per chunk to process (default 200k)")
    args = ap.parse_args()

    start_y, end_y = parse_years(args.years)
    if end_y < start_y:
        sys.exit("ERROR: end year must be ≥ start year")

    t_start = pd.Timestamp(f"{start_y}-01-01T00:00:00Z")
    t_stop  = pd.Timestamp(f"{end_y + 1}-01-01T00:00:00Z")

    writer: Optional[pq.ParquetWriter] = None
    total_rows = 0

    for cid in range(1, 5):
        pos_file = args.pos / f"cluster{cid}_gsm.txt"
        if not pos_file.is_file():
            sys.exit(f"Missing {pos_file}")

        cef_match = sorted((args.cef).glob(f"C{cid}_CT_AUX_GRMB__*.cef"))
        if not cef_match:
            sys.exit(f"No GRMB CEF for Cluster {cid} in {args.cef}")
        grmb = GRMBIntervals(cef_match[0], stop_before=t_stop)
        labeller = grmb.make_labeller()

        print(f"[{cid}] streaming {pos_file.name}")
        cols = ["time", "x", "y", "z"]
        for chunk in pd.read_csv(
                pos_file,
                sep=r"\s+",
                comment="#",
                names=cols,
                parse_dates=["time"],
                date_parser=lambda t: pd.to_datetime(t, utc=True),
                chunksize=args.chunksize):

            chunk = chunk[(chunk.time >= t_start) & (chunk.time < t_stop)]
            if chunk.empty:
                continue

            chunk["cluster"] = cid
            chunk["region"] = chunk.time.map(labeller)

            writer = write_chunk(chunk, writer, args.out)
            total_rows += len(chunk)
            print(f"    wrote {len(chunk):,} rows  (running total {total_rows:,})")

    if writer is not None:
        writer.close()
        print(f"✓ Finished.  Total rows written: {total_rows:,}")
        print(f"  File: {args.out.resolve()}")
    else:
        print("No data matched the time window; nothing written.")


if __name__ == "__main__":
    main()
