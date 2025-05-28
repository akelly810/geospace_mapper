#!/usr/bin/env python3
"""
Download one daily-tarball bundle from ESA/CSA TAP and unpack it.

Example:
    python scripts/csa_download.py \
        --dataset C3_CP_FGM_SPIN \
        --start  2008-01-01T00:00:00Z \
        --end    2009-01-01T00:00:00Z \
        --outdir data/C3_FGM_2008
"""
from __future__ import annotations
import argparse, tarfile
from pathlib import Path
import requests

URL = "https://csa.esac.esa.int/csa-sl-tap/data"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--start",   required=True)
    ap.add_argument("--end",     required=True)
    ap.add_argument("--outdir",  type=Path, required=True)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    tgz_path = args.outdir / "download.tgz"

    params = {
        "RETRIEVAL_TYPE":   "product",
        "DATASET_ID":       args.dataset,
        "START_DATE":       args.start,
        "END_DATE":         args.end,
        "DELIVERY_FORMAT":  "CEF",
        "DELIVERY_INTERVAL":"daily",
    }

    print(f"→ CSA download {args.dataset}")
    r = requests.get(URL, params=params, timeout=300)
    r.raise_for_status()
    tgz_path.write_bytes(r.content)
    print("✓ download complete")

    print("→ unpack")
    with tarfile.open(tgz_path) as tar:
        tar.extractall(args.outdir)
    tgz_path.unlink()                     # optional: keep directory clean

if __name__ == "__main__":
    main()
