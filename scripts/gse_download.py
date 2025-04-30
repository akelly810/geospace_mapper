"""
Download Cluster 1-4 GSE positions from NASA-SSCWeb.
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
from sscws.sscws import SscWs
from sscws.coordinates import CoordinateSystem

CADENCE_MIN = 1  # minutes

def year_chunks(start: datetime, stop: datetime):
    """Yield [start, stop] pairs covering the interval one calendar year each."""
    cur = start
    while cur <= stop:
        nxt = datetime(cur.year, 12, 31, 23, 59, 59)
        if nxt > stop:
            nxt = stop
        yield cur, nxt
        cur = nxt + timedelta(seconds=1)


def fetch_cluster_sat(sat_id: str, t0: datetime, t1: datetime, outfile: Path):
    """Download location records for one satellite and output to outfile."""
    ssc = SscWs()
    with outfile.open("w", buffering=1024 * 1024) as fh:
        fh.write(f"# {sat_id.upper()}  GSE position from {t0.isoformat()} to {t1.isoformat()}\n")
        fh.write("# UTC  X_km  Y_km  Z_km\n")

        for chunk_start, chunk_stop in year_chunks(t0, t1):
            iso = [chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                   chunk_stop.strftime("%Y-%m-%dT%H:%M:%SZ")]

            try:
                # we only have GSE coords
                res = ssc.get_locations2([sat_id], iso,
                                         coords=[CoordinateSystem.GSE])
            except Exception as exc:
                print(f"[{sat_id}] request failed for {iso}: {exc}", file=sys.stderr)
                continue

            data_block = res["Data"][0]
            times      = data_block["Time"]              # numpy.datetime64
            coords     = data_block["Coordinates"][0]    # GSE only
            xs, ys, zs = coords["X"], coords["Y"], coords["Z"]

            step = CADENCE_MIN
            for t, x, y, z in zip(times[::step], xs[::step], ys[::step], zs[::step]):
                fh.write(f"{str(t)}Z {x:.3f} {y:.3f} {z:.3f}\n")

    ssc.close()


def main():
    start = datetime(2001, 1, 1)
    stop  = datetime(2005, 12, 31, 23, 59, 59)

    probe = SscWs()
    obs = probe.get_observatories()["Observatory"]
    probe.close()

    cluster_ids = [o["Id"] for o in obs if o["Name"].lower().startswith("cluster")]
    if not cluster_ids:
        cluster_ids = ["cluster1", "cluster2", "cluster3", "cluster4"]

    for sat in sorted(cluster_ids):
        path = Path(f"{sat}.txt")
        print(f"Pulling {sat} -> {path}")
        fetch_cluster_sat(sat, start, stop, path)

    print("Done.")


if __name__ == "__main__":
    main()
