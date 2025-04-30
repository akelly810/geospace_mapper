from pathlib import Path
import re
import pandas as pd

class GRMBIntervals:
    # e.g. 2001-01-03T13:15:00Z/2001-01-03T14:20:00Z, "IN/PS"
    _pat = re.compile(r"(\d{4}-\d{2}-\d{2}T[0-9:]{8}Z)/"
                      r"(\d{4}-\d{2}-\d{2}T[0-9:]{8}Z),\s*\"([^\"]+)\"")

    def __init__(self, cef_file, stop_before=None):
        self.cef_file = Path(cef_file)
        self.stop_before = stop_before
        self.table = self._parse()

    def _parse(self) -> pd.DataFrame:
        recs: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
        with self.cef_file.open() as fh:
            for line in fh:
                m = self._pat.match(line)
                if not m:
                    continue
                t0, t1, label = m.groups()
                t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)
                if self.stop_before and t0 >= self.stop_before:
                    break
                recs.append((t0, t1, label))
        return pd.DataFrame(recs, columns=["start", "stop", "region"])

    def label_series(self, times: pd.Series) -> pd.Series:
        idx = pd.IntervalIndex.from_arrays(self.table["start"], self.table["stop"], closed="left")

        def _label(ts):
            try:
                return self.table["region"].iloc[idx.get_loc(ts)]
            except KeyError:
                return None

        return times.map(_label)
