import pandas as pd

class ClusterPositions:
    def __init__(self, txt_file):
        self.txt_file = txt_file
        self.df = self._load()

    def _load(self):
        cols = ["time", "x", "y", "z"]
        return pd.read_csv(
            self.txt_file, sep=r"\s+", comment="#", names=cols,
            parse_dates=["time"], date_parser=lambda t: pd.to_datetime(t, utc=True)
        )

    def to_re(self, grid):
        for ax in ("x", "y", "z"):
            self.df[ax] /= grid.re_km
