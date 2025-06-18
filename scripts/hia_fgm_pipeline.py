#!/usr/bin/env python3
"""
Convert one calendar year of CIS-HIA and FGM CEF files into

    • C{N}_HIA_FGM_labelled_{YEAR}.csv   (time-matched, GRMB-labelled)
    • C{N}_beta_cube_{YEAR}.parquet      (x,y,z,beta,Region -> ready for Dash)

Called by Snakemake or standalone (see Snakefile).
"""
from __future__ import annotations
import argparse, re, os
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from scipy.constants import mu_0
import pyarrow as pa
import pyarrow.parquet as pq

# ------------------------- helpers ----------------------------------
def extract_headers_from_cef(path: Path) -> list[str]:
    txt = path.read_text(encoding="utf-8")
    txt = re.sub(r"[ \t]+", " ", txt)
    hdr = txt.split("DATA_UNTIL = END_DATA")[0]
    blocks = re.findall(r"START_VARIABLE =(.*?)END_VARIABLE", hdr,
                        flags=re.DOTALL)
    out: list[str] = []
    for blk in blocks:
        field = re.search(r'FIELDNAM\s*=\s*"(.*?)"', blk)
        sizes = re.search(r'SIZES\s*=\s*([0-9,\s]+)', blk)
        if not field:
            continue
        name = field.group(1).strip()
        if sizes:
            dims = [int(x) for x in sizes.group(1).replace(" ", "").split(",")]
            n = dims[0] * (dims[1] if len(dims) > 1 else 1)
            out.extend([f"{name}_{i}" for i in range(n)] if n > 1 else [name])
        else:
            out.append(name)
    return out


def parse_cef(path: Path, keep_idx: list[int], keep_hdr: list[str]) -> pd.DataFrame:
    txt = path.read_text(encoding="utf-8")
    if "!RECORDS= 0" in txt:
        return pd.DataFrame()
    m = re.search(r"DATA_UNTIL\s*=\s*END_DATA\s*\n(.*?)\nEND_DATA", txt,
                  flags=re.DOTALL)
    if not m:
        return pd.DataFrame()
    rows = m.group(1).split(";" if ";" in m.group(1) else "$")
    dat = []
    for r in rows:
        vals = [v.strip() for v in r.split(",") if v.strip()]
        if len(vals) > max(keep_idx):
            dat.append([vals[i] for i in keep_idx])
    return pd.DataFrame(dat, columns=keep_hdr)


def concat_folder(folder: Path, keep_pat: dict[str, str],
                  label: str, out_csv: Path) -> pd.DataFrame:
    """Concat every science CEF in *folder* that contains ALL *keep_pat* fields."""
    def field_index_map(headers: list[str]) -> dict[str, int] | None:
        """Return {key: index} if *headers* satisfy every keep_pat regex."""
        idx = {}
        for k, pat in keep_pat.items():
            try:
                idx[k] = next(i for i, h in enumerate(headers)
                              if re.fullmatch(pat, h, flags=re.IGNORECASE))
            except StopIteration:
                return None          # one header missing -> reject this file set
        return idx

    # collect every *.cef* below folder (incl. sub-dirs)
    ceffiles = sorted(folder.rglob("*.cef"))

    hdrs, idx = None, None
    for f in ceffiles:
        cand = extract_headers_from_cef(f)
        idx   = field_index_map(cand)
        if idx:
            hdrs = cand
            break
    if hdrs is None:
        raise RuntimeError(f"{label}: no CEF in {folder} contains "
                           f"all required columns {list(keep_pat.values())}")

    keep_idx  = [idx[k] for k in keep_pat]   # preserve original order
    keep_hdrs = [hdrs[i] for i in keep_idx]

    # ---------- stream-concat every file that has those headers -------------
    frames = []
    for f in tqdm(ceffiles, desc=f"{label} CEF", colour="green"):
        h = extract_headers_from_cef(f)
        if field_index_map(h):                       # skip caveat / HK files
            df = parse_cef(f, keep_idx, keep_hdrs)
            if not df.empty:
                frames.append(df)

    if not frames:
        raise RuntimeError(f"{label}: no valid science CEFs in {folder}")

    comb = pd.concat(frames, ignore_index=True)
    comb.to_csv(out_csv, index=False)
    return comb



def match_nearest(left: pd.DataFrame, right: pd.DataFrame,
                  tol=pd.Timedelta(seconds=1)) -> pd.DataFrame:
    tl, tr = left.columns[0], right.columns[0]
    left[tl]  = pd.to_datetime(left[tl],  utc=True, errors="coerce")
    right[tr] = pd.to_datetime(right[tr], utc=True, errors="coerce")
    left.dropna(subset=[tl],  inplace=True)
    right.dropna(subset=[tr], inplace=True)
    left.sort_values(tl, inplace=True)
    right.sort_values(tr, inplace=True)
    out = pd.merge_asof(left, right, left_on=tl, right_on=tr,
                        direction="nearest", tolerance=tol)
    out.dropna(subset=[tr], inplace=True)
    out.drop(columns=[tr], inplace=True)
    return out


def parse_grmb(path: Path, year: int) -> pd.DataFrame:
    rows, data = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not data:
            data = line.startswith("DATA_UNTIL")
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        rng, lab = parts[0].strip('ENTRY = "'), parts[1].strip('"')
        if "/" in rng:
            rows.append((*rng.split("/"), lab))
    df = pd.DataFrame(rows, columns=["START", "END", "LABEL"])
    df["START"] = pd.to_datetime(df["START"], utc=True)
    df["END"]   = pd.to_datetime(df["END"],   utc=True)
    return df[(df["END"].dt.year >= year) & (df["START"].dt.year <= year)]\
             .reset_index(drop=True)


def add_labels(df: pd.DataFrame, lbl: pd.DataFrame) -> pd.DataFrame:
    t = df.columns[0]
    iv = pd.IntervalIndex.from_arrays(lbl["START"], lbl["END"], closed="left")
    idx = iv.get_indexer(df[t])
    df["Region"] = lbl["LABEL"].reindex(idx).fillna("UNLABELED").values
    return df

# ------------------------------ main ---------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year",    type=int, required=True)
    ap.add_argument("--cluster", type=int, required=True)
    ap.add_argument("--hia",     type=Path, required=True)
    ap.add_argument("--fgm",     type=Path, required=True)
    ap.add_argument("--grmb",    type=Path, required=True)
    ap.add_argument("--out",     type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # ---------- patterns we REALLY need --------------------------------
    NEEDED_HIA = {
        "ut":      r"^Center\s+time$",
        "density": r"^Density$",
        "temp":    r"^Temperature$",
    }
    # --- FGM headers: accept either the scalar magnitude OR the 3-vector ----
    NEEDED_FGM = {
        # time tag – any of these variants is fine
        "ut":     r"^(UT\s*Time|Universal\s*Time|UT|Epoch|Time.*)$",

        # magnetic field: accept either the scalar magnitude OR the 3-vector
        "B_mag":  r"^Magnetic\s*Field\s*Magnitude,\s*spin\s*resolution\s*$",
        "B_vec0": r"^Magnetic\s*Field\s*Vector,\s*spin\s*resolution\s*in\s*GSE[_ ]0$",
        "B_vec1": r"^Magnetic\s*Field\s*Vector,\s*spin\s*resolution\s*in\s*GSE[_ ]1$",
        "B_vec2": r"^Magnetic\s*Field\s*Vector,\s*spin\s*resolution\s*in\s*GSE[_ ]2$",

        # spacecraft position
        "pos_x":  r"^Position\s+in\s+GSE[_ ]0$",
        "pos_y":  r"^Position\s+in\s+GSE[_ ]1$",
        "pos_z":  r"^Position\s+in\s+GSE[_ ]2$",
    }

    hia_csv = args.out / f"C{args.cluster}_HIA_combined_{args.year}.csv"
    fgm_csv = args.out / f"C{args.cluster}_FGM_combined_{args.year}.csv"

    print("-> HIA   (extract & concat)")
    hia = concat_folder(args.hia, NEEDED_HIA, "HIA", hia_csv)

    print("-> FGM   (extract & concat)")
    fgm = concat_folder(args.fgm, NEEDED_FGM, "FGM", fgm_csv)

    print("-> time-matching HIA ⇄ FGM")
    matched = match_nearest(hia, fgm)

    print("-> GRMB label intervals")
    labels = parse_grmb(args.grmb, args.year)

    print("-> apply labels")
    labelled = add_labels(matched, labels)

    # ---------------------- compute beta & positions ----------------------
    print("-> compute plasma beta & stream-write")

    # open CSV writer
    csv_out  = args.out / f"C{args.cluster}_HIA_FGM_labelled_{args.year}.csv"
    cube_out = args.out / f"C{args.cluster}_beta_cube_{args.year}.parquet"

    chunksize = 1_000_000
    with csv_out.open("w", newline="") as fh:
        header_written = False
        writer = None
        for i in range(0, len(labelled), chunksize):
            chunk = labelled.iloc[i:i+chunksize].copy()

            # ---- compute beta for the chunk (same formula, float32 down-cast) ----
            n_cm3 = pd.to_numeric(chunk["Density"], errors="coerce", downcast="float")
            T_MK  = pd.to_numeric(chunk["Temperature"], errors="coerce", downcast="float")
            p_th  = n_cm3 * 1e6 * T_MK * 1e6 * 1.380649e-23

            if "Magnetic Field Magnitude, spin resolution" in chunk.columns:
                B_T = pd.to_numeric(chunk["Magnetic Field Magnitude, spin resolution"],
                                    errors="coerce", downcast="float") * 1e-9
            else:
                bx = pd.to_numeric(chunk["Magnetic Field Vector, spin resolution in GSE_0"],
                                errors="coerce", downcast="float")
                by = pd.to_numeric(chunk["Magnetic Field Vector, spin resolution in GSE_1"],
                                errors="coerce", downcast="float")
                bz = pd.to_numeric(chunk["Magnetic Field Vector, spin resolution in GSE_2"],
                                errors="coerce", downcast="float")
                B_T = (bx**2 + by**2 + bz**2).pow(0.5) * 1e-9

            chunk["beta"] = 2 * mu_0 * p_th / (B_T**2)
            chunk.dropna(subset=["beta"], inplace=True)

            # --- add position in R_E and select beta-cube columns -------------
            chunk["x"] = pd.to_numeric(chunk["Position in GSE_0"],
                                       errors="coerce",
                                       downcast="float") / 6371.0
            chunk["y"] = pd.to_numeric(chunk["Position in GSE_1"],
                                       errors="coerce",
                                       downcast="float") / 6371.0
            chunk["z"] = pd.to_numeric(chunk["Position in GSE_2"],
                                       errors="coerce",
                                       downcast="float") / 6371.0

            # ---- append to CSV -------------------------------------------------
            chunk.to_csv(fh, header=not header_written, index=False, mode="a")
            header_written = True

            # ---- append to Parquet
            cube_cols = chunk[["x", "y", "z", "beta", "Region"]]
            table = pa.Table.from_pandas(cube_cols, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(cube_out, table.schema, compression="zstd")
            writer.write_table(table)

        writer.close()

    print(f"wrote {csv_out}")
    print(f"wrote {cube_out}")

if __name__ == "__main__":
    main()
