##############################################################################
# geospace_mapper/Snakefile   —   v3  (β-scalar voxel + tidy-ups)
##############################################################################
# Keys / constants
#   • YEARS        multi-year range for the region-probability map
#   • YEAR_BETA    calendar year of plasma-β study
#   • CL_BETA      Cluster spacecraft number                  (1-4)
#
# New in v3
#   1. scalar-mode voxelisation for plasma-β  (voxelise.py --scalar beta)
#   2. target cube   data/C{CL}_beta_voxel_{YEAR}.npz   holds mean β per voxel
##############################################################################

configfile: "config.yaml"

import glob, sys
from pathlib import Path

# ---------------------------------------------------------------------------
# GLOBAL CONFIG --------------------------------------------------------------
YEARS         = config["years"]                       # e.g. "2001-2005"
CORES         = int(config.get("cores", 1))

# ---- plasma-β study --------------------------------------------------------
YEAR_BETA     = 2005
CL_BETA       = 3
YEAR_BETA_PLUS = YEAR_BETA + 1

# ---------------------------------------------------------------------------
# PATH CONSTANTS -------------------------------------------------------------
DATA          = "data"
SCRIPTS       = "scripts"

HIA_DIR       = f"{DATA}/C{CL_BETA}_HIA_{YEAR_BETA}"
FGM_DIR       = f"{DATA}/C{CL_BETA}_FGM_{YEAR_BETA}"
HIA_UNPACK    = f"{HIA_DIR}_unpack_ok"
FGM_UNPACK    = f"{FGM_DIR}_unpack_ok"

BETA_CSV          = f"{DATA}/C{CL_BETA}_HIA_FGM_labelled_{YEAR_BETA}.csv"
BETA_PARQ_GSM     = f"{DATA}/C{CL_BETA}_beta_cube_gsm_{YEAR_BETA}.parquet"
BETA_VOX_MEAN_NPZ = f"{DATA}/C{CL_BETA}_beta_voxel_{YEAR_BETA}.npz"

PARQUET_POS   = f"{DATA}/cluster_{YEARS}.parquet"
VOXEL_POS     = f"{DATA}/cluster_{YEARS}_voxel.npz"

# ---------------------------------------------------------------------------
# HELPERS -------------------------------------------------------------------
def find_grmb(cluster: int) -> str:
    files = glob.glob(f"{DATA}/C{cluster}_CT_AUX_GRMB__*.cef")
    if len(files) != 1:
        sys.exit(f"Expected exactly one GRMB file, found {len(files)} ➜ {files}")
    return files[0]

GRMB_CEF = find_grmb(CL_BETA)

# ---------------------------------------------------------------------------
# RULE all  —  final artefacts -----------------------------------------------
rule all:
    input:
        VOXEL_POS,            # region probabilities (multi-sc)
        BETA_CSV,             # time-matched HIA+FGM records
        BETA_PARQ_GSM,        # β points in GSM coords (Parquet)
        BETA_VOX_MEAN_NPZ     # mean β per voxel (scalar cube)

##############################################################################
# 1. POSITION → REGION PROBABILITIES → VOXEL MAP  (unchanged)  ---------------
##############################################################################

rule download:
    output: touch(f"{DATA}/.download_ok")
    shell:  "python {SCRIPTS}/gse_download.py"

rule gsm_single:
    input:  f"{DATA}/cluster{{cid}}.txt"
    output: f"{DATA}/cluster{{cid}}_gsm.txt"
    shell:  "python {SCRIPTS}/convert_gse_to_gsm.py {{input}}"

rule gsm_all:
    input: expand(f"{DATA}/cluster{{cid}}_gsm.txt", cid=[1,2,3,4])

rule parquet:
    input:  expand(f"{DATA}/cluster{{cid}}_gsm.txt", cid=[1,2,3,4])
    output: PARQUET_POS
    params:
        pos   = DATA,
        grmb  = Path(GRMB_CEF).parent,
        yrs   = YEARS
    shell:
        """
        python {SCRIPTS}/match_gsm_grmb.py \
               --pos {params.pos} --cef {params.grmb} \
               --years {params.yrs} --out {output}
        """

rule voxelise_regions:
    input:  PARQUET_POS
    output: VOXEL_POS
    params:
        vox = config["voxel"],
        dt  = config["dt"]
    shell:
        """
        python voxelise.py --parquet {input} --out {output} \
               --vox {params.vox} --dt {params.dt}
        """

##############################################################################
# 2. CSA DOWNLOAD & GUNZIP ---------------------------------------------------
##############################################################################

rule download_fgm:
    output: directory(FGM_DIR)
    shell: """
        python {SCRIPTS}/csa_download.py \
                --dataset C{CL_BETA}_CP_FGM_SPIN \
                --start {YEAR_BETA}-01-01T00:00:00Z \
                --end   {YEAR_BETA_PLUS}-01-01T00:00:00Z \
                --outdir {output}
            """
rule download_hia:
    output: directory(HIA_DIR)
    shell: """
        python {SCRIPTS}/csa_download.py \
                --dataset C{CL_BETA}_CP_CIS-HIA_ONBOARD_MOMENTS \
                --start {YEAR_BETA}-01-01T00:00:00Z \
                --end   {YEAR_BETA_PLUS}-01-01T00:00:00Z \
                --outdir {output}
            """

rule unpack_fgm:
    input:  FGM_DIR
    output: FGM_UNPACK
    shell: "find {input} -name '*.cef.gz' -exec gunzip -f {{}} + && touch {output}"

rule unpack_hia:
    input:  HIA_DIR
    output: HIA_UNPACK
    shell: "find {input} -name '*.cef.gz' -exec gunzip -f {{}} + && touch {output}"

##############################################################################
# 3. HIA+FGM ➜ β CSV  --------------------------------------------------------
##############################################################################

rule beta_csv:
    input:
        hia  = HIA_UNPACK,
        fgm  = FGM_UNPACK,
        grmb = GRMB_CEF
    output:
        csv  = BETA_CSV,
        cube = f"{DATA}/C{CL_BETA}_beta_cube_{YEAR_BETA}.parquet"  # legacy
    threads: CORES
    shell:
        """
        python {SCRIPTS}/hia_fgm_pipeline.py \
               --year {YEAR_BETA} --cluster {CL_BETA} \
               --hia {HIA_DIR}   --fgm {FGM_DIR} \
               --grmb {input.grmb} --out {DATA}
        """

##############################################################################
# 4. β-cube:  GSE → GSM  -----------------------------------------------------
##############################################################################

rule beta_cube_gsm:
    input:  BETA_CSV
    output: BETA_PARQ_GSM
    threads: 4
    shell:
        """
        python {SCRIPTS}/convert_beta_gse_to_gsm.py \
               --csv {input} --out {output}
        """

##############################################################################
# 5. β-voxel (mean per cell)  -------------------------------------------------
##############################################################################

rule voxelise_beta:
    input:  BETA_PARQ_GSM
    output: BETA_VOX_MEAN_NPZ
    params:
        vox = 0.5        # R_E
    shell:
        """
        python voxelise.py \
               --parquet {input} \
               --out {output} \
               --vox {params.vox} \
               --scalar beta
        """

##############################################################################
# NOTE -----------------------------------------------------------------------
# • hia_fgm_pipeline.py must already include rglob() and relaxed header regex.
# • voxelise.py v3 supports both probability (default) and --scalar beta modes.
##############################################################################
