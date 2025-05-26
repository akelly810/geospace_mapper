##############################################################################
# geospace_mapper/Snakefile
##############################################################################

configfile: "config.yaml"

YEARS = config["years"]
CORES = int(config.get("cores", 1))
CLIDS = [1, 2, 3, 4]

SCRIPTS = "scripts"
DATA    = "data"

rule all:
    input:
        f"{DATA}/cluster_{YEARS}_voxel.npz"

# 1. download -------------------------------------------------------
rule download:
    output: touch(f"{DATA}/.download_ok")
    shell:  "python {SCRIPTS}/gse_download.py"

# 2. convert --------------------------------------------------------
rule convert_gsm:
    input:
        f"{DATA}/.download_ok"
    output:
        expand(f"{DATA}/cluster{{cid}}_gsm.txt", cid=CLIDS)
    shell:
        "python {SCRIPTS}/convert_gse_to_gsm.py {wildcards.cid}"

# 3. label & merge --------------------------------------------------
rule label_regions:
    input:
        expand(f"{DATA}/cluster{{cid}}_gsm.txt", cid=CLIDS)
    output:
        f"{DATA}/cluster_{YEARS}.parquet"
    shell:
        ("python {SCRIPTS}/match_gsm_grmb.py "
         "--pos {DATA} --cef {DATA} "
         "--years {YEARS} --out {output}")

# 4. voxelise -------------------------------------------------------
rule voxelise:
    input:
        f"{DATA}/cluster_{YEARS}.parquet"
    output:
        f"{DATA}/cluster_{YEARS}_voxel.npz"
    params:
        voxel = config["voxel"],
        dt    = config["dt"]
    shell:
        ("python voxelise.py "
         "--parquet {input} "
         "--out {output} "
         "--vox {params.voxel} "
         "--dt {params.dt}")
