# ------------------------------------------------------------
# Paths (all relative to the project root)
# ------------------------------------------------------------
DATA_DIR := data
SCRIPTS  := scripts
CEF_DIR  := $(DATA_DIR)

YEARS    := 2001-2005
PARQUET  := $(DATA_DIR)/cluster_$(YEARS).parquet

# Spacecraft IDs
CL_IDS  := 1 2 3 4
CL_GSE  := $(foreach id,$(CL_IDS),$(DATA_DIR)/cluster$(id).txt)
CL_GSM  := $(foreach id,$(CL_IDS),$(DATA_DIR)/cluster$(id)_gsm.txt)

# Stamp file to guard the download step
GSE_STAMP := $(DATA_DIR)/.download_ok

# ------------------------------------------------------------
# Phony targets
# ------------------------------------------------------------
.PHONY: all download gsm clean

# Default: build the final Parquet
all: $(PARQUET)

# ------------------------------------------------------------
# Ensure data directory exists
# ------------------------------------------------------------
$(DATA_DIR):
	@mkdir -p $@

# ------------------------------------------------------------
# STEP 1: Download GSE files (runs once, creates the stamp)
# ------------------------------------------------------------
download: $(GSE_STAMP)

$(GSE_STAMP): | $(DATA_DIR)
	@echo "-> Downloading raw GSE positions..."
	python $(SCRIPTS)/gse_download.py
	@touch $@

# If any raw file is missing, trigger the download
$(CL_GSE): $(GSE_STAMP)

# ------------------------------------------------------------
# STEP 2: Convert GSE -> GSM
# ------------------------------------------------------------
gsm: $(CL_GSM)

$(DATA_DIR)/cluster%_gsm.txt: $(DATA_DIR)/cluster%.txt | $(DATA_DIR)
	@echo "-> Converting $< to GSM..."
	python $(SCRIPTS)/convert_gse_to_gsm.py $<

# ------------------------------------------------------------
# STEP 3: Merge with GRMB labels -> Parquet
# ------------------------------------------------------------
$(PARQUET): $(CL_GSE) $(CL_GSM) $(wildcard $(CEF_DIR)/C?_CT_AUX_GRMB__*.cef) | $(DATA_DIR)
	@echo "-> Merging positions + GRMB -> $(notdir $@)..."
	python $(SCRIPTS)/match_gsm_grmb.py \
	    --pos   $(DATA_DIR) \
	    --cef   $(CEF_DIR) \
	    --years $(YEARS) \
	    --out   $@

# ------------------------------------------------------------
# Clean up derived files (keeps raw GSE files)
# ------------------------------------------------------------
clean:
	@echo "-> Cleaning GSM, Parquet, and download stamp..."
	@rm -f $(CL_GSM) $(PARQUET) $(GSE_STAMP)
