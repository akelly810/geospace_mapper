## Download and Process Cluster Position Data
Spacecraft orbit data can be pulled and processed uning the two included scripts:
1. `gse_download.py`
2. `convert_gse_to_gsm.py`


### 1. `gse_download.py` – Download Cluster Positions (GSE)

This script downloads position data for Cluster 1–4 from NASA SSCWeb in **GSE coordinates** (in kilometers) and saves each spacecraft's data as a separate `.txt` file.

#### Instructions:
1. Open `gse_download.py`.
2. Modify the `start` and `stop` date variables near the bottom of the script to define your desired time range.
3. Specify `CADENCE_MIN` as the resolution in minutes (default 1 minute).
4. Run the script:
   ```bash
   python gse_download.py
   ```

Each `.txt` file will contain time-stamped position data, with columns:  
`UTC_timestamp   X_GSE_km   Y_GSE_km   Z_GSE_km`.

### 2. `convert_gse_to_gsm.py` – Convert to GSM Coordinates

This script converts all GSE `.txt` files matching the pattern `cluster*.txt` into their **GSM coordinate** equivalents using the SpacePy library.

#### Instructions:
1. Place `convert_gse_to_gsm.py` in the same directory as the GSE `.txt` files generated in the previous step.
2. Run the script:
   ```bash
   python convert_gse_to_gsm.py
   ```

This will create new `.txt` files with `_gsm` appended to the filename (e.g., `cluster1_gsm.txt`), containing:
`UTC_timestamp   X_GSM_km   Y_GSM_km   Z_GSM_km`.

## Main code
...
