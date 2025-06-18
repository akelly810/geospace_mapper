# Quick-Start • Geospace Mapper

Turn Cluster mission data + GRMB labels into 3-D voxel cubes and explore them with a Dash web-app.

---

## 1  Install
Configure your Python environment by installing the required packages.
I don't know if the below will work immediately, but you can figure it out.

Option 1: Conda
```bash
conda create -n gmapper python=3.11 \
    numpy pandas scipy pyarrow tqdm \
    dash plotly dash-bootstrap-components \
    snakemake spacepy -c conda-forge
conda activate gmapper
```

Option 2: pip + conda forge
```bash
python -m venv gm-env
source gm-env/bin/activate   # Windows: gm-env\Scripts\activate
pip install numpy pandas scipy pyarrow tqdm \
            dash plotly dash-bootstrap-components snakemake
conda install -c conda-forge spacepy   # SpacePy needs compiled libs

```

## 2 Edit config.yaml
Change years, voxel size, etc.

## 3 Run the pipeline
This produces the build artefacts and datacubes required for visualisation
```bash
snakemake -j 8
```

Outputs appear in data/:
datacube_YYYY-YYYY.npz     # region probabilities
C3_beta_voxel.npz          # plasma-β voxels (optional)

## 4 Launch the web app
Region datacube is required, beta cube is optional
```bash
python app_dash.py
```
