#!/usr/bin/env python
"""Interactive Dash dashboard for Cluster-GRMB voxel data (XY/XZ/YZ planes)."""
from pathlib import Path
import numpy as np
import dash, dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd

# ---------------------------------------------------------------------
# LOAD DATA -----------------------------------------------------------
VOXFILE = Path("data/cluster_2001-2005_voxel.npz")
data    = np.load(VOXFILE, allow_pickle=True)
regions = list(data["regions"])
P       = data["P"]                      # (nx,ny,nz,R)
edges   = data["edges"]
voxel   = float(data.get("voxel", edges[1]-edges[0]))
nx, ny, nz, R = P.shape

# Axis mid-points in R_E
x_cent = edges[0] + (np.arange(nx) + 0.5) * voxel
y_cent = edges[0] + (np.arange(ny) + 0.5) * voxel
z_cent = edges[0] + (np.arange(nz) + 0.5) * voxel

# ---------------------------------------------------------------------
# LOAD β-VOXEL DATA ---------------------------------------------------
BETA_VOXFILE = Path("data/C3_beta_voxel_2005.npz")
beta_data    = np.load(BETA_VOXFILE, allow_pickle=True)
β_P          = beta_data["P"]           # (nx, ny, nz) β per cell
β_edges      = beta_data["edges"]
β_voxel      = float(beta_data["voxel"])
β_cent = β_edges[:-1] + np.diff(β_edges)/2

# ---------------------------------------------------------------------
# COLOUR MAP ----------------------------------------------------------
region_colors = {
    "IN/UKN":"#8FBFFF","IN/PLS":"#6EA8FF","IN/PPTR":"#4A90FF",
    "IN/PSH":"#32CD32","IN/PSTR":"#228B22",
    "IN/LOB":"#0099CC","IN/POL":"#272A60",
    "IN/MP":"#A060E8","IN/MPTR":"#C184F5",
    "OUT/MSH":"#FF3D3D","OUT/BSTR":"#FF6A3D",
    "OUT/SWF":"#FF944D","OUT/UKN":"#FFB570",
    "UNKNOWN":"#9E9E9E","N/A":"#D3D3D3",
}
EMPTY_COLOR = "#000000"
color_lut = {lab: tuple(int(region_colors[lab].lstrip("#")[i:i+2], 16)
                        for i in (0, 2, 4))
             for lab in regions}
color_lut["empty"] = (0, 0, 0)

# ---------------------------------------------------------------------
# HELPERS -------------------------------------------------------------
axis_labels = {
    "xy": ("X [R<sub>E</sub>]", "Y [R<sub>E</sub>]"),
    "xz": ("X [R<sub>E</sub>]", "Z [R<sub>E</sub>]"),
    "yz": ("Y [R<sub>E</sub>]", "Z [R<sub>E</sub>]"),
}
centers = {
    "xy": (x_cent, y_cent),
    "xz": (x_cent, z_cent),
    "yz": (y_cent, z_cent),
}
max_index = {"xy": nz-1, "xz": ny-1, "yz": nx-1}

def slice_cube(plane: str, idx: int) -> np.ndarray:
    """Return (rows, cols, R) slice cube oriented for Plotly (rows = Y/Z)."""
    if plane == "xy":
        return P[:, :, idx, :].transpose(1, 0, 2)   # (ny,nx,R)
    if plane == "xz":
        return P[:, idx, :, :].transpose(1, 0, 2)   # (nz,nx,R)
    if plane == "yz":
        return P[idx, :, :, :].transpose(1, 0, 2)   # (nz,ny,R)
    raise ValueError(plane)

def slice_beta_cube(plane: str, idx: int) -> np.ndarray:
    """
    Return a 2D array of β for a given plane and index,
    oriented so rows = Y/Z and cols = X/Y as with slice_cube.
    """
    if plane == "xy":
        # Z = idx slice: β_P[:,:,idx] shape (nx,ny) → transpose to (ny,nx)
        return β_P[:, :, idx].T
    if plane == "xz":
        # Y = idx slice: β_P[:,idx,:] shape (nx,nz) → transpose to (nz,nx)
        return β_P[:, idx, :].T
    if plane == "yz":
        # X = idx slice: β_P[idx,:,:] shape (ny,nz) → transpose to (nz,ny)
        return β_P[idx, :, :].T
    raise ValueError(f"Unknown plane {plane!r}")


# ---------------------------------------------------------------------
# IMAGE (modal) -------------------------------------------------------
def modal_trace(cube: np.ndarray) -> go.Image:
    rows, cols, _ = cube.shape
    winner_idx  = cube.argmax(-1)
    winner_prob = cube[
        np.arange(rows)[:, None], np.arange(cols)[None, :], winner_idx]
    visited     = cube.sum(-1) > 0

    rgb       = np.zeros((rows, cols, 3), dtype=np.uint8)
    label_arr = np.full((rows, cols), "∅ (no data)", dtype=object)
    prob_arr  = np.zeros((rows, cols), dtype=float)

    for ridx, lab in enumerate(regions):
        m = (winner_idx == ridx) & visited
        rgb[m] = color_lut[lab]
        label_arr[m] = lab
        prob_arr[m]  = winner_prob[m]

    rgb[~visited] = color_lut["empty"]
    custom = np.dstack((label_arr, prob_arr))

    return go.Image(
        z=rgb,
        customdata=custom,
        hovertemplate="%{customdata[0]}<br>P=%{customdata[1]:.2f}<extra></extra>",
    )

# ---------------------------------------------------------------------
# PROBABILITY HEATMAP -------------------------------------------------
def prob_figure(cube: np.ndarray, region: str,
                xpts, ypts, xlab, ylab, title) -> go.Figure:
    ridx = regions.index(region)
    zmat = cube[..., ridx]                       # (rows, cols)

    # build hover text with top-3
    top_idx  = np.argsort(cube, axis=-1)[..., -1:-4:-1]
    top_prob = np.take_along_axis(cube, top_idx, axis=-1)
    top_lab  = np.vectorize(lambda i: regions[i])(top_idx)

    hover = np.empty(zmat.shape, dtype=object)
    for r in range(zmat.shape[0]):
        for c in range(zmat.shape[1]):
            if cube[r, c].sum() == 0:
                hover[r, c] = "∅ (no data)"
            else:
                lines = [f"{lab} : {p:.2f}"
                         for lab, p in zip(top_lab[r, c], top_prob[r, c])]
                hover[r, c] = "<br>".join(lines)

    fig = go.Figure(go.Heatmap(
        z=zmat, x=xpts, y=ypts,
        colorscale="Viridis", zmin=0, zmax=1,
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
        colorbar=dict(title=f"P({region})"),
    ))
    fig.update_layout(
        width=500, height=500,
        xaxis=dict(scaleanchor="y", title=xlab),
        yaxis=dict(title=ylab),
        margin=dict(l=40, r=40, t=40, b=40),
        title=title,
    )
    return fig

# DASH APP ------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Cluster GRMB Explorer"

# --- CONTROLS ---
plane_dd = dcc.Dropdown(
    id="plane-dd",
    options=[{"label":"X-Y","value":"xy"},
             {"label":"X-Z","value":"xz"},
             {"label":"Y-Z","value":"yz"}],
    value="xy", clearable=False)

region_dd = dcc.Dropdown(id="region-dd",
                         options=[{"label":r,"value":r} for r in regions],
                         value="OUT/MSH", clearable=False)

# slice / iso slider (will be repurposed in 3-D view)
slice_slider = dcc.Slider(id="slice-slider", min=0, max=max_index["xy"],
                          step=1, value=0)

# probability threshold for isosurface
iso_slider = dcc.Slider(id="iso", min=0.05, max=0.9, step=0.05,
                        value=0.25, tooltip={"always_visible":False},
                        marks={0.25:"0.25",0.5:"0.5"})

# view selector
view_tabs = dcc.Tabs(
    id="view", value="slice",
    children=[
        dcc.Tab(label="Slice view",  value="slice"),
        dcc.Tab(label="3-D volume",  value="vol"),
        dcc.Tab(label="Beta slice",     value="beta_slice"),
    ])

app.layout = dbc.Container([
    dbc.Row([view_tabs], className="mb-2"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="modal-graph"), md=6),
        dbc.Col(dcc.Graph(id="prob-graph"),  md=6),
    ]),
    dbc.Row([
        dbc.Col(plane_dd,     md=2),
        dbc.Col(slice_slider, md=5),
        dbc.Col(region_dd,    md=2),
        dbc.Col(html.Div("iso-P"), md=1, style={"textAlign":"right"}),
        dbc.Col(iso_slider,   md=2),
    ], className="mt-3"),
])
# ---------------------------------------------------------------------
# CALLBACK ------------------------------------------------------------
@app.callback(
    Output("modal-graph","figure"),
    Output("prob-graph", "figure"),
    Output("slice-slider","max"),
    Output("slice-slider","value"),
    Output("modal-graph","style"),   # hide in 3-D view
    Output("plane-dd","disabled"),   # disable plane select in 3-D
    Input("plane-dd",   "value"),
    Input("slice-slider","value"),
    Input("region-dd",  "value"),
    Input("view",       "value"),
    Input("iso",        "value"),
)
def refresh(plane, idx, region, view, iso_thr):
    max_int = max_index[plane]
    idx     = 0 if idx is None else min(idx, max_int)

    if view == "beta_slice":
        # compute slab coordinate
        if plane=="xy":
            slice_val = z_cent[idx]
        elif plane=="xz":
            slice_val = y_cent[idx]
        else:
            slice_val = x_cent[idx]

        # get the 2D β matrix
        β_mat = slice_beta_cube(plane, idx)

        xpts, ypts = centers[plane]
        finite     = np.isfinite(β_mat)
        z_max      = np.nanpercentile(β_mat[finite], 95) if finite.any() else 1

        figβ = go.Figure(go.Heatmap(
            z=β_mat,
            x=xpts,
            y=ypts,
            colorscale="Turbo",
            zmin=0,
            zmax=z_max,
            colorbar=dict(title="β"),
            hovertemplate="β=%{z:.2f}<br>" + axis_labels[plane][0] + "=%{x:.2f}<br>" + axis_labels[plane][1] + "=%{y:.2f}<extra></extra>"
        ))
        figβ.update_layout(
            width=500, height=500,
            xaxis=dict(title=axis_labels[plane][0]),
            yaxis=dict(title=axis_labels[plane][1]),
            margin=dict(l=40, r=40, t=40, b=40),
            title=f"β slice — {plane.upper()} = {slice_val:.2f} Rₑ"
        )

        # hide the other two plots and disable controls
        return figβ, go.Figure(), max_int, idx, {}, True

    cube    = slice_cube(plane, idx)

# -------- 3-D VOLUME VIEW ---------------------------------------
    if view == "vol":
        ridx = regions.index(region)
        fig3d = go.Figure(go.Isosurface(
            x=np.repeat(x_cent, ny*nz),
            y=np.tile(np.repeat(y_cent, nz), nx),
            z=np.tile(z_cent, nx*ny),
            value=P[..., ridx].ravel(),
            isomin=iso_thr, isomax=1.0,
            surface_count=1,
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorscale="Viridis",
            showscale=True,
        ))
        fig3d.update_layout(
            scene=dict(
                xaxis_title="X [R_E]", yaxis_title="Y [R_E]", zaxis_title="Z [R_E]",
                aspectmode="data"),
            margin=dict(l=0, r=0, b=0, t=40),
            title=f"Isosurface P({region}) ≥ {iso_thr}"
        )
        hide_modal = {"display":"none"}
        disable_plane = True

        # Return 3-D fig in right panel, leave left empty/hidden
        return go.Figure(), fig3d, max_int, idx, hide_modal, disable_plane

    xpts, ypts = centers[plane]
    xlab, ylab = axis_labels[plane]

    if plane == "xy":
        slice_title = f"Z = {z_cent[idx]:.2f} R<sub>E</sub>"
    elif plane == "xz":
        slice_title = f"Y = {y_cent[idx]:.2f} R<sub>E</sub>"
    else:
        slice_title = f"X = {x_cent[idx]:.2f} R<sub>E</sub>"

    modal = go.Figure(modal_trace(cube))
    modal.update_layout(
        width=500, height=500,
        xaxis=dict(scaleanchor="y", title=xlab),
        yaxis=dict(title=ylab),
        margin=dict(l=40,r=40,t=40,b=40),
        title=f"Dominant region -- {slice_title}",
    )

    prob = prob_figure(cube, region, xpts, ypts,
                       xlab, ylab,
                       title=f"P({region}) -- {slice_title}")

    return modal, prob, max_int, idx, {}, False

# ---------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
