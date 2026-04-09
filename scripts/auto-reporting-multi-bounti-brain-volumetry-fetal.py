import os
import sys
import base64
from io import BytesIO

import nibabel as nib
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.colors import ListedColormap

REPORT_TITLE = "Multi-BOUNTI report for fetal MRI"
GA_MIN = 20
GA_MAX = 40

LABEL_COLORS = {0: (0, 0, 0, 0.0), 1: (4, 51, 255, 1.0), 2: (21, 144, 156, 1.0), 3: (99, 22, 167, 1.0), 4: (103, 66, 230, 1.0), 5: (176, 51, 120, 1.0), 6: (190, 72, 8, 1.0), 7: (36, 131, 99, 1.0), 8: (23, 38, 95, 1.0), 9: (90, 40, 55, 1.0), 10: (131, 31, 44, 1.0), 11: (226, 121, 9, 1.0), 12: (212, 33, 20, 1.0), 13: (217, 75, 59, 1.0), 14: (188, 113, 217, 1.0), 15: (87, 224, 239, 1.0), 16: (171, 56, 91, 1.0), 17: (212, 122, 212, 1.0), 18: (233, 122, 174, 1.0), 19: (167, 90, 162, 1.0), 20: (171, 107, 130, 1.0), 21: (147, 112, 219, 1.0), 22: (55, 150, 190, 1.0), 23: (120, 34, 185, 1.0), 24: (29, 94, 199, 1.0), 25: (255, 191, 0, 1.0), 26: (255, 255, 255, 1.0), 27: (149, 132, 39, 1.0), 28: (153, 47, 33, 1.0), 29: (96, 0, 128, 1.0), 30: (134, 176, 92, 1.0), 31: (149, 6, 78, 1.0), 32: (131, 63, 24, 1.0), 33: (140, 35, 38, 1.0), 34: (141, 32, 158, 1.0), 35: (135, 47, 176, 1.0), 36: (162, 52, 104, 1.0), 37: (156, 255, 161, 1.0), 38: (174, 219, 255, 1.0), 39: (92, 167, 221, 1.0), 40: (115, 212, 175, 1.0), 41: (112, 139, 248, 1.0), 42: (171, 62, 25, 1.0), 43: (169, 82, 226, 1.0)}

COEFFICIENTS = {'DGM_caudate_nucleus_volume': {'a50': 0.00396901070925127,
                                'astd': -5.86084682909637e-06,
                                'b50': -0.144544873360137,
                                'bstd': 0.00720446615292477,
                                'c50': 1.51498173637729,
                                'cstd': -0.11662908394105},
 'DGM_lentiform_nucleus_volume': {'a50': 0.0061961627641139,
                                  'astd': 0.000897673714326503,
                                  'b50': -0.138715906260165,
                                  'bstd': -0.0354996504407343,
                                  'c50': 1.07514141158107,
                                  'cstd': 0.447699389627272},
 'DGM_thalamus_volume': {'a50': 0.00955522116813633,
                         'astd': 0.0003,
                         'b50': -0.235677179714089,
                         'bstd': -0.0015,
                         'c50': 1.9980690463482,
                         'cstd': 0.001},
 'GM_cingulate_volume': {'a50': 0.00725241123289672,
                         'astd': 0.0,
                         'b50': -0.204525212101765,
                         'bstd': 0.0165,
                         'c50': 1.69647501148509,
                         'cstd': -0.3},
 'GM_frontal_volume': {'a50': 0.0807,
                       'astd': 0.0069730214009038,
                       'b50': -3.1238,
                       'bstd': -0.274596736245797,
                       'c50': 33.0,
                       'cstd': 2.92539054001209},
 'GM_insular_volume': {'a50': 0.00489876140350367,
                       'astd': 9.21597910923961e-05,
                       'b50': -0.166087051284211,
                       'bstd': 0.00235338930143109,
                       'c50': 1.70170537368668,
                       'cstd': -0.0640889669501304},
 'GM_occipital_volume': {'a50': 0.0522,
                         'astd': 0.00522246567276285,
                         'b50': -2.1976,
                         'bstd': -0.23782423620518,
                         'c50': 25.0,
                         'cstd': 2.97201936200651},
 'GM_parietal_volume': {'a50': 0.068,
                        'astd': 0.00721991229188154,
                        'b50': -2.8658,
                        'bstd': -0.328232894396965,
                        'c50': 32.0,
                        'cstd': 3.98920903828335},
 'GM_temporal_volume': {'a50': 0.0533,
                        'astd': 0.00691489732001121,
                        'b50': -2.1688,
                        'bstd': -0.335237539622198,
                        'c50': 24.0,
                        'cstd': 4.34799834019574},
 'WM_cingulate_volume': {'a50': 0.0075, 'astd': 0.0002, 'b50': 0.0033, 'bstd': 0.0285, 'c50': -2.0, 'cstd': -0.5},
 'WM_frontal_volume': {'a50': 0.0375,
                       'astd': 0.016266611980808,
                       'b50': 0.9604,
                       'bstd': -0.63189332164781,
                       'c50': -30.0,
                       'cstd': 7.1073860580842},
 'WM_insular_volume': {'a50': 0.00423156582442052,
                       'astd': 0.00165228115360945,
                       'b50': 0.0840136045572269,
                       'bstd': -0.0664590616154003,
                       'c50': -2.97419860973309,
                       'cstd': 0.812608727985254},
 'WM_occipital_volume': {'a50': 0.002,
                         'astd': 0.00410861811688998,
                         'b50': 0.7953,
                         'bstd': -0.147399595291602,
                         'c50': -15.0,
                         'cstd': 1.8216795235438},
 'WM_parietal_volume': {'a50': 0.0016,
                        'astd': 0.00915497532531195,
                        'b50': 1.5921,
                        'bstd': -0.362102769861644,
                        'c50': -30.0,
                        'cstd': 4.34624905746771},
 'WM_temporal_volume': {'a50': 0.0003779264251499,
                        'astd': 0.00387492902328718,
                        'b50': 1.50633955759368,
                        'bstd': -0.10118723932875,
                        'c50': -28.8835869558016,
                        'cstd': 0.902406490021678},
 'brainstem_volume': {'a50': 0.00325533324578462,
                      'astd': -0.000251263825187535,
                      'b50': 0.108431411226112,
                      'bstd': 0.0295439063464388,
                      'c50': -2.44103040073891,
                      'cstd': -0.435844321785417},
 'cavum_volume': {'a50': -0.0059, 'astd': -0.0006, 'b50': 0.3724, 'bstd': 0.0425, 'c50': -5.0, 'cstd': -0.5},
 'cerebellum_volume': {'a50': 0.0466480977799726,
                       'astd': 0.00360987762439721,
                       'b50': -1.87938070214712,
                       'bstd': -0.139930405061914,
                       'c50': 20.0650374338015,
                       'cstd': 1.45705276266902},
 'eCSF_volume': {'a50': -0.1373,
                 'astd': -0.0163712639081,
                 'b50': 13.542,
                 'bstd': 1.59785166806484,
                 'c50': -200.0,
                 'cstd': -22.5734944518328},
 'fourth_ventricle_volume': {'a50': 0.00054566572903546,
                             'astd': 8e-05,
                             'b50': -0.018992164615086,
                             'bstd': -0.0018,
                             'c50': 0.208952525904693,
                             'cstd': 0.01},
 'lateral_ventricles_volume': {'a50': 0.0016, 'astd': 0.0, 'b50': 0.0209, 'bstd': 0.0692, 'c50': 2.0, 'cstd': -1.0},
 'third_ventricle_volume': {'a50': 0.00107157558561917,
                            'astd': 0.000139373400159916,
                            'b50': -0.0363541901140626,
                            'bstd': -0.00276526640102309,
                            'c50': 0.340438967164873,
                            'cstd': 0.00491863412204744},
 'total_DGM_volume': {'a50': 0.0197203946415015,
                      'astd': 0.000559296254541606,
                      'b50': -0.518937959334391,
                      'bstd': 0.000725552297728093,
                      'c50': 4.58819219430656,
                      'cstd': -0.110900894302876},
 'total_WM_volume': {'a50': -0.0113602088278119,
                     'astd': 0.0380689771119069,
                     'b50': 8.79713006822208,
                     'bstd': -1.55198182922729,
                     'c50': -165.36273860942,
                     'cstd': 19.009669220147},
 'total_brain_volume': {'a50': 0.0560772598864699,
                        'astd': -0.000714222351371512,
                        'b50': 18.384225646449,
                        'bstd': 1.1641566302025,
                        'c50': -360.048274167276,
                        'cstd': -16.6440446985629},
 'total_cortical_GM_volume': {'a50': 0.2637,
                              'astd': 0.0263547371654088,
                              'b50': -10.563,
                              'bstd': -1.18408376936538,
                              'c50': 115.0,
                              'cstd': 14.4135618576711},
 'total_parenchyma_volume': {'a50': 0.358001106623485,
                             'astd': 0.0446046125027615,
                             'b50': -5.21711385459465,
                             'bstd': -1.46460400627443,
                             'c50': -14.5840861004746,
                             'cstd': 14.2399945149215},
 'vermis_volume': {'a50': 0.0039, 'astd': 0.0003, 'b50': -0.0909, 'bstd': -0.0056, 'c50': 0.5, 'cstd': 0.005}}

ROI_LABELS = {
    "total_brain_volume": list(range(1, 44)),
    "total_parenchyma_volume": list(range(1, 37)),
    "total_cortical_GM_volume": list(range(1, 13)),
    "total_WM_volume": list(range(13, 25)),
    "total_DGM_volume": [31, 32, 33, 34, 35, 36],
    "GM_frontal_volume": [1, 2],
    "GM_parietal_volume": [3, 4],
    "GM_occipital_volume": [5, 6],
    "GM_insular_volume": [7, 8],
    "GM_temporal_volume": [9, 10],
    "GM_cingulate_volume": [11, 12],
    "WM_frontal_volume": [13, 14],
    "WM_parietal_volume": [15, 16],
    "WM_occipital_volume": [17, 18],
    "WM_insular_volume": [19, 20],
    "WM_temporal_volume": [21, 22],
    "WM_cingulate_volume": [23, 24],
    "brainstem_volume": [27],
    "cerebellum_volume": [28, 29],
    "vermis_volume": [30],
    "DGM_caudate_nucleus_volume": [31, 32],
    "DGM_lentiform_nucleus_volume": [33, 34],
    "DGM_thalamus_volume": [35, 36],
    "eCSF_volume": [37, 38],
    "lateral_ventricles_volume": [39, 40],
    "right_lateral_ventricle_volume": [39],
    "left_lateral_ventricle_volume": [40],
    "cavum_volume": [41],
    "third_ventricle_volume": [42],
    "fourth_ventricle_volume": [43],
}

DISPLAY_NAMES = {
    "total_brain_volume": "Total brain volume",
    "total_parenchyma_volume": "Total parenchyma volume",
    "total_cortical_GM_volume": "Total cortical GM volume",
    "total_WM_volume": "Total WM volume",
    "total_DGM_volume": "Total deep GM volume",
    "GM_frontal_volume": "GM frontal volume",
    "GM_parietal_volume": "GM parietal volume",
    "GM_occipital_volume": "GM occipital volume",
    "GM_insular_volume": "GM insular volume",
    "GM_temporal_volume": "GM temporal volume",
    "GM_cingulate_volume": "GM cingulate volume",
    "WM_frontal_volume": "WM frontal volume",
    "WM_parietal_volume": "WM parietal volume",
    "WM_occipital_volume": "WM occipital volume",
    "WM_insular_volume": "WM insular volume",
    "WM_temporal_volume": "WM temporal volume",
    "WM_cingulate_volume": "WM cingulate volume",
    "brainstem_volume": "Brainstem volume",
    "cerebellum_volume": "Cerebellum volume",
    "vermis_volume": "Vermis volume",
    "DGM_caudate_nucleus_volume": "Caudate nucleus volume",
    "DGM_lentiform_nucleus_volume": "Lentiform nucleus volume",
    "DGM_thalamus_volume": "Thalamus volume",
    "eCSF_volume": "Extracerebral CSF volume",
    "lateral_ventricles_volume": "Total lateral ventricles volume",
    "right_lateral_ventricle_volume": "Right lateral ventricle volume",
    "left_lateral_ventricle_volume": "Left lateral ventricle volume",
    "cavum_volume": "Cavum volume",
    "third_ventricle_volume": "Third ventricle volume",
    "fourth_ventricle_volume": "Fourth ventricle volume",
}

ORDERED_VARIABLES = [
    "total_brain_volume","total_parenchyma_volume","total_cortical_GM_volume","total_WM_volume","total_DGM_volume",
    "GM_frontal_volume","GM_parietal_volume","GM_occipital_volume","GM_insular_volume","GM_temporal_volume","GM_cingulate_volume",
    "WM_frontal_volume","WM_parietal_volume","WM_occipital_volume","WM_insular_volume","WM_temporal_volume","WM_cingulate_volume",
    "brainstem_volume","cerebellum_volume","vermis_volume","DGM_caudate_nucleus_volume","DGM_lentiform_nucleus_volume",
    "DGM_thalamus_volume","eCSF_volume","lateral_ventricles_volume","right_lateral_ventricle_volume",
    "left_lateral_ventricle_volume","cavum_volume","third_ventricle_volume","fourth_ventricle_volume"
]

label_rgba = np.zeros((44, 4), dtype=float)
for _idx, (r, g, b, a) in LABEL_COLORS.items():
    label_rgba[_idx] = [r/255.0, g/255.0, b/255.0, a]
jet_transparent = ListedColormap(label_rgba)

def compute_volume_cc(label_matrix, labels, voxel_dims):
    voxel_volume_mm3 = float(voxel_dims[0] * voxel_dims[1] * voxel_dims[2])
    n_vox = int(np.isin(label_matrix, labels).sum())
    return (n_vox * voxel_volume_mm3) / 1000.0

def model_mean_std(variable, ga):
    c = COEFFICIENTS[variable]
    ga_arr = np.asarray(ga, dtype=float)
    mean = c["a50"] * ga_arr**2 + c["b50"] * ga_arr + c["c50"]
    std = c["astd"] * ga_arr**2 + c["bstd"] * ga_arr + c["cstd"]
    std = np.maximum(std, 1e-6)
    return mean, std

def model_mean_std_side_lateral(ga):
    mean_total, std_total = model_mean_std("lateral_ventricles_volume", ga)
    return mean_total / 2.0, np.maximum(std_total / 2.0, 1e-6)

def centile_graph(variable, ga, measured_cc):
    x = np.linspace(GA_MIN, GA_MAX, 200)
    if variable in ("right_lateral_ventricle_volume", "left_lateral_ventricle_volume"):
        m, s = model_mean_std_side_lateral(x)
    else:
        m, s = model_mean_std(variable, x)

    m = np.asarray(m, dtype=float)
    s = np.asarray(s, dtype=float)
    y5 = m - 1.645 * s
    y95 = m + 1.645 * s

    finite = np.isfinite(x) & np.isfinite(m) & np.isfinite(y5) & np.isfinite(y95)
    x = x[finite]
    m = m[finite]
    y5 = y5[finite]
    y95 = y95[finite]

    x_plot = x.astype(float).tolist()
    m_plot = m.astype(float).tolist()
    y5_plot = y5.astype(float).tolist()
    y95_plot = y95.astype(float).tolist()
    y_all = np.concatenate([m, y5, y95, np.array([measured_cc], dtype=float)])
    y_min = float(np.nanmin(y_all))
    y_max = float(np.nanmax(y_all))
    pad = max((y_max - y_min) * 0.08, 1e-3)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_plot, y=m_plot, mode="lines", line=dict(color="black", width=2), name="50th"))
    fig.add_trace(go.Scatter(x=x_plot, y=y5_plot, mode="lines", line=dict(color="grey", dash="dot"), name="5th"))
    fig.add_trace(go.Scatter(x=x_plot, y=y95_plot, mode="lines", line=dict(color="grey", dash="dot"), name="95th"))
    fig.add_trace(go.Scatter(x=[float(ga)], y=[float(measured_cc)], mode="markers",
                             marker=dict(color="red", size=10, symbol="x"), name="Measured"))
    fig.update_layout(
        title={"text": DISPLAY_NAMES[variable], "x": 0.5, "xanchor": "center"},
        xaxis_title="GA [weeks]",
        yaxis_title="volume [cc]",
        xaxis=dict(range=[GA_MIN, GA_MAX], gridcolor="lightgrey"),
        yaxis=dict(range=[0, y_max + pad], gridcolor="lightgrey"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)

def evaluate_measurements(label_matrix, voxel_dims, ga):
    results = []
    for variable in ORDERED_VARIABLES:
        measured_cc = compute_volume_cc(label_matrix, ROI_LABELS[variable], voxel_dims)
        if variable in ("right_lateral_ventricle_volume", "left_lateral_ventricle_volume"):
            mean_cc, std_cc = model_mean_std_side_lateral(ga)
        else:
            mean_cc, std_cc = model_mean_std(variable, ga)
        mean_cc = float(mean_cc)
        std_cc = float(std_cc)
        z = (measured_cc - mean_cc) / std_cc
        centile = norm.cdf(z) * 100.0
        results.append({
            "variable": variable,
            "name": DISPLAY_NAMES[variable].replace(" volume", "").replace("Volume", ""),
            "volume_cc": measured_cc,
            "oe_ratio": float(measured_cc / mean_cc) if mean_cc != 0 else float("nan"),
            "mean_cc": mean_cc,
            "std_cc": std_cc,
            "z": float(z),
            "centile": float(centile),
            "graph": centile_graph(variable, ga, measured_cc),
        })
    return results

def _select_slice_indices(label_data, axis, n_slices=7):
    other_axes = [0, 1, 2]
    other_axes.remove(axis)
    presence = np.any(label_data > 0, axis=tuple(other_axes))
    idx = np.where(presence)[0]
    if len(idx) == 0:
        size = label_data.shape[axis]
        return np.linspace(int(size * 0.35), int(size * 0.65), n_slices).astype(int).tolist()
    if len(idx) == 1:
        return [int(idx[0])] * n_slices
    # Focus on the central brain and avoid edge-adjacent slices
    q = np.linspace(0.30, 0.70, n_slices)
    sel = np.quantile(idx, q)
    sel = np.clip(np.round(sel).astype(int), idx.min(), idx.max())
    return sel.tolist()

def _show_slice(ax, img2d, lab2d=None, alpha=0.4):
    ax.imshow(img2d.T, cmap="gray", origin="lower")
    if lab2d is not None:
        ax.imshow(lab2d.T, cmap=jet_transparent, origin="lower", alpha=alpha, vmin=0, vmax=43)
    ax.axis("off")

def plot_brain_image(t2w_data, label_data):
    n_slices = 7
    fig, axs = plt.subplots(6, n_slices, figsize=(24, 18))
    alpha = 0.4

    axial_idx = _select_slice_indices(label_data, axis=2, n_slices=n_slices)
    coronal_idx = _select_slice_indices(label_data, axis=1, n_slices=n_slices)
    sagittal_idx = _select_slice_indices(label_data, axis=0, n_slices=n_slices)

    for i, sl in enumerate(axial_idx):
        _show_slice(axs[0, i], t2w_data[:, :, sl], None, alpha)
        _show_slice(axs[1, i], t2w_data[:, :, sl], label_data[:, :, sl], alpha)

    for i, sl in enumerate(coronal_idx):
        _show_slice(axs[2, i], t2w_data[:, sl, :], None, alpha)
        _show_slice(axs[3, i], t2w_data[:, sl, :], label_data[:, sl, :], alpha)

    for i, sl in enumerate(sagittal_idx):
        _show_slice(axs[4, i], t2w_data[sl, :, :], None, alpha)
        _show_slice(axs[5, i], t2w_data[sl, :, :], label_data[sl, :, :], alpha)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=140)
    buf.seek(0)
    image_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return image_b64

def render_report(case_id, ga, scan_date, brain_image_b64, results):
    rows_html = "".join(
        (
            (f"<tr style='color:#c00000;font-weight:700;'><td>{r['name']}</td><td>{r['volume_cc']:.2f}</td><td>{r['oe_ratio']:.3f}</td>"
             f"<td>{r['centile']:.2f}</td><td>{r['z']:.2f}</td></tr>")
            if (r['centile'] < 5 or r['centile'] > 95)
            else
            (f"<tr><td>{r['name']}</td><td>{r['volume_cc']:.2f}</td><td>{r['oe_ratio']:.3f}</td>"
             f"<td>{r['centile']:.2f}</td><td>{r['z']:.2f}</td></tr>")
        )
        for r in results
    )
    graphs_html = "".join(f"<div class='graph'>{r['graph']}</div>" for r in results)
    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<meta name='viewport' content='width=device-width, initial-scale=1.0'>
<title>{REPORT_TITLE}</title>
<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
.info-table {{ width: 100%; max-width: 1100px; border-collapse: collapse; }}
.info-table td, .info-table th {{ border: 1px solid #bbb; padding: 6px 8px; text-align: left; font-size: 13px; }}
.brain-image {{ width: 100%; max-width: 1100px; height: auto; }}
.graph-container {{ display: grid; grid-template-columns: repeat(3, minmax(320px, 1fr)); gap: 12px; }}
.graph {{ border: 1px solid #ddd; padding: 4px; }}
</style>
</head>
<body>
<h1>{REPORT_TITLE}</h1>
<table class='info-table'>
<tr><td>Case ID</td><td>{case_id}</td></tr>
<tr><td>GA</td><td>{ga:.2f} weeks</td></tr>
<tr><td>Scan date</td><td>{scan_date}</td></tr>
</table>
<br>
<img src='data:image/png;base64,{brain_image_b64}' alt='Segmentation overlay' class='brain-image'>
<br><br>
<table class='info-table'>
<tr><th>ROI</th><th>Volume [cc]</th><th>O/E</th><th>Centile</th><th>Z-score</th></tr>
{rows_html}
</table>
<br><br>
<div class='graph-container'>{graphs_html}</div>
</body></html>"""

def main():
    if len(sys.argv) != 7:
        raise SystemExit("Usage: python script.py <case_id> <ga_weeks> <scan_date> <input_img_nii> <input_lab_nii> <output_html>")
    case_id = sys.argv[1]
    ga = float(sys.argv[2])
    scan_date = sys.argv[3]
    img_path = sys.argv[4]
    lab_path = sys.argv[5]
    output_html = sys.argv[6]

    img = nib.load(img_path)
    lab = nib.load(lab_path)
    img_data = np.asarray(img.get_fdata())
    lab_data = np.rint(np.asarray(lab.get_fdata())).astype(np.int16)
    voxel_dims = img.header.get_zooms()[:3]

    results = evaluate_measurements(lab_data, voxel_dims, ga)
    brain_image_b64 = plot_brain_image(img_data, lab_data)
    html = render_report(case_id, ga, scan_date, brain_image_b64, results)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(output_html)

if __name__ == "__main__":
    main()
