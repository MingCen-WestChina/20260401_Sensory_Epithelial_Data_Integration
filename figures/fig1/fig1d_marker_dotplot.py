"""Reproduce the representative marker-gene dot plot used in Figure 1D."""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.sparse import csr_matrix

COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
sys.path.insert(0, str(COMMON_DIR))
from editable_pdf import save_ai_editable_pdf  # noqa: E402
from figure_utils import (  # noqa: E402
    configure_matplotlib,
    data_directory,
    decode,
    read_categorical,
    resolve_input_path,
    results_directory,
)


DPI = 1201
MAX_DOT_AREA = 28.0
MARKER_GROUPS = OrderedDict(
    {
        "Early epithelium": ["Fabp5", "Mest", "Nfia", "Sox2"],
        "Domain patterning": ["Gata3", "Fst", "Sox11", "Nr2f1", "Bmp4"],
        "Prosensory lineages": ["Lgr5", "Fgfr3", "Socs2", "Lor", "Ntf3"],
        "Hair cells": ["Fgf8", "Otof", "Ccer2", "Atoh1", "Lhfpl5", "Kcnq4"],
        "Pillar / phalangeal": ["Npy", "Ngfr", "Gjb2", "Ccdc80", "Slc1a3", "Rorb"],
        "Lateral support": ["Fabp7", "Aqp4", "S100a13", "Nat8l", "Cd44", "Npnt", "Fbln2", "Otor", "Col3a1"],
        "Outer support": ["Hes5", "Gpx2", "Enho", "Pdzk1ip1", "Cox4i2", "Lfng", "Hey1"],
        "Inner epithelium": ["Igf1", "Meg3", "Otoa", "Cdkn1c"],
        "GER states": ["Calb1", "Crabp1", "Hmgn2", "Pttg1", "Rgcc", "Irx3", "Tecta", "Ebf1", "Isl1"],
        "KO states": ["Cpxm2", "Dkk3", "Hspa2", "Gsn", "Cldn4", "Tnfrsf12a", "Qpct", "Dhrs7"],
    }
)
LITERATURE_MARKERS = {
    "Sox2", "Gata3", "Fst", "Bmp4", "Lgr5", "Fgfr3", "Socs2", "Ntf3",
    "Fgf8", "Otof", "Atoh1", "Lhfpl5", "Kcnq4", "Npy", "Gjb2", "Slc1a3",
    "Rorb", "Fabp7", "Aqp4", "Cd44", "Otor", "Col3a1", "Hes5", "Pdzk1ip1",
    "Lfng", "Hey1", "Igf1", "Otoa", "Calb1", "Crabp1", "Tecta", "Isl1",
    "Gsn", "Dkk3",
}
CELLTYPE_ORDER = [
    "OV epithelial cells", "Prosensory domain", "Lateral domain", "Medial domain",
    "LER", "L.PsC", "M.PsC", "IHC", "iOHC", "OHC", "IPC", "IPhC", "PC",
    "HeC", "IB", "OSC", "CC/OSC", "CC/OS", "OPC", "DC", "DC-1/2", "DC-3",
    "ISC", "IdC", "GER", "GER/Hmgn2", "GER/KO", "L.GER", "KO-1", "KO-2",
    "KO-3", "KO-4", "L KO", "M KO", "ML KO",
]
GROUP_COLORS = [
    "#b22222", "#c27d00", "#2ca25f", "#7a3db8", "#de2d26",
    "#ff7f00", "#9467bd", "#1f9e89", "#6baed6", "#e39d25",
]


def extract_celltype(stage_celltype, stage) -> np.ndarray:
    labels = []
    for raw_label, raw_stage in zip(stage_celltype, stage):
        label = str(raw_label)
        for prefix in (f"{raw_stage}_", f"{raw_stage} "):
            if label.startswith(prefix):
                label = label[len(prefix):]
                break
        else:
            if "_" in label:
                label = label.split("_", 1)[1]
        labels.append(label.replace("_", " "))
    return np.asarray(labels, dtype=object)


def load_and_summarize(h5ad_file: Path):
    marker_genes = [gene for genes in MARKER_GROUPS.values() for gene in genes]
    if len(marker_genes) != 63 or len(marker_genes) != len(set(marker_genes)):
        raise ValueError("The final marker list must contain 63 unique genes.")
    if not h5ad_file.is_file():
        raise FileNotFoundError(f"Input h5AD not found: {h5ad_file}")
    with h5py.File(h5ad_file, "r") as handle:
        var_names = decode(handle["var/_index"][()])
        stage, _ = read_categorical(handle, "stage")
        stage_celltype, _ = read_categorical(handle, "stage_celltype")
        celltype_labels = extract_celltype(stage_celltype, stage)
        gene_lookup = {gene.upper(): index for index, gene in enumerate(var_names)}
        missing_genes = [gene for gene in marker_genes if gene.upper() not in gene_lookup]
        if missing_genes:
            raise KeyError(f"Marker genes missing from the expression matrix: {missing_genes}")
        marker_indices = [gene_lookup[gene.upper()] for gene in marker_genes]
        matrix = csr_matrix(
            (handle["X/data"][()], handle["X/indices"][()], handle["X/indptr"][()]),
            shape=(celltype_labels.shape[0], len(var_names)),
        )[:, marker_indices].tocsr()

    observed_celltypes = set(np.unique(celltype_labels))
    missing_celltypes = [label for label in CELLTYPE_ORDER if label not in observed_celltypes]
    unexpected_celltypes = sorted(observed_celltypes.difference(CELLTYPE_ORDER))
    if missing_celltypes or unexpected_celltypes:
        raise ValueError(f"Cell-type mismatch. Missing: {missing_celltypes}; unexpected: {unexpected_celltypes}")

    n_cells, n_genes = matrix.shape
    n_rows = len(CELLTYPE_ORDER)
    pct_expr = np.zeros((n_rows, n_genes), dtype=float)
    avg_expr = np.zeros_like(pct_expr)
    rest_avg_expr = np.zeros_like(pct_expr)
    total_sum = np.asarray(matrix.sum(axis=0)).ravel()
    for row_index, label in enumerate(CELLTYPE_ORDER):
        rows = np.flatnonzero(celltype_labels == label)
        subset = matrix[rows, :]
        group_sum = np.asarray(subset.sum(axis=0)).ravel()
        pct_expr[row_index, :] = np.asarray((subset > 0).sum(axis=0)).ravel() / len(rows) * 100
        avg_expr[row_index, :] = group_sum / len(rows)
        rest_avg_expr[row_index, :] = (total_sum - group_sum) / (n_cells - len(rows))
    gene_min = avg_expr.min(axis=0)
    gene_range = avg_expr.max(axis=0) - gene_min
    scaled_expr = np.full_like(avg_expr, -1.0)
    variable = gene_range > 0
    scaled_expr[:, variable] = ((avg_expr[:, variable] - gene_min[variable]) / gene_range[variable]) * 2 - 1
    qualifying = (pct_expr >= 20.0) & ((avg_expr - rest_avg_expr) >= 0.20) & (scaled_expr >= -0.05)
    coverage = qualifying.sum(axis=1)
    if coverage.min() < 2:
        failed = {label: int(count) for label, count in zip(CELLTYPE_ORDER, coverage) if count < 2}
        raise ValueError(f"Cell types with fewer than two qualifying differential markers: {failed}")
    weights = np.clip((scaled_expr + 1.0) / 2.0, 0, 1) ** 4 * np.sqrt(pct_expr / 100.0)
    x_position = np.arange(n_genes) / max(n_genes - 1, 1)
    y_position = np.arange(n_rows) / max(n_rows - 1, 1)
    diagonal_score = (weights * np.abs(y_position[:, None] - x_position[None, :])).sum() / np.maximum(weights.sum(), 1e-12)
    return marker_genes, pct_expr, scaled_expr, coverage, float(diagonal_score), n_cells


def draw_figure(marker_genes, pct_expr, scaled_expr):
    n_rows, n_genes = pct_expr.shape
    fig = plt.figure(figsize=(11.0, 6.35), facecolor="white")
    ax = fig.add_axes([0.14, 0.21, 0.65, 0.66])
    x = np.tile(np.arange(n_genes), n_rows)
    y = np.repeat(np.arange(n_rows)[::-1], n_genes)
    flat_pct = pct_expr.reshape(-1)
    sizes = 0.5 + flat_pct / 100.0 * MAX_DOT_AREA
    sizes[flat_pct <= 0] = 0
    cmap = LinearSegmentedColormap.from_list(
        "scanpy_like_redpurple",
        ["#f2f2f2", "#e8c7df", "#c77bb9", "#96307e", "#6a005f"],
    )
    ax.scatter(x, y, c=scaled_expr.reshape(-1), s=sizes, cmap=cmap, vmin=-1,
               vmax=1, edgecolors="none")
    ax.set_xlim(-0.6, n_genes - 0.4)
    ax.set_ylim(-0.7, n_rows + 2.1)
    ax.set_xticks(np.arange(n_genes))
    ax.set_xticklabels(marker_genes, rotation=70, ha="right", fontsize=4.7, fontstyle="italic")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(CELLTYPE_ORDER[::-1], fontsize=4.7)
    ax.tick_params(axis="both", length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    current = 0
    y_bar = n_rows + 0.42
    for color, (group_name, genes) in zip(GROUP_COLORS, MARKER_GROUPS.items()):
        start, end = current, current + len(genes) - 1
        current = end + 1
        ax.plot([start - 0.35, end + 0.35], [y_bar, y_bar], color=color, lw=1.5, clip_on=False)
        ax.text((start + end) / 2.0, y_bar + 0.10, group_name, ha="center", va="bottom",
                rotation=40, fontsize=4.8, clip_on=False)

    size_ax = fig.add_axes([0.82, 0.57, 0.16, 0.11])
    size_ax.axis("off")
    size_ax.text(0.0, 0.88, "% Exp.", fontsize=5.2, ha="left", va="bottom")
    legend_percents = [0, 25, 50, 75, 100]
    legend_x = np.linspace(0.08, 0.92, len(legend_percents))
    for xpos, percent in zip(legend_x, legend_percents):
        size = 0 if percent == 0 else 0.5 + percent / 100.0 * MAX_DOT_AREA
        size_ax.scatter([xpos], [0.48], s=size, color="black")
        size_ax.text(xpos, 0.13, str(percent), fontsize=4.7, ha="center", va="center")
    size_ax.set_xlim(0, 1)
    size_ax.set_ylim(0, 1)

    cax = fig.add_axes([0.835, 0.46, 0.13, 0.028])
    for step in range(192):
        lower = -1.0 + 2.0 * step / 192
        cax.add_patch(Rectangle((lower, 0), 2.0 / 192 + 1e-4, 1,
                                facecolor=cmap(step / 191), edgecolor="none"))
    cax.set_xlim(-1, 1)
    cax.set_ylim(0, 1)
    cax.set_yticks([])
    cax.set_xticks([-1, -0.5, 0, 0.5, 1])
    cax.tick_params(axis="x", labelsize=4.7, length=0, pad=2)
    cax.set_title("Avg. Exp.", fontsize=5.2, pad=5)
    for spine in cax.spines.values():
        spine.set_visible(False)

    fig.canvas.draw()
    main_bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    maximum_diameter = np.sqrt(0.5 + MAX_DOT_AREA)
    x_pitch = main_bbox.width * 72.0 / (n_genes - 0.4 + 0.6)
    y_pitch = main_bbox.height * 72.0 / (n_rows + 2.1 + 0.7)
    x_gap_ratio = (x_pitch - maximum_diameter) / maximum_diameter
    y_gap_ratio = (y_pitch - maximum_diameter) / maximum_diameter
    if not (0.45 <= x_gap_ratio <= 0.60 and 0.45 <= y_gap_ratio <= 0.60):
        raise ValueError(f"Dot spacing is outside the requested range: x={x_gap_ratio:.3f}, y={y_gap_ratio:.3f}")
    return fig, x_gap_ratio, y_gap_ratio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-inputs", action="store_true", help="Validate and summarize input without drawing files.")
    args = parser.parse_args()
    configure_matplotlib()
    h5ad_file = resolve_input_path(
        results_directory(__file__) / "02_scanvi_integration/all_stages/all_stages_scanvi.h5ad",
        data_directory(__file__) / "Integration_3types_after_training/Entire period files/all_retained_scanvi_HVG5000_integrated.h5ad",
        data_directory(__file__) / "Entire period/EntirePeriod_scanvi_HVG5000/all_retained_scanvi_HVG5000_integrated.h5ad",
    )
    marker_genes, pct_expr, scaled_expr, coverage, diagonal_score, n_cells = load_and_summarize(h5ad_file)
    if args.check_inputs:
        print(f"Validated Figure 1D input: {n_cells:,} cells; minimum marker coverage={int(coverage.min())}; diagonal score={diagonal_score:.4f}")
        return
    output_dir = results_directory(__file__) / "figures/fig1"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, x_gap, y_gap = draw_figure(marker_genes, pct_expr, scaled_expr)
    save_ai_editable_pdf(fig, output_dir / "fig1d_marker_dotplot.pdf", bbox_inches="tight", title="Figure 1D marker dot plot")
    fig.savefig(output_dir / "fig1d_marker_dotplot.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved Figure 1D; dot edge-gap ratios: x={x_gap:.3f}, y={y_gap:.3f}")


if __name__ == "__main__":
    main()
