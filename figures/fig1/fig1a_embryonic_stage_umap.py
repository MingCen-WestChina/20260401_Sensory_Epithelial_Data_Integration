"""Reproduce the developmental-stage inset used in Figure 1A."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
sys.path.insert(0, str(COMMON_DIR))
from editable_pdf import save_ai_editable_pdf  # noqa: E402
from figure_utils import (  # noqa: E402
    configure_matplotlib,
    data_directory,
    integer_ticks,
    read_categorical,
    resolve_input_path,
    results_directory,
)


RANDOM_STATE = 0
DPI = 1201
TARGET_PIXEL_SIZE = (8907, 6659)
EXPORT_BBOX = Bbox.from_bounds(
    0.922098,
    0.127004,
    TARGET_PIXEL_SIZE[0] / DPI,
    TARGET_PIXEL_SIZE[1] / DPI,
)
FIGSIZE = (10.2, 6.4)
POINT_SIZE = 1.35
POINT_ALPHA = 0.82
LEGEND_FONTSIZE = 8.5
LEGEND_MARKER_SIZE = 5.5
PERIOD_STAGES = ["E9.5", "E11.5", "E13.5", "E14.5", "E16.5"]
STAGE_COLORS = {
    "E9.5": "#1F77B4",
    "E11.5": "#FF7F0E",
    "E13.5": "#2CA02C",
    "E14.5": "#D62728",
    "E16.5": "#9467BD",
}


def load_data(h5ad_file: Path):
    if not h5ad_file.is_file():
        raise FileNotFoundError(f"Input h5AD not found: {h5ad_file}")
    with h5py.File(h5ad_file, "r") as handle:
        coords = handle["obsm/X_umap"][()]
        stage_labels, _ = read_categorical(handle, "stage")
    unexpected = sorted(set(stage_labels).difference(PERIOD_STAGES))
    stage_counts = {stage: int(np.sum(stage_labels == stage)) for stage in PERIOD_STAGES}
    if unexpected or sum(stage_counts.values()) != coords.shape[0]:
        raise ValueError(f"Stage annotations do not match the embryonic UMAP: unexpected={unexpected}")
    return coords, stage_labels, stage_counts


def draw_figure(coords, stage_labels, stage_counts):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(right=0.66)
    draw_stages = sorted(PERIOD_STAGES, key=lambda stage: stage_counts[stage], reverse=True)
    rng = np.random.default_rng(RANDOM_STATE)
    for stage in draw_stages:
        indices = rng.permutation(np.flatnonzero(stage_labels == stage))
        ax.scatter(
            coords[indices, 0], coords[indices, 1], s=POINT_SIZE,
            c=STAGE_COLORS[stage], alpha=POINT_ALPHA, linewidths=0, zorder=1,
        )
    x_pad = np.ptp(coords[:, 0]) * 0.08
    y_pad = np.ptp(coords[:, 1]) * 0.08
    ax.set_xlim(coords[:, 0].min() - x_pad, coords[:, 0].max() + x_pad)
    ax.set_ylim(coords[:, 1].min() - y_pad, coords[:, 1].max() + y_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=STAGE_COLORS[stage],
               markeredgecolor=STAGE_COLORS[stage], markersize=LEGEND_MARKER_SIZE,
               label=f"{stage} (n = {stage_counts[stage]:,})")
        for stage in PERIOD_STAGES
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.50),
              frameon=False, fontsize=LEGEND_FONTSIZE, handlelength=0.8,
              handletextpad=0.45, labelspacing=0.85, borderaxespad=0)
    ax.set_xlabel("UMAP1", fontsize=18)
    ax.set_ylabel("UMAP2", fontsize=18)
    ax.set_xticks(integer_ticks(ax.get_xlim()))
    ax.set_yticks(integer_ticks(ax.get_ylim()))
    ax.tick_params(axis="both", length=4, width=1.0, labelsize=10, direction="out")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-inputs", action="store_true", help="Validate the input without drawing files.")
    args = parser.parse_args()
    configure_matplotlib()
    h5ad_file = resolve_input_path(
        results_directory(__file__) / "02_scanvi_integration/embryonic/Embryonic_scanvi_refined_integrated.h5ad",
        data_directory(__file__) / "Integration_3types_after_training/Embryonic stage files/Embryonic_scanvi_refined_integrated.h5ad",
    )
    coords, labels, counts = load_data(h5ad_file)
    if args.check_inputs:
        print(f"Validated Figure 1A stage input: {coords.shape[0]:,} cells")
        return
    output_dir = results_directory(__file__) / "figures/fig1"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = draw_figure(coords, labels, counts)
    options = {"bbox_inches": EXPORT_BBOX, "pad_inches": 0, "facecolor": "white"}
    fig.savefig(output_dir / "fig1a_embryonic_stage_umap.png", dpi=DPI, **options)
    save_ai_editable_pdf(
        fig,
        output_dir / "fig1a_embryonic_stage_umap.pdf",
        bbox_inches=EXPORT_BBOX,
        title="Embryonic stage UMAP",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
