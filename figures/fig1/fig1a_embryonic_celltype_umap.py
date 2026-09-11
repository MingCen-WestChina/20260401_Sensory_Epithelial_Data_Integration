"""Reproduce the cell-state UMAP used in Figure 1A."""

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
LABEL_FONTSIZE = 8.0
LABEL_STROKE_WIDTH = 1.6
LABEL_REPEL_MAX_ITER = 240
LABEL_REPEL_STEP_PX = 5
LABEL_MAX_MOVE_PX = 65
CONNECT_DISTANCE_PX = 45
CONNECT_LABEL_GAP_PX = 2
CONNECT_WIDTH = 0.45
CONNECT_ALPHA = 0.85
LEGEND_FONTSIZE = 8.5
LEGEND_MARKER_SIZE = 5.5

STAGE_ORDER = ["E9.5", "E11.5", "E13.5", "E14.5", "E16.5"]
PALETTE_COLORS = [
    "#cedb9c", "#637939", "#b5cf6b", "#c49c94", "#9467bd", "#8ca252",
    "#9c9ede", "#d62728", "#dbdb8d", "#ff9896", "#ff7f0e", "#393b79",
    "#1f77b4", "#e377c2", "#98df8a", "#7f7f7f", "#ffbb78", "#9edae5",
    "#aec7e8", "#5254a3", "#8c564b", "#6b6ecf", "#c5b0d5", "#2ca02c",
    "#17becf", "#c7c7c7", "#bcbd22", "#f7b6d2",
]
MANUAL_LABEL_SHIFT_PX = {
    "OV epithelial cells": (0, 34),
    "GER": (-38, -10),
    "L.GER": (4, 6),
    "GER/Hmgn2": (0, -6),
    "Prosensory domain": (0, 18),
    "Medial domain": (0, 10),
    "IBC": (-8, 10),
    "IPhC": (-14, -14),
    "IPC": (-8, -4),
    "L.PsC": (10, -6),
    "OHC": (-8, -4),
    "IHC": (-6, 2),
    "ISC": (82, 20),
    "LER": (14, 0),
    "HeC": (0, 10),
    "CC/OSC": (0, 10),
}
SKIP_CONNECT_CELLTYPES = {"OV epithelial cells"}


def dense_anchor(anchor_coords: np.ndarray) -> np.ndarray:
    anchor_coords = np.asarray(anchor_coords)
    if anchor_coords.shape[0] <= 2:
        return anchor_coords.mean(axis=0)
    bins = int(np.clip(np.sqrt(anchor_coords.shape[0]) / 2, 10, 45))
    hist, x_edges, y_edges = np.histogram2d(
        anchor_coords[:, 0], anchor_coords[:, 1], bins=bins
    )
    max_x, max_y = np.unravel_index(np.argmax(hist), hist.shape)
    in_bin = (
        (anchor_coords[:, 0] >= x_edges[max_x])
        & (anchor_coords[:, 0] <= x_edges[max_x + 1])
        & (anchor_coords[:, 1] >= y_edges[max_y])
        & (anchor_coords[:, 1] <= y_edges[max_y + 1])
    )
    anchor = np.median(anchor_coords[in_bin], axis=0)
    return anchor_coords[np.argmin(np.sum((anchor_coords - anchor) ** 2, axis=1))]


def move_text_pixel(ax, text, origin, dx, dy) -> None:
    text_px = ax.transData.transform(text.get_position())
    origin_px = ax.transData.transform(origin)
    next_px = text_px + np.array([dx, dy])
    delta = next_px - origin_px
    distance = np.linalg.norm(delta)
    if distance > LABEL_MAX_MOVE_PX:
        next_px = origin_px + delta / distance * LABEL_MAX_MOVE_PX
    text.set_position(ax.transData.inverted().transform(next_px))


def repel_labels(ax, texts, origins) -> None:
    for _ in range(LABEL_REPEL_MAX_ITER):
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        boxes = [text.get_window_extent(renderer=renderer).expanded(1.06, 1.15) for text in texts]
        moved = False
        for i in range(len(texts) - 1):
            for j in range(i + 1, len(texts)):
                if not boxes[i].overlaps(boxes[j]):
                    continue
                center_i = np.array([(boxes[i].x0 + boxes[i].x1) / 2, (boxes[i].y0 + boxes[i].y1) / 2])
                center_j = np.array([(boxes[j].x0 + boxes[j].x1) / 2, (boxes[j].y0 + boxes[j].y1) / 2])
                direction = center_i - center_j
                if np.linalg.norm(direction) == 0:
                    direction = np.array([1.0, 0.0])
                direction /= np.linalg.norm(direction)
                move_text_pixel(ax, texts[i], origins[i], *(direction * LABEL_REPEL_STEP_PX))
                move_text_pixel(ax, texts[j], origins[j], *(-direction * LABEL_REPEL_STEP_PX))
                moved = True
        if not moved:
            break


def text_edge_position(ax, text, target) -> np.ndarray:
    ax.figure.canvas.draw()
    bbox = text.get_window_extent(renderer=ax.figure.canvas.get_renderer()).expanded(1.03, 1.10)
    target_px = ax.transData.transform(target)
    center_px = np.array([(bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2])
    direction = target_px - center_px
    if np.linalg.norm(direction) == 0:
        return np.asarray(text.get_position())
    dx, dy = direction
    scale_x = bbox.width / 2 / abs(dx) if dx else np.inf
    scale_y = bbox.height / 2 / abs(dy) if dy else np.inf
    start_px = center_px + direction * min(scale_x, scale_y)
    start_px += direction / np.linalg.norm(direction) * CONNECT_LABEL_GAP_PX
    return ax.transData.inverted().transform(start_px)


def load_data(h5ad_file: Path):
    if not h5ad_file.is_file():
        raise FileNotFoundError(f"Input h5AD not found: {h5ad_file}")
    with h5py.File(h5ad_file, "r") as handle:
        coords = handle["obsm/X_umap"][()]
        stage_labels, _ = read_categorical(handle, "stage")
        celltype_labels, celltype_order = read_categorical(handle, "final_celltype_refined")
    stage_celltype_labels = np.asarray(
        [f"{stage} {celltype}" for stage, celltype in zip(stage_labels, celltype_labels)],
        dtype=object,
    )
    present_labels = set(stage_celltype_labels)
    stage_celltype_order = [
        f"{stage} {celltype}"
        for stage in STAGE_ORDER
        for celltype in celltype_order
        if f"{stage} {celltype}" in present_labels
    ]
    if len(stage_celltype_order) != len(PALETTE_COLORS):
        raise ValueError("The saved label set no longer matches the fixed Figure 1A palette.")
    if coords.shape[0] != len(stage_labels):
        raise ValueError("UMAP coordinates and cell annotations have different lengths.")
    return coords, stage_labels, celltype_labels, celltype_order, stage_celltype_labels, stage_celltype_order


def draw_figure(coords, celltype_labels, celltype_order, stage_celltype_labels, stage_celltype_order):
    palette = dict(zip(stage_celltype_order, PALETTE_COLORS))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(right=0.66)
    draw_order = np.random.default_rng(RANDOM_STATE).permutation(coords.shape[0])
    ax.scatter(
        coords[draw_order, 0], coords[draw_order, 1],
        c=[palette[label] for label in stage_celltype_labels[draw_order]],
        s=POINT_SIZE, alpha=POINT_ALPHA, linewidths=0, zorder=1,
    )
    x_pad = np.ptp(coords[:, 0]) * 0.08
    y_pad = np.ptp(coords[:, 1]) * 0.08
    ax.set_xlim(coords[:, 0].min() - x_pad, coords[:, 0].max() + x_pad)
    ax.set_ylim(coords[:, 1].min() - y_pad, coords[:, 1].max() + y_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)

    celltype_info = {}
    for celltype in celltype_order:
        celltype_mask = celltype_labels == celltype
        label_anchor = dense_anchor(coords[celltype_mask])
        stage_anchors = []
        for stage_celltype in stage_celltype_order:
            if stage_celltype.split(" ", 1)[1] != celltype:
                continue
            mask = stage_celltype_labels == stage_celltype
            stage_anchors.append({"label": stage_celltype, "anchor": dense_anchor(coords[mask])})
        distances = [np.linalg.norm(item["anchor"] - label_anchor) for item in stage_anchors]
        main_index = int(np.argmin(distances))
        celltype_info[celltype] = {
            "label_anchor": label_anchor,
            "stage_anchors": stage_anchors,
            "main_label": stage_anchors[main_index]["label"],
        }

    texts, origins, text_celltypes = [], [], []
    for celltype in celltype_order:
        origin = celltype_info[celltype]["label_anchor"]
        shift = np.asarray(MANUAL_LABEL_SHIFT_PX.get(celltype, (0, 0)))
        position = ax.transData.inverted().transform(ax.transData.transform(origin) + shift)
        origins.append(origin)
        text_celltypes.append(celltype)
        texts.append(ax.text(*position, celltype, ha="center", va="center", fontsize=LABEL_FONTSIZE, color="black", zorder=5))
    repel_labels(ax, texts, origins)

    outline_radius = LABEL_STROKE_WIDTH / 2
    outline_offsets = [
        (-outline_radius, 0), (outline_radius, 0), (0, -outline_radius), (0, outline_radius),
        (-outline_radius, -outline_radius), (-outline_radius, outline_radius),
        (outline_radius, -outline_radius), (outline_radius, outline_radius),
    ]
    for text in texts:
        for dx, dy in outline_offsets:
            ax.annotate(
                text.get_text(), xy=text.get_position(), xycoords="data", xytext=(dx, dy),
                textcoords="offset points", ha=text.get_ha(), va=text.get_va(),
                fontsize=text.get_fontsize(), color="white", zorder=4, annotation_clip=False,
            )

    for text, celltype in zip(texts, text_celltypes):
        if celltype in SKIP_CONNECT_CELLTYPES:
            continue
        info = celltype_info[celltype]
        for item in info["stage_anchors"]:
            if item["label"] == info["main_label"]:
                continue
            anchor = item["anchor"]
            anchor_distance = np.linalg.norm(
                ax.transData.transform(anchor) - ax.transData.transform(info["label_anchor"])
            )
            if anchor_distance < CONNECT_DISTANCE_PX:
                continue
            start = text_edge_position(ax, text, anchor)
            ax.plot([start[0], anchor[0]], [start[1], anchor[1]], color="black", lw=CONNECT_WIDTH, alpha=CONNECT_ALPHA, zorder=3)

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=palette[label],
               markeredgecolor=palette[label], markersize=LEGEND_MARKER_SIZE, label=label)
        for label in stage_celltype_order
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.00), ncol=1,
              frameon=False, fontsize=LEGEND_FONTSIZE, handlelength=0.8,
              handletextpad=0.35, columnspacing=1.0, borderaxespad=0)
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
    loaded = load_data(h5ad_file)
    if args.check_inputs:
        print(f"Validated Figure 1A input: {loaded[0].shape[0]:,} cells, {len(loaded[-1])} stage-cell-state groups")
        return
    output_dir = results_directory(__file__) / "figures/fig1"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = draw_figure(loaded[0], loaded[2], loaded[3], loaded[4], loaded[5])
    options = {"bbox_inches": EXPORT_BBOX, "pad_inches": 0, "facecolor": "white"}
    fig.savefig(output_dir / "fig1a_embryonic_celltype_umap.png", dpi=DPI, **options)
    fig.savefig(output_dir / "fig1a_embryonic_celltype_umap.pdf", format="pdf", **options)
    plt.close(fig)


if __name__ == "__main__":
    main()
