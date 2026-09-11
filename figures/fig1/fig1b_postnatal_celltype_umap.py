"""Reproduce the stage-cell-type UMAP used in Figure 1B."""

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
    decode,
    integer_ticks,
    read_categorical,
    resolve_input_path,
    results_directory,
)


RANDOM_STATE = 0
DPI = 1201
TARGET_PIXEL_SIZE = (7325, 5673)
EXPORT_BBOX = Bbox.from_bounds(
    0.119544,
    1.699986,
    TARGET_PIXEL_SIZE[0] / DPI,
    TARGET_PIXEL_SIZE[1] / DPI,
)
FIGSIZE = (8.4, 8.4)
POINT_SIZE = 1.35
POINT_ALPHA = 0.82
LABEL_FONTSIZE = 8.0
LABEL_STROKE_WIDTH = 1.6
LABEL_REPEL_MAX_ITER = 60
LABEL_REPEL_STEP_PX = 1.4
LABEL_MAX_MOVE_PX = 25
LEGEND_FONTSIZE = 6.8
LEGEND_MARKER_SIZE = 5.0
MANUAL_LABEL_POS = {
    "KO": (-4.65, 3.35),
    "IBC_IPhC": (-2.45, 7.55),
    "IHC": (5.85, 8.95),
    "OHC": (13.55, 1.55),
    "HeC": (0.85, -0.05),
    "ISC/OSC/CC/IDC": (-2.15, -1.55),
    "DC_PC": (6.0, -4.15),
}
DISPLAY_LABEL = {"ISC/OSC/CC/IDC": "ISC/OSC\nCC/IDC"}


def dense_anchor(anchor_coords: np.ndarray) -> np.ndarray:
    anchor_coords = np.asarray(anchor_coords)
    if anchor_coords.shape[0] <= 2:
        return anchor_coords.mean(axis=0)
    bins = int(np.clip(np.sqrt(anchor_coords.shape[0]) / 2, 10, 45))
    hist, x_edges, y_edges = np.histogram2d(anchor_coords[:, 0], anchor_coords[:, 1], bins=bins)
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
    next_px = text_px + np.asarray([dx, dy])
    delta = next_px - origin_px
    distance = np.linalg.norm(delta)
    if distance > LABEL_MAX_MOVE_PX:
        next_px = origin_px + delta / distance * LABEL_MAX_MOVE_PX
    text.set_position(ax.transData.inverted().transform(next_px))


def repel_labels(ax, texts, origins) -> None:
    for _ in range(LABEL_REPEL_MAX_ITER):
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        boxes = [text.get_window_extent(renderer=renderer).expanded(1.04, 1.10) for text in texts]
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


def load_data(h5ad_file: Path):
    if not h5ad_file.is_file():
        raise FileNotFoundError(f"Input h5AD not found: {h5ad_file}")
    with h5py.File(h5ad_file, "r") as handle:
        coords = handle["obsm/X_umap"][()]
        labels, _ = read_categorical(handle, "stage_celltype")
        stage_order = decode(handle["uns/stage_order"][()])
        celltype_order = decode(handle["uns/revised_celltype_order"][()])
    if coords.shape[0] != len(labels):
        raise ValueError("UMAP coordinates and cell annotations have different lengths.")
    present_labels = set(labels)
    stage_celltype_order = [
        f"{stage}_{celltype}"
        for celltype in celltype_order
        for stage in stage_order
        if f"{stage}_{celltype}" in present_labels
    ]
    unexpected = sorted(present_labels.difference(stage_celltype_order))
    if unexpected:
        raise ValueError(f"Unexpected stage-cell-type labels: {unexpected}")
    return coords, labels, celltype_order, stage_celltype_order


def draw_figure(coords, labels, celltype_order, stage_celltype_order):
    base_colors = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
    if len(stage_celltype_order) > len(base_colors):
        raise ValueError("The fixed postnatal palette does not contain enough colors.")
    palette = {label: base_colors[index] for index, label in enumerate(stage_celltype_order)}
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.08, right=0.58, bottom=0.12, top=0.90)
    draw_order = np.random.default_rng(RANDOM_STATE).permutation(coords.shape[0])
    ax.scatter(coords[draw_order, 0], coords[draw_order, 1],
               c=[palette[label] for label in labels[draw_order]], s=POINT_SIZE,
               alpha=POINT_ALPHA, linewidths=0, zorder=1)
    x_pad = np.ptp(coords[:, 0]) * 0.07
    y_pad = np.ptp(coords[:, 1]) * 0.07
    ax.set_xlim(coords[:, 0].min() - x_pad, coords[:, 0].max() + x_pad)
    ax.set_ylim(coords[:, 1].min() - y_pad, coords[:, 1].max() + y_pad)
    ax.set_box_aspect(1)

    celltype_to_stage_labels = {
        celltype: [label for label in stage_celltype_order if label.split("_", 1)[1] == celltype]
        for celltype in celltype_order
    }
    texts, origins = [], []
    for celltype in celltype_order:
        stage_labels = celltype_to_stage_labels[celltype]
        if not stage_labels:
            continue
        origin = dense_anchor(coords[np.isin(labels, stage_labels)])
        position = np.asarray(MANUAL_LABEL_POS.get(celltype, origin), dtype=float)
        origins.append(position)
        texts.append(ax.text(*position, DISPLAY_LABEL.get(celltype, celltype), ha="center",
                             va="center", fontsize=LABEL_FONTSIZE, color="black", zorder=5))
    repel_labels(ax, texts, origins)

    outline_radius = LABEL_STROKE_WIDTH / 2
    outline_offsets = [
        (-outline_radius, 0), (outline_radius, 0), (0, -outline_radius), (0, outline_radius),
        (-outline_radius, -outline_radius), (-outline_radius, outline_radius),
        (outline_radius, -outline_radius), (outline_radius, outline_radius),
    ]
    for text in texts:
        for dx, dy in outline_offsets:
            ax.annotate(text.get_text(), xy=text.get_position(), xycoords="data", xytext=(dx, dy),
                        textcoords="offset points", ha=text.get_ha(), va=text.get_va(),
                        fontsize=text.get_fontsize(), color="white", zorder=4,
                        annotation_clip=False)

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=palette[label],
               markeredgecolor=palette[label], markersize=LEGEND_MARKER_SIZE,
               label=label.replace("_", " "))
        for label in stage_celltype_order
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.04, 1.00), ncol=1,
              frameon=False, fontsize=LEGEND_FONTSIZE, handlelength=0.8,
              handletextpad=0.35, columnspacing=0.9, borderaxespad=0)
    ax.set_xlabel("UMAP1", fontsize=11)
    ax.set_ylabel("UMAP2", fontsize=11)
    ax.set_xticks(integer_ticks(ax.get_xlim()))
    ax.set_yticks(integer_ticks(ax.get_ylim()))
    ax.tick_params(axis="both", length=4, width=1.0, labelsize=10, direction="out")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("black")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-inputs", action="store_true", help="Validate the input without drawing files.")
    args = parser.parse_args()
    configure_matplotlib()
    h5ad_file = resolve_input_path(
        results_directory(__file__) / "02_scanvi_integration/postnatal/postnatal_scanvi.h5ad",
        data_directory(__file__) / "Integration_3types_after_training/Postnatal period files/PostBirth_scanvi_HVG5000_integrated.h5ad",
    )
    loaded = load_data(h5ad_file)
    if args.check_inputs:
        print(f"Validated Figure 1B input: {loaded[0].shape[0]:,} cells, {len(loaded[-1])} groups")
        return
    output_dir = results_directory(__file__) / "figures/fig1"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = draw_figure(*loaded)
    options = {"bbox_inches": EXPORT_BBOX, "pad_inches": 0, "facecolor": "white"}
    fig.savefig(output_dir / "fig1b_postnatal_celltype_umap.png", dpi=DPI, **options)
    fig.savefig(output_dir / "fig1b_postnatal_celltype_umap.pdf", format="pdf", **options)
    plt.close(fig)


if __name__ == "__main__":
    main()
