"""Embryonic scANVI integration for E9.5-E16.5.

The script keeps the analysis steps that affect the final embryonic integrated
object and final refined UMAP panels. It does not save historical Harmony or
subtype-check diagnostic figures.
"""

from __future__ import annotations

import contextlib
import io
import logging
import warnings
from pathlib import Path
import os

import anndata as ad
import matplotlib.patheffects as pe
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scvi
import torch


# 1. ===========Paths and parameters
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("COCHLEA_DATA_DIR", REPO_ROOT / "Data")).expanduser().resolve()
RESULTS_ROOT = Path(os.environ.get("COCHLEA_RESULTS_DIR", REPO_ROOT / "results")).expanduser().resolve()
STAGE_RESULT_DIR = RESULTS_ROOT / "01_stage_clustering" / "h5ad_for_integration"
PRECOMPUTED_STAGE_DIR = DATA_ROOT / "h5ad_for_integration"
OUTPUT_DIR = RESULTS_ROOT / "02_scanvi_integration" / "embryonic" / "figures"
DATA_DIR = RESULTS_ROOT / "02_scanvi_integration" / "embryonic"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 0
SCVI_MAX_EPOCHS = 200
SCANVI_MAX_EPOCHS = 80
N_LATENT = 30
N_LAYERS = 2
N_NEIGHBORS = 25
UMAP_MIN_DIST = 0.35
GENE_LIKELIHOOD = "nb"
SAVE_DPI = 1201

STAGE_ORDER = ["E9.5", "E11.5", "E13.5", "E14.5", "E16.5"]


def resolve_stage_file(filename: str) -> Path:
    """Prefer freshly generated stage objects, with a documented-data fallback."""
    generated = STAGE_RESULT_DIR / filename
    return generated if generated.exists() else PRECOMPUTED_STAGE_DIR / filename


STAGE_FILES = {
    "E9.5": resolve_stage_file("E9_for_scnvi_normalized.h5ad"),
    "E11.5": resolve_stage_file("E11_for_scnvi_normalized.h5ad"),
    "E13.5": resolve_stage_file("E13.5_for_scnvi_normalized.h5ad"),
    "E14.5": resolve_stage_file("E14.5_for_scnvi_normalized.h5ad"),
    "E16.5": resolve_stage_file("E16.5_for_scnvi_normalized.h5ad"),
}

EXCLUDE_MODEL_GENES = [
    "Oc90", "Otx2",
    "Ptprc", "Lyz2", "Cd74",
    "Hbb-bs", "Hbb-bt", "Hba-a1", "Hba-a2",
]

REFINED_COLOR_HINTS = {
    "IHC": "#1f77b4",
    "OHC": "#ff7f0e",
    "HeC": "#bcbd22",
    "IBC": "#2ca02c",
    "IPhC": "#2ca02c",
    "IPC": "#9467bd",
    "DC_PC": "#9467bd",
    "ISC/OSC/CC/IDC": "#17becf",
    "KO": "#c7b6bd",
}

MARKER_GROUPS = {
    "Core": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "Early_OV_domain": ["Pax2", "Pax8", "Dlx5", "Dlx6", "Jag1", "Fgf20", "Bmp4"],
    "Hair_cell": ["Atoh1", "Pou4f3", "Myo6", "Myo7a", "Cib2", "Pvalb", "Otof", "Tmc1"],
    "Supporting_cell": ["Fgfr3", "Prox1", "S100a1", "Gjb2", "Smpx", "Aqp4", "Dcn", "Col3a1"],
    "Exclusion": ["Oc90", "Otx2", "Ptprc", "Lyz2", "Cd74", "Hbb-bs", "Hbb-bt", "Hba-a1", "Hba-a2"],
}

warnings.filterwarnings("ignore", category=FutureWarning)
for logger_name in ["scanpy", "scvi", "lightning", "lightning.pytorch", "pytorch_lightning"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
sc.settings.verbosity = 1
scvi.settings.seed = RANDOM_STATE
scvi.settings.dl_num_workers = 0
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# 2. ===========General helpers
def to_csr_float32(adata: ad.AnnData) -> ad.AnnData:
    if sp.issparse(adata.X):
        adata.X = adata.X.astype(np.float32).tocsr()
    else:
        adata.X = sp.csr_matrix(np.asarray(adata.X, dtype=np.float32))
    return adata


def get_expr_vector(adata: ad.AnnData, gene: str) -> np.ndarray:
    if gene not in adata.var_names:
        return np.zeros(adata.n_obs, dtype=float)
    x = adata[:, gene].X
    if sp.issparse(x):
        return np.asarray(x.toarray()).ravel()
    return np.asarray(x).ravel()


def match_present_genes(adata: ad.AnnData, genes: list[str]) -> list[str]:
    lookup = {str(gene).upper(): str(gene) for gene in adata.var_names}
    present = []
    for gene in genes:
        matched = lookup.get(gene.upper())
        if matched is not None and matched not in present:
            present.append(matched)
    return present


def train_model(model, max_epochs: int) -> None:
    train_kwargs = {
        "max_epochs": max_epochs,
        "early_stopping": True,
        "batch_size": 512,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
    }
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if torch.cuda.is_available():
            try:
                model.train(**train_kwargs, accelerator="gpu", devices=1)
            except TypeError:
                model.train(**train_kwargs, use_gpu=True)
        else:
            model.train(**train_kwargs)


# 3. ===========Read and merge embryonic inputs
def load_embryonic_inputs() -> ad.AnnData:
    adatas = []
    for stage in STAGE_ORDER:
        path = STAGE_FILES[stage]
        if not path.exists():
            raise FileNotFoundError(path)

        adata = sc.read_h5ad(path).copy()
        adata.var_names = adata.var_names.astype(str)
        adata.obs_names = [f"{stage}_{x}" for x in adata.obs_names.astype(str)]
        adata.var_names_make_unique()
        adata.obs_names_make_unique()

        adata.obs["stage"] = pd.Categorical([stage] * adata.n_obs, categories=STAGE_ORDER, ordered=True)
        if "final_celltype" not in adata.obs:
            adata.obs["final_celltype"] = "Unknown"
        adata.obs["final_celltype"] = adata.obs["final_celltype"].astype(str)

        if "sample" not in adata.obs:
            adata.obs["sample"] = stage
        adata.obs["sample"] = adata.obs["sample"].astype(str)

        if "source_object" not in adata.obs:
            adata.obs["source_object"] = path.stem
        adata.obs["source_object"] = adata.obs["source_object"].astype(str)
        adata.obs["source_stage"] = stage
        adata.obs["source_file"] = path.name

        # The source objects store log-like normalized values in X. Keeping X
        # reproduces the manuscript integration; no raw-count layer is swapped in.
        adata.raw = None
        adata.layers.clear()
        adata.obsm.clear()
        adata.uns.clear()
        adatas.append(to_csr_float32(adata))

    adata_merged = ad.concat(adatas, join="inner", merge="same", uns_merge=None, index_unique=None)
    adata_merged.obs["stage"] = pd.Categorical(
        adata_merged.obs["stage"].astype(str),
        categories=STAGE_ORDER,
        ordered=True,
    )

    mt_mask = adata_merged.var_names.str.startswith(("mt-", "Mt-", "MT-"))
    adata_merged = adata_merged[:, ~mt_mask].copy()
    sc.pp.filter_genes(adata_merged, min_cells=10)
    return adata_merged


# 4. ===========Pre-correction embedding retained in final h5ad
def add_pre_harmony_embedding(adata: ad.AnnData) -> ad.AnnData:
    adata_pre = adata.copy()
    sc.pp.highly_variable_genes(
        adata_pre,
        n_top_genes=min(3000, adata_pre.n_vars),
        flavor="seurat",
    )
    hvg_mask = adata_pre.var["highly_variable"].to_numpy().copy()
    hvg_mask &= ~adata_pre.var_names.isin(EXCLUDE_MODEL_GENES)

    adata_pre = adata_pre[:, hvg_mask].copy()
    sc.pp.scale(adata_pre, max_value=10)
    sc.tl.pca(adata_pre, svd_solver="arpack", n_comps=50, random_state=RANDOM_STATE)
    sc.pp.neighbors(adata_pre, n_neighbors=25, n_pcs=40)
    sc.tl.umap(adata_pre, min_dist=UMAP_MIN_DIST, random_state=RANDOM_STATE)
    sc.tl.leiden(adata_pre, resolution=0.5, key_added="pre_harmony_leiden", random_state=RANDOM_STATE)

    adata.obs["pre_harmony_leiden"] = adata_pre.obs["pre_harmony_leiden"].astype(str).to_numpy()
    adata.obsm["X_umap_pre_harmony"] = adata_pre.obsm["X_umap"].copy()
    return adata


# 5. ===========Train embryonic scANVI
def run_scanvi(adata: ad.AnnData) -> tuple[ad.AnnData, scvi.model.SCANVI]:
    adata_scanvi = adata.copy()

    if "Cd74" in adata_scanvi.var_names:
        adata_scanvi = adata_scanvi[get_expr_vector(adata_scanvi, "Cd74") == 0].copy()

    model_exclude = [gene for gene in EXCLUDE_MODEL_GENES if gene in adata_scanvi.var_names]
    if model_exclude:
        adata_scanvi = adata_scanvi[:, ~adata_scanvi.var_names.isin(model_exclude)].copy()

    to_csr_float32(adata_scanvi)
    adata_scanvi.obs["stage"] = pd.Categorical(
        adata_scanvi.obs["stage"].astype(str),
        categories=STAGE_ORDER,
        ordered=True,
    )
    adata_scanvi.obs["scanvi_label"] = adata_scanvi.obs["final_celltype"].astype(str)
    label_categories = sorted(adata_scanvi.obs["scanvi_label"].unique().tolist())
    if "Unknown" not in label_categories:
        label_categories.append("Unknown")
    adata_scanvi.obs["scanvi_label"] = pd.Categorical(
        adata_scanvi.obs["scanvi_label"],
        categories=label_categories,
    )
    adata_scanvi.obs["scanvi_batch"] = adata_scanvi.obs["source_object"].astype(str).astype("category")

    scvi.model.SCVI.setup_anndata(
        adata_scanvi,
        batch_key="scanvi_batch",
        labels_key="scanvi_label",
    )
    scvi_model = scvi.model.SCVI(
        adata_scanvi,
        n_layers=N_LAYERS,
        n_latent=N_LATENT,
        gene_likelihood=GENE_LIKELIHOOD,
    )
    train_model(scvi_model, SCVI_MAX_EPOCHS)

    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        labels_key="scanvi_label",
        unlabeled_category="Unknown",
    )
    train_model(scanvi_model, SCANVI_MAX_EPOCHS)

    adata_scanvi.obsm["X_scanvi"] = scanvi_model.get_latent_representation()
    sc.pp.neighbors(adata_scanvi, use_rep="X_scanvi", n_neighbors=N_NEIGHBORS)
    sc.tl.umap(adata_scanvi, min_dist=UMAP_MIN_DIST, random_state=RANDOM_STATE)
    sc.tl.leiden(adata_scanvi, resolution=0.6, key_added="scanvi_leiden", random_state=RANDOM_STATE)
    return adata_scanvi, scanvi_model


# 6. ===========Refine IPhC and IBC labels
def refine_celltypes(adata_scanvi: ad.AnnData) -> ad.AnnData:
    iphc_mask = adata_scanvi.obs["final_celltype"].astype(str).isin(["IPhC", "IPHC"])
    adata_iphc = adata_scanvi[iphc_mask].copy()

    adata_scanvi.obs["final_celltype_refined"] = adata_scanvi.obs["final_celltype"].astype(str)
    if adata_iphc.n_obs >= 20:
        sc.pp.neighbors(
            adata_iphc,
            use_rep="X_scanvi",
            n_neighbors=min(20, max(5, adata_iphc.n_obs // 5)),
        )
        sc.tl.umap(adata_iphc, min_dist=UMAP_MIN_DIST, random_state=RANDOM_STATE)
        sc.tl.leiden(adata_iphc, resolution=0.4, key_added="iphc_sub_leiden", random_state=RANDOM_STATE)

        iphc_sub = adata_iphc.obs["iphc_sub_leiden"].astype(str)
        adata_scanvi.obs.loc[iphc_sub.index[iphc_sub == "0"], "final_celltype_refined"] = "IPhC"
        adata_scanvi.obs.loc[iphc_sub.index[iphc_sub == "1"], "final_celltype_refined"] = "IBC"

    adata_scanvi.obs["final_celltype_refined"] = (
        adata_scanvi.obs["final_celltype_refined"]
        .astype(str)
        .replace({"IBC-like": "IBC", "IBC_Like": "IBC", "IBC_like": "IBC"})
    )
    adata_scanvi.obs["final_celltype_refined"] = pd.Categorical(
        adata_scanvi.obs["final_celltype_refined"]
    )
    return adata_scanvi


# 7. ===========Final figure helpers
def set_refined_palette(adata: ad.AnnData, key: str) -> list[str]:
    categories = list(adata.obs[key].cat.categories)
    base_colors = (
        list(plt.cm.tab20.colors)
        + list(plt.cm.tab20b.colors)
        + list(plt.cm.tab20c.colors)
        + list(plt.cm.Set3.colors)
        + list(plt.cm.Dark2.colors)
    )
    palette = {
        label: REFINED_COLOR_HINTS.get(label, base_colors[i % len(base_colors)])
        for i, label in enumerate(categories)
    }
    adata.uns[f"{key}_colors"] = [palette[label] for label in categories]
    return categories


def save_plot_cache(adata: ad.AnnData) -> None:
    cache = adata.obs[["stage", "final_celltype", "final_celltype_refined", "scanvi_leiden"]].copy()
    cache["UMAP1"] = adata.obsm["X_umap"][:, 0]
    cache["UMAP2"] = adata.obsm["X_umap"][:, 1]
    cache.to_csv(OUTPUT_DIR / "scANVI_refined_celltype_umap_plot_cache.csv")


def save_refined_umap_panels(adata: ad.AnnData) -> None:
    set_refined_palette(adata, "final_celltype_refined")

    ax = sc.pl.umap(
        adata,
        color="final_celltype_refined",
        size=8,
        frameon=True,
        legend_loc="on data",
        legend_fontsize=7,
        legend_fontoutline=2,
        title="scANVI: refined cell type",
        show=False,
    )
    fig = ax.figure
    fig.set_size_inches(6, 5)
    fig.savefig(OUTPUT_DIR / "scANVI_refined_celltype_label_on_data.png", dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "Embryonic UMAP label.pdf", bbox_inches="tight")
    plt.close(fig)

    ax = sc.pl.umap(
        adata,
        color="final_celltype_refined",
        size=8,
        frameon=True,
        legend_loc="right margin",
        title="scANVI: refined cell type",
        show=False,
    )
    fig = ax.figure
    fig.savefig(OUTPUT_DIR / "scANVI_refined_celltype_legend.png", dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "Embryonic UMAP legend.pdf", bbox_inches="tight")
    plt.close(fig)


def save_stage_celltype_label_panel(adata: ad.AnnData) -> None:
    label_key = "stage_celltype_plot"
    labels = (
        adata.obs["stage"].astype(str)
        + " "
        + adata.obs["final_celltype_refined"].astype(str).replace({"IBC-like": "IBC"})
    )
    adata.obs[label_key] = labels
    label_order = labels.drop_duplicates().tolist()
    adata.obs[label_key] = pd.Categorical(adata.obs[label_key], categories=label_order, ordered=True)

    coords = np.asarray(adata.obsm["X_umap"])
    base_colors = list(plt.cm.tab20.colors) + list(plt.cm.Set3.colors) + list(plt.cm.Dark2.colors)
    palette = {label: base_colors[i % len(base_colors)] for i, label in enumerate(label_order)}

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    rng = np.random.default_rng(RANDOM_STATE)
    draw_order = rng.permutation(adata.n_obs)
    label_values = adata.obs[label_key].astype(str).to_numpy()
    ax.scatter(
        coords[draw_order, 0],
        coords[draw_order, 1],
        c=[palette[x] for x in label_values[draw_order]],
        s=2.0,
        alpha=0.85,
        linewidths=0,
    )

    texts = []
    anchors = []
    for label in label_order:
        mask = label_values == label
        if not np.any(mask):
            continue
        pts = coords[mask]
        anchor = np.median(pts, axis=0)
        anchor = pts[np.argmin(np.sum((pts - anchor) ** 2, axis=1))]
        anchors.append(anchor)
        texts.append(
            ax.text(
                anchor[0],
                anchor[1],
                label,
                ha="center",
                va="center",
                fontsize=7.0,
                color="black",
                path_effects=[pe.withStroke(linewidth=1.7, foreground="white")],
            )
        )

    repel_label_texts(ax, texts, anchors)
    for text, anchor in zip(texts, anchors):
        text_pos = np.asarray(text.get_position())
        if np.linalg.norm(text_pos - anchor) > 1e-8:
            ax.plot([anchor[0], text_pos[0]], [anchor[1], text_pos[1]], color="0.72", lw=0.42, alpha=0.70)

    ax.set_title("scANVI: refined cell type", fontsize=18)
    ax.set_xlabel("UMAP1", fontsize=18)
    ax.set_ylabel("UMAP2", fontsize=18)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(OUTPUT_DIR / "Embryonic UMAP label_2.pdf", dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    del adata.obs[label_key]


def repel_label_texts(ax, texts: list[plt.Text], anchors: list[np.ndarray], max_iter: int = 160) -> None:
    if len(texts) < 2:
        return

    fig = ax.figure
    anchor_px = ax.transData.transform(np.asarray(anchors))
    for _ in range(max_iter):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [text.get_window_extent(renderer=renderer).expanded(1.05, 1.12) for text in texts]
        shifts = np.zeros((len(texts), 2), dtype=float)

        for i in range(len(texts) - 1):
            for j in range(i + 1, len(texts)):
                if not boxes[i].overlaps(boxes[j]):
                    continue
                center_i = np.array([(boxes[i].x0 + boxes[i].x1) / 2, (boxes[i].y0 + boxes[i].y1) / 2])
                center_j = np.array([(boxes[j].x0 + boxes[j].x1) / 2, (boxes[j].y0 + boxes[j].y1) / 2])
                direction = center_i - center_j
                if np.linalg.norm(direction) == 0:
                    direction = np.array([1.0, 0.0])
                direction = direction / np.linalg.norm(direction)
                shifts[i] += direction * 4.0
                shifts[j] -= direction * 4.0

        if not np.any(shifts):
            break

        for i, text in enumerate(texts):
            current_px = ax.transData.transform(text.get_position())
            next_px = current_px + shifts[i]
            delta = next_px - anchor_px[i]
            dist = np.linalg.norm(delta)
            if dist > 150:
                next_px = anchor_px[i] + delta * (150 / dist)
            text.set_position(ax.transData.inverted().transform(next_px))


def save_marker_dotplot(adata: ad.AnnData) -> None:
    stage_celltype = adata.obs["stage"].astype(str) + "_" + adata.obs["final_celltype_refined"].astype(str)
    order = stage_celltype.drop_duplicates().tolist()
    adata_plot = adata.copy()
    adata_plot.obs["stage_celltype"] = pd.Categorical(stage_celltype, categories=order, ordered=True)

    marker_groups = {
        group: match_present_genes(adata_plot, genes)
        for group, genes in MARKER_GROUPS.items()
    }
    marker_groups = {group: genes for group, genes in marker_groups.items() if genes}

    dp = sc.pl.dotplot(
        adata_plot,
        var_names=marker_groups,
        groupby="stage_celltype",
        categories_order=order,
        standard_scale="var",
        color_map="Reds",
        dot_min=0,
        dot_max=0.85,
        dendrogram=False,
        var_group_rotation=90,
        figsize=(16, max(6, 0.22 * len(order) + 2.5)),
        show=False,
        return_fig=True,
    )
    dp.savefig(str(OUTPUT_DIR / "Embryonic marker dotplot.pdf"))
    plt.close("all")


# 8. ===========Save h5ad and model
def save_integrated_outputs(adata: ad.AnnData, scanvi_model: scvi.model.SCANVI) -> None:
    adata.write_h5ad(DATA_DIR / "Embryonic_scanvi_refined_integrated.h5ad")
    scanvi_model.save(
        DATA_DIR / "Embryonic_scanvi_refined_model",
        overwrite=True,
        save_anndata=False,
    )


# 9. ===========Run workflow
def main() -> None:
    print("Running embryonic scANVI integration.")
    adata_emb = load_embryonic_inputs()
    adata_emb = add_pre_harmony_embedding(adata_emb)
    adata_scanvi, scanvi_model = run_scanvi(adata_emb)
    adata_scanvi = refine_celltypes(adata_scanvi)

    save_integrated_outputs(adata_scanvi, scanvi_model)
    save_plot_cache(adata_scanvi)
    save_refined_umap_panels(adata_scanvi)
    save_stage_celltype_label_panel(adata_scanvi)
    save_marker_dotplot(adata_scanvi)
    print("Embryonic scANVI integration finished.")


if __name__ == "__main__":
    main()
