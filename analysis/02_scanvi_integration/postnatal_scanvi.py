"""Postnatal scANVI integration for P1-P28 with HVG5000.

The executable workflow uses the revised cell-type labels shown in the final
postnatal panels. Helper functions for the original-label diagnostic run are
kept for provenance but are not called by :func:`main`.
"""

from __future__ import annotations

import contextlib
import itertools
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
DATA_BASE_DIR = RESULTS_ROOT / "02_scanvi_integration" / "postnatal"
POSTNATAL_DATA_DIR = DATA_BASE_DIR
POSTNATAL_OUTPUT_DIR = DATA_BASE_DIR / "figures"
POSTBIRTH_OUTPUT_DIR = POSTNATAL_OUTPUT_DIR / "original_label_diagnostics"
GRID_METRICS_FILE = DATA_BASE_DIR / "grid_metrics.csv"

for path in [POSTBIRTH_OUTPUT_DIR, POSTNATAL_OUTPUT_DIR, DATA_BASE_DIR, POSTNATAL_DATA_DIR]:
    path.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 0
N_HVG = 5000
HVG_FLAVOR = "seurat"
N_LATENT = 30
N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.35
SAVE_DPI = 1201

ORIGINAL_SCVI_MAX_EPOCHS = 300
ORIGINAL_SCANVI_MAX_EPOCHS = 120
REVISED_SCVI_MAX_EPOCHS = 400
REVISED_SCANVI_MAX_EPOCHS = 150

STAGE_ORDER = ["P1", "P7", "P14", "P28"]


def resolve_stage_file(filename: str) -> Path:
    """Prefer freshly generated stage objects, with a documented-data fallback."""
    generated = STAGE_RESULT_DIR / filename
    return generated if generated.exists() else PRECOMPUTED_STAGE_DIR / filename


STAGE_FILES = {
    "P1": resolve_stage_file("P1_for_scnvi_normalized.h5ad"),
    "P7": resolve_stage_file("P7_for_scnvi_normalized.h5ad"),
    "P14": resolve_stage_file("P14_for_scnvi_normalized.h5ad"),
    "P28": resolve_stage_file("P28_for_scnvi_normalized.h5ad"),
}
LABEL_KEYS = {
    "P1": "P1_final_celltype",
    "P7": "P7_final_celltype",
    "P14": "P14_final_celltype",
    "P28": "P28_final_celltype",
}
SAMPLE_KEYS = {
    "P1": "sample",
    "P7": "batch",
    "P14": "sample",
    "P28": "sample",
}

REVISED_MAP = {
    "ISC": "ISC/OSC/CC/IDC",
    "OSC": "ISC/OSC/CC/IDC",
    "CC/OSC": "ISC/OSC/CC/IDC",
    "CC/OS": "ISC/OSC/CC/IDC",
    "ISC/OSC/CC": "ISC/OSC/CC/IDC",
    "ISC/OSC/CC/IdC": "ISC/OSC/CC/IDC",
    "ISC/OSC/CC/IDC": "ISC/OSC/CC/IDC",
    "IdC": "ISC/OSC/CC/IDC",
    "IDC": "ISC/OSC/CC/IDC",
    "IB": "IBC_IPhC",
    "IBC": "IBC_IPhC",
    "IPhC": "IBC_IPhC",
    "IPHC": "IBC_IPhC",
    "IPHCh": "IBC_IPhC",
    "IBC_IPHC": "IBC_IPhC",
    "IBC_IPhC": "IBC_IPhC",
    "DC": "DC_PC",
    "PC": "DC_PC",
    "IPC": "DC_PC",
    "OPC": "DC_PC",
    "DC-1/2": "DC_PC",
    "DC-3": "DC_PC",
    "KO-1": "KO",
    "KO-2": "KO",
    "KO-3": "KO",
    "KO-4": "KO",
    "L_KO": "KO",
    "M_KO": "KO",
    "ML_KO": "KO",
}
REVISED_CELLTYPE_ORDER = ["IHC", "OHC", "HeC", "IBC_IPhC", "DC_PC", "ISC/OSC/CC/IDC", "KO"]

ORIGINAL_CELLTYPE_ORDER = [
    "IHC", "OHC",
    "KO-1", "KO-2", "KO-3", "KO-4", "L_KO", "M_KO", "ML_KO",
    "IPhC", "IPC", "OPC", "PC",
    "DC-1/2", "DC-3", "DC",
    "HeC", "CC/OSC", "CC/OS", "OSC", "ISC", "IB", "IdC",
]

POSTNATAL_MARKER_GROUPS = {
    "Core": ["Epcam", "Gata3", "Sox2", "Isl1", "Lgr5"],
    "IHC": ["Otof", "Slc17a8", "Atp2a3", "Cabp2", "Fgf8", "Calb2", "Tmc1", "Myo6", "Myo7a"],
    "OHC": ["Slc26a5", "Ocm", "Pcp4", "Aqp11", "Cib2", "Six2"],
    "HeC": ["Pmch", "Smpx", "Epyc", "Kazald1", "Fabp7", "Sostdc1"],
    "IBC_IPhC": ["Anxa5", "Matn4", "Gjb2", "Fabp7", "Igfbp4", "Slc1a3"],
    "DC_PC": ["Fgfr3", "Prox1", "Ceacam16", "Rbp7", "Lgr6", "Npy", "Pcp4", "Calb2"],
    "ISC/OSC/CC/IDC": [
        "Igf1", "Matn1", "Meg3", "Rgcc", "Tm4sf1", "Aqp4",
        "Npnt", "Bmp4", "Fbln2", "Fst", "Apoe", "Coch",
        "Sparcl1", "Otor", "Col3a1", "Ibsp", "Smoc2",
        "Cdkn1c", "Fxyd6", "Otoa", "Ptgds", "Gsn",
    ],
    "KO": [
        "Dcn", "Rcn3", "Pdia6", "Sdf2l1", "Ddost", "Cpxm2", "Ctgf", "Kazald1",
        "Tectb", "Fkbp9", "Cst3", "Gjb6", "Net1", "Tsen15", "Calb1", "Crabp1",
        "Epyc", "Itm2a", "Stmn2", "Anxa5", "Foxq1", "Igfbp3", "Gsn", "Hspa2",
        "Cldn4", "Upp1", "Ly6h", "Serpina3a", "Ier5", "Clu", "Grb14", "Fam159b", "Qpct",
    ],
    "Exclusion": ["Oc90", "Otx2", "Ptprc", "Lyz2", "Cd74", "Hbb-bs", "Hbb-bt", "Hba-a1", "Hba-a2"],
}

POSTNATAL_DOTPLOT_PARTS = {
    "part1_core_hair_hec": ["Core", "IHC", "OHC", "HeC"],
    "part2_support_epithelium": ["IBC_IPhC", "DC_PC", "ISC/OSC/CC/IDC"],
    "part3_KO_exclusion": ["KO", "Exclusion"],
}

POSTBIRTH_DOTPLOT_PARTS = {
    "part1_HC_PC_DC": ["Core", "IHC", "OHC", "DC_PC"],
    "part2_support": ["HeC", "IBC_IPhC", "ISC/OSC/CC/IDC"],
    "part3_KO": ["KO", "Exclusion"],
}

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*does not contain unnormalized count data.*")
for logger_name in ["scanpy", "scvi", "lightning", "lightning.pytorch", "pytorch_lightning"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
sc.settings.verbosity = 1
scvi.settings.seed = RANDOM_STATE
scvi.settings.dl_num_workers = 0

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


def match_gene(gene: str, var_names: pd.Index) -> str | None:
    if gene in var_names:
        return gene
    lookup = {str(x).upper(): str(x) for x in var_names}
    return lookup.get(gene.upper())


def marker_groups_present(adata: ad.AnnData, groups: dict[str, list[str]], max_markers: int = 8) -> dict[str, list[str]]:
    used = set()
    present = {}
    for group, genes in groups.items():
        group_genes = []
        for gene in genes:
            matched = match_gene(gene, adata.var_names)
            if matched is None or matched in used:
                continue
            group_genes.append(matched)
            used.add(matched)
            if len(group_genes) >= max_markers:
                break
        if group_genes:
            present[group] = group_genes
    return present


def flatten_groups(groups: dict[str, list[str]]) -> list[str]:
    genes = []
    for values in groups.values():
        genes.extend(values)
    return list(dict.fromkeys(genes))


def make_palette(order: list[str]) -> dict[str, object]:
    cmaps = [plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c, plt.cm.Set3, plt.cm.Dark2]
    palette = {}
    for i, item in enumerate(order):
        cmap = cmaps[(i // 20) % len(cmaps)]
        palette[item] = cmap(i % cmap.N)
    return palette


def map_original_scanvi_label(label: str) -> str:
    label = str(label)
    if label.startswith("KO-") or label in {"L_KO", "M_KO", "ML_KO"}:
        return "KO"
    if label in {"DC-1/2", "DC-3", "DC"}:
        return "DC"
    if label in {"IPC", "OPC", "PC"}:
        return "PC"
    if label in {"CC/OSC", "CC/OS", "OSC"}:
        return "OSC"
    return label


def make_revised_celltype(label: str) -> str:
    return REVISED_MAP.get(str(label), str(label))


def train_model(model, max_epochs: int, early_stopping: bool) -> None:
    train_kwargs = {
        "max_epochs": max_epochs,
        "early_stopping": early_stopping,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
    }
    if early_stopping:
        train_kwargs["early_stopping_patience"] = 30
        train_kwargs["check_val_every_n_epoch"] = 10

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if torch.cuda.is_available():
            try:
                model.train(**train_kwargs, accelerator="gpu", devices=1)
            except TypeError:
                model.train(**train_kwargs, use_gpu=True)
        else:
            model.train(**train_kwargs)


# 3. ===========Read and merge post-birth inputs
def read_stage_adata(stage: str, revised: bool) -> ad.AnnData:
    path = STAGE_FILES[stage]
    if not path.exists():
        raise FileNotFoundError(path)

    adata = sc.read_h5ad(path).copy()
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    label_key = LABEL_KEYS[stage]
    sample_key = SAMPLE_KEYS[stage]
    if label_key not in adata.obs:
        raise KeyError(f"{stage} missing obs['{label_key}'].")
    if sample_key not in adata.obs:
        raise KeyError(f"{stage} missing obs['{sample_key}'].")

    adata.obs_names = [f"{stage}__{x}" for x in adata.obs_names.astype(str)]
    adata.obs["stage"] = stage
    adata.obs["sample_raw"] = adata.obs[sample_key].astype(str)
    adata.obs["sample_id"] = adata.obs["stage"].astype(str) + "__" + adata.obs["sample_raw"].astype(str)
    adata.obs["original_celltype"] = adata.obs[label_key].astype(str)

    if revised:
        adata.obs["integration_batch"] = "single_sample" if stage == "P14" else adata.obs["sample_id"].astype(str)
        adata.obs["revised_celltype"] = adata.obs["original_celltype"].map(make_revised_celltype)
        adata.obs["scanvi_label"] = adata.obs["revised_celltype"].astype(str)
        adata.obs["stage_celltype"] = adata.obs["stage"].astype(str) + "_" + adata.obs["revised_celltype"].astype(str)
        keep_obs = [
            "stage", "sample_raw", "sample_id", "integration_batch",
            "original_celltype", "revised_celltype", "scanvi_label", "stage_celltype",
        ]
    else:
        adata.obs["integration_batch"] = adata.obs["sample_id"].astype(str)
        adata.obs["scanvi_label"] = adata.obs["original_celltype"].map(map_original_scanvi_label)
        adata.obs["stage_celltype"] = adata.obs["stage"].astype(str) + "_" + adata.obs["original_celltype"].astype(str)
        keep_obs = [
            "stage", "sample_raw", "sample_id", "integration_batch",
            "original_celltype", "scanvi_label", "stage_celltype",
        ]

    to_csr_float32(adata)
    adata.obs = adata.obs[keep_obs].copy()
    adata.var = pd.DataFrame(index=adata.var_names)
    # The source objects store log-like normalized values in X. Keeping X
    # reproduces the fitted manuscript model; no raw-count layer is substituted.
    adata.raw = None
    adata.layers.clear()
    adata.obsm.clear()
    adata.varm.clear()
    adata.obsp.clear()
    adata.uns = {}
    return adata


def merge_postbirth_inputs(revised: bool) -> ad.AnnData:
    adatas = [read_stage_adata(stage, revised=revised) for stage in STAGE_ORDER]
    gene_sets = [set(adata.var_names) for adata in adatas]
    common_genes = [gene for gene in adatas[0].var_names if all(gene in genes for genes in gene_sets[1:])]
    if not common_genes:
        raise ValueError("No common genes were found for the post-birth merge.")

    adata_post = ad.concat(
        [adata[:, common_genes].copy() for adata in adatas],
        join="inner",
        merge="same",
        uns_merge=None,
        index_unique=None,
    )
    adata_post.obs["stage"] = pd.Categorical(adata_post.obs["stage"].astype(str), categories=STAGE_ORDER, ordered=True)

    if revised:
        present = [x for x in REVISED_CELLTYPE_ORDER if x in set(adata_post.obs["revised_celltype"].astype(str))]
        adata_post.obs["revised_celltype"] = pd.Categorical(
            adata_post.obs["revised_celltype"].astype(str),
            categories=present,
            ordered=True,
        )
        adata_post.obs["scanvi_label"] = pd.Categorical(adata_post.obs["scanvi_label"].astype(str), categories=present)
    else:
        label_order = [
            x for x in ["IHC", "OHC", "KO", "IPhC", "PC", "DC", "HeC", "OSC", "ISC", "IB", "IdC"]
            if x in set(adata_post.obs["scanvi_label"].astype(str))
        ]
        adata_post.obs["scanvi_label"] = pd.Categorical(
            adata_post.obs["scanvi_label"].astype(str),
            categories=label_order,
            ordered=True,
        )
    return adata_post


def selected_hvg_genes(adata: ad.AnnData) -> list[str]:
    adata_hvg = adata.copy()
    sc.pp.highly_variable_genes(
        adata_hvg,
        flavor=HVG_FLAVOR,
        n_top_genes=min(N_HVG, adata_hvg.n_vars),
        subset=False,
    )
    return adata_hvg.var_names[adata_hvg.var["highly_variable"].to_numpy()].tolist()


# 4. ===========Train HVG5000 scVI and scANVI
def train_postbirth_scanvi(
    adata_post: ad.AnnData,
    scvi_epochs: int,
    scanvi_epochs: int,
    early_stopping: bool,
) -> tuple[ad.AnnData, scvi.model.SCANVI]:
    selected_genes = selected_hvg_genes(adata_post)
    adata_model = adata_post[:, selected_genes].copy()
    adata_model.obs["integration_batch"] = adata_model.obs["integration_batch"].astype("category")
    adata_model.obs["scanvi_label"] = adata_model.obs["scanvi_label"].astype("category")
    if "Unknown" not in adata_model.obs["scanvi_label"].cat.categories:
        adata_model.obs["scanvi_label"] = adata_model.obs["scanvi_label"].cat.add_categories(["Unknown"])

    adata_model.var[f"hvg_{N_HVG}"] = True
    scvi.model.SCVI.setup_anndata(
        adata_model,
        batch_key="integration_batch",
        labels_key="scanvi_label",
    )
    scvi_model = scvi.model.SCVI(adata_model, n_latent=N_LATENT)
    train_model(scvi_model, scvi_epochs, early_stopping=early_stopping)

    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        labels_key="scanvi_label",
        unlabeled_category="Unknown",
    )
    train_model(scanvi_model, scanvi_epochs, early_stopping=early_stopping)

    adata_model.obsm["X_scanvi"] = scanvi_model.get_latent_representation()
    sc.pp.neighbors(
        adata_model,
        use_rep="X_scanvi",
        n_neighbors=N_NEIGHBORS,
        random_state=RANDOM_STATE,
    )
    sc.tl.umap(adata_model, min_dist=UMAP_MIN_DIST, random_state=RANDOM_STATE)
    return adata_model, scanvi_model


# 5. ===========Original-celltype outputs
def original_stage_celltype_order(adata: ad.AnnData) -> list[str]:
    present = set(adata.obs["stage_celltype"].astype(str))
    order = []
    for celltype in ORIGINAL_CELLTYPE_ORDER:
        for stage in STAGE_ORDER:
            label = f"{stage}_{celltype}"
            if label in present:
                order.append(label)
    return order + sorted(present - set(order))


def save_original_h5ad_and_model(adata_model: ad.AnnData, scanvi_model: scvi.model.SCANVI) -> None:
    out_dir = DATA_BASE_DIR / "PostBirth_scanvi_HVG5000-1"
    out_dir.mkdir(parents=True, exist_ok=True)
    adata_save = adata_model.copy()
    adata_save.uns["postbirth_scanvi"] = {
        "hvg": N_HVG,
        "batch_key": "integration_batch",
        "labels_key": "scanvi_label",
        "stage_order": list(STAGE_ORDER),
        "obsm": ["X_scanvi", "X_umap"],
    }
    adata_save.write_h5ad(out_dir / "PostBirth_scanvi_HVG5000_integrated.h5ad", compression="gzip")
    scanvi_model.save(out_dir, overwrite=True, save_anndata=False)


def plot_umap_labels(adata: ad.AnnData, order: list[str], out_path: Path, title: str) -> None:
    coords = adata.obsm["X_umap"]
    labels = adata.obs["stage_celltype"].astype(str).to_numpy()
    palette = make_palette(order)
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    rng = np.random.default_rng(RANDOM_STATE)
    draw_order = rng.permutation(adata.n_obs)
    ax.scatter(
        coords[draw_order, 0],
        coords[draw_order, 1],
        c=[palette[x] for x in labels[draw_order]],
        s=1.35,
        alpha=0.82,
        linewidths=0,
        rasterized=True,
    )
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    texts = []
    anchors = []
    for label in order:
        mask = labels == label
        if not np.any(mask):
            continue
        anchor = np.median(coords[mask], axis=0)
        anchors.append(anchor)
        texts.append(
            ax.text(
                anchor[0],
                anchor[1],
                label.replace("_", "\n", 1),
                ha="center",
                va="center",
                fontsize=6.75,
                color="black",
                path_effects=[pe.withStroke(linewidth=1.45, foreground="white")],
            )
        )
    repel_label_texts(ax, texts, anchors)
    for text, anchor in zip(texts, anchors):
        text_pos = np.asarray(text.get_position())
        if np.linalg.norm(ax.transData.transform(text_pos) - ax.transData.transform(anchor)) > 12:
            ax.plot([anchor[0], text_pos[0]], [anchor[1], text_pos[1]], color="0.32", lw=0.42, alpha=0.70)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("UMAP1", fontsize=7)
    ax.set_ylabel("UMAP2", fontsize=7)
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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


def plot_umap_legend(adata: ad.AnnData, order: list[str], out_path: Path, title: str) -> None:
    coords = adata.obsm["X_umap"]
    labels = adata.obs["stage_celltype"].astype(str).to_numpy()
    palette = make_palette(order)
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for label in order:
        mask = labels == label
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=1.35,
            c=[palette[label]],
            alpha=0.82,
            linewidths=0,
            rasterized=True,
            label=f"{label} ({mask.sum()})",
        )
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("UMAP1", fontsize=7)
    ax.set_ylabel("UMAP2", fontsize=7)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=5.8, markerscale=1.8)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_original_figures(adata: ad.AnnData, adata_expression: ad.AnnData) -> None:
    order = original_stage_celltype_order(adata)
    plot_umap_labels(
        adata,
        order,
        POSTBIRTH_OUTPUT_DIR / "PostBirth_scanvi_HVG5000_stage_celltype_umap_label.png",
        "PostBirth scANVI HVG5000 stage_celltype",
    )
    plot_umap_legend(
        adata,
        order,
        POSTBIRTH_OUTPUT_DIR / "PostBirth_scanvi_HVG5000_stage_celltype_umap_legend.png",
        "PostBirth scANVI HVG5000 stage_celltype",
    )
    save_dotplot_set(
        adata_result=adata,
        adata_expression=adata_expression,
        stage_celltype_order=order,
        output_dir=POSTBIRTH_OUTPUT_DIR,
        png_prefix="PostBirth_scanvi_HVG5000_stage_celltype_dotplot",
        pdf_prefix=None,
        parts=POSTBIRTH_DOTPLOT_PARTS,
    )


# 6. ===========Revised-celltype h5ad outputs
def save_revised_h5ad(adata_model: ad.AnnData) -> None:
    for col in adata_model.obs.columns:
        if adata_model.obs[col].dtype == "object":
            adata_model.obs[col] = adata_model.obs[col].astype(str)

    adata_model.uns = {
        "hvg": int(N_HVG),
        "hvg_flavor": HVG_FLAVOR,
        "random_state": int(RANDOM_STATE),
        "n_latent": int(N_LATENT),
        "n_neighbors": int(N_NEIGHBORS),
        "umap_min_dist": float(UMAP_MIN_DIST),
        "scvi_max_epochs": int(REVISED_SCVI_MAX_EPOCHS),
        "scanvi_max_epochs": int(REVISED_SCANVI_MAX_EPOCHS),
        "early_stopping": False,
        "batch_key": "integration_batch",
        "labels_key": "scanvi_label",
        "unlabeled_category": "Unknown",
        "latent_key": "X_scanvi",
        "umap_key": "X_umap",
        "stage_order": list(STAGE_ORDER),
        "revised_celltype_order": list(REVISED_CELLTYPE_ORDER),
    }

    adata_model.write_h5ad(DATA_BASE_DIR / "postnatal_scanvi.h5ad", compression="gzip")


# 7. ===========Dotplot outputs
def stage_celltype_order_revised(adata: ad.AnnData) -> list[str]:
    present = set(adata.obs["stage_celltype"].astype(str))
    order = []
    for celltype in REVISED_CELLTYPE_ORDER:
        for stage in STAGE_ORDER:
            label = f"{stage}_{celltype}"
            if label in present:
                order.append(label)
    return order + sorted(present - set(order))


def save_one_dotplot(
    adata_dot: ad.AnnData,
    marker_groups: dict[str, list[str]],
    stage_celltype_order: list[str],
    out_path: Path,
    title: str,
) -> None:
    dot_genes = flatten_groups(marker_groups)
    fig_width = min(max(8.5, len(dot_genes) * 0.18 + 3.5), 15)
    fig_height = max(6.2, len(stage_celltype_order) * 0.20 + 2.4)
    dp = sc.pl.dotplot(
        adata_dot[:, dot_genes].copy(),
        var_names=marker_groups,
        groupby="stage_celltype",
        categories_order=stage_celltype_order,
        standard_scale="var",
        color_map="Reds",
        dendrogram=False,
        dot_min=0,
        dot_max=0.85,
        var_group_rotation=90,
        figsize=(fig_width, fig_height),
        title=None,
        use_raw=False,
        show=False,
        return_fig=True,
    )
    dp.make_figure()
    fig = dp.fig
    fig.subplots_adjust(top=0.70)
    fig.suptitle(title, fontsize=10, y=0.985)
    fig.savefig(str(out_path), dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.18, facecolor="white")
    plt.close(fig)


def save_dotplot_set(
    adata_result: ad.AnnData,
    adata_expression: ad.AnnData,
    stage_celltype_order: list[str],
    output_dir: Path,
    png_prefix: str,
    pdf_prefix: str | None,
    parts: dict[str, list[str]],
) -> None:
    marker_groups = marker_groups_present(adata_expression, POSTNATAL_MARKER_GROUPS)
    dot_genes = flatten_groups(marker_groups)
    adata_dot = adata_expression[adata_result.obs_names, dot_genes].copy()
    adata_dot.obs["stage_celltype"] = pd.Categorical(
        adata_result.obs.loc[adata_dot.obs_names, "stage_celltype"].astype(str),
        categories=stage_celltype_order,
        ordered=True,
    )

    save_one_dotplot(
        adata_dot,
        marker_groups,
        stage_celltype_order,
        output_dir / f"{png_prefix}_full.png",
        f"HVG {N_HVG} stage_celltype marker dotplot",
    )
    if pdf_prefix is not None:
        save_one_dotplot(
            adata_dot,
            marker_groups,
            stage_celltype_order,
            output_dir / f"{pdf_prefix}.pdf",
            f"HVG {N_HVG} stage_celltype marker dotplot",
        )

    for part_name, group_names in parts.items():
        part_groups = {group: marker_groups[group] for group in group_names if group in marker_groups}
        if not part_groups:
            continue
        save_one_dotplot(
            adata_dot,
            part_groups,
            stage_celltype_order,
            output_dir / f"{png_prefix}_{part_name}.png",
            f"HVG {N_HVG} stage_celltype marker dotplot {part_name}",
        )


def save_revised_figures(adata: ad.AnnData, adata_expression: ad.AnnData) -> None:
    order = stage_celltype_order_revised(adata)
    plot_umap_labels(
        adata,
        order,
        POSTNATAL_OUTPUT_DIR / "Postnatal UMAP label.pdf",
        "Postnatal scANVI HVG5000 stage_celltype",
    )
    plot_umap_legend(
        adata,
        order,
        POSTNATAL_OUTPUT_DIR / "Postnatal UMAP legend.pdf",
        "Postnatal scANVI HVG5000 stage_celltype",
    )
    save_dotplot_set(
        adata_result=adata,
        adata_expression=adata_expression,
        stage_celltype_order=order,
        output_dir=POSTNATAL_OUTPUT_DIR,
        png_prefix="HVG5000_stage_celltype_dotplot",
        pdf_prefix="Postnatal marker dotplot",
        parts=POSTNATAL_DOTPLOT_PARTS,
    )

    pdf_part_names = {
        "part1_core_hair_hec": "Postnatal marker dotplot core hair HeC.pdf",
        "part2_support_epithelium": "Postnatal marker dotplot support epithelium.pdf",
        "part3_KO_exclusion": "Postnatal marker dotplot KO exclusion.pdf",
    }
    marker_groups = marker_groups_present(adata_expression, POSTNATAL_MARKER_GROUPS)
    dot_genes = flatten_groups(marker_groups)
    adata_dot = adata_expression[adata.obs_names, dot_genes].copy()
    adata_dot.obs["stage_celltype"] = pd.Categorical(
        adata.obs.loc[adata_dot.obs_names, "stage_celltype"].astype(str),
        categories=order,
        ordered=True,
    )
    for part_name, pdf_name in pdf_part_names.items():
        part_groups = {
            group: marker_groups[group]
            for group in POSTNATAL_DOTPLOT_PARTS[part_name]
            if group in marker_groups
        }
        if part_groups:
            save_one_dotplot(
                adata_dot,
                part_groups,
                order,
                POSTNATAL_OUTPUT_DIR / pdf_name,
                f"HVG {N_HVG} stage_celltype marker dotplot {part_name}",
            )


# 8. ===========Selected postnatal grid UMAP
def group_connectivity(adata: ad.AnnData, label_key: str, neighbors_key: str) -> pd.DataFrame:
    connectivities_key = adata.uns[neighbors_key]["connectivities_key"]
    labels = pd.Categorical(adata.obs[label_key].astype(str))
    codes = labels.codes
    valid = codes >= 0
    onehot = sp.csr_matrix(
        (np.ones(valid.sum(), dtype=np.float32), (np.where(valid)[0], codes[valid])),
        shape=(len(codes), len(labels.categories)),
    )
    raw = onehot.T @ adata.obsp[connectivities_key] @ onehot
    raw = raw.astype(np.float64)
    raw.setdiag(0)
    row_sum = np.asarray(raw.sum(axis=1)).ravel()
    row_sum[row_sum == 0] = 1.0
    matrix = raw.multiply(1.0 / row_sum[:, None]).toarray()
    matrix = (matrix + matrix.T) / 2.0
    return pd.DataFrame(matrix, index=labels.categories, columns=labels.categories)


def centroid_distance(adata: ad.AnnData, label_key: str, a: str, b: str) -> float:
    if a not in set(adata.obs[label_key]) or b not in set(adata.obs[label_key]):
        return np.nan
    coords = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
    coords[label_key] = adata.obs[label_key].to_numpy()
    centers = coords.groupby(label_key)[["UMAP1", "UMAP2"]].median()
    return float(np.linalg.norm(centers.loc[a].to_numpy() - centers.loc[b].to_numpy()))


def get_edge(matrix: pd.DataFrame, a: str, b: str) -> float:
    if a not in matrix.index or b not in matrix.columns:
        return np.nan
    return float(matrix.loc[a, b])


def weighted_mean(values: list[tuple[float, float]]) -> float:
    valid = [(value, weight) for value, weight in values if not np.isnan(value)]
    if not valid:
        return np.nan
    return float(np.average([value for value, _ in valid], weights=[weight for _, weight in valid]))


def save_selected_grid_umap(adata: ad.AnnData, run_grid: bool = True) -> None:
    expected_edges = [
        ("OHC", "DC_PC", 2.0),
        ("IHC", "IBC_IPhC", 2.0),
        ("IBC_IPhC", "KO", 1.0),
        ("KO", "ISC/OSC/CC/IDC", 1.0),
        ("HeC", "DC_PC", 0.7),
        ("HeC", "OHC", 0.5),
    ]
    bad_edges = [
        ("OHC", "KO", 2.0),
        ("OHC", "IBC_IPhC", 1.0),
        ("IHC", "HeC", 1.0),
        ("IHC", "KO", 1.0),
        ("DC_PC", "KO", 0.7),
    ]
    metric_grid = ["euclidean", "cosine"]
    n_neighbors_grid = [15, 30, 50, 75, 100, 150, 200]
    min_dist_grid = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9]
    spread_grid = [0.8, 1.0, 1.5, 2.0]

    neighbors_key = "grid_neighbors"
    if run_grid:
        rows = []
        for metric, n_neighbors in itertools.product(metric_grid, n_neighbors_grid):
            sc.pp.neighbors(
                adata,
                use_rep="X_scanvi",
                metric=metric,
                n_neighbors=n_neighbors,
                random_state=RANDOM_STATE,
                key_added=neighbors_key,
            )
            matrix = group_connectivity(adata, "revised_celltype", neighbors_key)
            expected = weighted_mean([(get_edge(matrix, a, b), weight) for a, b, weight in expected_edges])
            bad = weighted_mean([(get_edge(matrix, a, b), weight) for a, b, weight in bad_edges])
            bio_score = expected - bad

            for min_dist, spread in itertools.product(min_dist_grid, spread_grid):
                sc.tl.umap(
                    adata,
                    neighbors_key=neighbors_key,
                    min_dist=min_dist,
                    spread=spread,
                    random_state=RANDOM_STATE,
                )
                d_ohc_dcpc = centroid_distance(adata, "revised_celltype", "OHC", "DC_PC")
                d_ihc_ibc = centroid_distance(adata, "revised_celltype", "IHC", "IBC_IPhC")
                geometry_score = np.nanmean([1.0 / (1.0 + d_ohc_dcpc), 1.0 / (1.0 + d_ihc_ibc)])
                rows.append({
                    "status": "ok",
                    "metric": metric,
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "spread": spread,
                    "expected_connectivity": expected,
                    "bad_connectivity": bad,
                    "bio_score": bio_score,
                    "umap_OHC_to_DC_PC_distance": d_ohc_dcpc,
                    "umap_IHC_to_IBC_IPhC_distance": d_ihc_ibc,
                    "geometry_score": geometry_score,
                    "total_score": 5 * bio_score + geometry_score,
                })
        grid_metrics = pd.DataFrame(rows)
        grid_metrics.to_csv(GRID_METRICS_FILE, index=False)
    elif not GRID_METRICS_FILE.exists():
        raise FileNotFoundError(GRID_METRICS_FILE)
    else:
        grid_metrics = pd.read_csv(GRID_METRICS_FILE)

    grid_metrics.to_csv(POSTNATAL_OUTPUT_DIR / "grid_metrics.csv", index=False)

    sc.pp.neighbors(
        adata,
        use_rep="X_scanvi",
        metric="euclidean",
        n_neighbors=15,
        random_state=RANDOM_STATE,
        key_added=neighbors_key,
    )
    sc.tl.umap(
        adata,
        neighbors_key=neighbors_key,
        min_dist=0.7,
        spread=0.8,
        random_state=RANDOM_STATE,
    )
    order = stage_celltype_order_revised(adata)
    palette = make_palette(order)
    coords = adata.obsm["X_umap"]
    labels = adata.obs["stage_celltype"].astype(str).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.9), constrained_layout=True)
    for label in order:
        mask = labels == label
        if np.any(mask):
            axes[0].scatter(coords[mask, 0], coords[mask, 1], s=1.35, c=[palette[label]], alpha=0.82, label=f"{label} ({mask.sum()})")
            axes[1].scatter(coords[mask, 0], coords[mask, 1], s=1.35, c=[palette[label]], alpha=0.82)
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=4.5)
    axes[0].set_title("HVG5000 selected UMAP legend", fontsize=8)

    texts = []
    anchors = []
    for label in order:
        mask = labels == label
        if not np.any(mask):
            continue
        anchor = np.median(coords[mask], axis=0)
        anchors.append(anchor)
        texts.append(
            axes[1].text(
                anchor[0],
                anchor[1],
                label.replace("_", "\n", 1),
                fontsize=6.75,
                ha="center",
                va="center",
                path_effects=[pe.withStroke(linewidth=1.45, foreground="white")],
            )
        )
    repel_label_texts(axes[1], texts, anchors)
    for text, anchor in zip(texts, anchors):
        text_pos = np.asarray(text.get_position())
        if np.linalg.norm(axes[1].transData.transform(text_pos) - axes[1].transData.transform(anchor)) > 12:
            axes[1].plot([anchor[0], text_pos[0]], [anchor[1], text_pos[1]], color="0.32", lw=0.42, alpha=0.70)
    axes[1].set_title("HVG5000 selected UMAP labels", fontsize=8)
    for ax in axes:
        ax.set_xlabel("UMAP1", fontsize=7)
        ax.set_ylabel("UMAP2", fontsize=7)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#d0d0d0", linewidth=0.45, alpha=0.8)
    fig.savefig(
        POSTNATAL_OUTPUT_DIR / "HVG5000_euclidean_nn15_mindist0p7_spread0p8.png",
        dpi=SAVE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


# 9. ===========Run workflow
def main() -> None:
    print("Running postnatal HVG5000 revised-celltype scANVI integration.")
    adata_revised = merge_postbirth_inputs(revised=True)
    revised_result, _ = train_postbirth_scanvi(
        adata_revised,
        scvi_epochs=REVISED_SCVI_MAX_EPOCHS,
        scanvi_epochs=REVISED_SCANVI_MAX_EPOCHS,
        early_stopping=False,
    )
    save_revised_h5ad(revised_result)
    save_revised_figures(revised_result, adata_revised)
    save_selected_grid_umap(revised_result.copy())
    print("Postnatal HVG5000 scANVI integration finished.")


if __name__ == "__main__":
    main()
