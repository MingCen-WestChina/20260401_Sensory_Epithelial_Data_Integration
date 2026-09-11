"""All-stage scANVI integration with retained labels and 5,000 HVGs.

The final h5ad keeps the full common-gene matrix plus X_scvi and X_scanvi.
The final label and legend UMAPs are redrawn from saved X_scanvi with R/uwot.
"""

from __future__ import annotations

import contextlib
import io
import logging
import shutil
import subprocess
import warnings
from pathlib import Path
import os

import anndata as ad
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
DATA_DIR = RESULTS_ROOT / "02_scanvi_integration" / "all_stages"
ENTIRE_OUTPUT_DIR = DATA_DIR / "figures"
R_UMAP_SCRIPT = Path(__file__).with_name("all_stages_umap.R")

for path in [ENTIRE_OUTPUT_DIR, DATA_DIR]:
    path.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 0
RUN_VERSION = "all_retained"
N_HVG = 5000
HVG_FLAVOR = "seurat"
N_LATENT = 30
N_LAYERS = 2
GENE_LIKELIHOOD = "nb"
N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.35
SCVI_MAX_EPOCHS = 400
SCANVI_MAX_EPOCHS = 150
EARLY_STOPPING = False
SAVE_DPI = 1201

STAGE_ORDER = ["E9.5", "E11.5", "E13.5", "E14.5", "E16.5", "P1", "P7", "P14", "P28"]


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
    "P1": resolve_stage_file("P1_for_scnvi_normalized.h5ad"),
    "P7": resolve_stage_file("P7_for_scnvi_normalized.h5ad"),
    "P14": resolve_stage_file("P14_for_scnvi_normalized.h5ad"),
    "P28": resolve_stage_file("P28_for_scnvi_normalized.h5ad"),
}
LABEL_KEYS = {
    "E9.5": "final_celltype",
    "E11.5": "final_celltype",
    "E13.5": "final_celltype",
    "E14.5": "final_celltype",
    "E16.5": "final_celltype",
    "P1": "P1_final_celltype",
    "P7": "P7_final_celltype",
    "P14": "P14_final_celltype",
    "P28": "P28_final_celltype",
}
SAMPLE_KEY_OPTIONS = {
    "E9.5": [],
    "E11.5": ["sample", "batch"],
    "E13.5": ["sample"],
    "E14.5": ["sample"],
    "E16.5": [],
    "P1": ["sample"],
    "P7": ["batch"],
    "P14": [],
    "P28": ["sample", "batch"],
}
SINGLE_SAMPLE_STAGES = {"E9.5", "E16.5", "P14"}
EXCLUDE_MODEL_GENES = [
    "Oc90", "Otx2",
    "Ptprc", "Lyz2", "Cd74",
    "Hbb-bs", "Hbb-bt", "Hba-a1", "Hba-a2",
]

DOTPLOT_MARKER_GROUPS = {
    "Epithelium": ["Epcam", "Krt8", "Krt18", "Sox2", "Gata3", "Pax2", "Pax8"],
    "Early_domain": ["Dlx5", "Jag1", "Fgf20", "Bmp4"],
    "Hair_cell": ["Atoh1", "Myo6", "Myo7a", "Cib2", "Pvalb", "Otof", "Tmc1", "Pcp4"],
    "Supporting_cell": ["Fgfr3", "Prox1", "S100a1", "Ntf3", "Socs2", "Gjb2", "Smpx", "Aqp4", "Dcn", "Col3a1"],
    "Exclusion": ["Oc90", "Otx2", "Cd74", "Hbb-bs", "Hbb-bt", "Hba-a1", "Hba-a2"],
}
DOTPLOT_PARTS = {
    "epithelium hair": ["Epithelium", "Early_domain", "Hair_cell"],
    "support": ["Supporting_cell"],
    "exclusion": ["Exclusion"],
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


def choose_sample_key(stage: str, adata: ad.AnnData) -> str | None:
    for key in SAMPLE_KEY_OPTIONS[stage]:
        if key in adata.obs:
            return key
    return None


def make_all_retained_label(label: str) -> str:
    return str(label).strip()


def train_model(model, max_epochs: int) -> None:
    train_kwargs = {
        "max_epochs": max_epochs,
        "early_stopping": EARLY_STOPPING,
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


def match_gene(gene: str, var_names: pd.Index) -> str | None:
    if gene in var_names:
        return gene
    lookup = {str(x).upper(): str(x) for x in var_names}
    return lookup.get(gene.upper())


def marker_groups_present(adata: ad.AnnData, groups: dict[str, list[str]]) -> dict[str, list[str]]:
    present = {}
    used = set()
    for group, genes in groups.items():
        values = []
        for gene in genes:
            matched = match_gene(gene, adata.var_names)
            if matched is None or matched in used:
                continue
            values.append(matched)
            used.add(matched)
        if values:
            present[group] = values
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


# 3. ===========Read and merge all stages
def read_stage_adata(stage: str) -> ad.AnnData:
    path = STAGE_FILES[stage]
    if not path.exists():
        raise FileNotFoundError(path)

    adata = sc.read_h5ad(path).copy()
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    label_key = LABEL_KEYS[stage]
    if label_key not in adata.obs:
        raise KeyError(f"{stage} missing obs['{label_key}'].")

    sample_key = choose_sample_key(stage, adata)
    adata.obs_names = [f"{stage}__{x}" for x in adata.obs_names.astype(str)]
    adata.obs["stage"] = stage
    adata.obs["period"] = "Embryo" if stage.startswith("E") else "Post_birth"

    if stage in SINGLE_SAMPLE_STAGES or sample_key is None:
        adata.obs["sample_raw"] = "single_sample"
        adata.obs["sample_id"] = stage + "__single_sample"
        adata.obs["integration_batch"] = "single_sample"
        adata.obs["real_sample_batch"] = False
    else:
        adata.obs["sample_raw"] = adata.obs[sample_key].astype(str)
        adata.obs["sample_id"] = stage + "__" + adata.obs["sample_raw"].astype(str)
        if adata.obs["sample_raw"].nunique() > 1:
            adata.obs["integration_batch"] = adata.obs["sample_id"].astype(str)
            adata.obs["real_sample_batch"] = True
        else:
            adata.obs["integration_batch"] = "single_sample"
            adata.obs["real_sample_batch"] = False

    adata.obs["original_celltype"] = adata.obs[label_key].astype(str).str.strip()
    adata.obs["revised_celltype"] = adata.obs["original_celltype"].map(make_all_retained_label)
    adata.obs["scanvi_label"] = adata.obs["revised_celltype"]
    keep_obs = [
        "stage", "period", "sample_raw", "sample_id", "integration_batch",
        "real_sample_batch", "original_celltype", "revised_celltype", "scanvi_label",
    ]
    adata.obs = adata.obs[keep_obs].copy()
    adata.var = pd.DataFrame(index=adata.var_names)
    # The archived integration inputs store log-like normalized values in X.
    # Retaining X (and clearing auxiliary layers) reproduces the fitted model
    # used for the manuscript figures; this workflow intentionally does not
    # substitute a raw-count layer.
    adata.raw = None
    adata.layers.clear()
    adata.obsm.clear()
    adata.varm.clear()
    adata.obsp.clear()
    adata.uns = {}
    return to_csr_float32(adata)


def load_all_stages() -> ad.AnnData:
    adatas = [read_stage_adata(stage) for stage in STAGE_ORDER]
    gene_sets = [set(adata.var_names) for adata in adatas]
    common_genes = [gene for gene in adatas[0].var_names if all(gene in genes for genes in gene_sets[1:])]
    if not common_genes:
        raise ValueError("No common genes were found.")

    adata_all = ad.concat(
        [adata[:, common_genes].copy() for adata in adatas],
        join="inner",
        merge="same",
        uns_merge=None,
        index_unique=None,
    )
    adata_all.obs["stage"] = pd.Categorical(adata_all.obs["stage"].astype(str), categories=STAGE_ORDER, ordered=True)
    adata_all.obs["stage_celltype"] = (
        adata_all.obs["stage"].astype(str) + "_" + adata_all.obs["revised_celltype"].astype(str)
    )
    return adata_all


# 4. ===========HVG selection and scANVI training
def selected_hvg_genes(adata_all: ad.AnnData) -> list[str]:
    mt_mask = adata_all.var_names.str.startswith(("mt-", "Mt-", "MT-"))
    exclude_mask = adata_all.var_names.isin(EXCLUDE_MODEL_GENES)
    adata_for_hvg = adata_all[:, ~(mt_mask | exclude_mask)].copy()
    sc.pp.highly_variable_genes(
        adata_for_hvg,
        flavor=HVG_FLAVOR,
        n_top_genes=min(N_HVG, adata_for_hvg.n_vars),
        subset=False,
    )
    return adata_for_hvg.var_names[adata_for_hvg.var["highly_variable"].to_numpy()].tolist()


def run_scanvi(adata_all: ad.AnnData) -> tuple[ad.AnnData, list[str]]:
    selected_genes = selected_hvg_genes(adata_all)
    adata_run = adata_all[:, selected_genes].copy()
    adata_run.obs["scanvi_label"] = pd.Categorical(adata_run.obs["scanvi_label"].astype(str))
    if "Unknown" not in adata_run.obs["scanvi_label"].cat.categories:
        adata_run.obs["scanvi_label"] = adata_run.obs["scanvi_label"].cat.add_categories(["Unknown"])

    unknown_cells = int((adata_run.obs["scanvi_label"].astype(str) == "Unknown").sum())
    if unknown_cells != 0:
        raise ValueError(f"Unknown cells found: {unknown_cells}")

    keep_obs = [
        "stage", "period", "sample_raw", "sample_id", "integration_batch",
        "real_sample_batch", "original_celltype", "revised_celltype",
        "scanvi_label", "stage_celltype",
    ]
    adata_run.obs = adata_run.obs[keep_obs].copy()
    adata_run.var["selected_hvg"] = True
    adata_run.var["hvg_n"] = int(N_HVG)
    to_csr_float32(adata_run)

    scvi.model.SCVI.setup_anndata(
        adata_run,
        batch_key="integration_batch",
        labels_key="scanvi_label",
    )
    scvi_model = scvi.model.SCVI(
        adata_run,
        n_layers=N_LAYERS,
        n_latent=N_LATENT,
        gene_likelihood=GENE_LIKELIHOOD,
    )
    train_model(scvi_model, SCVI_MAX_EPOCHS)
    adata_run.obsm["X_scvi"] = scvi_model.get_latent_representation()

    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        labels_key="scanvi_label",
        unlabeled_category="Unknown",
    )
    train_model(scanvi_model, SCANVI_MAX_EPOCHS)
    adata_run.obsm["X_scanvi"] = scanvi_model.get_latent_representation()

    sc.pp.neighbors(adata_run, use_rep="X_scanvi", n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE)
    sc.tl.umap(adata_run, min_dist=UMAP_MIN_DIST, random_state=RANDOM_STATE)
    return adata_run, selected_genes


# 5. ===========Save final h5ad data
def save_integrated_h5ad(adata_all: ad.AnnData, adata_run: ad.AnnData, selected_genes: list[str]) -> None:
    adata_save = adata_all[adata_run.obs_names, :].copy()
    obs_keep = [
        "stage", "period", "sample_raw", "sample_id", "integration_batch",
        "real_sample_batch", "original_celltype", "revised_celltype",
        "scanvi_label", "stage_celltype",
    ]
    adata_save.obs = adata_run.obs[obs_keep].copy()
    adata_save.var = pd.DataFrame(index=adata_save.var_names)
    adata_save.var[f"hvg_{N_HVG}"] = adata_save.var_names.isin(selected_genes)

    for key in list(adata_save.obsm.keys()):
        del adata_save.obsm[key]
    for key in list(adata_save.obsp.keys()):
        del adata_save.obsp[key]
    for key in list(adata_save.uns.keys()):
        del adata_save.uns[key]

    adata_save.obsm["X_scvi"] = np.asarray(adata_run.obsm["X_scvi"]).copy()
    adata_save.obsm["X_scanvi"] = np.asarray(adata_run.obsm["X_scanvi"]).copy()
    adata_save.uns["entire_period_scanvi"] = {
        "version": RUN_VERSION,
        "hvg": int(N_HVG),
        "batch_key": "integration_batch",
        "label_key": "scanvi_label",
        "stage_order": list(STAGE_ORDER),
        "saved_obsm": ["X_scvi", "X_scanvi"],
        "not_saved": ["X_umap", "neighbors", "distances", "connectivities"],
    }

    adata_save.write_h5ad(DATA_DIR / "all_stages_scanvi.h5ad", compression="gzip")


# 6. ===========UMAP figure helpers
def stage_celltype_sort_key(label: str) -> tuple[int, str]:
    stage = label.split("_", 1)[0]
    stage_rank = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER)
    return stage_rank, label


def stage_celltype_order(adata: ad.AnnData) -> list[str]:
    labels = adata.obs["stage_celltype"].astype(str)
    return sorted(labels.unique().tolist(), key=stage_celltype_sort_key)


def save_stage_revised_umap(adata: ad.AnnData) -> None:
    coords = adata.obsm["X_umap"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, ax, title in [
        ("stage", axes[0], "all_retained HVG5000 stage"),
        ("revised_celltype", axes[1], "all_retained HVG5000 revised_celltype"),
    ]:
        labels = adata.obs[key].astype(str).to_numpy()
        order = pd.Series(labels).value_counts().index.tolist()
        palette = make_palette(order)
        for label in order:
            mask = labels == label
            ax.scatter(coords[mask, 0], coords[mask, 1], s=2.0, alpha=0.82, c=[palette[label]], label=f"{label} ({mask.sum()})")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=5.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(
        ENTIRE_OUTPUT_DIR / "all_stages_stage_revised_celltype.png",
        dpi=SAVE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def find_rscript() -> str:
    configured = os.environ.get("COCHLEA_RSCRIPT")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"COCHLEA_RSCRIPT does not exist: {candidate}")

    rscript = shutil.which("Rscript")
    if rscript:
        return rscript

    raise FileNotFoundError(
        "Rscript was not found. Install R or add Rscript.exe to PATH before running the R/uwot UMAP plotting step."
    )


def save_stage_celltype_r_uwot_umaps() -> None:
    """Draw final full-period stage_celltype UMAPs from saved X_scanvi with R/uwot."""
    input_h5ad = DATA_DIR / "all_stages_scanvi.h5ad"
    if not input_h5ad.exists():
        raise FileNotFoundError(input_h5ad)
    if not R_UMAP_SCRIPT.exists():
        raise FileNotFoundError(R_UMAP_SCRIPT)

    cmd = [
        find_rscript(),
        str(R_UMAP_SCRIPT),
        str(input_h5ad),
        str(ENTIRE_OUTPUT_DIR),
    ]
    print("Running R/uwot full-period UMAP plotting from saved X_scanvi.")
    subprocess.run(cmd, check=True)


# 7. ===========Dotplot figures
def make_dotplot_stage_celltype_order(adata: ad.AnnData) -> list[str]:
    order = []
    for stage in STAGE_ORDER:
        labels_stage = (
            adata.obs.loc[adata.obs["stage"].astype(str) == stage, "stage_celltype"]
            .astype(str)
            .value_counts()
            .index
            .tolist()
        )
        order.extend(labels_stage)
    return list(dict.fromkeys(order))


def save_one_dotplot(
    adata_dot: ad.AnnData,
    marker_groups: dict[str, list[str]],
    order: list[str],
    out_path: Path,
    figsize: tuple[float, float] = (22, 14),
) -> None:
    dp = sc.pl.dotplot(
        adata_dot,
        var_names=marker_groups,
        groupby="stage_celltype",
        categories_order=order,
        standard_scale="var",
        color_map="Reds",
        dot_min=0,
        dot_max=0.85,
        dendrogram=False,
        var_group_rotation=90,
        figsize=figsize,
        show=False,
        return_fig=True,
    )
    dp.savefig(str(out_path), dpi=SAVE_DPI)
    plt.close("all")


def save_dotplots(adata_all: ad.AnnData) -> None:
    adata_dot = adata_all.copy()
    order = make_dotplot_stage_celltype_order(adata_dot)
    adata_dot.obs["stage_celltype"] = pd.Categorical(
        adata_dot.obs["stage_celltype"].astype(str),
        categories=order,
        ordered=True,
    )
    marker_groups = marker_groups_present(adata_dot, DOTPLOT_MARKER_GROUPS)

    save_one_dotplot(
        adata_dot,
        marker_groups,
        order,
        ENTIRE_OUTPUT_DIR / "all_stages_marker_dotplot.png",
    )
    save_one_dotplot(adata_dot, marker_groups, order, ENTIRE_OUTPUT_DIR / "all_stages_marker_dotplot.pdf")

    for suffix, group_names in DOTPLOT_PARTS.items():
        part_groups = {group: marker_groups[group] for group in group_names if group in marker_groups}
        if not part_groups:
            continue
        pdf_name = (
            "Entire period exclusion dotplot.pdf"
            if suffix == "exclusion"
            else f"Entire period marker dotplot {suffix}.pdf"
        )
        save_one_dotplot(
            adata_dot,
            part_groups,
            order,
            ENTIRE_OUTPUT_DIR / pdf_name,
            figsize=(14, 14),
        )


# 8. ===========Run workflow
def main() -> None:
    print("Running entire-period all_retained HVG5000 scANVI integration.")
    adata_all = load_all_stages()
    adata_run, selected_genes = run_scanvi(adata_all)
    save_integrated_h5ad(adata_all, adata_run, selected_genes)
    save_stage_revised_umap(adata_run)
    save_stage_celltype_r_uwot_umaps()
    save_dotplots(adata_all)
    print("Entire-period all_retained HVG5000 scANVI integration finished.")


if __name__ == "__main__":
    main()
