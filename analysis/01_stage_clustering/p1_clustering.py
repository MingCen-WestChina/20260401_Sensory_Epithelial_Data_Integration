#1. ===========Purpose and reproducibility settings
"""Preprocess, cluster, annotate, and export the P1 integration inputs."""

from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("COCHLEA_DATA_DIR", REPO_ROOT / "Data")).expanduser().resolve()
RESULTS_ROOT = Path(os.environ.get("COCHLEA_RESULTS_DIR", REPO_ROOT / "results")).expanduser().resolve()
STAGE_OBJECT_DIR = RESULTS_ROOT / "01_stage_clustering" / "h5ad_for_integration"
OUTPUT_DIR = RESULTS_ROOT / "01_stage_clustering" / "p1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STAGE_OBJECT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 0
SAVE_DPI = 1201
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_STATE))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def display(*objects: object) -> None:
    """Print notebook-style diagnostics when the script runs at the command line."""
    for obj in objects:
        print(obj)

# Notebook cell 1

#1. ===========Import and settings

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=300, facecolor="white")
sc.set_figure_params(figsize=(5, 4))


#2. ===========Read P1 normalized matrix

base_dir = REPO_ROOT
data_path = DATA_ROOT / "raw" / "p1" / "P1_Normalized_Counts.txt"

df = pd.read_csv(data_path, sep="\t", index_col=0)

adata = ad.AnnData(X=df.T.copy())
adata.obs_names = df.columns.astype(str)
adata.var_names = df.index.astype(str)

sc.pp.filter_genes(adata, min_cells=3)

adata.obs["sample"] = (
    adata.obs_names.to_series()
    .str.extract(r"^(se_\d+_P1)", expand=False)
    .fillna("unknown")
    .astype("category")
    .values
)


#3. ===========QC metrics

def row_sum(x):
    return np.asarray(x.sum(axis=1)).ravel()

def row_detected(x):
    return np.asarray((x > 0).sum(axis=1)).ravel()

def gene_sum(adata_obj, genes):
    genes = [g for g in genes if g in adata_obj.var_names]
    if len(genes) == 0:
        return np.zeros(adata_obj.n_obs)
    return row_sum(adata_obj[:, genes].X)

adata.var["mt"] = adata.var_names.str.startswith(("mt-", "Mt-", "MT-"))
adata.var["ribo"] = adata.var_names.str.startswith(("Rps", "Rpl"))
adata.var["hb"] = adata.var_names.str.startswith(("Hba", "Hbb"))

adata.obs["total_expr_norm"] = row_sum(adata.X)
adata.obs["n_genes_detected"] = row_detected(adata.X)

total_expr = np.maximum(adata.obs["total_expr_norm"].to_numpy(), 1e-12)
adata.obs["norm_pct_mt"] = gene_sum(adata, adata.var_names[adata.var["mt"]]) / total_expr * 100
adata.obs["norm_pct_ribo"] = gene_sum(adata, adata.var_names[adata.var["ribo"]]) / total_expr * 100
adata.obs["norm_pct_hb"] = gene_sum(adata, adata.var_names[adata.var["hb"]]) / total_expr * 100

qc_summary = adata.obs.groupby("sample", observed=True)[
    ["total_expr_norm", "n_genes_detected", "norm_pct_mt", "norm_pct_ribo", "norm_pct_hb"]
].agg(["count", "median", "min", "max"])

display(qc_summary)


#4. ===========QC plots

fig, axes = plt.subplots(2, 3, figsize=(11, 6))

axes[0, 0].hist(adata.obs["total_expr_norm"], bins=60)
axes[0, 0].set_title("Total normalized expression")

axes[0, 1].hist(adata.obs["n_genes_detected"], bins=60)
axes[0, 1].set_title("Detected genes per cell")

axes[0, 2].hist(adata.obs["norm_pct_mt"], bins=60)
axes[0, 2].set_title("Mito fraction")

axes[1, 0].hist(adata.obs["norm_pct_ribo"], bins=60)
axes[1, 0].set_title("Ribo fraction")

axes[1, 1].hist(adata.obs["norm_pct_hb"], bins=60)
axes[1, 1].set_title("Hb fraction")

adata.obs["sample"].value_counts().sort_index().plot(kind="bar", ax=axes[1, 2])
axes[1, 2].set_title("Cells per sample")
axes[1, 2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.close("all")


#5. ===========Remove Oc90 / Otx2 positive roof cells

def gene_positive_mask(adata_obj, gene, cutoff=0):
    if gene not in adata_obj.var_names:
        return np.zeros(adata_obj.n_obs, dtype=bool)
    x = adata_obj[:, gene].X
    x = x.toarray().ravel() if sparse.issparse(x) else np.asarray(x).ravel()
    return x > cutoff

oc90_pos = gene_positive_mask(adata, "Oc90", cutoff=0)
otx2_pos = gene_positive_mask(adata, "Otx2", cutoff=0)
exclude_roof = oc90_pos | otx2_pos

roof_summary = pd.DataFrame({
    "group": ["all_cells", "Oc90_positive", "Otx2_positive", "Oc90_or_Otx2_positive", "kept_cells"],
    "n_cells": [
        adata.n_obs,
        int(oc90_pos.sum()),
        int(otx2_pos.sum()),
        int(exclude_roof.sum()),
        int((~exclude_roof).sum()),
    ],
})

roof_by_sample = (
    pd.DataFrame({
        "sample": adata.obs["sample"].astype(str).values,
        "Oc90_positive": oc90_pos,
        "Otx2_positive": otx2_pos,
        "exclude_roof": exclude_roof,
    })
    .groupby("sample", observed=True)
    .agg(
        n_cells=("exclude_roof", "size"),
        Oc90_positive=("Oc90_positive", "sum"),
        Otx2_positive=("Otx2_positive", "sum"),
        removed=("exclude_roof", "sum"),
    )
)

roof_by_sample["removed_pct"] = roof_by_sample["removed"] / roof_by_sample["n_cells"] * 100

display(roof_summary)
display(roof_by_sample.round(2))

adata_clean = adata[~exclude_roof].copy()

print(f"P1 QC finished: {adata.n_obs} cells before filtering, {adata_clean.n_obs} cells kept.")


plt.close("all")


# Notebook cell 2

#1. ===========Harmony batch correction

import harmonypy as hm
import contextlib
import io

sc.pp.highly_variable_genes(
    adata_clean,
    flavor="seurat",
    n_top_genes=3000,
)

technical_hvg = (
    adata_clean.var_names.str.startswith(("mt-", "Mt-", "MT-"))
    | adata_clean.var_names.str.startswith(("Rps", "Rpl"))
    | adata_clean.var_names.str.startswith(("Hba", "Hbb"))
)

adata_clean.var.loc[technical_hvg, "highly_variable"] = False

adata_clean_hvg = adata_clean[:, adata_clean.var["highly_variable"]].copy()

sc.pp.scale(adata_clean_hvg, max_value=10)
sc.tl.pca(adata_clean_hvg, svd_solver="arpack", n_comps=50)

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    ho = hm.run_harmony(
        adata_clean_hvg.obsm["X_pca"],
        adata_clean_hvg.obs,
        vars_use=["sample"],
        max_iter_harmony=20,
    )

Z_corr = ho.Z_corr
adata_clean_hvg.obsm["X_pca_harmony"] = (
    Z_corr if Z_corr.shape[0] == adata_clean_hvg.n_obs else Z_corr.T
)


#2. ===========UMAP and Leiden on Harmony PCs

sc.pp.neighbors(
    adata_clean_hvg,
    n_neighbors=15,
    n_pcs=30,
    use_rep="X_pca_harmony",
)

sc.tl.umap(adata_clean_hvg, random_state=0)

sc.tl.leiden(
    adata_clean_hvg,
    resolution=0.6,
    key_added="harmony_leiden",
    random_state=0,
)

adata_clean.obs["harmony_leiden"] = adata_clean_hvg.obs["harmony_leiden"].astype(str).values
adata_clean.obsm["X_umap_harmony"] = adata_clean_hvg.obsm["X_umap"].copy()

leiden_order = sorted(adata_clean.obs["harmony_leiden"].unique(), key=lambda x: int(x))
adata_clean.obs["harmony_leiden"] = pd.Categorical(
    adata_clean.obs["harmony_leiden"],
    categories=leiden_order,
    ordered=True,
)

leiden_counts = adata_clean.obs["harmony_leiden"].value_counts().sort_index()
adata_clean.obs["harmony_leiden_label"] = adata_clean.obs["harmony_leiden"].map(
    {x: f"{x} (n={leiden_counts[x]})" for x in leiden_counts.index}
).astype("category")


#3. ===========Batch check

batch_table = pd.crosstab(
    adata_clean.obs["harmony_leiden"],
    adata_clean.obs["sample"],
    margins=True,
)

display(batch_table)


#4. ===========Harmony UMAP check

sc.pl.embedding(
    adata_clean,
    basis="umap_harmony",
    color=["harmony_leiden_label", "sample"],
    ncols=2,
    size=7,
    frameon=False,
    wspace=0.35,
    show=False,
)


#5. ===========Broad marker dotplot

p1_broad_markers = {
    "Core": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "HC": ["Myo6", "Myo7a", "Cib2", "Pvalb", "Pou4f3", "Atoh1", "Ccer2", "Pcp4"],
    "IPhC": ["Anxa5", "Matn4", "Gjb2", "Fabp7", "Prss23"],
    "IPC_OPC_DC": ["Npy", "Igfbpl1", "S100b", "Fam159b", "Smagp", "Serpine2", "Pdzk1ip1", "Hes5", "Fgf3", "S100a1"],
    "HeC_CC_OSC": ["Pmch", "Fst", "Npnt", "Bmp4", "Apoe", "Fbln2"],
    "ISC_KO": ["Igf1", "Matn1", "Meg3", "Dcn", "Rcn3", "Pdia6", "Cpxm2", "Ctgf", "Kazald1", "Cst3", "Epyc"],
    "IdC": ["Cdkn1c", "Fxyd6", "Otoa", "Ptgds", "Smoc2"],
    "Exclude": ["Oc90", "Otx2", "Pecam1", "Kdr", "Ptprc", "Lyz2", "Sox10", "Mbp", "Mpz", "Snap25", "Tubb3"],
}

p1_broad_markers = {
    group: [gene for gene in genes if gene in adata_clean.var_names]
    for group, genes in p1_broad_markers.items()
}

p1_broad_markers = {
    group: genes for group, genes in p1_broad_markers.items()
    if len(genes) > 0
}

sc.pl.dotplot(
    adata_clean,
    var_names=p1_broad_markers,
    groupby="harmony_leiden",
    standard_scale="var",
    figsize=(18, 5),
    dendrogram=False,
    show=False,
)


plt.close("all")


# Notebook cell 3

#1. ===========Check HC parent clusters

np.random.seed(0)

required_cols = ["harmony_leiden", "sample"]
missing_cols = [col for col in required_cols if col not in adata_clean.obs]
assert len(missing_cols) == 0, f"Missing columns in adata_clean.obs: {missing_cols}"

hc_parent_clusters = ["6", "11"]
available_clusters = set(adata_clean.obs["harmony_leiden"].astype(str).unique())
missing_clusters = sorted(set(hc_parent_clusters) - available_clusters)
assert len(missing_clusters) == 0, f"Missing HC parent clusters: {missing_clusters}"

def mean_marker_expr(adata_obj, genes):
    genes = [gene for gene in genes if gene in adata_obj.var_names]
    if len(genes) == 0:
        return np.zeros(adata_obj.n_obs)
    return np.asarray(adata_obj[:, genes].X.mean(axis=1)).ravel()

p1_parent_check_genes = {
    "HC": ["Myo6", "Myo7a", "Cib2", "Pvalb", "Pou4f3", "Atoh1", "Ccer2", "Pcp4"],
    "IHC": ["Otof", "Atp2a3", "Cabp2", "Fgf8", "Dlk2", "Nefl", "Calb2"],
    "OHC": ["Ocm", "Aqp11", "Pcp4", "Calb1", "Six2"],
    "IPhC": ["Anxa5", "Matn4", "Gjb2", "Fabp7", "Prss23"],
    "IPC_OPC_DC": ["Npy", "Igfbpl1", "S100b", "Fam159b", "Smagp", "Serpine2", "Pdzk1ip1", "Hes5", "Fgf3", "S100a1"],
    "HeC_CC_OSC": ["Pmch", "Fst", "Npnt", "Bmp4", "Apoe", "Fbln2"],
    "ISC_KO": ["Igf1", "Matn1", "Meg3", "Dcn", "Rcn3", "Pdia6", "Cpxm2", "Ctgf", "Kazald1", "Cst3", "Epyc"],
    "IdC": ["Cdkn1c", "Fxyd6", "Otoa", "Ptgds", "Smoc2"],
    "Exclude": ["Oc90", "Otx2", "Pecam1", "Kdr", "Ptprc", "Lyz2", "Sox10", "Mbp", "Mpz", "Snap25", "Tubb3"],
}
p1_parent_check_genes = {
    group: [gene for gene in genes if gene in adata_clean.var_names]
    for group, genes in p1_parent_check_genes.items()
}
assert all(len(p1_parent_check_genes[group]) > 0 for group in ["HC", "IHC", "OHC"])

for group, genes in p1_parent_check_genes.items():
    adata_clean.obs[f"parent_{group}_score"] = mean_marker_expr(adata_clean, genes)

parent_score_cols = [f"parent_{group}_score" for group in p1_parent_check_genes]
parent_main_cols = [col for col in parent_score_cols if col != "parent_Exclude_score"]

parent_score_summary = (
    adata_clean.obs
    .assign(harmony_leiden_str=adata_clean.obs["harmony_leiden"].astype(str))
    .groupby("harmony_leiden_str", observed=True)[parent_score_cols]
    .mean()
    .round(3)
)
parent_score_summary["n_cells"] = (
    adata_clean.obs["harmony_leiden"].astype(str)
    .value_counts()
    .reindex(parent_score_summary.index)
)
parent_score_summary["top_call"] = (
    parent_score_summary[parent_main_cols]
    .idxmax(axis=1)
    .str.replace("parent_", "", regex=False)
    .str.replace("_score", "", regex=False)
)
parent_score_summary = parent_score_summary.sort_index(key=lambda x: x.astype(int))

display(parent_score_summary[parent_score_cols + ["n_cells", "top_call"]])
display(parent_score_summary.loc[hc_parent_clusters, parent_score_cols + ["n_cells", "top_call"]])


#2. ===========Split HC candidate and support cells

adata_hc = adata_clean[
    adata_clean.obs["harmony_leiden"].astype(str).isin(hc_parent_clusters)
].copy()

adata_support = adata_clean[
    ~adata_clean.obs["harmony_leiden"].astype(str).isin(hc_parent_clusters)
].copy()

adata_hc.obs["parent_harmony_leiden"] = adata_hc.obs["harmony_leiden"].astype(str)
adata_support.obs["parent_harmony_leiden"] = adata_support.obs["harmony_leiden"].astype(str)

parent_split_table = pd.crosstab(
    adata_hc.obs["parent_harmony_leiden"],
    adata_hc.obs["sample"],
    margins=True,
)
assert int(parent_split_table.loc["All", "All"]) == adata_hc.n_obs

display(parent_split_table)


#3. ===========HC Harmony reclustering

sc.pp.highly_variable_genes(
    adata_hc,
    flavor="seurat",
    n_top_genes=1500,
)

technical_hvg = (
    adata_hc.var_names.str.startswith(("mt-", "Mt-", "MT-"))
    | adata_hc.var_names.str.startswith(("Rps", "Rpl"))
    | adata_hc.var_names.str.startswith(("Hba", "Hbb"))
)
adata_hc.var.loc[technical_hvg, "highly_variable"] = False

adata_hc_hvg = adata_hc[:, adata_hc.var["highly_variable"]].copy()
n_hc_pcs = min(30, adata_hc_hvg.n_obs - 1, adata_hc_hvg.n_vars - 1)
assert n_hc_pcs >= 2, "Too few cells or HVGs for HC PCA."

sc.pp.scale(adata_hc_hvg, max_value=10)
sc.tl.pca(adata_hc_hvg, svd_solver="arpack", n_comps=n_hc_pcs)

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    hc_harmony = hm.run_harmony(
        adata_hc_hvg.obsm["X_pca"],
        adata_hc_hvg.obs,
        vars_use=["sample"],
        max_iter_harmony=20,
    )

Z_corr = hc_harmony.Z_corr
adata_hc_hvg.obsm["X_pca_harmony"] = (
    Z_corr if Z_corr.shape[0] == adata_hc_hvg.n_obs else Z_corr.T
)

sc.pp.neighbors(
    adata_hc_hvg,
    n_neighbors=10,
    n_pcs=min(20, n_hc_pcs),
    use_rep="X_pca_harmony",
)
sc.tl.umap(adata_hc_hvg, random_state=0)

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    sc.tl.leiden(
        adata_hc_hvg,
        resolution=0.25,
        key_added="hc_leiden",
        random_state=0,
    )

adata_hc.obs["hc_leiden"] = adata_hc_hvg.obs["hc_leiden"].astype(str).values
adata_hc.obsm["X_umap_hc"] = adata_hc_hvg.obsm["X_umap"].copy()


#4. ===========HC marker score check

hc_score_genes = {
    "HC": ["Myo6", "Myo7a", "Cib2", "Pvalb", "Pou4f3", "Atoh1", "Ccer2", "Pcp4"],
    "IHC": ["Otof", "Atp2a3", "Cabp2", "Fgf8", "Dlk2", "Nefl", "Calb2"],
    "OHC": ["Ocm", "Aqp11", "Pcp4", "Calb1", "Six2"],
    "Exclude": ["Oc90", "Otx2", "Sox10", "Mbp", "Mpz", "Ptprc", "Lyz2"],
}
for group, genes in hc_score_genes.items():
    adata_hc.obs[f"{group}_score"] = mean_marker_expr(adata_hc, genes)

hc_score_cols = ["HC_score", "IHC_score", "OHC_score", "Exclude_score"]
hc_score_summary = (
    adata_hc.obs
    .groupby("hc_leiden", observed=True)[hc_score_cols]
    .mean()
    .round(3)
)
hc_score_summary["n_cells"] = adata_hc.obs["hc_leiden"].value_counts().reindex(hc_score_summary.index)
hc_score_summary["top_call"] = (
    hc_score_summary[["HC_score", "IHC_score", "OHC_score"]]
    .idxmax(axis=1)
    .str.replace("_score", "", regex=False)
)

hc_call_rank = {"IHC": 0, "OHC": 1, "HC": 2}
hc_score_for_order = hc_score_summary.copy()
hc_score_for_order["call_rank"] = hc_score_for_order["top_call"].map(hc_call_rank).fillna(9)
hc_score_for_order["cluster_rank"] = hc_score_for_order.index.astype(str).astype(int)
hc_plot_order = hc_score_for_order.sort_values(["call_rank", "cluster_rank"]).index.astype(str).tolist()

adata_hc.obs["hc_leiden"] = pd.Categorical(
    adata_hc.obs["hc_leiden"].astype(str),
    categories=hc_plot_order,
    ordered=True,
)
hc_counts = adata_hc.obs["hc_leiden"].value_counts().reindex(hc_plot_order)
hc_label_map = {cluster: f"{cluster} (n={int(hc_counts.loc[cluster])})" for cluster in hc_plot_order}
adata_hc.obs["hc_leiden_label"] = pd.Categorical(
    adata_hc.obs["hc_leiden"].astype(str).map(hc_label_map),
    categories=[hc_label_map[cluster] for cluster in hc_plot_order],
    ordered=True,
)

display(hc_score_summary.loc[hc_plot_order, hc_score_cols + ["n_cells", "top_call"]])
display(pd.crosstab(adata_hc.obs["hc_leiden"], adata_hc.obs["parent_harmony_leiden"], margins=True))


#5. ===========HC UMAP and dotplot

sc.pl.embedding(
    adata_hc,
    basis="umap_hc",
    color=["hc_leiden_label", "sample", "parent_harmony_leiden"],
    ncols=3,
    size=15,
    frameon=False,
    wspace=0.35,
    show=False,
)

hc_marker_dict = {
    "Core": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "HC": ["Myo6", "Myo7a", "Cib2", "Pvalb", "Pou4f3", "Atoh1", "Ccer2", "Pcp4"],
    "IHC": ["Otof", "Atp2a3", "Cabp2", "Fgf8", "Dlk2", "Nefl", "Calb2"],
    "OHC": ["Ocm", "Aqp11", "Pcp4", "Calb1", "Six2"],
    "Exclude": ["Oc90", "Otx2", "Sox10", "Mbp", "Mpz", "Ptprc", "Lyz2"],
}
hc_marker_dict = {
    group: [gene for gene in genes if gene in adata_hc.var_names]
    for group, genes in hc_marker_dict.items()
}
hc_marker_dict = {group: genes for group, genes in hc_marker_dict.items() if len(genes) > 0}

sc.pl.dotplot(
    adata_hc,
    var_names=hc_marker_dict,
    groupby="hc_leiden",
    standard_scale=None,
    cmap="Reds",
    dendrogram=False,
    swap_axes=False,
    figsize=(12, 4),
    show=False,
)


plt.close("all")


# Notebook cell 4

#1. ===========Write corrected HC labels to step1

hc_label_map = {
    "0": "OHC",
    "1": "OHC",
    "2": "IHC",
    "3": "OHC",
    "4": "OHC",
    "5": "OHC",
    "6": "OHC",
}

observed_hc_leiden = set(adata_hc.obs["hc_leiden"].astype(str).unique())
missing_hc_map = sorted(observed_hc_leiden - set(hc_label_map))
assert len(missing_hc_map) == 0, f"Need HC label for hc_leiden: {missing_hc_map}"

adata_hc.obs["hc_celltype"] = adata_hc.obs["hc_leiden"].astype(str).map(hc_label_map)
adata_hc.obs["hc_celltype"] = pd.Categorical(
    adata_hc.obs["hc_celltype"],
    categories=["IHC", "OHC"],
    ordered=True,
)

adata_clean.obs["P1_step1_celltype"] = "Unassigned"
adata_clean.obs.loc[adata_hc.obs_names, "P1_step1_celltype"] = adata_hc.obs["hc_celltype"].astype(str).values
adata_clean.obs["P1_step1_celltype"] = pd.Categorical(
    adata_clean.obs["P1_step1_celltype"],
    categories=["IHC", "OHC", "Unassigned"],
    ordered=True,
)

display(pd.crosstab(adata_hc.obs["hc_celltype"], adata_hc.obs["parent_harmony_leiden"], margins=True))
display(pd.crosstab(adata_clean.obs["P1_step1_celltype"], adata_clean.obs["sample"], margins=True))


#2. ===========Support-cell Harmony reclustering

logging.getLogger("harmonypy").setLevel(logging.ERROR)
logging.getLogger("harmonypy.harmony").setLevel(logging.ERROR)

assert "parent_harmony_leiden" in adata_support.obs.columns
assert not adata_support.obs["harmony_leiden"].astype(str).isin(hc_parent_clusters).any()

sc.pp.highly_variable_genes(
    adata_support,
    flavor="seurat",
    n_top_genes=3000,
)

exclude_from_hvg = (
    adata_support.var_names.str.startswith(("mt-", "Mt-", "MT-"))
    | adata_support.var_names.str.startswith(("Rps", "Rpl"))
    | adata_support.var_names.str.startswith(("Hba", "Hbb"))
    | adata_support.var_names.isin([
        "Oc90", "Otx2",
        "Myo6", "Myo7a", "Cib2", "Pvalb", "Pou4f3", "Atoh1", "Ccer2", "Pcp4",
        "Pecam1", "Kdr", "Cdh5", "Ptprc", "Lyz2", "C1qa", "C1qb",
        "Sox10", "Mbp", "Mpz", "Plp1",
        "Snap25", "Tubb3", "Elavl3",
    ])
)
adata_support.var.loc[exclude_from_hvg, "highly_variable"] = False

adata_support_hvg = adata_support[:, adata_support.var["highly_variable"]].copy()
n_support_pcs = min(50, adata_support_hvg.n_obs - 1, adata_support_hvg.n_vars - 1)
assert n_support_pcs >= 2, "Too few cells or HVGs for support PCA."

sc.pp.scale(adata_support_hvg, max_value=10)
sc.tl.pca(adata_support_hvg, svd_solver="arpack", n_comps=n_support_pcs)

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    support_harmony = hm.run_harmony(
        adata_support_hvg.obsm["X_pca"],
        adata_support_hvg.obs,
        vars_use=["sample"],
        max_iter_harmony=20,
    )

Z_corr = support_harmony.Z_corr
adata_support_hvg.obsm["X_pca_harmony"] = (
    Z_corr if Z_corr.shape[0] == adata_support_hvg.n_obs else Z_corr.T
)

sc.pp.neighbors(
    adata_support_hvg,
    n_neighbors=15,
    n_pcs=min(30, n_support_pcs),
    use_rep="X_pca_harmony",
)
sc.tl.umap(adata_support_hvg, random_state=0)

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    sc.tl.leiden(
        adata_support_hvg,
        resolution=0.7,
        key_added="support_leiden",
        random_state=0,
    )

adata_support.obs["support_leiden"] = adata_support_hvg.obs["support_leiden"].astype(str).values
adata_support.obsm["X_umap_support"] = adata_support_hvg.obsm["X_umap"].copy()


#3. ===========Support-cell broad marker scores

support_score_genes = {
    "IPhC": ["Anxa5", "Matn4", "Gjb2", "Fabp7", "Prss23"],
    "IPC_OPC_DC": ["Npy", "Igfbpl1", "S100b", "Fam159b", "Smagp", "Serpine2", "Pdzk1ip1", "Hes5", "Fgf3", "S100a1"],
    "HeC_CC_OSC": ["Pmch", "Fst", "Npnt", "Bmp4", "Apoe", "Fbln2"],
    "ISC_KO": ["Igf1", "Matn1", "Meg3", "Dcn", "Rcn3", "Pdia6", "Cpxm2", "Ctgf", "Kazald1", "Cst3", "Epyc"],
    "IdC": ["Cdkn1c", "Fxyd6", "Otoa", "Ptgds", "Smoc2"],
    "Exclude": ["Oc90", "Otx2", "Myo6", "Myo7a", "Pecam1", "Kdr", "Ptprc", "Lyz2", "Sox10", "Mbp", "Mpz", "Snap25", "Tubb3"],
}

for group, genes in support_score_genes.items():
    adata_support.obs[f"{group}_score"] = mean_marker_expr(adata_support, genes)

support_score_cols = [f"{group}_score" for group in support_score_genes]
support_main_score_cols = [col for col in support_score_cols if col != "Exclude_score"]

support_score_summary = (
    adata_support.obs
    .groupby("support_leiden", observed=True)[support_score_cols]
    .mean()
    .round(3)
)
support_score_summary["n_cells"] = adata_support.obs["support_leiden"].value_counts().reindex(support_score_summary.index)
support_score_summary["top_call"] = (
    support_score_summary[support_main_score_cols]
    .idxmax(axis=1)
    .str.replace("_score", "", regex=False)
)

display(support_score_summary.sort_index())
display(pd.crosstab(adata_support.obs["support_leiden"], adata_support.obs["sample"], margins=True))
display(pd.crosstab(adata_support.obs["support_leiden"], adata_support.obs["parent_harmony_leiden"], margins=True))


#4. ===========Support-cell UMAP and dotplot

support_counts = adata_support.obs["support_leiden"].value_counts().sort_index()
support_label_map = {cluster: f"{cluster} (n={int(support_counts.loc[cluster])})" for cluster in support_counts.index}
adata_support.obs["support_leiden_label"] = adata_support.obs["support_leiden"].astype(str).map(support_label_map)

sc.pl.embedding(
    adata_support,
    basis="umap_support",
    color=["support_leiden_label", "sample", "parent_harmony_leiden"],
    ncols=3,
    size=7,
    frameon=False,
    wspace=0.35,
    show=False,
)

support_marker_dict = {
    "Core": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "IPhC": ["Anxa5", "Matn4", "Gjb2", "Fabp7", "Prss23"],
    "IPC_OPC_DC": ["Npy", "Igfbpl1", "S100b", "Fam159b", "Smagp", "Serpine2", "Pdzk1ip1", "Hes5", "Fgf3", "S100a1"],
    "HeC_CC_OSC": ["Pmch", "Fst", "Npnt", "Bmp4", "Apoe", "Fbln2"],
    "ISC_KO": ["Igf1", "Matn1", "Meg3", "Dcn", "Rcn3", "Pdia6", "Cpxm2", "Ctgf", "Kazald1", "Cst3", "Epyc"],
    "IdC": ["Cdkn1c", "Fxyd6", "Otoa", "Ptgds", "Smoc2"],
    "Exclude": ["Oc90", "Otx2", "Myo6", "Myo7a", "Pecam1", "Kdr", "Ptprc", "Lyz2", "Sox10", "Mbp", "Mpz", "Snap25", "Tubb3"],
}
support_marker_dict = {
    group: [gene for gene in genes if gene in adata_support.var_names]
    for group, genes in support_marker_dict.items()
}
support_marker_dict = {group: genes for group, genes in support_marker_dict.items() if len(genes) > 0}

sc.pl.dotplot(
    adata_support,
    var_names=support_marker_dict,
    groupby="support_leiden",
    standard_scale=None,
    cmap="Reds",
    dendrogram=False,
    swap_axes=False,
    figsize=(18, 5),
    show=False,
)


plt.close("all")


# Notebook cell 5

#1. ===========Assign support broad labels

support_broad_map = {
    "0": "ISC_KO",
    "1": "ISC_KO",
    "2": "ISC_KO",
    "3": "ISC_KO",
    "4": "IdC",
    "5": "IPhC",
    "6": "IPC_OPC_DC",
    "7": "IPhC",
    "8": "HeC_CC_OSC",
    "9": "IPC_OPC_DC",
}

observed_support_leiden = set(adata_support.obs["support_leiden"].astype(str).unique())
missing_support_map = sorted(observed_support_leiden - set(support_broad_map))
assert len(missing_support_map) == 0, f"Need support broad label for support_leiden: {missing_support_map}"

support_broad_order = ["IHC", "OHC", "IPhC", "IPC_OPC_DC", "HeC_CC_OSC", "ISC_KO", "IdC"]

adata_support.obs["support_broad_celltype"] = (
    adata_support.obs["support_leiden"].astype(str).map(support_broad_map)
)
adata_support.obs["support_broad_celltype"] = pd.Categorical(
    adata_support.obs["support_broad_celltype"],
    categories=["IPhC", "IPC_OPC_DC", "HeC_CC_OSC", "ISC_KO", "IdC"],
    ordered=True,
)

adata_clean.obs["P1_step1_celltype"] = "Unassigned"
adata_clean.obs.loc[adata_hc.obs_names, "P1_step1_celltype"] = adata_hc.obs["hc_celltype"].astype(str).values
adata_clean.obs.loc[adata_support.obs_names, "P1_step1_celltype"] = adata_support.obs["support_broad_celltype"].astype(str).values
assert not (adata_clean.obs["P1_step1_celltype"] == "Unassigned").any()

adata_clean.obs["P1_step1_celltype"] = pd.Categorical(
    adata_clean.obs["P1_step1_celltype"],
    categories=support_broad_order,
    ordered=True,
)

display(pd.crosstab(adata_clean.obs["P1_step1_celltype"], adata_clean.obs["sample"], margins=True))


#2. ===========Extract ISC / KO branch

isc_ko_clusters = ["0", "1", "2", "3"]

adata_isc_ko = adata_support[
    adata_support.obs["support_leiden"].astype(str).isin(isc_ko_clusters)
].copy()

adata_isc_ko.obs["parent_support_leiden"] = pd.Categorical(
    adata_isc_ko.obs["support_leiden"].astype(str),
    categories=isc_ko_clusters,
    ordered=True,
)

assert adata_isc_ko.n_obs > 0, "No cells found for ISC / KO branch."

display(pd.crosstab(adata_isc_ko.obs["parent_support_leiden"], adata_isc_ko.obs["sample"], margins=True))


#3. ===========ISC / KO branch Harmony reclustering

sc.pp.highly_variable_genes(
    adata_isc_ko,
    flavor="seurat",
    n_top_genes=3000,
    batch_key="sample",
)

exclude_from_hvg = (
    adata_isc_ko.var_names.str.startswith(("mt-", "Mt-", "MT-"))
    | adata_isc_ko.var_names.str.startswith(("Rps", "Rpl"))
    | adata_isc_ko.var_names.str.startswith(("Hba", "Hbb"))
    | adata_isc_ko.var_names.isin([
        "Oc90", "Otx2",
        "Myo6", "Myo7a", "Cib2", "Pvalb", "Pou4f3", "Atoh1", "Ccer2", "Pcp4",
        "Pecam1", "Kdr", "Cdh5", "Ptprc", "Lyz2", "C1qa", "C1qb",
        "Sox10", "Mbp", "Mpz", "Plp1",
        "Snap25", "Tubb3", "Elavl3",
    ])
)
adata_isc_ko.var.loc[exclude_from_hvg, "highly_variable"] = False

adata_isc_ko_hvg = adata_isc_ko[:, adata_isc_ko.var["highly_variable"]].copy()
n_isc_ko_pcs = min(40, adata_isc_ko_hvg.n_obs - 1, adata_isc_ko_hvg.n_vars - 1)
assert n_isc_ko_pcs >= 2, "Too few cells or HVGs for ISC / KO PCA."

sc.pp.scale(adata_isc_ko_hvg, max_value=10)
sc.tl.pca(adata_isc_ko_hvg, svd_solver="arpack", n_comps=n_isc_ko_pcs)

logging.getLogger("harmonypy").setLevel(logging.ERROR)
logging.getLogger("harmonypy.harmony").setLevel(logging.ERROR)

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    isc_ko_harmony = hm.run_harmony(
        adata_isc_ko_hvg.obsm["X_pca"],
        adata_isc_ko_hvg.obs,
        vars_use=["sample"],
        max_iter_harmony=20,
    )

Z_corr = isc_ko_harmony.Z_corr
adata_isc_ko_hvg.obsm["X_pca_harmony"] = (
    Z_corr if Z_corr.shape[0] == adata_isc_ko_hvg.n_obs else Z_corr.T
)

sc.pp.neighbors(
    adata_isc_ko_hvg,
    n_neighbors=15,
    n_pcs=min(30, n_isc_ko_pcs),
    use_rep="X_pca_harmony",
)
sc.tl.umap(adata_isc_ko_hvg, random_state=0)

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    sc.tl.leiden(
        adata_isc_ko_hvg,
        resolution=0.8,
        key_added="isc_ko_leiden",
        random_state=0,
    )

adata_isc_ko.obs["isc_ko_leiden"] = adata_isc_ko_hvg.obs["isc_ko_leiden"].astype(str).values
adata_isc_ko.obsm["X_umap_isc_ko"] = adata_isc_ko_hvg.obsm["X_umap"].copy()


#4. ===========ISC / KO marker scores

isc_ko_score_genes = {
    "ISC": ["Igf1", "Matn1", "Meg3", "Rgcc", "Tm4sf1"],
    "KO1": ["Dcn", "Rcn3", "Pdia6", "Sdf2l1", "Ddost"],
    "KO2": ["Cpxm2", "Ctgf", "Kazald1", "Tectb", "Fkbp9"],
    "KO3": ["Cst3", "Gjb6", "Net1", "Tectb", "Tsen15"],
    "IdC": ["Cdkn1c", "Fxyd6", "Otoa", "Ptgds", "Smoc2"],
    "IPhC": ["Anxa5", "Matn4", "Gjb2", "Fabp7", "Prss23"],
    "Exclude": ["Oc90", "Otx2", "Myo6", "Myo7a", "Pecam1", "Kdr", "Ptprc", "Lyz2", "Sox10", "Mbp", "Mpz", "Snap25", "Tubb3"],
}

for group, genes in isc_ko_score_genes.items():
    adata_isc_ko.obs[f"{group}_score"] = mean_marker_expr(adata_isc_ko, genes)

isc_ko_score_cols = [f"{group}_score" for group in isc_ko_score_genes]
isc_ko_main_score_cols = [col for col in isc_ko_score_cols if col != "Exclude_score"]

isc_ko_score_summary = (
    adata_isc_ko.obs
    .groupby("isc_ko_leiden", observed=True)[isc_ko_score_cols]
    .mean()
    .round(3)
)
isc_ko_score_summary["n_cells"] = adata_isc_ko.obs["isc_ko_leiden"].value_counts().reindex(isc_ko_score_summary.index)
isc_ko_score_summary["top_call"] = (
    isc_ko_score_summary[isc_ko_main_score_cols]
    .idxmax(axis=1)
    .str.replace("_score", "", regex=False)
)

isc_ko_call_rank = {"ISC": 0, "KO1": 1, "KO2": 2, "KO3": 3, "IdC": 4, "IPhC": 5}
isc_ko_score_for_order = isc_ko_score_summary.copy()
isc_ko_score_for_order["call_rank"] = isc_ko_score_for_order["top_call"].map(isc_ko_call_rank).fillna(9)
isc_ko_score_for_order["cluster_rank"] = isc_ko_score_for_order.index.astype(str).astype(int)
isc_ko_order = isc_ko_score_for_order.sort_values(["call_rank", "cluster_rank"]).index.astype(str).tolist()

adata_isc_ko.obs["isc_ko_leiden"] = pd.Categorical(
    adata_isc_ko.obs["isc_ko_leiden"].astype(str),
    categories=isc_ko_order,
    ordered=True,
)

isc_ko_counts = adata_isc_ko.obs["isc_ko_leiden"].value_counts().reindex(isc_ko_order)
isc_ko_label_map = {cluster: f"{cluster} (n={int(isc_ko_counts.loc[cluster])})" for cluster in isc_ko_order}
adata_isc_ko.obs["isc_ko_leiden_label"] = pd.Categorical(
    adata_isc_ko.obs["isc_ko_leiden"].astype(str).map(isc_ko_label_map),
    categories=[isc_ko_label_map[cluster] for cluster in isc_ko_order],
    ordered=True,
)

display(isc_ko_score_summary.loc[isc_ko_order, isc_ko_score_cols + ["n_cells", "top_call"]])
display(pd.crosstab(adata_isc_ko.obs["isc_ko_leiden"], adata_isc_ko.obs["sample"], margins=True))
display(pd.crosstab(adata_isc_ko.obs["isc_ko_leiden"], adata_isc_ko.obs["parent_support_leiden"], margins=True))


#5. ===========ISC / KO UMAP and dotplot

sc.pl.embedding(
    adata_isc_ko,
    basis="umap_isc_ko",
    color=["isc_ko_leiden_label", "sample", "parent_support_leiden"],
    ncols=3,
    size=7,
    frameon=False,
    legend_loc="right margin",
    wspace=0.35,
    show=False,
)

sc.pl.embedding(
    adata_isc_ko,
    basis="umap_isc_ko",
    color="isc_ko_leiden_label",
    size=7,
    frameon=False,
    legend_loc="on data",
    legend_fontoutline=2,
    show=False,
)

isc_ko_marker_dict = {
    group: [gene for gene in genes if gene in adata_isc_ko.var_names]
    for group, genes in isc_ko_score_genes.items()
}
isc_ko_marker_dict = {group: genes for group, genes in isc_ko_marker_dict.items() if len(genes) > 0}

sc.pl.dotplot(
    adata_isc_ko,
    var_names=isc_ko_marker_dict,
    groupby="isc_ko_leiden",
    categories_order=isc_ko_order,
    standard_scale=None,
    cmap="Reds",
    dendrogram=False,
    swap_axes=False,
    figsize=(16, 5),
    show=False,
)


plt.close("all")


# Notebook cell 6

#1. ===========Assign ISC / KO branch labels

isc_ko_call_map = {
    "0": "ISC",
    "1": "ISC_KO_transition",
    "2": "KO1_like",
    "4": "KO1_like",
    "6": "KO1_like",
    "3": "KO2_like",
    "5": "KO2_like",
}

observed_isc_ko_leiden = set(adata_isc_ko.obs["isc_ko_leiden"].astype(str).unique())
missing_isc_ko_map = sorted(observed_isc_ko_leiden - set(isc_ko_call_map))
assert len(missing_isc_ko_map) == 0, f"Need ISC / KO label for isc_ko_leiden: {missing_isc_ko_map}"

isc_ko_call_order = ["ISC", "ISC_KO_transition", "KO1_like", "KO2_like"]

adata_isc_ko.obs["isc_ko_call"] = adata_isc_ko.obs["isc_ko_leiden"].astype(str).map(isc_ko_call_map)
adata_isc_ko.obs["isc_ko_call"] = pd.Categorical(
    adata_isc_ko.obs["isc_ko_call"],
    categories=isc_ko_call_order,
    ordered=True,
)

display(pd.crosstab(adata_isc_ko.obs["isc_ko_call"], adata_isc_ko.obs["sample"], margins=True))
display(pd.crosstab(adata_isc_ko.obs["isc_ko_call"], adata_isc_ko.obs["parent_support_leiden"], margins=True))


#2. ===========Extract IPC / OPC / DC branch

ipc_opc_dc_clusters = ["6", "9"]

adata_ipc_opc_dc = adata_support[
    adata_support.obs["support_leiden"].astype(str).isin(ipc_opc_dc_clusters)
].copy()

adata_ipc_opc_dc.obs["parent_support_leiden"] = pd.Categorical(
    adata_ipc_opc_dc.obs["support_leiden"].astype(str),
    categories=ipc_opc_dc_clusters,
    ordered=True,
)

assert adata_ipc_opc_dc.n_obs > 0, "No cells found for IPC / OPC / DC branch."

display(pd.crosstab(adata_ipc_opc_dc.obs["parent_support_leiden"], adata_ipc_opc_dc.obs["sample"], margins=True))


#3. ===========IPC / OPC / DC branch Harmony reclustering

sc.pp.highly_variable_genes(
    adata_ipc_opc_dc,
    flavor="seurat",
    n_top_genes=1200,
)

exclude_from_hvg = (
    adata_ipc_opc_dc.var_names.str.startswith(("mt-", "Mt-", "MT-"))
    | adata_ipc_opc_dc.var_names.str.startswith(("Rps", "Rpl"))
    | adata_ipc_opc_dc.var_names.str.startswith(("Hba", "Hbb"))
    | adata_ipc_opc_dc.var_names.isin([
        "Oc90", "Otx2",
        "Myo6", "Myo7a", "Cib2", "Pvalb", "Pou4f3", "Atoh1", "Ccer2", "Pcp4",
        "Pecam1", "Kdr", "Cdh5", "Ptprc", "Lyz2", "C1qa", "C1qb",
        "Sox10", "Mbp", "Mpz", "Plp1",
        "Snap25", "Tubb3", "Elavl3",
    ])
)
adata_ipc_opc_dc.var.loc[exclude_from_hvg, "highly_variable"] = False

adata_ipc_opc_dc_hvg = adata_ipc_opc_dc[:, adata_ipc_opc_dc.var["highly_variable"]].copy()
n_ipc_opc_dc_pcs = min(30, adata_ipc_opc_dc_hvg.n_obs - 1, adata_ipc_opc_dc_hvg.n_vars - 1)
assert n_ipc_opc_dc_pcs >= 2, "Too few cells or HVGs for IPC / OPC / DC PCA."

sc.pp.scale(adata_ipc_opc_dc_hvg, max_value=10)
sc.tl.pca(adata_ipc_opc_dc_hvg, svd_solver="arpack", n_comps=n_ipc_opc_dc_pcs)

logging.getLogger("harmonypy").setLevel(logging.ERROR)
logging.getLogger("harmonypy.harmony").setLevel(logging.ERROR)

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    ipc_opc_dc_harmony = hm.run_harmony(
        adata_ipc_opc_dc_hvg.obsm["X_pca"],
        adata_ipc_opc_dc_hvg.obs,
        vars_use=["sample"],
        max_iter_harmony=20,
    )

Z_corr = ipc_opc_dc_harmony.Z_corr
adata_ipc_opc_dc_hvg.obsm["X_pca_harmony"] = (
    Z_corr if Z_corr.shape[0] == adata_ipc_opc_dc_hvg.n_obs else Z_corr.T
)

sc.pp.neighbors(
    adata_ipc_opc_dc_hvg,
    n_neighbors=10,
    n_pcs=min(20, n_ipc_opc_dc_pcs),
    use_rep="X_pca_harmony",
)
sc.tl.umap(adata_ipc_opc_dc_hvg, random_state=0)

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    sc.tl.leiden(
        adata_ipc_opc_dc_hvg,
        resolution=0.5,
        key_added="ipc_opc_dc_leiden",
        random_state=0,
    )

adata_ipc_opc_dc.obs["ipc_opc_dc_leiden"] = adata_ipc_opc_dc_hvg.obs["ipc_opc_dc_leiden"].astype(str).values
adata_ipc_opc_dc.obsm["X_umap_ipc_opc_dc"] = adata_ipc_opc_dc_hvg.obsm["X_umap"].copy()


#4. ===========IPC / OPC / DC marker scores

ipc_opc_dc_score_genes = {
    "IPC": ["Npy", "Igfbpl1", "Tuba1a", "Tuba1b", "Cep41", "Cryab", "Emid1"],
    "OPC": ["Fam159b", "Smagp", "Serpine2", "Ppp1r2", "Fzd9", "S100b"],
    "DC": ["Pdzk1ip1", "Hes5", "Fgf3", "S100a1", "Rbp7", "Car14", "Mansc4"],
    "IPhC": ["Anxa5", "Matn4", "Gjb2", "Fabp7", "Prss23"],
    "Exclude": ["Oc90", "Otx2", "Myo6", "Myo7a", "Pecam1", "Kdr", "Ptprc", "Lyz2", "Sox10", "Mbp", "Mpz", "Snap25", "Tubb3"],
}

for group, genes in ipc_opc_dc_score_genes.items():
    adata_ipc_opc_dc.obs[f"{group}_score"] = mean_marker_expr(adata_ipc_opc_dc, genes)

ipc_opc_dc_score_cols = [f"{group}_score" for group in ipc_opc_dc_score_genes]
ipc_opc_dc_main_score_cols = [col for col in ipc_opc_dc_score_cols if col != "Exclude_score"]

ipc_opc_dc_score_summary = (
    adata_ipc_opc_dc.obs
    .groupby("ipc_opc_dc_leiden", observed=True)[ipc_opc_dc_score_cols]
    .mean()
    .round(3)
)
ipc_opc_dc_score_summary["n_cells"] = adata_ipc_opc_dc.obs["ipc_opc_dc_leiden"].value_counts().reindex(ipc_opc_dc_score_summary.index)
ipc_opc_dc_score_summary["top_call"] = (
    ipc_opc_dc_score_summary[ipc_opc_dc_main_score_cols]
    .idxmax(axis=1)
    .str.replace("_score", "", regex=False)
)

ipc_opc_dc_call_rank = {"IPC": 0, "OPC": 1, "DC": 2, "IPhC": 3}
ipc_opc_dc_score_for_order = ipc_opc_dc_score_summary.copy()
ipc_opc_dc_score_for_order["call_rank"] = ipc_opc_dc_score_for_order["top_call"].map(ipc_opc_dc_call_rank).fillna(9)
ipc_opc_dc_score_for_order["cluster_rank"] = ipc_opc_dc_score_for_order.index.astype(str).astype(int)
ipc_opc_dc_order = ipc_opc_dc_score_for_order.sort_values(["call_rank", "cluster_rank"]).index.astype(str).tolist()

adata_ipc_opc_dc.obs["ipc_opc_dc_leiden"] = pd.Categorical(
    adata_ipc_opc_dc.obs["ipc_opc_dc_leiden"].astype(str),
    categories=ipc_opc_dc_order,
    ordered=True,
)

ipc_opc_dc_counts = adata_ipc_opc_dc.obs["ipc_opc_dc_leiden"].value_counts().reindex(ipc_opc_dc_order)
ipc_opc_dc_label_map = {cluster: f"{cluster} (n={int(ipc_opc_dc_counts.loc[cluster])})" for cluster in ipc_opc_dc_order}
adata_ipc_opc_dc.obs["ipc_opc_dc_leiden_label"] = pd.Categorical(
    adata_ipc_opc_dc.obs["ipc_opc_dc_leiden"].astype(str).map(ipc_opc_dc_label_map),
    categories=[ipc_opc_dc_label_map[cluster] for cluster in ipc_opc_dc_order],
    ordered=True,
)

display(ipc_opc_dc_score_summary.loc[ipc_opc_dc_order, ipc_opc_dc_score_cols + ["n_cells", "top_call"]])
display(pd.crosstab(adata_ipc_opc_dc.obs["ipc_opc_dc_leiden"], adata_ipc_opc_dc.obs["sample"], margins=True))
display(pd.crosstab(adata_ipc_opc_dc.obs["ipc_opc_dc_leiden"], adata_ipc_opc_dc.obs["parent_support_leiden"], margins=True))


#5. ===========IPC / OPC / DC UMAP and dotplot

sc.pl.embedding(
    adata_ipc_opc_dc,
    basis="umap_ipc_opc_dc",
    color=["ipc_opc_dc_leiden_label", "sample", "parent_support_leiden"],
    ncols=3,
    size=14,
    frameon=False,
    legend_loc="right margin",
    wspace=0.35,
    show=False,
)

sc.pl.embedding(
    adata_ipc_opc_dc,
    basis="umap_ipc_opc_dc",
    color="ipc_opc_dc_leiden_label",
    size=14,
    frameon=False,
    legend_loc="on data",
    legend_fontoutline=2,
    show=False,
)

ipc_opc_dc_marker_dict = {
    group: [gene for gene in genes if gene in adata_ipc_opc_dc.var_names]
    for group, genes in ipc_opc_dc_score_genes.items()
}
ipc_opc_dc_marker_dict = {group: genes for group, genes in ipc_opc_dc_marker_dict.items() if len(genes) > 0}

sc.pl.dotplot(
    adata_ipc_opc_dc,
    var_names=ipc_opc_dc_marker_dict,
    groupby="ipc_opc_dc_leiden",
    categories_order=ipc_opc_dc_order,
    standard_scale=None,
    cmap="Reds",
    dendrogram=False,
    swap_axes=False,
    figsize=(12, 4),
    show=False,
)


plt.close("all")


# Notebook cell 7

#1. ===========Assign ISC / KO branch labels

isc_ko_call_map = {
    "0": "ISC",
    "1": "ISC_KO_transition",
    "2": "KO1_like",
    "4": "KO1_like",
    "6": "KO1_like",
    "3": "KO2_like",
    "5": "KO2_like",
}

observed_isc_ko_leiden = set(adata_isc_ko.obs["isc_ko_leiden"].astype(str).unique())
missing_isc_ko_map = sorted(observed_isc_ko_leiden - set(isc_ko_call_map))
assert len(missing_isc_ko_map) == 0, f"Need ISC / KO label for isc_ko_leiden: {missing_isc_ko_map}"

isc_ko_call_order = ["ISC", "ISC_KO_transition", "KO1_like", "KO2_like"]

adata_isc_ko.obs["isc_ko_call"] = adata_isc_ko.obs["isc_ko_leiden"].astype(str).map(isc_ko_call_map)
adata_isc_ko.obs["isc_ko_call"] = pd.Categorical(
    adata_isc_ko.obs["isc_ko_call"],
    categories=isc_ko_call_order,
    ordered=True,
)

display(pd.crosstab(adata_isc_ko.obs["isc_ko_call"], adata_isc_ko.obs["sample"], margins=True))
display(pd.crosstab(adata_isc_ko.obs["isc_ko_call"], adata_isc_ko.obs["parent_support_leiden"], margins=True))


#2. ===========Extract IPC / OPC / DC branch

ipc_opc_dc_clusters = ["6", "9"]

adata_ipc_opc_dc = adata_support[
    adata_support.obs["support_leiden"].astype(str).isin(ipc_opc_dc_clusters)
].copy()

adata_ipc_opc_dc.obs["parent_support_leiden"] = pd.Categorical(
    adata_ipc_opc_dc.obs["support_leiden"].astype(str),
    categories=ipc_opc_dc_clusters,
    ordered=True,
)

assert adata_ipc_opc_dc.n_obs > 0, "No cells found for IPC / OPC / DC branch."

display(pd.crosstab(adata_ipc_opc_dc.obs["parent_support_leiden"], adata_ipc_opc_dc.obs["sample"], margins=True))


#3. ===========IPC / OPC / DC branch Harmony reclustering

sc.pp.highly_variable_genes(
    adata_ipc_opc_dc,
    flavor="seurat",
    n_top_genes=1200,
)

exclude_from_hvg = (
    adata_ipc_opc_dc.var_names.str.startswith(("mt-", "Mt-", "MT-"))
    | adata_ipc_opc_dc.var_names.str.startswith(("Rps", "Rpl"))
    | adata_ipc_opc_dc.var_names.str.startswith(("Hba", "Hbb"))
    | adata_ipc_opc_dc.var_names.isin([
        "Oc90", "Otx2",
        "Myo6", "Myo7a", "Cib2", "Pvalb", "Pou4f3", "Atoh1", "Ccer2", "Pcp4",
        "Pecam1", "Kdr", "Cdh5", "Ptprc", "Lyz2", "C1qa", "C1qb",
        "Sox10", "Mbp", "Mpz", "Plp1",
        "Snap25", "Tubb3", "Elavl3",
    ])
)
adata_ipc_opc_dc.var.loc[exclude_from_hvg, "highly_variable"] = False

adata_ipc_opc_dc_hvg = adata_ipc_opc_dc[:, adata_ipc_opc_dc.var["highly_variable"]].copy()
n_ipc_opc_dc_pcs = min(30, adata_ipc_opc_dc_hvg.n_obs - 1, adata_ipc_opc_dc_hvg.n_vars - 1)
assert n_ipc_opc_dc_pcs >= 2, "Too few cells or HVGs for IPC / OPC / DC PCA."

sc.pp.scale(adata_ipc_opc_dc_hvg, max_value=10)
sc.tl.pca(adata_ipc_opc_dc_hvg, svd_solver="arpack", n_comps=n_ipc_opc_dc_pcs)

logging.getLogger("harmonypy").setLevel(logging.ERROR)
logging.getLogger("harmonypy.harmony").setLevel(logging.ERROR)

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    ipc_opc_dc_harmony = hm.run_harmony(
        adata_ipc_opc_dc_hvg.obsm["X_pca"],
        adata_ipc_opc_dc_hvg.obs,
        vars_use=["sample"],
        max_iter_harmony=20,
    )

Z_corr = ipc_opc_dc_harmony.Z_corr
adata_ipc_opc_dc_hvg.obsm["X_pca_harmony"] = (
    Z_corr if Z_corr.shape[0] == adata_ipc_opc_dc_hvg.n_obs else Z_corr.T
)

sc.pp.neighbors(
    adata_ipc_opc_dc_hvg,
    n_neighbors=10,
    n_pcs=min(20, n_ipc_opc_dc_pcs),
    use_rep="X_pca_harmony",
)
sc.tl.umap(adata_ipc_opc_dc_hvg, random_state=0)

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    sc.tl.leiden(
        adata_ipc_opc_dc_hvg,
        resolution=0.5,
        key_added="ipc_opc_dc_leiden",
        random_state=0,
    )

adata_ipc_opc_dc.obs["ipc_opc_dc_leiden"] = adata_ipc_opc_dc_hvg.obs["ipc_opc_dc_leiden"].astype(str).values
adata_ipc_opc_dc.obsm["X_umap_ipc_opc_dc"] = adata_ipc_opc_dc_hvg.obsm["X_umap"].copy()


#4. ===========IPC / OPC / DC marker scores

ipc_opc_dc_score_genes = {
    "IPC": ["Npy", "Igfbpl1", "Tuba1a", "Tuba1b", "Cep41", "Cryab", "Emid1"],
    "OPC": ["Fam159b", "Smagp", "Serpine2", "Ppp1r2", "Fzd9", "S100b"],
    "DC": ["Pdzk1ip1", "Hes5", "Fgf3", "S100a1", "Rbp7", "Car14", "Mansc4"],
    "IPhC": ["Anxa5", "Matn4", "Gjb2", "Fabp7", "Prss23"],
    "Exclude": ["Oc90", "Otx2", "Myo6", "Myo7a", "Pecam1", "Kdr", "Ptprc", "Lyz2", "Sox10", "Mbp", "Mpz", "Snap25", "Tubb3"],
}

for group, genes in ipc_opc_dc_score_genes.items():
    adata_ipc_opc_dc.obs[f"{group}_score"] = mean_marker_expr(adata_ipc_opc_dc, genes)

ipc_opc_dc_score_cols = [f"{group}_score" for group in ipc_opc_dc_score_genes]
ipc_opc_dc_main_score_cols = [col for col in ipc_opc_dc_score_cols if col != "Exclude_score"]

ipc_opc_dc_score_summary = (
    adata_ipc_opc_dc.obs
    .groupby("ipc_opc_dc_leiden", observed=True)[ipc_opc_dc_score_cols]
    .mean()
    .round(3)
)
ipc_opc_dc_score_summary["n_cells"] = adata_ipc_opc_dc.obs["ipc_opc_dc_leiden"].value_counts().reindex(ipc_opc_dc_score_summary.index)
ipc_opc_dc_score_summary["top_call"] = (
    ipc_opc_dc_score_summary[ipc_opc_dc_main_score_cols]
    .idxmax(axis=1)
    .str.replace("_score", "", regex=False)
)

ipc_opc_dc_call_rank = {"IPC": 0, "OPC": 1, "DC": 2, "IPhC": 3}
ipc_opc_dc_score_for_order = ipc_opc_dc_score_summary.copy()
ipc_opc_dc_score_for_order["call_rank"] = ipc_opc_dc_score_for_order["top_call"].map(ipc_opc_dc_call_rank).fillna(9)
ipc_opc_dc_score_for_order["cluster_rank"] = ipc_opc_dc_score_for_order.index.astype(str).astype(int)
ipc_opc_dc_order = ipc_opc_dc_score_for_order.sort_values(["call_rank", "cluster_rank"]).index.astype(str).tolist()

adata_ipc_opc_dc.obs["ipc_opc_dc_leiden"] = pd.Categorical(
    adata_ipc_opc_dc.obs["ipc_opc_dc_leiden"].astype(str),
    categories=ipc_opc_dc_order,
    ordered=True,
)

ipc_opc_dc_counts = adata_ipc_opc_dc.obs["ipc_opc_dc_leiden"].value_counts().reindex(ipc_opc_dc_order)
ipc_opc_dc_label_map = {cluster: f"{cluster} (n={int(ipc_opc_dc_counts.loc[cluster])})" for cluster in ipc_opc_dc_order}
adata_ipc_opc_dc.obs["ipc_opc_dc_leiden_label"] = pd.Categorical(
    adata_ipc_opc_dc.obs["ipc_opc_dc_leiden"].astype(str).map(ipc_opc_dc_label_map),
    categories=[ipc_opc_dc_label_map[cluster] for cluster in ipc_opc_dc_order],
    ordered=True,
)

display(ipc_opc_dc_score_summary.loc[ipc_opc_dc_order, ipc_opc_dc_score_cols + ["n_cells", "top_call"]])
display(pd.crosstab(adata_ipc_opc_dc.obs["ipc_opc_dc_leiden"], adata_ipc_opc_dc.obs["sample"], margins=True))
display(pd.crosstab(adata_ipc_opc_dc.obs["ipc_opc_dc_leiden"], adata_ipc_opc_dc.obs["parent_support_leiden"], margins=True))


#5. ===========IPC / OPC / DC UMAP and dotplot

sc.pl.embedding(
    adata_ipc_opc_dc,
    basis="umap_ipc_opc_dc",
    color=["ipc_opc_dc_leiden_label", "sample", "parent_support_leiden"],
    ncols=3,
    size=14,
    frameon=False,
    legend_loc="right margin",
    wspace=0.35,
    show=False,
)

sc.pl.embedding(
    adata_ipc_opc_dc,
    basis="umap_ipc_opc_dc",
    color="ipc_opc_dc_leiden_label",
    size=14,
    frameon=False,
    legend_loc="on data",
    legend_fontoutline=2,
    show=False,
)

ipc_opc_dc_marker_dict = {
    group: [gene for gene in genes if gene in adata_ipc_opc_dc.var_names]
    for group, genes in ipc_opc_dc_score_genes.items()
}
ipc_opc_dc_marker_dict = {group: genes for group, genes in ipc_opc_dc_marker_dict.items() if len(genes) > 0}

sc.pl.dotplot(
    adata_ipc_opc_dc,
    var_names=ipc_opc_dc_marker_dict,
    groupby="ipc_opc_dc_leiden",
    categories_order=ipc_opc_dc_order,
    standard_scale=None,
    cmap="Reds",
    dendrogram=False,
    swap_axes=False,
    figsize=(12, 4),
    show=False,
)


plt.close("all")


# Notebook cell 8

#1. ===========P1 Excel marker score helper

p1_excel_celltype_order = [
    "IHC", "OHC", "IPhC", "IPC", "OPC", "DC-1/2", "DC-3",
    "HeC", "CC/OSC", "ISC", "KO-1", "KO-2", "KO-3", "KO-4", "IdC",
]

def summarize_excel_scores(adata, group_key, score_genes):
    score_genes = {
        group: [gene for gene in genes if gene in adata.var_names]
        for group, genes in score_genes.items()
    }
    score_genes = {group: genes for group, genes in score_genes.items() if len(genes) > 0}
    score_cols = [f"{group}_score" for group in score_genes]

    for group, genes in score_genes.items():
        adata.obs[f"{group}_score"] = mean_marker_expr(adata, genes)

    main_cols = [col for col in score_cols if col != "Exclude_score"]
    summary = adata.obs.groupby(group_key, observed=True)[score_cols].mean().round(3)
    summary["n_cells"] = adata.obs[group_key].value_counts().reindex(summary.index)
    summary["top_call"] = summary[main_cols].idxmax(axis=1).str.replace("_score", "", regex=False)
    summary["score_margin"] = summary[main_cols].apply(
        lambda row: row.nlargest(2).iloc[0] - row.nlargest(2).iloc[1],
        axis=1,
    ).round(3)

    return score_genes, score_cols, summary


#2. ===========ISC / KO branch Excel marker scores

isc_ko_excel_score_genes = {
    "ISC": ["Igf1", "Matn1", "Meg3", "Rgcc", "Tm4sf1"],
    "KO-1": ["Dcn", "Rcn3", "Pdia6", "Sdf2l1", "Ddost"],
    "KO-2": ["Cpxm2", "Ctgf", "Kazald1", "Tectb", "Fkbp9"],
    "KO-3": ["Cst3", "Gjb6", "Net1", "Tectb", "Tsen15"],
    "KO-4": ["Calb1", "Crabp1", "Epyc", "Itm2a", "Stmn2"],
    "IdC": ["Cdkn1c", "Fxyd6", "Otoa", "Ptgds", "Smoc2"],
    "IPhC": ["Anxa5", "Matn4", "Gjb2", "Fabp7", "Prss23"],
    "Exclude": ["Oc90", "Otx2", "Myo6", "Myo7a", "Pecam1", "Kdr", "Ptprc", "Lyz2", "Sox10", "Mbp", "Mpz", "Snap25", "Tubb3"],
}

isc_ko_excel_marker_dict, isc_ko_excel_score_cols, isc_ko_excel_score_summary = summarize_excel_scores(
    adata_isc_ko,
    "isc_ko_leiden",
    isc_ko_excel_score_genes,
)

isc_ko_excel_call_map = isc_ko_excel_score_summary["top_call"].astype(str).to_dict()
adata_isc_ko.obs["isc_ko_excel_call"] = adata_isc_ko.obs["isc_ko_leiden"].astype(str).map(isc_ko_excel_call_map)
adata_isc_ko.obs["isc_ko_excel_call"] = pd.Categorical(
    adata_isc_ko.obs["isc_ko_excel_call"],
    categories=[x for x in p1_excel_celltype_order if x in adata_isc_ko.obs["isc_ko_excel_call"].unique()],
    ordered=True,
)

display(isc_ko_excel_score_summary[isc_ko_excel_score_cols + ["n_cells", "top_call", "score_margin"]])
display(pd.crosstab(adata_isc_ko.obs["isc_ko_excel_call"], adata_isc_ko.obs["sample"], margins=True))

sc.pl.dotplot(
    adata_isc_ko,
    var_names=isc_ko_excel_marker_dict,
    groupby="isc_ko_leiden",
    categories_order=isc_ko_order,
    standard_scale=None,
    cmap="Reds",
    dendrogram=False,
    swap_axes=False,
    figsize=(17, 5),
    show=False,
)


#3. ===========IPC / OPC / DC branch Excel marker scores

ipc_opc_dc_excel_score_genes = {
    "IPC": ["Npy", "Emid1", "Cryab", "Igfbpl1", "Matn4", "S100b"],
    "OPC": ["Fam159b", "Smagp", "Serpine2", "Ppp1r2", "S100b"],
    "DC-1/2": ["Hes5", "Pdzk1ip1", "S100a1", "Serpine2", "Ppp1r2"],
    "DC-3": ["Lgr5", "Fgf3", "Hes5", "Igfbpl1", "Lfng", "Prss23", "S100a1"],
    "IPhC": ["Anxa5", "Matn4", "Gjb2", "Fabp7", "Prss23"],
    "Exclude": ["Oc90", "Otx2", "Myo6", "Myo7a", "Pecam1", "Kdr", "Ptprc", "Lyz2", "Sox10", "Mbp", "Mpz", "Snap25", "Tubb3"],
}

ipc_opc_dc_excel_marker_dict, ipc_opc_dc_excel_score_cols, ipc_opc_dc_excel_score_summary = summarize_excel_scores(
    adata_ipc_opc_dc,
    "ipc_opc_dc_leiden",
    ipc_opc_dc_excel_score_genes,
)

ipc_opc_dc_excel_call_map = ipc_opc_dc_excel_score_summary["top_call"].astype(str).to_dict()
adata_ipc_opc_dc.obs["ipc_opc_dc_excel_call"] = adata_ipc_opc_dc.obs["ipc_opc_dc_leiden"].astype(str).map(ipc_opc_dc_excel_call_map)
adata_ipc_opc_dc.obs["ipc_opc_dc_excel_call"] = pd.Categorical(
    adata_ipc_opc_dc.obs["ipc_opc_dc_excel_call"],
    categories=[x for x in p1_excel_celltype_order if x in adata_ipc_opc_dc.obs["ipc_opc_dc_excel_call"].unique()],
    ordered=True,
)

display(ipc_opc_dc_excel_score_summary[ipc_opc_dc_excel_score_cols + ["n_cells", "top_call", "score_margin"]])
display(pd.crosstab(adata_ipc_opc_dc.obs["ipc_opc_dc_excel_call"], adata_ipc_opc_dc.obs["sample"], margins=True))

sc.pl.dotplot(
    adata_ipc_opc_dc,
    var_names=ipc_opc_dc_excel_marker_dict,
    groupby="ipc_opc_dc_leiden",
    categories_order=ipc_opc_dc_order,
    standard_scale=None,
    cmap="Reds",
    dendrogram=False,
    swap_axes=False,
    figsize=(14, 4),
    show=False,
)


plt.close("all")


# Notebook cell 9

#1. ===========Assign Excel-based labels for completed branches

isc_ko_excel_final_map = {
    "0": "KO-4",
    "1": "KO-4",
    "2": "KO-1",
    "4": "KO-1",
    "6": "KO-1",
    "3": "KO-2",
    "5": "KO-2",
}

isc_ko_excel_confidence_map = {
    "0": "high",
    "1": "high",
    "2": "high",
    "4": "low_KO1_KO4_tie",
    "6": "medium",
    "3": "medium_KO2_KO3_close",
    "5": "low_KO2_KO1_close",
}

observed_isc_ko_leiden = set(adata_isc_ko.obs["isc_ko_leiden"].astype(str).unique())
missing_isc_ko_final = sorted(observed_isc_ko_leiden - set(isc_ko_excel_final_map))
assert len(missing_isc_ko_final) == 0, f"Need final Excel label for isc_ko_leiden: {missing_isc_ko_final}"

adata_isc_ko.obs["isc_ko_excel_call"] = adata_isc_ko.obs["isc_ko_leiden"].astype(str).map(isc_ko_excel_final_map)
adata_isc_ko.obs["isc_ko_excel_confidence"] = adata_isc_ko.obs["isc_ko_leiden"].astype(str).map(isc_ko_excel_confidence_map)
adata_isc_ko.obs["isc_ko_excel_call"] = pd.Categorical(
    adata_isc_ko.obs["isc_ko_excel_call"],
    categories=[x for x in p1_excel_celltype_order if x in adata_isc_ko.obs["isc_ko_excel_call"].unique()],
    ordered=True,
)

ipc_opc_dc_excel_final_map = {
    "1": "IPC",
    "4": "DC-1/2",
    "0": "DC-1/2",
    "2": "DC-1/2",
    "3": "DC-1/2",
    "5": "DC-1/2",
    "6": "DC-3",
    "7": "DC-1/2",
    "8": "DC-1/2",
    "9": "DC-1/2",
}

ipc_opc_dc_excel_confidence_map = {
    "1": "high",
    "4": "low_DC12_DC3_close",
    "0": "medium",
    "2": "high",
    "3": "low_DC12_OPC_close",
    "5": "medium",
    "6": "low_DC3_DC12_tie",
    "7": "medium",
    "8": "medium",
    "9": "medium",
}

observed_ipc_opc_dc_leiden = set(adata_ipc_opc_dc.obs["ipc_opc_dc_leiden"].astype(str).unique())
missing_ipc_opc_dc_final = sorted(observed_ipc_opc_dc_leiden - set(ipc_opc_dc_excel_final_map))
assert len(missing_ipc_opc_dc_final) == 0, f"Need final Excel label for ipc_opc_dc_leiden: {missing_ipc_opc_dc_final}"

adata_ipc_opc_dc.obs["ipc_opc_dc_excel_call"] = adata_ipc_opc_dc.obs["ipc_opc_dc_leiden"].astype(str).map(ipc_opc_dc_excel_final_map)
adata_ipc_opc_dc.obs["ipc_opc_dc_excel_confidence"] = adata_ipc_opc_dc.obs["ipc_opc_dc_leiden"].astype(str).map(ipc_opc_dc_excel_confidence_map)
adata_ipc_opc_dc.obs["ipc_opc_dc_excel_call"] = pd.Categorical(
    adata_ipc_opc_dc.obs["ipc_opc_dc_excel_call"],
    categories=[x for x in p1_excel_celltype_order if x in adata_ipc_opc_dc.obs["ipc_opc_dc_excel_call"].unique()],
    ordered=True,
)

display(pd.crosstab(adata_isc_ko.obs["isc_ko_excel_call"], adata_isc_ko.obs["sample"], margins=True))
display(pd.crosstab(adata_ipc_opc_dc.obs["ipc_opc_dc_excel_call"], adata_ipc_opc_dc.obs["sample"], margins=True))


#2. ===========Extract HeC / CC / OSC branch

hec_cc_osc_clusters = ["8"]

adata_hec_cc_osc = adata_support[
    adata_support.obs["support_leiden"].astype(str).isin(hec_cc_osc_clusters)
].copy()

adata_hec_cc_osc.obs["parent_support_leiden"] = pd.Categorical(
    adata_hec_cc_osc.obs["support_leiden"].astype(str),
    categories=hec_cc_osc_clusters,
    ordered=True,
)

assert adata_hec_cc_osc.n_obs > 0, "No cells found for HeC / CC / OSC branch."

display(pd.crosstab(adata_hec_cc_osc.obs["parent_support_leiden"], adata_hec_cc_osc.obs["sample"], margins=True))


#3. ===========HeC / CC / OSC branch Harmony reclustering

sc.pp.highly_variable_genes(
    adata_hec_cc_osc,
    flavor="seurat",
    n_top_genes=800,
)

exclude_from_hvg = (
    adata_hec_cc_osc.var_names.str.startswith(("mt-", "Mt-", "MT-"))
    | adata_hec_cc_osc.var_names.str.startswith(("Rps", "Rpl"))
    | adata_hec_cc_osc.var_names.str.startswith(("Hba", "Hbb"))
    | adata_hec_cc_osc.var_names.isin([
        "Oc90", "Otx2",
        "Myo6", "Myo7a", "Cib2", "Pvalb", "Pou4f3", "Atoh1", "Ccer2", "Pcp4",
        "Pecam1", "Kdr", "Cdh5", "Ptprc", "Lyz2", "C1qa", "C1qb",
        "Sox10", "Mbp", "Mpz", "Plp1",
        "Snap25", "Tubb3", "Elavl3",
    ])
)
adata_hec_cc_osc.var.loc[exclude_from_hvg, "highly_variable"] = False

adata_hec_cc_osc_hvg = adata_hec_cc_osc[:, adata_hec_cc_osc.var["highly_variable"]].copy()
n_hec_cc_osc_pcs = min(25, adata_hec_cc_osc_hvg.n_obs - 1, adata_hec_cc_osc_hvg.n_vars - 1)
assert n_hec_cc_osc_pcs >= 2, "Too few cells or HVGs for HeC / CC / OSC PCA."

sc.pp.scale(adata_hec_cc_osc_hvg, max_value=10)
sc.tl.pca(adata_hec_cc_osc_hvg, svd_solver="arpack", n_comps=n_hec_cc_osc_pcs)

logging.getLogger("harmonypy").setLevel(logging.ERROR)
logging.getLogger("harmonypy.harmony").setLevel(logging.ERROR)

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    hec_cc_osc_harmony = hm.run_harmony(
        adata_hec_cc_osc_hvg.obsm["X_pca"],
        adata_hec_cc_osc_hvg.obs,
        vars_use=["sample"],
        max_iter_harmony=20,
    )

Z_corr = hec_cc_osc_harmony.Z_corr
adata_hec_cc_osc_hvg.obsm["X_pca_harmony"] = (
    Z_corr if Z_corr.shape[0] == adata_hec_cc_osc_hvg.n_obs else Z_corr.T
)

sc.pp.neighbors(
    adata_hec_cc_osc_hvg,
    n_neighbors=8,
    n_pcs=min(15, n_hec_cc_osc_pcs),
    use_rep="X_pca_harmony",
)
sc.tl.umap(adata_hec_cc_osc_hvg, random_state=0)

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    sc.tl.leiden(
        adata_hec_cc_osc_hvg,
        resolution=0.45,
        key_added="hec_cc_osc_leiden",
        random_state=0,
    )

adata_hec_cc_osc.obs["hec_cc_osc_leiden"] = adata_hec_cc_osc_hvg.obs["hec_cc_osc_leiden"].astype(str).values
adata_hec_cc_osc.obsm["X_umap_hec_cc_osc"] = adata_hec_cc_osc_hvg.obsm["X_umap"].copy()


#4. ===========HeC / CC / OSC Excel marker scores

hec_cc_osc_excel_score_genes = {
    "HeC": ["Pmch", "Fst", "Nupr1", "Fam159b", "Egfl6"],
    "CC/OSC": ["Npnt", "Bmp4", "Fbln2", "Fst", "Apoe"],
    "IPhC": ["Anxa5", "Matn4", "Gjb2", "Fabp7", "Prss23"],
    "ISC": ["Igf1", "Matn1", "Meg3", "Rgcc", "Tm4sf1"],
    "KO-1": ["Dcn", "Rcn3", "Pdia6", "Sdf2l1", "Ddost"],
    "KO-2": ["Cpxm2", "Ctgf", "Kazald1", "Tectb", "Fkbp9"],
    "KO-3": ["Cst3", "Gjb6", "Net1", "Tectb", "Tsen15"],
    "KO-4": ["Calb1", "Crabp1", "Epyc", "Itm2a", "Stmn2"],
    "IdC": ["Cdkn1c", "Fxyd6", "Otoa", "Ptgds", "Smoc2"],
    "Exclude": ["Oc90", "Otx2", "Myo6", "Myo7a", "Pecam1", "Kdr", "Ptprc", "Lyz2", "Sox10", "Mbp", "Mpz", "Snap25", "Tubb3"],
}

hec_cc_osc_excel_marker_dict, hec_cc_osc_excel_score_cols, hec_cc_osc_excel_score_summary = summarize_excel_scores(
    adata_hec_cc_osc,
    "hec_cc_osc_leiden",
    hec_cc_osc_excel_score_genes,
)

hec_cc_osc_score_for_order = hec_cc_osc_excel_score_summary.copy()
hec_cc_osc_score_for_order["call_rank"] = hec_cc_osc_score_for_order["top_call"].map(
    {celltype: i for i, celltype in enumerate(p1_excel_celltype_order)}
).fillna(99)
hec_cc_osc_score_for_order["cluster_rank"] = hec_cc_osc_score_for_order.index.astype(str).astype(int)
hec_cc_osc_order = hec_cc_osc_score_for_order.sort_values(["call_rank", "cluster_rank"]).index.astype(str).tolist()

adata_hec_cc_osc.obs["hec_cc_osc_leiden"] = pd.Categorical(
    adata_hec_cc_osc.obs["hec_cc_osc_leiden"].astype(str),
    categories=hec_cc_osc_order,
    ordered=True,
)

hec_cc_osc_counts = adata_hec_cc_osc.obs["hec_cc_osc_leiden"].value_counts().reindex(hec_cc_osc_order)
hec_cc_osc_label_map = {cluster: f"{cluster} (n={int(hec_cc_osc_counts.loc[cluster])})" for cluster in hec_cc_osc_order}
adata_hec_cc_osc.obs["hec_cc_osc_leiden_label"] = pd.Categorical(
    adata_hec_cc_osc.obs["hec_cc_osc_leiden"].astype(str).map(hec_cc_osc_label_map),
    categories=[hec_cc_osc_label_map[cluster] for cluster in hec_cc_osc_order],
    ordered=True,
)

hec_cc_osc_excel_call_map = hec_cc_osc_excel_score_summary["top_call"].astype(str).to_dict()
adata_hec_cc_osc.obs["hec_cc_osc_excel_call"] = adata_hec_cc_osc.obs["hec_cc_osc_leiden"].astype(str).map(hec_cc_osc_excel_call_map)
adata_hec_cc_osc.obs["hec_cc_osc_excel_call"] = pd.Categorical(
    adata_hec_cc_osc.obs["hec_cc_osc_excel_call"],
    categories=[x for x in p1_excel_celltype_order if x in adata_hec_cc_osc.obs["hec_cc_osc_excel_call"].unique()],
    ordered=True,
)

display(hec_cc_osc_excel_score_summary.loc[hec_cc_osc_order, hec_cc_osc_excel_score_cols + ["n_cells", "top_call", "score_margin"]])
display(pd.crosstab(adata_hec_cc_osc.obs["hec_cc_osc_excel_call"], adata_hec_cc_osc.obs["sample"], margins=True))


#5. ===========HeC / CC / OSC UMAP and dotplot

sc.pl.embedding(
    adata_hec_cc_osc,
    basis="umap_hec_cc_osc",
    color=["hec_cc_osc_leiden_label", "sample", "hec_cc_osc_excel_call"],
    ncols=3,
    size=18,
    frameon=False,
    legend_loc="right margin",
    wspace=0.35,
    show=False,
)

sc.pl.embedding(
    adata_hec_cc_osc,
    basis="umap_hec_cc_osc",
    color="hec_cc_osc_leiden_label",
    size=18,
    frameon=False,
    legend_loc="on data",
    legend_fontoutline=2,
    show=False,
)

sc.pl.dotplot(
    adata_hec_cc_osc,
    var_names=hec_cc_osc_excel_marker_dict,
    groupby="hec_cc_osc_leiden",
    categories_order=hec_cc_osc_order,
    standard_scale=None,
    cmap="Reds",
    dendrogram=False,
    swap_axes=False,
    figsize=(17, 4.5),
    show=False,
)


plt.close("all")


# Notebook cell 10

#1. ===========Assign HeC / CC / OSC final Excel labels

hec_cc_osc_excel_final_map = {
    "7": "IPhC",
    "2": "HeC",
    "3": "HeC",
    "1": "CC/OSC",
    "5": "CC/OSC",
    "6": "CC/OSC",
    "0": "KO-1",
    "4": "KO-1",
}

hec_cc_osc_excel_confidence_map = {
    "7": "medium_IPhC_KO1_close",
    "2": "medium",
    "3": "medium_HeC_KO1_close",
    "1": "high",
    "5": "high",
    "6": "medium",
    "0": "low_KO1_KO3_close",
    "4": "medium_KO1_HeC_close",
}

observed_hec_cc_osc_leiden = set(adata_hec_cc_osc.obs["hec_cc_osc_leiden"].astype(str).unique())
missing_hec_cc_osc_final = sorted(observed_hec_cc_osc_leiden - set(hec_cc_osc_excel_final_map))
assert len(missing_hec_cc_osc_final) == 0, f"Need final Excel label for hec_cc_osc_leiden: {missing_hec_cc_osc_final}"

adata_hec_cc_osc.obs["hec_cc_osc_excel_call"] = (
    adata_hec_cc_osc.obs["hec_cc_osc_leiden"].astype(str).map(hec_cc_osc_excel_final_map)
)
adata_hec_cc_osc.obs["hec_cc_osc_excel_confidence"] = (
    adata_hec_cc_osc.obs["hec_cc_osc_leiden"].astype(str).map(hec_cc_osc_excel_confidence_map)
)
adata_hec_cc_osc.obs["hec_cc_osc_excel_call"] = pd.Categorical(
    adata_hec_cc_osc.obs["hec_cc_osc_excel_call"],
    categories=[x for x in p1_excel_celltype_order if x in adata_hec_cc_osc.obs["hec_cc_osc_excel_call"].unique()],
    ordered=True,
)

display(pd.crosstab(adata_hec_cc_osc.obs["hec_cc_osc_excel_call"], adata_hec_cc_osc.obs["sample"], margins=True))


#2. ===========Merge P1 final Excel cell types

adata_clean.obs["P1_final_celltype"] = "Unassigned"
adata_clean.obs["P1_final_confidence"] = "unassigned"

adata_clean.obs.loc[adata_hc.obs_names, "P1_final_celltype"] = adata_hc.obs["hc_celltype"].astype(str).values
adata_clean.obs.loc[adata_hc.obs_names, "P1_final_confidence"] = "high"

direct_support_map = {
    "4": "IdC",
    "5": "IPhC",
    "7": "IPhC",
}
direct_support_confidence_map = {
    "4": "high",
    "5": "medium_IPhC_ISC_close",
    "7": "high",
}

direct_support_mask = adata_support.obs["support_leiden"].astype(str).isin(direct_support_map)
direct_support_cells = adata_support.obs_names[direct_support_mask]
direct_support_leiden = adata_support.obs.loc[direct_support_cells, "support_leiden"].astype(str)

adata_clean.obs.loc[direct_support_cells, "P1_final_celltype"] = direct_support_leiden.map(direct_support_map).values
adata_clean.obs.loc[direct_support_cells, "P1_final_confidence"] = direct_support_leiden.map(direct_support_confidence_map).values

adata_clean.obs.loc[adata_isc_ko.obs_names, "P1_final_celltype"] = adata_isc_ko.obs["isc_ko_excel_call"].astype(str).values
adata_clean.obs.loc[adata_isc_ko.obs_names, "P1_final_confidence"] = adata_isc_ko.obs["isc_ko_excel_confidence"].astype(str).values

adata_clean.obs.loc[adata_ipc_opc_dc.obs_names, "P1_final_celltype"] = adata_ipc_opc_dc.obs["ipc_opc_dc_excel_call"].astype(str).values
adata_clean.obs.loc[adata_ipc_opc_dc.obs_names, "P1_final_confidence"] = adata_ipc_opc_dc.obs["ipc_opc_dc_excel_confidence"].astype(str).values

adata_clean.obs.loc[adata_hec_cc_osc.obs_names, "P1_final_celltype"] = adata_hec_cc_osc.obs["hec_cc_osc_excel_call"].astype(str).values
adata_clean.obs.loc[adata_hec_cc_osc.obs_names, "P1_final_confidence"] = adata_hec_cc_osc.obs["hec_cc_osc_excel_confidence"].astype(str).values

assert not (adata_clean.obs["P1_final_celltype"] == "Unassigned").any()
assert set(adata_clean.obs["P1_final_celltype"].astype(str)).issubset(set(p1_excel_celltype_order))

p1_final_celltype_order = [
    celltype for celltype in p1_excel_celltype_order
    if celltype in adata_clean.obs["P1_final_celltype"].astype(str).unique()
]

adata_clean.obs["P1_final_celltype"] = pd.Categorical(
    adata_clean.obs["P1_final_celltype"].astype(str),
    categories=p1_final_celltype_order,
    ordered=True,
)

p1_final_counts = adata_clean.obs["P1_final_celltype"].value_counts().reindex(p1_final_celltype_order)
p1_final_label_map = {
    celltype: f"{celltype} (n={int(p1_final_counts.loc[celltype])})"
    for celltype in p1_final_celltype_order
}
p1_final_label_order = [p1_final_label_map[celltype] for celltype in p1_final_celltype_order]

adata_clean.obs["P1_final_celltype_label"] = adata_clean.obs["P1_final_celltype"].astype(str).map(p1_final_label_map)
adata_clean.obs["P1_final_celltype_label"] = pd.Categorical(
    adata_clean.obs["P1_final_celltype_label"].astype(str),
    categories=p1_final_label_order,
    ordered=True,
)

display(pd.crosstab(adata_clean.obs["P1_final_celltype"], adata_clean.obs["sample"], margins=True))
display(pd.crosstab(adata_clean.obs["P1_final_celltype"], adata_clean.obs["P1_final_confidence"], margins=True))


#3. ===========P1 final UMAP check

sc.pl.embedding(
    adata_clean,
    basis="umap_harmony",
    color=["P1_final_celltype_label", "sample"],
    ncols=2,
    size=5,
    frameon=False,
    legend_loc="right margin",
    wspace=0.35,
    show=False,
)

sc.pl.embedding(
    adata_clean,
    basis="umap_harmony",
    color="P1_final_celltype_label",
    size=5,
    frameon=False,
    legend_loc="on data",
    legend_fontoutline=2,
    show=False,
)


#4. ===========P1 final Excel marker dotplot

p1_final_marker_dict = {
    "IHC": ["Fgf8", "Atp2a3", "Cabp2", "Tbx2", "Kcnj13", "Fam19a3", "Shtn1"],
    "OHC": ["Calca", "Serpina1c", "Veph1", "Strip2", "Msln1"],
    "IPhC": ["Anxa5", "Matn4", "Gjb2", "Fabp7", "Prss23"],
    "IPC": ["Npy", "Emid1", "Cryab", "Igfbpl1", "Matn4", "S100b"],
    "OPC": ["Fam159b", "Smagp", "Serpine2", "Ppp1r2", "S100b"],
    "DC-1/2": ["Hes5", "Pdzk1ip1", "S100a1", "Serpine2", "Ppp1r2"],
    "DC-3": ["Lgr5", "Fgf3", "Hes5", "Igfbpl1", "Lfng", "Prss23", "S100a1"],
    "HeC": ["Pmch", "Fst", "Nupr1", "Fam159b", "Egfl6"],
    "CC/OSC": ["Npnt", "Bmp4", "Fbln2", "Fst", "Apoe"],
    "ISC": ["Igf1", "Matn1", "Meg3", "Rgcc", "Tm4sf1"],
    "KO-1": ["Dcn", "Rcn3", "Pdia6", "Sdf2l1", "Ddost"],
    "KO-2": ["Cpxm2", "Ctgf", "Kazald1", "Tectb", "Fkbp9"],
    "KO-3": ["Cst3", "Gjb6", "Net1", "Tectb", "Tsen15"],
    "KO-4": ["Calb1", "Crabp1", "Epyc", "Itm2a", "Stmn2"],
    "IdC": ["Cdkn1c", "Fxyd6", "Otoa", "Ptgds", "Smoc2"],
    "Exclude": ["Oc90", "Otx2", "Myo6", "Myo7a", "Pecam1", "Kdr", "Ptprc", "Lyz2", "Sox10", "Mbp", "Mpz", "Snap25", "Tubb3"],
}

p1_final_marker_dict = {
    group: [gene for gene in genes if gene in adata_clean.var_names]
    for group, genes in p1_final_marker_dict.items()
}
p1_final_marker_dict = {
    group: genes for group, genes in p1_final_marker_dict.items()
    if len(genes) > 0
}

sc.pl.dotplot(
    adata_clean,
    var_names=p1_final_marker_dict,
    groupby="P1_final_celltype",
    categories_order=p1_final_celltype_order,
    standard_scale=None,
    cmap="Reds",
    dendrogram=False,
    swap_axes=False,
    figsize=(24, 7),
    show=False,
)


plt.close("all")


# Notebook cell 11

#1. ===========Final label score audit

p1_final_audit_marker_dict = {
    "HC": ["Myo6", "Myo7a", "Cib2", "Pvalb", "Pou4f3", "Atoh1", "Ccer2", "Pcp4"],
    "IHC": ["Fgf8", "Atp2a3", "Cabp2", "Tbx2", "Kcnj13", "Fam19a3", "Shtn1"],
    "OHC": ["Calca", "Serpina1c", "Veph1", "Strip2", "Msln1", "Pcp4", "Calb1"],
    "IPhC": ["Anxa5", "Matn4", "Gjb2", "Fabp7", "Prss23"],
    "IPC": ["Npy", "Emid1", "Cryab", "Igfbpl1", "Matn4", "S100b"],
    "OPC": ["Fam159b", "Smagp", "Serpine2", "Ppp1r2", "S100b"],
    "DC-1/2": ["Hes5", "Pdzk1ip1", "S100a1", "Serpine2", "Ppp1r2"],
    "DC-3": ["Lgr5", "Fgf3", "Hes5", "Igfbpl1", "Lfng", "Prss23", "S100a1"],
    "HeC": ["Pmch", "Fst", "Nupr1", "Fam159b", "Egfl6"],
    "CC/OSC": ["Npnt", "Bmp4", "Fbln2", "Fst", "Apoe"],
    "ISC": ["Igf1", "Matn1", "Meg3", "Rgcc", "Tm4sf1"],
    "KO-1": ["Dcn", "Rcn3", "Pdia6", "Sdf2l1", "Ddost"],
    "KO-2": ["Cpxm2", "Ctgf", "Kazald1", "Tectb", "Fkbp9"],
    "KO-3": ["Cst3", "Gjb6", "Net1", "Tectb", "Tsen15"],
    "KO-4": ["Calb1", "Crabp1", "Epyc", "Itm2a", "Stmn2"],
    "IdC": ["Cdkn1c", "Fxyd6", "Otoa", "Ptgds", "Smoc2"],
}

p1_final_audit_marker_dict = {
    group: [gene for gene in genes if gene in adata_clean.var_names]
    for group, genes in p1_final_audit_marker_dict.items()
}
p1_final_audit_marker_dict = {
    group: genes for group, genes in p1_final_audit_marker_dict.items()
    if len(genes) > 0
}

for group, genes in p1_final_audit_marker_dict.items():
    adata_clean.obs[f"audit_{group}_score"] = mean_marker_expr(adata_clean, genes)

audit_score_cols = [f"audit_{group}_score" for group in p1_final_audit_marker_dict]

p1_final_audit_summary = (
    adata_clean.obs
    .groupby("P1_final_celltype", observed=True)[audit_score_cols]
    .mean()
    .round(3)
)
p1_final_audit_summary["n_cells"] = adata_clean.obs["P1_final_celltype"].value_counts().reindex(
    p1_final_audit_summary.index
)
p1_final_audit_summary["top_call"] = (
    p1_final_audit_summary[audit_score_cols]
    .idxmax(axis=1)
    .str.replace("audit_", "", regex=False)
    .str.replace("_score", "", regex=False)
)
p1_final_audit_summary["score_margin"] = p1_final_audit_summary[audit_score_cols].apply(
    lambda row: row.nlargest(2).iloc[0] - row.nlargest(2).iloc[1],
    axis=1,
).round(3)

display(p1_final_audit_summary[audit_score_cols + ["n_cells", "top_call", "score_margin"]])


#2. ===========Focused KO / ISC / IdC check

ko_isc_idc_order = [
    x for x in ["ISC", "KO-1", "KO-2", "KO-3", "KO-4", "IdC"]
    if x in adata_clean.obs["P1_final_celltype"].astype(str).unique()
]

sc.pl.dotplot(
    adata_clean[adata_clean.obs["P1_final_celltype"].astype(str).isin(ko_isc_idc_order)].copy(),
    var_names={
        group: p1_final_audit_marker_dict[group]
        for group in ["ISC", "KO-1", "KO-2", "KO-3", "KO-4", "IdC"]
        if group in p1_final_audit_marker_dict
    },
    groupby="P1_final_celltype",
    categories_order=ko_isc_idc_order,
    standard_scale=None,
    cmap="Reds",
    dendrogram=False,
    swap_axes=False,
    figsize=(14, 4),
    show=False,
)


#3. ===========Focused IPC / OPC / DC check

ipc_opc_dc_order_final = [
    x for x in ["IPC", "OPC", "DC-1/2", "DC-3"]
    if x in adata_clean.obs["P1_final_celltype"].astype(str).unique()
]

sc.pl.dotplot(
    adata_clean[adata_clean.obs["P1_final_celltype"].astype(str).isin(ipc_opc_dc_order_final)].copy(),
    var_names={
        group: p1_final_audit_marker_dict[group]
        for group in ["IPC", "OPC", "DC-1/2", "DC-3"]
        if group in p1_final_audit_marker_dict
    },
    groupby="P1_final_celltype",
    categories_order=ipc_opc_dc_order_final,
    standard_scale=None,
    cmap="Reds",
    dendrogram=False,
    swap_axes=False,
    figsize=(10, 3.5),
    show=False,
)


plt.close("all")


# Notebook cell 12

#1. ===========Extract unstable marker-refinement branch

if "p1_excel_celltype_order" not in globals():
    p1_excel_celltype_order = [
        "IHC", "OHC", "IPhC", "IPC", "OPC", "DC-1/2", "DC-3",
        "HeC", "CC/OSC", "ISC", "KO-1", "KO-2", "KO-3", "KO-4", "IdC",
    ]

refine_parent_clusters = ["0", "1", "2", "3", "4", "5", "7"]
refine_cell_set = set(
    adata_support.obs_names[
        adata_support.obs["support_leiden"].astype(str).isin(refine_parent_clusters)
    ]
)

hec_problem_cells = []
if "adata_hec_cc_osc" in globals() and "hec_cc_osc_leiden" in adata_hec_cc_osc.obs.columns:
    hec_problem_cells = adata_hec_cc_osc.obs_names[
        adata_hec_cc_osc.obs["hec_cc_osc_leiden"].astype(str).isin(["0", "4", "7"])
    ]
    refine_cell_set.update(hec_problem_cells)

refine_cells = adata_support.obs_names[adata_support.obs_names.isin(refine_cell_set)]
adata_refine = adata_support[refine_cells].copy()

adata_refine.obs["parent_support_leiden"] = pd.Categorical(
    adata_refine.obs["support_leiden"].astype(str),
    categories=["0", "1", "2", "3", "4", "5", "7", "8"],
    ordered=True,
)

adata_refine.obs["refine_source"] = "support_" + adata_refine.obs["support_leiden"].astype(str)
if len(hec_problem_cells) > 0:
    hec_problem_map = adata_hec_cc_osc.obs.loc[hec_problem_cells, "hec_cc_osc_leiden"].astype(str)
    adata_refine.obs.loc[hec_problem_cells, "refine_source"] = "hec_cc_osc_" + hec_problem_map

assert adata_refine.n_obs > 0, "No cells found for refinement branch."

display(pd.crosstab(adata_refine.obs["refine_source"], adata_refine.obs["sample"], margins=True))
display(pd.crosstab(adata_refine.obs["parent_support_leiden"], adata_refine.obs["sample"], margins=True))


#2. ===========Marker-guided Harmony reclustering

refine_z_score_genes = {
    "IPhC": ["Anxa5", "Matn4", "Fabp7", "Prss23"],
    "ISC": ["Igf1", "Matn1", "Meg3", "Rgcc", "Tm4sf1"],
    "KO-1": ["Dcn", "Rcn3", "Pdia6", "Sdf2l1", "Ddost"],
    "KO-2": ["Cpxm2", "Ctgf", "Kazald1", "Fkbp9"],
    "KO-3": ["Cst3", "Gjb6", "Net1", "Tsen15"],
    "KO-4": ["Crabp1", "Itm2a", "Stmn2"],
    "IdC": ["Cdkn1c", "Fxyd6", "Otoa", "Ptgds", "Smoc2"],
    "HeC": ["Pmch", "Fst", "Nupr1", "Egfl6"],
    "CC/OSC": ["Npnt", "Bmp4", "Fbln2", "Apoe"],
}

refine_dotplot_marker_dict = {
    "IPhC": ["Anxa5", "Matn4", "Gjb2", "Fabp7", "Prss23"],
    "ISC": ["Igf1", "Matn1", "Meg3", "Rgcc", "Tm4sf1"],
    "KO-1": ["Dcn", "Rcn3", "Pdia6", "Sdf2l1", "Ddost"],
    "KO-2": ["Cpxm2", "Ctgf", "Kazald1", "Tectb", "Fkbp9"],
    "KO-3": ["Cst3", "Gjb6", "Net1", "Tectb", "Tsen15"],
    "KO-4": ["Calb1", "Crabp1", "Epyc", "Itm2a", "Stmn2"],
    "IdC": ["Cdkn1c", "Fxyd6", "Otoa", "Ptgds", "Smoc2"],
    "HeC": ["Pmch", "Fst", "Nupr1", "Fam159b", "Egfl6"],
    "CC/OSC": ["Npnt", "Bmp4", "Fbln2", "Fst", "Apoe"],
    "Exclude": ["Oc90", "Otx2", "Myo6", "Myo7a", "Pecam1", "Kdr", "Ptprc", "Lyz2", "Sox10", "Mbp", "Mpz", "Snap25", "Tubb3"],
}

refine_force_genes = sorted({
    gene
    for genes in refine_z_score_genes.values()
    for gene in genes
    if gene in adata_refine.var_names
})
assert len(refine_force_genes) > 0, "No diagnostic marker genes found in adata_refine."

sc.pp.highly_variable_genes(
    adata_refine,
    flavor="seurat",
    n_top_genes=3000,
    batch_key="sample",
)

exclude_from_hvg = (
    adata_refine.var_names.str.startswith(("mt-", "Mt-", "MT-"))
    | adata_refine.var_names.str.startswith(("Rps", "Rpl"))
    | adata_refine.var_names.str.startswith(("Hba", "Hbb"))
    | adata_refine.var_names.isin([
        "Oc90", "Otx2",
        "Myo6", "Myo7a", "Cib2", "Pvalb", "Pou4f3", "Atoh1", "Ccer2", "Pcp4",
        "Pecam1", "Kdr", "Cdh5", "Ptprc", "Lyz2", "C1qa", "C1qb",
        "Sox10", "Mbp", "Mpz", "Plp1",
        "Snap25", "Tubb3", "Elavl3",
    ])
)
adata_refine.var.loc[exclude_from_hvg, "highly_variable"] = False
adata_refine.var.loc[refine_force_genes, "highly_variable"] = True

adata_refine_hvg = adata_refine[:, adata_refine.var["highly_variable"]].copy()
n_refine_pcs = min(50, adata_refine_hvg.n_obs - 1, adata_refine_hvg.n_vars - 1)
assert n_refine_pcs >= 2, "Too few cells or HVGs for refinement PCA."

sc.pp.scale(adata_refine_hvg, max_value=10)
sc.tl.pca(adata_refine_hvg, svd_solver="arpack", n_comps=n_refine_pcs)

logging.getLogger("harmonypy").setLevel(logging.ERROR)
logging.getLogger("harmonypy.harmony").setLevel(logging.ERROR)

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    refine_harmony = hm.run_harmony(
        adata_refine_hvg.obsm["X_pca"],
        adata_refine_hvg.obs,
        vars_use=["sample"],
        max_iter_harmony=20,
    )

Z_corr = refine_harmony.Z_corr
adata_refine_hvg.obsm["X_pca_harmony"] = (
    Z_corr if Z_corr.shape[0] == adata_refine_hvg.n_obs else Z_corr.T
)


def marker_z_matrix(adata, genes):
    genes = [gene for gene in genes if gene in adata.var_names]
    X = adata[:, genes].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=float)
    sd = X.std(axis=0)
    sd[sd == 0] = 1
    return (X - X.mean(axis=0)) / sd

X_marker_z = marker_z_matrix(adata_refine, refine_force_genes)
adata_refine_hvg.obsm["X_marker_guided"] = np.concatenate(
    [
        adata_refine_hvg.obsm["X_pca_harmony"][:, :min(25, n_refine_pcs)],
        X_marker_z * 1.5,
    ],
    axis=1,
)

sc.pp.neighbors(
    adata_refine_hvg,
    n_neighbors=12,
    use_rep="X_marker_guided",
)
sc.tl.umap(adata_refine_hvg, random_state=0)

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    sc.tl.leiden(
        adata_refine_hvg,
        resolution=1.2,
        key_added="refine_leiden",
        random_state=0,
    )

adata_refine.obs["refine_leiden"] = adata_refine_hvg.obs["refine_leiden"].astype(str).values
adata_refine.obsm["X_umap_refine"] = adata_refine_hvg.obsm["X_umap"].copy()


#3. ===========Refinement z-score assignment

for group, genes in refine_z_score_genes.items():
    genes = [gene for gene in genes if gene in adata_refine.var_names]
    if len(genes) > 0:
        adata_refine.obs[f"{group}_z_score"] = marker_z_matrix(adata_refine, genes).mean(axis=1)
        adata_refine.obs[f"{group}_raw_score"] = mean_marker_expr(adata_refine, genes)

refine_z_score_cols = [col for col in adata_refine.obs.columns if col.endswith("_z_score")]
refine_raw_score_cols = [col for col in adata_refine.obs.columns if col.endswith("_raw_score")]

refine_score_summary = (
    adata_refine.obs
    .groupby("refine_leiden", observed=True)[refine_z_score_cols]
    .mean()
    .round(3)
)
refine_score_summary["n_cells"] = adata_refine.obs["refine_leiden"].value_counts().reindex(refine_score_summary.index)
refine_score_summary["top_call"] = (
    refine_score_summary[refine_z_score_cols]
    .idxmax(axis=1)
    .str.replace("_z_score", "", regex=False)
)
refine_score_summary["score_margin"] = refine_score_summary[refine_z_score_cols].apply(
    lambda row: row.nlargest(2).iloc[0] - row.nlargest(2).iloc[1],
    axis=1,
).round(3)

refine_score_for_order = refine_score_summary.copy()
refine_score_for_order["call_rank"] = refine_score_for_order["top_call"].map(
    {celltype: i for i, celltype in enumerate(p1_excel_celltype_order)}
).fillna(99)
refine_score_for_order["cluster_rank"] = refine_score_for_order.index.astype(str).astype(int)
refine_order = refine_score_for_order.sort_values(["call_rank", "cluster_rank"]).index.astype(str).tolist()

adata_refine.obs["refine_leiden"] = pd.Categorical(
    adata_refine.obs["refine_leiden"].astype(str),
    categories=refine_order,
    ordered=True,
)

refine_counts = adata_refine.obs["refine_leiden"].value_counts().reindex(refine_order)
refine_label_map = {cluster: f"{cluster} (n={int(refine_counts.loc[cluster])})" for cluster in refine_order}
adata_refine.obs["refine_leiden_label"] = pd.Categorical(
    adata_refine.obs["refine_leiden"].astype(str).map(refine_label_map),
    categories=[refine_label_map[cluster] for cluster in refine_order],
    ordered=True,
)

refine_call_map = refine_score_summary["top_call"].astype(str).to_dict()
adata_refine.obs["refine_excel_call"] = adata_refine.obs["refine_leiden"].astype(str).map(refine_call_map)
adata_refine.obs["refine_excel_call"] = pd.Categorical(
    adata_refine.obs["refine_excel_call"].astype(str),
    categories=[x for x in p1_excel_celltype_order if x in adata_refine.obs["refine_excel_call"].unique()],
    ordered=True,
)

display(refine_score_summary.loc[refine_order, refine_z_score_cols + ["n_cells", "top_call", "score_margin"]])
display(pd.crosstab(adata_refine.obs["refine_excel_call"], adata_refine.obs["sample"], margins=True))
display(pd.crosstab(adata_refine.obs["refine_excel_call"], adata_refine.obs["parent_support_leiden"], margins=True))


#4. ===========Refinement UMAP and dotplot

sc.pl.embedding(
    adata_refine,
    basis="umap_refine",
    color=["refine_leiden_label", "refine_excel_call", "sample", "parent_support_leiden"],
    ncols=4,
    size=5,
    frameon=False,
    legend_loc="right margin",
    wspace=0.35,
    show=False,
)

sc.pl.embedding(
    adata_refine,
    basis="umap_refine",
    color="refine_leiden_label",
    size=5,
    frameon=False,
    legend_loc="on data",
    legend_fontoutline=2,
    show=False,
)

refine_dotplot_marker_dict = {
    group: [gene for gene in genes if gene in adata_refine.var_names]
    for group, genes in refine_dotplot_marker_dict.items()
}
refine_dotplot_marker_dict = {
    group: genes for group, genes in refine_dotplot_marker_dict.items()
    if len(genes) > 0
}

sc.pl.dotplot(
    adata_refine,
    var_names=refine_dotplot_marker_dict,
    groupby="refine_leiden",
    categories_order=refine_order,
    standard_scale=None,
    cmap="Reds",
    dendrogram=False,
    swap_axes=False,
    figsize=(18, 5.5),
    show=False,
)


plt.close("all")


# Notebook cell 13

#1. ===========Merge refined calls into final P1 labels

# Purpose:
# Combine the corrected HC split, IPC/DC branch, stable HeC/CC/OSC branch,
# and the marker-guided refinement branch into one final P1 label.
#
# Method:
# The marker-guided refine result has priority and overwrites earlier broad support calls.
# Low-margin clusters are still assigned, but marked by confidence.
#
# Parameters:
# high confidence: margin >= 0.75
# medium confidence: 0.25 <= margin < 0.75
# low confidence: margin < 0.25

def confidence_from_margin(margin):
    if pd.isna(margin):
        return "high"
    if margin >= 0.75:
        return "high"
    if margin >= 0.25:
        return "medium"
    return "low"


refine_final_map = {
    str(k): str(v)
    for k, v in refine_score_summary["top_call"].items()
}
refine_margin_map = {
    str(k): float(v)
    for k, v in refine_score_summary["score_margin"].items()
}

adata_refine.obs["refine_final_celltype"] = (
    adata_refine.obs["refine_leiden"].astype(str).map(refine_final_map)
)
adata_refine.obs["refine_final_margin"] = (
    adata_refine.obs["refine_leiden"].astype(str).map(refine_margin_map)
)
adata_refine.obs["refine_final_confidence"] = (
    adata_refine.obs["refine_final_margin"].map(confidence_from_margin)
)

assert adata_refine.obs["refine_final_celltype"].notna().all()
assert adata_refine.obs["refine_final_margin"].notna().all()

adata_clean.obs["P1_final_celltype"] = "Unassigned"
adata_clean.obs["P1_final_confidence"] = "unassigned"
adata_clean.obs["P1_final_score_margin"] = np.nan
adata_clean.obs["P1_final_source"] = "unassigned"

adata_clean.obs.loc[adata_hc.obs_names, "P1_final_celltype"] = adata_hc.obs["hc_celltype"].astype(str).values
adata_clean.obs.loc[adata_hc.obs_names, "P1_final_confidence"] = "high"
adata_clean.obs.loc[adata_hc.obs_names, "P1_final_source"] = "HC_branch"

ipc_final_map = {
    str(k): str(v)
    for k, v in ipc_opc_dc_excel_score_summary["top_call"].items()
}
ipc_margin_map = {
    str(k): float(v)
    for k, v in ipc_opc_dc_excel_score_summary["score_margin"].items()
}

ipc_cells = adata_ipc_opc_dc.obs_names
ipc_leiden = adata_ipc_opc_dc.obs["ipc_opc_dc_leiden"].astype(str)

adata_clean.obs.loc[ipc_cells, "P1_final_celltype"] = ipc_leiden.map(ipc_final_map).values
adata_clean.obs.loc[ipc_cells, "P1_final_score_margin"] = ipc_leiden.map(ipc_margin_map).values
adata_clean.obs.loc[ipc_cells, "P1_final_confidence"] = [
    confidence_from_margin(x) for x in ipc_leiden.map(ipc_margin_map).values
]
adata_clean.obs.loc[ipc_cells, "P1_final_source"] = "IPC_OPC_DC_branch"

hec_final_map = {
    str(k): str(v)
    for k, v in hec_cc_osc_excel_score_summary["top_call"].items()
}
hec_margin_map = {
    str(k): float(v)
    for k, v in hec_cc_osc_excel_score_summary["score_margin"].items()
}

hec_cells = adata_hec_cc_osc.obs_names
hec_leiden = adata_hec_cc_osc.obs["hec_cc_osc_leiden"].astype(str)

adata_clean.obs.loc[hec_cells, "P1_final_celltype"] = hec_leiden.map(hec_final_map).values
adata_clean.obs.loc[hec_cells, "P1_final_score_margin"] = hec_leiden.map(hec_margin_map).values
adata_clean.obs.loc[hec_cells, "P1_final_confidence"] = [
    confidence_from_margin(x) for x in hec_leiden.map(hec_margin_map).values
]
adata_clean.obs.loc[hec_cells, "P1_final_source"] = "HeC_CC_OSC_branch"

refine_cells = adata_refine.obs_names

adata_clean.obs.loc[refine_cells, "P1_final_celltype"] = (
    adata_refine.obs["refine_final_celltype"].astype(str).values
)
adata_clean.obs.loc[refine_cells, "P1_final_score_margin"] = (
    adata_refine.obs["refine_final_margin"].astype(float).values
)
adata_clean.obs.loc[refine_cells, "P1_final_confidence"] = (
    adata_refine.obs["refine_final_confidence"].astype(str).values
)
adata_clean.obs.loc[refine_cells, "P1_final_source"] = "marker_guided_refine"

assert adata_clean.obs["P1_final_celltype"].notna().all()
assert not (adata_clean.obs["P1_final_celltype"] == "Unassigned").any()
assert set(adata_clean.obs["P1_final_celltype"].astype(str)).issubset(set(p1_excel_celltype_order))

p1_final_celltype_order = [
    x for x in p1_excel_celltype_order
    if x in adata_clean.obs["P1_final_celltype"].astype(str).unique()
]

adata_clean.obs["P1_final_celltype"] = pd.Categorical(
    adata_clean.obs["P1_final_celltype"].astype(str),
    categories=p1_final_celltype_order,
    ordered=True,
)

display(pd.crosstab(adata_clean.obs["P1_final_celltype"], adata_clean.obs["sample"], margins=True))
display(pd.crosstab(adata_clean.obs["P1_final_celltype"], adata_clean.obs["P1_final_confidence"], margins=True))
display(pd.crosstab(adata_clean.obs["P1_final_celltype"], adata_clean.obs["P1_final_source"], margins=True))


plt.close("all")


# Notebook cell 14

#3. ===========Plot final P1 labels

# Purpose:
# Visual check of the final P1 labels on UMAP and marker dotplot.
#
# Method:
# Use right-margin legend for count-readable UMAP, on-data labels for spatial inspection,
# and Excel marker groups for final dotplot.

p1_final_counts = adata_clean.obs["P1_final_celltype"].value_counts().reindex(p1_final_celltype_order)

p1_final_label_map = {
    celltype: f"{celltype} (n={int(p1_final_counts.loc[celltype])})"
    for celltype in p1_final_celltype_order
}

p1_final_label_order = [
    p1_final_label_map[celltype]
    for celltype in p1_final_celltype_order
]

adata_clean.obs["P1_final_celltype_label"] = (
    adata_clean.obs["P1_final_celltype"].astype(str).map(p1_final_label_map)
)

adata_clean.obs["P1_final_celltype_label"] = pd.Categorical(
    adata_clean.obs["P1_final_celltype_label"].astype(str),
    categories=p1_final_label_order,
    ordered=True,
)

p1_final_plot_marker_order = [
    "IHC", "OHC", "IPhC", "IPC", "OPC", "DC-1/2", "DC-3",
    "HeC", "CC/OSC", "ISC", "KO-1", "KO-2", "KO-3", "KO-4",
    "IdC", "Exclude",
]

p1_final_plot_marker_dict = {
    group: p1_final_audit_marker_dict[group]
    for group in p1_final_plot_marker_order
    if group in p1_final_audit_marker_dict
}

sc.pl.embedding(
    adata_clean,
    basis="umap_harmony",
    color=["P1_final_celltype_label", "sample"],
    ncols=2,
    size=5,
    frameon=False,
    legend_loc="right margin",
    wspace=0.35,
    show=False,
)

sc.pl.embedding(
    adata_clean,
    basis="umap_harmony",
    color="P1_final_celltype_label",
    size=5,
    frameon=False,
    legend_loc="on data",
    legend_fontoutline=2,
    show=False,
)

sc.pl.dotplot(
    adata_clean,
    var_names=p1_final_plot_marker_dict,
    groupby="P1_final_celltype",
    categories_order=p1_final_celltype_order,
    standard_scale=None,
    cmap="Reds",
    dendrogram=False,
    swap_axes=False,
    figsize=(24, 7),
    show=False,
)


plt.close("all")


# Notebook cell 15

#1. ===========Add core epithelial markers to final dotplot

# Purpose:
# Add broad epithelial/support markers to the final P1 dotplot for QC.
#
# Method:
# Put Epcam, Gata3, Sox2, Sox9, Lgr5, and Isl1 as a Core marker group
# before the cell-type-specific Excel marker groups.

core_marker_genes = ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"]

missing_core_markers = [
    gene for gene in core_marker_genes
    if gene not in adata_clean.var_names
]

assert len(missing_core_markers) == 0, f"Missing core markers: {missing_core_markers}"

p1_final_plot_marker_dict_with_core = {
    "Core": core_marker_genes,
    **p1_final_plot_marker_dict,
}

sc.pl.dotplot(
    adata_clean,
    var_names=p1_final_plot_marker_dict_with_core,
    groupby="P1_final_celltype",
    categories_order=p1_final_celltype_order,
    standard_scale=None,
    cmap="Reds",
    dendrogram=False,
    swap_axes=False,
    figsize=(26, 7),
    show=False,
)


plt.close("all")


# Notebook cell 16

#1. ===========P1 OPC-search setup

np.random.seed(0)
sc.settings.verbosity = 0

p1_opc_random_state = 0
p1_opc_n_top_genes = 1200
p1_opc_n_neighbors = 8
p1_opc_resolution = 0.85
p1_opc_marker_weight = 2.0

adata_p1_opc_search = adata_ipc_opc_dc.copy()
adata_p1_opc_search.obs["parent_ipc_opc_dc_leiden"] = (
    adata_p1_opc_search.obs["ipc_opc_dc_leiden"].astype(str)
)
adata_p1_opc_search.obs["P1_final_celltype"] = (
    adata_clean.obs.loc[adata_p1_opc_search.obs_names, "P1_final_celltype"]
    .astype(str)
    .values
)

p1_opc_core_score_genes = {
    "IPC": ["Npy", "Emid1", "Cryab", "Igfbpl1", "Matn4"],
    "OPC": ["Fam159b", "Smagp"],
    "DC-1/2": ["Hes5", "Pdzk1ip1", "S100a1"],
    "DC-3": ["Lgr5", "Fgf3", "Lfng"],
    "IPhC": ["Anxa5", "Fabp7", "Prss23"],
}

p1_opc_dotplot_marker_dict = {
    "IPC": ["Npy", "Emid1", "Cryab", "Igfbpl1", "Matn4", "S100b"],
    "OPC core": ["Fam159b", "Smagp"],
    "OPC shared": ["Serpine2", "Ppp1r2", "S100b"],
    "DC-1/2 core": ["Hes5", "Pdzk1ip1", "S100a1"],
    "DC-3": ["Lgr5", "Fgf3", "Lfng", "Prss23"],
    "IPhC": ["Anxa5", "Matn4", "Gjb2", "Fabp7", "Prss23"],
    "Exclude": ["Oc90", "Otx2", "Myo6", "Myo7a", "Pecam1", "Kdr", "Ptprc", "Lyz2", "Sox10", "Mbp", "Mpz", "Snap25", "Tubb3"],
}

p1_opc_dotplot_marker_dict = {
    group: [gene for gene in genes if gene in adata_p1_opc_search.var_names]
    for group, genes in p1_opc_dotplot_marker_dict.items()
}
p1_opc_dotplot_marker_dict = {
    group: genes for group, genes in p1_opc_dotplot_marker_dict.items()
    if len(genes) > 0
}

p1_opc_force_genes = sorted({
    gene
    for genes in p1_opc_dotplot_marker_dict.values()
    for gene in genes
    if gene in adata_p1_opc_search.var_names
})

p1_opc_core_force_genes = sorted({
    gene
    for genes in p1_opc_core_score_genes.values()
    for gene in genes
    if gene in adata_p1_opc_search.var_names
})

assert len(p1_opc_core_force_genes) > 0, "No OPC-search core marker genes found."


#2. ===========Marker-guided reclustering

def p1_opc_marker_z_matrix(adata_obj, genes):
    genes = [gene for gene in genes if gene in adata_obj.var_names]
    X = adata_obj[:, genes].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=float)
    sd = X.std(axis=0)
    sd[sd == 0] = 1
    return (X - X.mean(axis=0)) / sd


sc.pp.highly_variable_genes(
    adata_p1_opc_search,
    flavor="seurat",
    n_top_genes=p1_opc_n_top_genes,
)

p1_opc_exclude_from_hvg = (
    adata_p1_opc_search.var_names.str.startswith(("mt-", "Mt-", "MT-"))
    | adata_p1_opc_search.var_names.str.startswith(("Rps", "Rpl"))
    | adata_p1_opc_search.var_names.str.startswith(("Hba", "Hbb"))
    | adata_p1_opc_search.var_names.isin([
        "Oc90", "Otx2",
        "Myo6", "Myo7a", "Cib2", "Pvalb", "Pou4f3", "Atoh1", "Ccer2", "Pcp4",
        "Pecam1", "Kdr", "Cdh5", "Ptprc", "Lyz2", "C1qa", "C1qb",
        "Sox10", "Mbp", "Mpz", "Plp1",
        "Snap25", "Tubb3", "Elavl3",
    ])
)
adata_p1_opc_search.var.loc[p1_opc_exclude_from_hvg, "highly_variable"] = False
adata_p1_opc_search.var.loc[p1_opc_force_genes, "highly_variable"] = True

adata_p1_opc_search_hvg = adata_p1_opc_search[:, adata_p1_opc_search.var["highly_variable"]].copy()
p1_opc_n_pcs = min(30, adata_p1_opc_search_hvg.n_obs - 1, adata_p1_opc_search_hvg.n_vars - 1)
assert p1_opc_n_pcs >= 2, "Too few cells or HVGs for P1 OPC-search PCA."

sc.pp.scale(adata_p1_opc_search_hvg, max_value=10)
sc.tl.pca(adata_p1_opc_search_hvg, svd_solver="arpack", n_comps=p1_opc_n_pcs)

logging.getLogger("harmonypy").setLevel(logging.ERROR)
logging.getLogger("harmonypy.harmony").setLevel(logging.ERROR)

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    p1_opc_harmony = hm.run_harmony(
        adata_p1_opc_search_hvg.obsm["X_pca"],
        adata_p1_opc_search_hvg.obs,
        vars_use=["sample"],
        max_iter_harmony=20,
    )

Z_corr = p1_opc_harmony.Z_corr
adata_p1_opc_search_hvg.obsm["X_pca_harmony"] = (
    Z_corr if Z_corr.shape[0] == adata_p1_opc_search_hvg.n_obs else Z_corr.T
)

p1_opc_marker_z = p1_opc_marker_z_matrix(adata_p1_opc_search, p1_opc_core_force_genes)
adata_p1_opc_search_hvg.obsm["X_p1_opc_guided"] = np.concatenate(
    [
        adata_p1_opc_search_hvg.obsm["X_pca_harmony"][:, :min(20, p1_opc_n_pcs)],
        p1_opc_marker_z * p1_opc_marker_weight,
    ],
    axis=1,
)

sc.pp.neighbors(
    adata_p1_opc_search_hvg,
    n_neighbors=p1_opc_n_neighbors,
    use_rep="X_p1_opc_guided",
)
sc.tl.umap(adata_p1_opc_search_hvg, random_state=p1_opc_random_state)

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    sc.tl.leiden(
        adata_p1_opc_search_hvg,
        resolution=p1_opc_resolution,
        key_added="p1_opc_search_leiden",
        random_state=p1_opc_random_state,
    )

adata_p1_opc_search.obs["p1_opc_search_leiden"] = (
    adata_p1_opc_search_hvg.obs["p1_opc_search_leiden"].astype(str).values
)
adata_p1_opc_search.obsm["X_umap_p1_opc_search"] = (
    adata_p1_opc_search_hvg.obsm["X_umap"].copy()
)


#3. ===========OPC-search score summary

for group, genes in p1_opc_core_score_genes.items():
    genes = [gene for gene in genes if gene in adata_p1_opc_search.var_names]
    if len(genes) > 0:
        adata_p1_opc_search.obs[f"{group}_core_z_score"] = (
            p1_opc_marker_z_matrix(adata_p1_opc_search, genes).mean(axis=1)
        )
        adata_p1_opc_search.obs[f"{group}_core_score"] = (
            mean_marker_expr(adata_p1_opc_search, genes)
        )

p1_opc_core_score_cols = [
    f"{group}_core_z_score"
    for group in p1_opc_core_score_genes
    if f"{group}_core_z_score" in adata_p1_opc_search.obs.columns
]

p1_opc_search_summary = (
    adata_p1_opc_search.obs
    .groupby("p1_opc_search_leiden", observed=True)[p1_opc_core_score_cols]
    .mean()
    .round(3)
)
p1_opc_search_summary["n_cells"] = (
    adata_p1_opc_search.obs["p1_opc_search_leiden"]
    .value_counts()
    .reindex(p1_opc_search_summary.index)
)
p1_opc_search_summary["top_call"] = (
    p1_opc_search_summary[p1_opc_core_score_cols]
    .idxmax(axis=1)
    .str.replace("_core_z_score", "", regex=False)
)
p1_opc_search_summary["score_margin"] = p1_opc_search_summary[p1_opc_core_score_cols].apply(
    lambda row: row.nlargest(2).iloc[0] - row.nlargest(2).iloc[1],
    axis=1,
).round(3)

p1_opc_search_summary["OPC_minus_DC12"] = (
    p1_opc_search_summary["OPC_core_z_score"]
    - p1_opc_search_summary["DC-1/2_core_z_score"]
).round(3)
p1_opc_search_summary["OPC_minus_IPC"] = (
    p1_opc_search_summary["OPC_core_z_score"]
    - p1_opc_search_summary["IPC_core_z_score"]
).round(3)
p1_opc_search_summary["OPC_minus_DC3"] = (
    p1_opc_search_summary["OPC_core_z_score"]
    - p1_opc_search_summary["DC-3_core_z_score"]
).round(3)

p1_opc_search_summary["opc_candidate"] = np.where(
    (p1_opc_search_summary["OPC_minus_DC12"] > 0)
    & (p1_opc_search_summary["OPC_minus_IPC"] > 0)
    & (p1_opc_search_summary["OPC_minus_DC3"] > 0),
    "check_OPC",
    "",
)

p1_opc_call_rank = {"IPC": 0, "OPC": 1, "DC-1/2": 2, "DC-3": 3, "IPhC": 4}
p1_opc_score_for_order = p1_opc_search_summary.copy()
p1_opc_score_for_order["call_rank"] = (
    p1_opc_score_for_order["top_call"].map(p1_opc_call_rank).fillna(9)
)
p1_opc_score_for_order["cluster_rank"] = p1_opc_score_for_order.index.astype(str).astype(int)
p1_opc_search_order = (
    p1_opc_score_for_order
    .sort_values(["call_rank", "cluster_rank"])
    .index.astype(str)
    .tolist()
)

adata_p1_opc_search.obs["p1_opc_search_leiden"] = pd.Categorical(
    adata_p1_opc_search.obs["p1_opc_search_leiden"].astype(str),
    categories=p1_opc_search_order,
    ordered=True,
)

p1_opc_search_counts = (
    adata_p1_opc_search.obs["p1_opc_search_leiden"]
    .value_counts()
    .reindex(p1_opc_search_order)
)
p1_opc_search_label_map = {
    cluster: (
        f"{cluster}:{p1_opc_search_summary.loc[cluster, 'top_call']} "
        f"(n={int(p1_opc_search_counts.loc[cluster])})"
    )
    for cluster in p1_opc_search_order
}
p1_opc_search_label_order = [
    p1_opc_search_label_map[cluster]
    for cluster in p1_opc_search_order
]

adata_p1_opc_search.obs["p1_opc_search_call"] = (
    adata_p1_opc_search.obs["p1_opc_search_leiden"]
    .astype(str)
    .map(p1_opc_search_summary["top_call"].astype(str).to_dict())
)
adata_p1_opc_search.obs["p1_opc_search_label"] = pd.Categorical(
    adata_p1_opc_search.obs["p1_opc_search_leiden"].astype(str).map(p1_opc_search_label_map),
    categories=p1_opc_search_label_order,
    ordered=True,
)

p1_opc_show_cols = (
    p1_opc_core_score_cols
    + ["n_cells", "top_call", "score_margin", "OPC_minus_DC12", "OPC_minus_IPC", "OPC_minus_DC3", "opc_candidate"]
)

display(p1_opc_search_summary.loc[p1_opc_search_order, p1_opc_show_cols])
display(pd.crosstab(adata_p1_opc_search.obs["p1_opc_search_label"], adata_p1_opc_search.obs["sample"], margins=True))


#4. ===========OPC-search UMAP

sc.pl.embedding(
    adata_p1_opc_search,
    basis="umap_p1_opc_search",
    color="p1_opc_search_label",
    size=16,
    frameon=False,
    legend_loc="right margin",
    show=False,
)
plt.gcf().set_size_inches(5.8, 4.2)
plt.close("all")
plt.close("all")

sc.pl.embedding(
    adata_p1_opc_search,
    basis="umap_p1_opc_search",
    color="p1_opc_search_label",
    size=16,
    frameon=False,
    legend_loc="on data",
    legend_fontoutline=2,
    show=False,
)
plt.gcf().set_size_inches(4.8, 4.2)
plt.close("all")
plt.close("all")


#5. ===========OPC-search dotplot

sc.pl.dotplot(
    adata_p1_opc_search,
    var_names=p1_opc_dotplot_marker_dict,
    groupby="p1_opc_search_label",
    categories_order=p1_opc_search_label_order,
    standard_scale=None,
    cmap="Reds",
    dendrogram=False,
    swap_axes=False,
    figsize=(14, 4.8),
    show=False,
)


plt.close("all")


# Notebook cell 17

#1. ===========Merge OPC-search result into final P1 labels

p1_opc_search_final_map = p1_opc_search_summary["top_call"].astype(str).to_dict()
p1_opc_search_margin_map = p1_opc_search_summary["score_margin"].astype(float).to_dict()

adata_clean.obs["P1_final_celltype_v2"] = adata_clean.obs["P1_final_celltype"].astype(str)
adata_clean.obs["P1_final_confidence_v2"] = adata_clean.obs["P1_final_confidence"].astype(str)

if "P1_final_source" in adata_clean.obs.columns:
    adata_clean.obs["P1_final_source_v2"] = adata_clean.obs["P1_final_source"].astype(str)
else:
    adata_clean.obs["P1_final_source_v2"] = "previous_P1_final"

p1_opc_search_cells = adata_p1_opc_search.obs_names
p1_opc_search_leiden = adata_p1_opc_search.obs["p1_opc_search_leiden"].astype(str)

adata_clean.obs.loc[p1_opc_search_cells, "P1_final_celltype_v2"] = (
    p1_opc_search_leiden.map(p1_opc_search_final_map).values
)
adata_clean.obs.loc[p1_opc_search_cells, "P1_final_score_margin_v2"] = (
    p1_opc_search_leiden.map(p1_opc_search_margin_map).values
)
adata_clean.obs.loc[p1_opc_search_cells, "P1_final_source_v2"] = "P1_OPC_search"

adata_clean.obs.loc[p1_opc_search_cells, "P1_final_confidence_v2"] = [
    "high" if x >= 0.75 else "medium" if x >= 0.25 else "low"
    for x in p1_opc_search_leiden.map(p1_opc_search_margin_map).values
]

assert set(adata_clean.obs["P1_final_celltype_v2"].astype(str)).issubset(set(p1_excel_celltype_order))
assert (adata_clean.obs["P1_final_celltype_v2"].astype(str) == "OPC").sum() > 0

p1_final_celltype_order_v2 = [
    x for x in p1_excel_celltype_order
    if x in adata_clean.obs["P1_final_celltype_v2"].astype(str).unique()
]

adata_clean.obs["P1_final_celltype_v2"] = pd.Categorical(
    adata_clean.obs["P1_final_celltype_v2"].astype(str),
    categories=p1_final_celltype_order_v2,
    ordered=True,
)


#2. ===========Check final label changes

p1_opc_change_cells = adata_clean.obs_names[
    adata_clean.obs["P1_final_source_v2"].astype(str) == "P1_OPC_search"
]

display(pd.crosstab(
    adata_clean.obs["P1_final_celltype_v2"],
    adata_clean.obs["sample"],
    margins=True,
))

display(pd.crosstab(
    adata_clean.obs.loc[p1_opc_change_cells, "P1_final_celltype"],
    adata_clean.obs.loc[p1_opc_change_cells, "P1_final_celltype_v2"],
    margins=True,
))

display(pd.crosstab(
    adata_clean.obs["P1_final_celltype_v2"],
    adata_clean.obs["P1_final_confidence_v2"],
    margins=True,
))


#3. ===========P1 final v2 UMAP

p1_final_counts_v2 = (
    adata_clean.obs["P1_final_celltype_v2"]
    .value_counts()
    .reindex(p1_final_celltype_order_v2)
)

p1_final_label_map_v2 = {
    celltype: f"{celltype} (n={int(p1_final_counts_v2.loc[celltype])})"
    for celltype in p1_final_celltype_order_v2
}
p1_final_label_order_v2 = [
    p1_final_label_map_v2[celltype]
    for celltype in p1_final_celltype_order_v2
]

adata_clean.obs["P1_final_celltype_label_v2"] = pd.Categorical(
    adata_clean.obs["P1_final_celltype_v2"].astype(str).map(p1_final_label_map_v2),
    categories=p1_final_label_order_v2,
    ordered=True,
)

sc.pl.embedding(
    adata_clean,
    basis="umap_harmony",
    color="P1_final_celltype_label_v2",
    size=5,
    frameon=False,
    legend_loc="right margin",
    show=False,
)
plt.gcf().set_size_inches(7.2, 4.6)
plt.close("all")
plt.close("all")

sc.pl.embedding(
    adata_clean,
    basis="umap_harmony",
    color="P1_final_celltype_label_v2",
    size=5,
    frameon=False,
    legend_loc="on data",
    legend_fontoutline=2,
    show=False,
)
plt.gcf().set_size_inches(5.2, 4.6)
plt.close("all")
plt.close("all")


#4. ===========Focused IPC / OPC / DC dotplot

p1_ipc_opc_dc_order_v2 = [
    x for x in ["IPC", "OPC", "DC-1/2", "DC-3"]
    if x in adata_clean.obs["P1_final_celltype_v2"].astype(str).unique()
]

p1_ipc_opc_dc_marker_dict_v2 = {
    "IPC": ["Npy", "Emid1", "Cryab", "Igfbpl1", "Matn4", "S100b"],
    "OPC core": ["Fam159b", "Smagp"],
    "OPC shared": ["Serpine2", "Ppp1r2", "S100b"],
    "DC-1/2": ["Hes5", "Pdzk1ip1", "S100a1"],
    "DC-3": ["Lgr5", "Fgf3", "Lfng", "Prss23"],
    "IPhC": ["Anxa5", "Matn4", "Gjb2", "Fabp7", "Prss23"],
    "Exclude": ["Oc90", "Otx2", "Myo6", "Myo7a", "Pecam1", "Kdr", "Ptprc", "Lyz2", "Sox10", "Mbp", "Mpz", "Snap25", "Tubb3"],
}

p1_ipc_opc_dc_marker_dict_v2 = {
    group: [gene for gene in genes if gene in adata_clean.var_names]
    for group, genes in p1_ipc_opc_dc_marker_dict_v2.items()
}
p1_ipc_opc_dc_marker_dict_v2 = {
    group: genes for group, genes in p1_ipc_opc_dc_marker_dict_v2.items()
    if len(genes) > 0
}

sc.pl.dotplot(
    adata_clean[adata_clean.obs["P1_final_celltype_v2"].astype(str).isin(p1_ipc_opc_dc_order_v2)].copy(),
    var_names=p1_ipc_opc_dc_marker_dict_v2,
    groupby="P1_final_celltype_v2",
    categories_order=p1_ipc_opc_dc_order_v2,
    standard_scale=None,
    cmap="Reds",
    dendrogram=False,
    swap_axes=False,
    figsize=(11, 3.8),
    show=False,
)


plt.close("all")


# Notebook cell 18

#1. ===========Set P1 v2 labels as final labels

if "P1_final_celltype_before_OPC" not in adata_clean.obs.columns:
    adata_clean.obs["P1_final_celltype_before_OPC"] = adata_clean.obs["P1_final_celltype"].astype(str)
    adata_clean.obs["P1_final_confidence_before_OPC"] = adata_clean.obs["P1_final_confidence"].astype(str)

adata_clean.obs["P1_final_celltype"] = adata_clean.obs["P1_final_celltype_v2"].astype(str)
adata_clean.obs["P1_final_confidence"] = adata_clean.obs["P1_final_confidence_v2"].astype(str)
adata_clean.obs["P1_final_source"] = adata_clean.obs["P1_final_source_v2"].astype(str)

p1_final_celltype_order = [
    x for x in p1_excel_celltype_order
    if x in adata_clean.obs["P1_final_celltype"].astype(str).unique()
]

assert "OPC" in p1_final_celltype_order
assert set(p1_excel_celltype_order).issubset(set(p1_final_celltype_order))

adata_clean.obs["P1_final_celltype"] = pd.Categorical(
    adata_clean.obs["P1_final_celltype"].astype(str),
    categories=p1_final_celltype_order,
    ordered=True,
)

display(pd.crosstab(
    adata_clean.obs["P1_final_celltype"],
    adata_clean.obs["sample"],
    margins=True,
))

display(pd.crosstab(
    adata_clean.obs["P1_final_celltype"],
    adata_clean.obs["P1_final_confidence"],
    margins=True,
))


#2. ===========Final marker-score audit

for group, genes in p1_final_audit_marker_dict.items():
    adata_clean.obs[f"audit_v2_{group}_score"] = mean_marker_expr(adata_clean, genes)

audit_v2_score_cols = [f"audit_v2_{group}_score" for group in p1_final_audit_marker_dict]

p1_final_audit_summary_v2 = (
    adata_clean.obs
    .groupby("P1_final_celltype", observed=True)[audit_v2_score_cols]
    .mean()
    .round(3)
)
p1_final_audit_summary_v2["n_cells"] = (
    adata_clean.obs["P1_final_celltype"]
    .value_counts()
    .reindex(p1_final_audit_summary_v2.index)
)
p1_final_audit_summary_v2["top_call"] = (
    p1_final_audit_summary_v2[audit_v2_score_cols]
    .idxmax(axis=1)
    .str.replace("audit_v2_", "", regex=False)
    .str.replace("_score", "", regex=False)
)
p1_final_audit_summary_v2["score_margin"] = p1_final_audit_summary_v2[audit_v2_score_cols].apply(
    lambda row: row.nlargest(2).iloc[0] - row.nlargest(2).iloc[1],
    axis=1,
).round(3)

display(p1_final_audit_summary_v2[["n_cells", "top_call", "score_margin"]])


#3. ===========Final P1 UMAP

p1_final_counts = adata_clean.obs["P1_final_celltype"].value_counts().reindex(p1_final_celltype_order)
p1_final_label_map = {
    celltype: f"{celltype} (n={int(p1_final_counts.loc[celltype])})"
    for celltype in p1_final_celltype_order
}
p1_final_label_order = [p1_final_label_map[celltype] for celltype in p1_final_celltype_order]

adata_clean.obs["P1_final_celltype_label"] = pd.Categorical(
    adata_clean.obs["P1_final_celltype"].astype(str).map(p1_final_label_map),
    categories=p1_final_label_order,
    ordered=True,
)

sc.pl.embedding(
    adata_clean,
    basis="umap_harmony",
    color="P1_final_celltype_label",
    size=5,
    frameon=False,
    legend_loc="right margin",
    show=False,
)
plt.gcf().set_size_inches(7.2, 4.6)
plt.close("all")
plt.close("all")

sc.pl.embedding(
    adata_clean,
    basis="umap_harmony",
    color="P1_final_celltype_label",
    size=5,
    frameon=False,
    legend_loc="on data",
    legend_fontsize=8,
    legend_fontoutline=2,
    show=False,
)
plt.gcf().set_size_inches(5.2, 4.6)
plt.close("all")
plt.close("all")


#4. ===========Final P1 marker dotplot

core_marker_genes = ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"]

p1_final_plot_marker_dict = {
    "Core": [gene for gene in core_marker_genes if gene in adata_clean.var_names],
    **{
        group: p1_final_audit_marker_dict[group]
        for group in p1_final_celltype_order
        if group in p1_final_audit_marker_dict
    },
}

sc.pl.dotplot(
    adata_clean,
    var_names=p1_final_plot_marker_dict,
    groupby="P1_final_celltype",
    categories_order=p1_final_celltype_order,
    standard_scale=None,
    cmap="Reds",
    dendrogram=False,
    swap_axes=False,
    figsize=(25, 7),
    show=False,
)


plt.close("all")


# Notebook cell 19

# Notebook note omitted.

p1_fig_dir = OUTPUT_DIR
p1_harmony_dir = RESULTS_ROOT / "01_stage_clustering" / "harmony_inputs" / "p1"
p1_scanvi_dir = STAGE_OBJECT_DIR

p1_fig_dir.mkdir(parents=True, exist_ok=True)
p1_harmony_dir.mkdir(parents=True, exist_ok=True)
p1_scanvi_dir.mkdir(parents=True, exist_ok=True)

p1_label_key = "P1_final_celltype"
p1_label_n_key = "P1_final_celltype_n"

p1_celltype_order = [
    "IHC", "OHC", "IPhC", "IPC", "OPC", "DC-1/2", "DC-3", "HeC",
    "CC/OSC", "ISC", "KO-1", "KO-2", "KO-3", "KO-4", "IdC"
]

if p1_label_key not in adata_clean.obs.columns:
    raise KeyError(f"{p1_label_key} not found in adata_clean.obs")

adata_P1_final_save = adata_clean.copy()
adata_P1_final_save.obs[p1_label_key] = pd.Categorical(
    adata_P1_final_save.obs[p1_label_key].astype(str),
    categories=p1_celltype_order,
    ordered=True
)

p1_counts = adata_P1_final_save.obs[p1_label_key].value_counts().reindex(p1_celltype_order).fillna(0).astype(int)
p1_label_n_map = {ct: f"{ct} (n={p1_counts.loc[ct]})" for ct in p1_celltype_order}
adata_P1_final_save.obs[p1_label_n_key] = adata_P1_final_save.obs[p1_label_key].map(p1_label_n_map).astype(str)
adata_P1_final_save.obs[p1_label_n_key] = pd.Categorical(
    adata_P1_final_save.obs[p1_label_n_key],
    categories=[p1_label_n_map[ct] for ct in p1_celltype_order],
    ordered=True
)

if "X_umap_harmony" in adata_P1_final_save.obsm:
    adata_P1_final_save.obsm["X_umap"] = adata_P1_final_save.obsm["X_umap_harmony"].copy()

p1_dot_marker_dict = {
    "Core": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "IHC": ["Fgf8", "Atp2a3", "Cabp2", "Tbx2", "Kcnj13", "Fam19a3", "Shtn1"],
    "OHC": ["Calca", "Serpina1c", "Veph1", "Strip2", "Msln1", "Pcp4", "Calb1"],
    "IPhC": ["Anxa5", "Matn4", "Gjb2", "Fabp7", "Prss23"],
    "IPC": ["Npy", "Emid1", "Cryab", "Igfbpl1", "Matn4", "S100b"],
    "OPC": ["Fam159b", "Smagp", "Serpine2", "Ppp1r2", "S100b"],
    "DC-1/2": ["Hes5", "Pdzk1ip1", "S100a1", "Serpine2", "Ppp1r2"],
    "DC-3": ["Lgr5", "Fgf3", "Hes5", "Igfbpl1", "Lfng", "Prss23", "S100a1"],
    "HeC": ["Pmch", "Fst", "Nupr1", "Fam159b", "Egfl6"],
    "CC/OSC": ["Npnt", "Bmp4", "Fbln2", "Fst", "Apoe"],
    "ISC": ["Igf1", "Matn1", "Meg3", "Rgcc", "Tm4sf1"],
    "KO-1": ["Dcn", "Rcn3", "Pdia6", "Sdf2l1", "Ddost"],
    "KO-2": ["Cpxm2", "Ctgf", "Kazald1", "Tectb", "Fkbp9"],
    "KO-3": ["Cst3", "Gjb6", "Net1", "Tectb", "Tsen15"],
    "KO-4": ["Calb1", "Crabp1", "Epyc", "Itm2a", "Stmn2"],
    "IdC": ["Cdkn1c", "Fxyd6", "Otoa", "Ptgds", "Smoc2"],
    "Exclude": ["Oc90", "Otx2", "Myo6", "Myo7a", "Pecam1", "Kdr", "Ptprc", "Lyz2", "Sox10", "Mbp", "Mpz", "Snap25", "Tubb3"]
}
p1_dot_marker_dict = {
    group: [gene for gene in genes if gene in adata_P1_final_save.var_names]
    for group, genes in p1_dot_marker_dict.items()
}
p1_dot_marker_dict = {group: genes for group, genes in p1_dot_marker_dict.items() if len(genes) > 0}

p1_dot_marker_part1 = {
    group: p1_dot_marker_dict[group]
    for group in ["Core", "IHC", "OHC", "IPhC", "IPC", "OPC"]
    if group in p1_dot_marker_dict
}
p1_dot_marker_part2 = {
    group: p1_dot_marker_dict[group]
    for group in ["DC-1/2", "DC-3", "HeC", "CC/OSC", "ISC", "KO-1", "KO-2", "KO-3", "KO-4", "IdC", "Exclude"]
    if group in p1_dot_marker_dict
}

dot_kwargs = {}
if "plot_expr" in adata_P1_final_save.layers:
    dot_kwargs["layer"] = "plot_expr"

sc.set_figure_params(dpi=100, fontsize=10)

fig = sc.pl.umap(
    adata_P1_final_save,
    color=[p1_label_n_key, "sample"],
    size=8,
    ncols=2,
    wspace=0.45,
    frameon=True,
    show=False,
    return_fig=True
)
fig.savefig(p1_fig_dir / "P1_final_sensory_epithelium_umap.png", dpi=SAVE_DPI, bbox_inches="tight")
fig.clf()

fig = sc.pl.umap(
    adata_P1_final_save,
    color=p1_label_n_key,
    legend_loc="on data",
    legend_fontsize=9,
    legend_fontoutline=2,
    size=8,
    frameon=True,
    show=False,
    return_fig=True
)
fig.savefig(p1_fig_dir / "P1_final_umap.png", dpi=SAVE_DPI, bbox_inches="tight")
fig.clf()

dp = sc.pl.dotplot(
    adata_P1_final_save,
    var_names=p1_dot_marker_dict,
    groupby=p1_label_key,
    categories_order=p1_celltype_order,
    standard_scale="var",
    dot_min=0,
    dot_max=1,
    dendrogram=False,
    figsize=(24, 7),
    show=False,
    return_fig=True,
    **dot_kwargs
)
dp.savefig(p1_fig_dir / "P1_final_sensory_epithelium_dotplot.png", dpi=SAVE_DPI)
dp.savefig(p1_fig_dir / "P1_final_dotplot.png", dpi=SAVE_DPI)

dp = sc.pl.dotplot(
    adata_P1_final_save,
    var_names=p1_dot_marker_part1,
    groupby=p1_label_key,
    categories_order=p1_celltype_order,
    standard_scale="var",
    dot_min=0,
    dot_max=1,
    dendrogram=False,
    figsize=(16, 7),
    show=False,
    return_fig=True,
    **dot_kwargs
)
dp.savefig(p1_fig_dir / "P1_final_sensory_epithelium_dotplot_part1.png", dpi=SAVE_DPI)

dp = sc.pl.dotplot(
    adata_P1_final_save,
    var_names=p1_dot_marker_part2,
    groupby=p1_label_key,
    categories_order=p1_celltype_order,
    standard_scale="var",
    dot_min=0,
    dot_max=1,
    dendrogram=False,
    figsize=(18, 7),
    show=False,
    return_fig=True,
    **dot_kwargs
)
dp.savefig(p1_fig_dir / "P1_final_sensory_epithelium_dotplot_part2.png", dpi=SAVE_DPI)

adata_P1_harmony_save = adata_P1_final_save.copy()
adata_P1_harmony_save.obs = adata_P1_harmony_save.obs[["sample", p1_label_key]].copy()
adata_P1_harmony_save.obs["stage"] = "P1"
adata_P1_harmony_save.raw = None
adata_P1_harmony_save.layers.clear()
adata_P1_harmony_save.obsm.clear()
adata_P1_harmony_save.varm.clear()
adata_P1_harmony_save.obsp.clear()
adata_P1_harmony_save.uns = {}

adata_P1_scanvi_save = adata_P1_harmony_save.copy()

p1_harmony_path = p1_harmony_dir / "P1_for_Harmony_normalized.h5ad"
p1_scanvi_path = p1_scanvi_dir / "P1_for_scnvi_normalized.h5ad"

adata_P1_harmony_save.write_h5ad(p1_harmony_path, compression="gzip")
adata_P1_scanvi_save.write_h5ad(p1_scanvi_path, compression="gzip")



plt.close("all")
