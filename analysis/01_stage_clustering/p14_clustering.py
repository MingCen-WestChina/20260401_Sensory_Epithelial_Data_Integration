#1. ===========Purpose and reproducibility settings
"""Preprocess, cluster, annotate, and export the P14 integration inputs."""

from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("COCHLEA_DATA_DIR", REPO_ROOT / "Data")).expanduser().resolve()
RESULTS_ROOT = Path(os.environ.get("COCHLEA_RESULTS_DIR", REPO_ROOT / "results")).expanduser().resolve()
STAGE_OBJECT_DIR = RESULTS_ROOT / "01_stage_clustering" / "h5ad_for_integration"
OUTPUT_DIR = RESULTS_ROOT / "01_stage_clustering" / "p14"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STAGE_OBJECT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 0
SAVE_DPI = 1201
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_STATE))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Notebook cell 43


# =========================================================
# 0. import

import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.io import mmread
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=300, facecolor="white")
sc.set_figure_params(figsize=(8, 8))


# =========================================================
# 1. paths

base_dir = REPO_ROOT
figure_dir = OUTPUT_DIR
data_dir = DATA_ROOT / "raw" / "p14"

os.chdir(base_dir)

p14_barcodes = data_dir / "P14_barcodes.tsv"
p14_features = data_dir / "P14_features.tsv"
p14_matrix = data_dir / "P14_matrix.mtx"

p28_data_dir = DATA_ROOT / "raw" / "p28"
p28_1_barcodes = p28_data_dir / "P28_1_barcodes.tsv"
p28_1_features = p28_data_dir / "P28_1_features.tsv"
p28_1_matrix = p28_data_dir / "P28_1_matrix.mtx"

p28_2_barcodes = p28_data_dir / "P28_2_barcodes.tsv"
p28_2_features = p28_data_dir / "P28_2_features.tsv"
p28_2_matrix = p28_data_dir / "P28_2_matrix.mtx"
# =========================================================
# 2. function: read one 10x mtx folder

def read_10x_mtx_manual(
    barcodes_path: str,
    features_path: str,
    matrix_path: str,
    sample_name: str,
) -> sc.AnnData:
    barcodes = pd.read_csv(barcodes_path, sep="\t", header=None)
    features = pd.read_csv(features_path, sep="\t", header=None)
    matrix = mmread(matrix_path).tocsr()

    if matrix.shape[0] != features.shape[0]:
        raise ValueError(f"{sample_name}: gene dimension does not match features.tsv")
    if matrix.shape[1] != barcodes.shape[0]:
        raise ValueError(f"{sample_name}: cell dimension does not match barcodes.tsv")

    gene_ids = features.iloc[:, 0].astype(str).values
    gene_names = features.iloc[:, 1].astype(str).values if features.shape[1] > 1 else gene_ids.copy()
    cell_barcodes = barcodes.iloc[:, 0].astype(str).values

    adata = sc.AnnData(X=matrix.T.copy())
    adata.obs_names = pd.Index(cell_barcodes)
    adata.var_names = pd.Index(gene_names)
    adata.var["gene_id"] = gene_ids
    adata.var["gene_symbol"] = gene_names
    if features.shape[1] > 2:
        adata.var["feature_type"] = features.iloc[:, 2].astype(str).values

    adata.obs["sample"] = sample_name
    adata.obs["age"] = sample_name.split("_")[0]
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    return adata
# =========================================================
# 3. read data and split P14 after shared QC
# keep exactly the same object flow as your original notebook

adata_p14 = read_10x_mtx_manual(
    barcodes_path=p14_barcodes,
    features_path=p14_features,
    matrix_path=p14_matrix,
    sample_name="P14",
)
adata_p28_1 = read_10x_mtx_manual(
    barcodes_path=p28_1_barcodes,
    features_path=p28_1_features,
    matrix_path=p28_1_matrix,
    sample_name="P28_1",
)
adata_p28_2 = read_10x_mtx_manual(
    barcodes_path=p28_2_barcodes,
    features_path=p28_2_features,
    matrix_path=p28_2_matrix,
    sample_name="P28_2",
)

print("P14 :", adata_p14.shape)
print("P28_1 :", adata_p28_1.shape)
print("P28_2 :", adata_p28_2.shape)

adatas = [adata_p14, adata_p28_1, adata_p28_2]
adata = ad.concat(adatas, join="outer", merge="same")
adata.var_names_make_unique()

adata.var["mt"] = adata.var_names.str.lower().str.startswith("mt-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

# exactly same shared QC as original
adata = adata[
    (adata.obs["n_genes_by_counts"] >= 200)
    & (adata.obs["n_genes_by_counts"] <= 3000)
    & (adata.obs["total_counts"] <= 15000)
    & (adata.obs["pct_counts_mt"] <= 15)
].copy()

print(adata)
print(adata.obs["sample"].value_counts())

adata_P14 = adata[adata.obs["age"] == "P14"].copy()

print("\nP14")
print(adata_P14)
print(adata_P14.obs["sample"].value_counts())


# Notebook cell 44

# =========================================================
# 4. P14: preprocess + coarse clustering

adata_p14 = adata_P14.copy()
adata_p14.layers["counts"] = adata_p14.X.copy()

sc.pp.normalize_total(adata_p14, target_sum=1e4)
sc.pp.log1p(adata_p14)

sc.pp.highly_variable_genes(
    adata_p14,
    n_top_genes=2000,
    flavor="seurat",
)

adata_p14 = adata_p14[:, adata_p14.var["highly_variable"]].copy()
sc.pp.scale(adata_p14, max_value=10)
sc.tl.pca(adata_p14, svd_solver="arpack")
sc.pp.neighbors(adata_p14, n_neighbors=12, n_pcs=20)
sc.tl.umap(adata_p14)
sc.tl.tsne(adata_p14, n_pcs=20)
sc.tl.leiden(adata_p14, resolution=0.4, key_added="cluster0")

print(adata_p14)
print("\nP14 cluster sizes:")
print(adata_p14.obs["cluster0"].value_counts().sort_index())


# =========================================================
# 5. P14: attach cluster back to full-gene object

adata_p14_full = adata_P14.copy()
adata_p14_full = adata_p14_full[adata_p14.obs_names].copy()
adata_p14_full.obs["cluster0"] = adata_p14.obs["cluster0"].values
adata_p14_full.obsm["X_tsne"] = adata_p14.obsm["X_tsne"].copy()
adata_p14_full.obsm["X_umap"] = adata_p14.obsm["X_umap"].copy()

marker_final = [
    "Myo6", "Slc26a5", "Ocm", "Pcp4",
    "Slc17a8", "Otof", "Calb2", "Acbd7",
    "Sox2", "Matn4", "Gata3",
    "Gjb2", "Lgr5", "Fgf3", "Prox1", "Pmch"
]
marker_final = [g for g in marker_final if g in adata_p14_full.var_names]

df_sub_full = sc.get.obs_df(adata_p14_full, keys=["cluster0"] + marker_final)
subcluster_mean0 = df_sub_full.groupby("cluster0")[marker_final].mean().round(2)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)
print(subcluster_mean0)


# =========================================================
# 6. P14: subset target epithelial/supporting/hair clusters

target_clusters_p14 = ["1", "2", "4", "5", "9"]
adata_p14_sub = adata_p14_full[adata_p14_full.obs["cluster0"].isin(target_clusters_p14)].copy()
adata_p14_sub.layers["counts"] = adata_p14_sub.X.copy()

print(adata_p14_sub)
print(adata_p14_sub.obs["cluster0"].value_counts().sort_index())


# =========================================================
# 7. P14: re-cluster within target cells

sc.pp.normalize_total(adata_p14_sub, target_sum=1e4)
sc.pp.log1p(adata_p14_sub)

sc.pp.highly_variable_genes(
    adata_p14_sub,
    n_top_genes=2000,
    flavor="seurat",
)

adata_p14_sub = adata_p14_sub[:, adata_p14_sub.var["highly_variable"]].copy()
sc.pp.scale(adata_p14_sub, max_value=10)
sc.tl.pca(adata_p14_sub, svd_solver="arpack")
sc.pp.neighbors(adata_p14_sub, n_neighbors=10, n_pcs=20)
sc.tl.umap(adata_p14_sub)
sc.tl.tsne(adata_p14_sub, n_pcs=20)
sc.tl.leiden(adata_p14_sub, resolution=0.35, key_added="subcluster")

print("\nP14 subcluster sizes:")
print(adata_p14_sub.obs["subcluster"].value_counts().sort_index())


# =========================================================
# 8. P14: full-gene subset with subcluster labels

adata_p14_sub_full = adata_p14_full[adata_p14_sub.obs_names].copy()
adata_p14_sub_full.obs["subcluster"] = adata_p14_sub.obs["subcluster"].values
adata_p14_sub_full.obsm["X_umap"] = adata_p14_sub.obsm["X_umap"].copy()
adata_p14_sub_full.obsm["X_tsne"] = adata_p14_sub.obsm["X_tsne"].copy()

df_sub_full = sc.get.obs_df(adata_p14_sub_full, keys=["subcluster"] + marker_final)
subcluster_mean = df_sub_full.groupby("subcluster")[marker_final].mean().round(2)
print(subcluster_mean)


# =========================================================
# 9. P14: remove HC, keep supporting/sulcus candidates

non_hc_subclusters = ["0", "2", "3", "5", "6", "7"]
adata_p14_sc = adata_p14_sub_full[adata_p14_sub_full.obs["subcluster"].isin(non_hc_subclusters)].copy()
adata_p14_sc.layers["counts"] = adata_p14_sc.X.copy()

print(adata_p14_sc)
print(adata_p14_sc.obs["subcluster"].value_counts().sort_index())


# =========================================================
# 10. P14: re-cluster supporting/sulcus only

sc.pp.normalize_total(adata_p14_sc, target_sum=1e4)
sc.pp.log1p(adata_p14_sc)

sc.pp.highly_variable_genes(
    adata_p14_sc,
    n_top_genes=2000,
    flavor="seurat",
)

adata_p14_sc = adata_p14_sc[:, adata_p14_sc.var["highly_variable"]].copy()
sc.pp.scale(adata_p14_sc, max_value=10)
sc.tl.pca(adata_p14_sc, svd_solver="arpack")
sc.pp.neighbors(adata_p14_sc, n_neighbors=8, n_pcs=15)
sc.tl.umap(adata_p14_sc)
sc.tl.tsne(adata_p14_sc, n_pcs=15)
sc.tl.leiden(adata_p14_sc, resolution=0.45, key_added="sc_cluster")

print(adata_p14_sc)
print("\nP14 supporting/sulcus cluster sizes:")
print(adata_p14_sc.obs["sc_cluster"].value_counts().sort_index())


# Notebook cell 45

# =========================================================
# 11. P14: attach new labels back to full-gene object

adata_p14_sc_full = adata_p14_sub_full[adata_p14_sc.obs_names].copy()
adata_p14_sc_full.obs["sc_cluster"] = adata_p14_sc.obs["sc_cluster"].values
adata_p14_sc_full.obsm["X_umap"] = adata_p14_sc.obsm["X_umap"].copy()
adata_p14_sc_full.obsm["X_tsne"] = adata_p14_sc.obsm["X_tsne"].copy()

marker_sc = ["Sox2", "Matn4", "Gata3", "Gjb2", "Lgr5", "Fgf3", "Prox1", "Pmch"]
marker_sc = [g for g in marker_sc if g in adata_p14_sc_full.var_names]

df_sc = sc.get.obs_df(adata_p14_sc_full, keys=["sc_cluster"] + marker_sc)
sc_mean = df_sc.groupby("sc_cluster")[marker_sc].mean().round(2)
print(sc_mean)


# =========================================================
# 12. P14: keep only unresolved 5 supporting-cell candidates

keep_unresolved = ["1", "2", "3", "6"]
adata_p14_5cand = adata_p14_sc_full[adata_p14_sc_full.obs["sc_cluster"].isin(keep_unresolved)].copy()
adata_p14_5cand.layers["counts"] = adata_p14_5cand.X.copy()

print(adata_p14_5cand)
print(adata_p14_5cand.obs["sc_cluster"].value_counts().sort_index())


# =========================================================
# 13. P14: re-cluster unresolved 5-cell-type candidates

sc.pp.normalize_total(adata_p14_5cand, target_sum=1e4)
sc.pp.log1p(adata_p14_5cand)

sc.pp.highly_variable_genes(
    adata_p14_5cand,
    n_top_genes=2000,
    flavor="seurat",
)

adata_p14_5cand = adata_p14_5cand[:, adata_p14_5cand.var["highly_variable"]].copy()
sc.pp.scale(adata_p14_5cand, max_value=10)
sc.tl.pca(adata_p14_5cand, svd_solver="arpack")
sc.pp.neighbors(adata_p14_5cand, n_neighbors=8, n_pcs=15)
sc.tl.umap(adata_p14_5cand)
sc.tl.tsne(adata_p14_5cand, n_pcs=15)
sc.tl.leiden(adata_p14_5cand, resolution=0.5, key_added="cand5_cluster")

print("\nP14 unresolved 5-type cluster sizes:")
print(adata_p14_5cand.obs["cand5_cluster"].value_counts().sort_index())


# Notebook cell 46

# =========================================================
# 14. P14: full-gene object for unresolved 5-cell-type candidates

adata_p14_5cand_full = adata_p14_sc_full[adata_p14_5cand.obs_names].copy()
adata_p14_5cand_full.obs["cand5_cluster"] = adata_p14_5cand.obs["cand5_cluster"].values
adata_p14_5cand_full.obsm["X_umap"] = adata_p14_5cand.obsm["X_umap"].copy()
adata_p14_5cand_full.obsm["X_tsne"] = adata_p14_5cand.obsm["X_tsne"].copy()

marker_5cand = ["Sox2", "Matn4", "Gata3", "Gjb2", "Lgr5", "Fgf3", "Prox1", "Pmch"]
marker_5cand = [g for g in marker_5cand if g in adata_p14_5cand_full.var_names]

df_5cand = sc.get.obs_df(adata_p14_5cand_full, keys=["cand5_cluster"] + marker_5cand)
cand5_mean = df_5cand.groupby("cand5_cluster")[marker_5cand].mean().round(2)
print(cand5_mean)


# =========================================================
# 15. P14: start final annotation from full object

adata_p14_final = adata_p14_full.copy()
adata_p14_final.obs["celltype"] = "Other"

# fixed clusters from subcluster
ohc_cells = adata_p14_sub_full.obs_names[adata_p14_sub_full.obs["subcluster"] == "1"]
ihc_cells = adata_p14_sub_full.obs_names[adata_p14_sub_full.obs["subcluster"] == "4"]
adata_p14_final.obs.loc[ohc_cells, "celltype"] = "OHC"
adata_p14_final.obs.loc[ihc_cells, "celltype"] = "IHC"

# fixed clusters from sc_cluster
isc_cells = adata_p14_sc_full.obs_names[adata_p14_sc_full.obs["sc_cluster"] == "4"]
osc_cells = adata_p14_sc_full.obs_names[adata_p14_sc_full.obs["sc_cluster"] == "0"]
adata_p14_final.obs.loc[isc_cells, "celltype"] = "ISC"
adata_p14_final.obs.loc[osc_cells, "celltype"] = "OSC"

# unresolved 5 classes
map_5 = {
    "0": "IPhC",
    "1": "IB",
    "2": "HeC",
    "3": "DC",
    "4": "PC",
}
for clu, ct in map_5.items():
    cells = adata_p14_5cand_full.obs_names[adata_p14_5cand_full.obs["cand5_cluster"] == clu]
    adata_p14_final.obs.loc[cells, "celltype"] = ct

celltype_order = ["OHC", "IHC", "IPhC", "IB", "DC", "PC", "HeC", "OSC", "ISC"]
adata_9 = adata_p14_final[adata_p14_final.obs["celltype"].isin(celltype_order)].copy()
adata_9.obs["celltype"] = pd.Categorical(adata_9.obs["celltype"], categories=celltype_order, ordered=True)

print(adata_9)
print(adata_9.obs["celltype"].value_counts())


# =========================================================
# 16. P14: recompute embedding on final 9 classes

adata_9.layers["counts"] = adata_9.X.copy()

sc.pp.normalize_total(adata_9, target_sum=1e4)
sc.pp.log1p(adata_9)

sc.pp.highly_variable_genes(
    adata_9,
    n_top_genes=2000,
    flavor="seurat",
)

adata_9 = adata_9[:, adata_9.var["highly_variable"]].copy()
sc.pp.scale(adata_9, max_value=10)
sc.tl.pca(adata_9, svd_solver="arpack")
sc.pp.neighbors(adata_9, n_neighbors=10, n_pcs=20)
sc.tl.umap(adata_9)
sc.tl.tsne(adata_9, n_pcs=20)


# Notebook cell 47

# =========================================================
# 17. P14: rebuild full-gene final object
# keep original clustering, labels, cell numbers, and UMAP unchanged

sc.settings.verbosity = 0

celltype_order = ["OHC", "IHC", "IPhC", "IB", "DC", "PC", "HeC", "OSC", "ISC"]

adata_P14_fullgene = adata_p14_full.copy()
adata_P14_fullgene.obs["celltype"] = "Other"

ohc_cells = adata_p14_sub_full.obs_names[adata_p14_sub_full.obs["subcluster"] == "1"]
ihc_cells = adata_p14_sub_full.obs_names[adata_p14_sub_full.obs["subcluster"] == "4"]
adata_P14_fullgene.obs.loc[ohc_cells, "celltype"] = "OHC"
adata_P14_fullgene.obs.loc[ihc_cells, "celltype"] = "IHC"

isc_cells = adata_p14_sc_full.obs_names[adata_p14_sc_full.obs["sc_cluster"] == "4"]
osc_cells = adata_p14_sc_full.obs_names[adata_p14_sc_full.obs["sc_cluster"] == "0"]
adata_P14_fullgene.obs.loc[isc_cells, "celltype"] = "ISC"
adata_P14_fullgene.obs.loc[osc_cells, "celltype"] = "OSC"

map_5 = {
    "0": "IPhC",
    "1": "IB",
    "2": "HeC",
    "3": "DC",
    "4": "PC",
}
for clu, ct in map_5.items():
    cells = adata_p14_5cand_full.obs_names[adata_p14_5cand_full.obs["cand5_cluster"] == clu]
    adata_P14_fullgene.obs.loc[cells, "celltype"] = ct

adata_P14_fullgene = adata_P14_fullgene[
    adata_P14_fullgene.obs["celltype"].isin(celltype_order)
].copy()

adata_P14_fullgene = adata_P14_fullgene[adata_9.obs_names].copy()

adata_P14_fullgene.obs["celltype"] = pd.Categorical(
    adata_P14_fullgene.obs["celltype"].astype(str),
    categories=celltype_order,
    ordered=True
)

print(adata_P14_fullgene.obs["celltype"].value_counts().reindex(celltype_order))


# =========================================================
# 18. P14: normalized full-gene object for validation

adata_P14_rich = adata_P14_fullgene.copy()
adata_P14_rich.layers["counts"] = adata_P14_rich.X.copy()

sc.pp.normalize_total(adata_P14_rich, target_sum=1e4)
sc.pp.log1p(adata_P14_rich)
adata_P14_rich.raw = adata_P14_rich

adata_P14_rich.obs["stage"] = "P14"
adata_P14_rich.obs["final_celltype"] = adata_P14_rich.obs["celltype"].astype(str)
adata_P14_rich.obs["source_object"] = "P14_final_9celltypes_fullgene"

# use original final embedding from your analysis
adata_P14_rich.obsm["X_umap"] = adata_9.obsm["X_umap"].copy()
adata_P14_rich.obsm["X_tsne"] = adata_9.obsm["X_tsne"].copy()
if "umap" in adata_9.uns:
    adata_P14_rich.uns["umap"] = adata_9.uns["umap"].copy()


# =========================================================
# 19. P14: final UMAP for validation only

sc.settings.set_figure_params(
    dpi=120,
    dpi_save=1201,
    facecolor="white",
    fontsize=9
)

sc.pl.umap(
    adata_P14_rich,
    color="final_celltype",
    legend_loc="right margin",
    title="P14 final 9 cell types",
    show=False,
)
plt.close("all")
plt.close()


# =========================================================
# 20. P14: final dotPlot 1, main marker genes

def keep_present(adata_in, genes):
    return list(dict.fromkeys([g for g in genes if g in adata_in.var_names]))

p14_dotplot_genes_1 = keep_present(
    adata_P14_rich,
    [
        "Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1",
        "Myo6", "Slc26a5", "Ocm", "Pcp4",
        "Slc17a8", "Otof", "Calb2", "Acbd7",
        "Matn4", "Gjb2", "Fgf3", "Prox1", "Pmch",
        "Lmo7", "Calm2", "Calm1", "Fus"
    ]
)

sc.pl.dotplot(
    adata_P14_rich,
    var_names=p14_dotplot_genes_1,
    groupby="final_celltype",
    standard_scale="var",
    cmap="Reds",
    dot_max=0.9,
    figsize=(13, 3.8),
    show=False,
)
plt.close("all")
plt.close()


# =========================================================
# 21. P14: final dotPlot 2, supporting-cell and auxiliary genes

p14_dotplot_genes_2 = keep_present(
    adata_P14_rich,
    [
        "BC006965", "Mir100hg", "Cdkn1b", "Cobl", "Tgfb2",
        "Gsn", "Farl1", "Slc1a3", "Selenom", "Tubb4b",
        "Gas2", "Ptma", "Ptms", "Ppia", "Lgals1", "Tmsb4x",
        "S100a1", "App", "Epyc", "Ctsl",
        "Smpx", "Ceacam16", "Rbp", "Rbp1", "Rbp7", "Fabp3",
        "Lgr6", "Ednrb", "Prdm12", "Aqp4"
    ]
)

sc.pl.dotplot(
    adata_P14_rich,
    var_names=p14_dotplot_genes_2,
    groupby="final_celltype",
    standard_scale="var",
    cmap="Reds",
    dot_max=0.9,
    figsize=(15, 3.8),
    show=False,
)
plt.close("all")
plt.close()


# =========================================================
# 22. P14: final dotPlot 3, exclusion markers

p14_exclude_genes = keep_present(
    adata_P14_rich,
    [
        "Oc90", "Otx2",
        "Neurod1", "Neurog1", "Snap25", "Tubb3", "Elavl3",
        "Ptprc", "Lyz2", "Cd74", "Hba-a1", "Hbb-bs",
        "Sox10", "Mbp", "Mpz", "Plp1", "Pecam1", "Kdr"
    ]
)

sc.pl.dotplot(
    adata_P14_rich,
    var_names=p14_exclude_genes,
    groupby="final_celltype",
    standard_scale="var",
    cmap="Reds",
    dot_max=0.9,
    figsize=(10, 3.5),
    show=False,
)
plt.close("all")
plt.close()


# Notebook cell 48

# =========================================================
# 20. P14: final dotPlot for validation only
# follow the dotPlot style used in Grouping by periods-1.pdf

p14_marker_dict_final = {
    "Context_check": [
        "Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"
    ],
    "HC_common": [
        "Myo6"
    ],
    "OHC": [
        "Slc26a5", "Ocm", "Pcp4", "Lmo7"
    ],
    "IHC": [
        "Slc17a8", "Otof", "Calb2", "Acbd7", "Calm2", "Calm1", "Fus"
    ],
    "IPhC": [
        "Sox2", "Gata3", "Slc1a3", "Matn4", "Smpx", "Epyc"
    ],
    "IB": [
        "Sox2", "Gata3", "Smpx", "Epyc"
    ],
    "DC": [
        "Ceacam16", "Rbp", "Rbp1", "Rbp7", "Fabp3",
        "Sox2", "Gjb2", "Lgr5", "Fgf3", "Prox1",
        "S100a1"
    ],
    "PC": [
        "Sox2", "Gjb2", "Lgr6", "Ednrb", "Prdm12", "Prox1"
    ],
    "HeC": [
        "Sox2", "Gata3", "Pmch", "Smpx", "Epyc"
    ],
    "OSC": [
        "Gjb2", "Aqp4", "Epyc"
    ],
    "ISC": [
        "Gjb2", "Gata3", "Aqp4", "Epyc"
    ],
    "Support_aux": [
        "BC006965", "Mir100hg", "Cdkn1b", "Cobl", "Tgfb2",
        "Gsn", "Farl1", "Selenom", "Tubb4b", "Gas2",
        "Ptma", "Ptms", "Ppia", "Lgals1", "Tmsb4x",
        "App", "Ctsl"
    ],
    "Exclude": [
        "Oc90", "Otx2",
        "Neurod1", "Neurog1", "Snap25", "Tubb3", "Elavl3",
        "Ptprc", "Lyz2", "Cd74", "Hba-a1", "Hbb-bs",
        "Sox10", "Mbp", "Mpz", "Plp1", "Pecam1", "Kdr"
    ],
}

p14_marker_dict_final = {
    group: [gene for gene in genes if gene in adata_P14_rich.var_names]
    for group, genes in p14_marker_dict_final.items()
}
p14_marker_dict_final = {
    group: genes
    for group, genes in p14_marker_dict_final.items()
    if len(genes) > 0
}

sc.pl.dotplot(
    adata_P14_rich,
    var_names=p14_marker_dict_final,
    groupby="final_celltype",
    standard_scale="var",
    figsize=(26, 8),
    show=True
)


# Notebook cell 50

# =========================================================
# 26. P14 rescue re-clustering
# remove accepted cell types, rescue epithelial/supporting candidates from remaining P14 cells

sc.settings.verbosity = 0

accepted_types = ["OHC", "IHC", "DC", "OSC", "ISC"]
unresolved_types = ["IPhC", "IB", "HeC", "PC"]

accepted_cells = adata_P14_rich.obs_names[
    adata_P14_rich.obs["final_celltype"].isin(accepted_types)
]

adata_P14_rescue = adata_p14_full[
    ~adata_p14_full.obs_names.isin(accepted_cells)
].copy()

# record previous labels if cells were already in the final 9-cell object
previous_label = pd.Series("Other_rescue", index=adata_P14_rescue.obs_names)
known_labels = adata_P14_rich.obs["final_celltype"].astype(str)
overlap_cells = previous_label.index.intersection(known_labels.index)
previous_label.loc[overlap_cells] = known_labels.loc[overlap_cells]
adata_P14_rescue.obs["previous_label"] = previous_label.values


# =========================================================
# 27. strict Oc90/Otx2 removal

strict_exclude_genes = [g for g in ["Oc90", "Otx2"] if g in adata_P14_rescue.var_names]

if len(strict_exclude_genes) > 0:
    strict_df = sc.get.obs_df(adata_P14_rescue, keys=strict_exclude_genes)
    keep_strict = (strict_df[strict_exclude_genes].fillna(0) <= 0).all(axis=1)
    adata_P14_rescue = adata_P14_rescue[keep_strict.values].copy()

print(adata_P14_rescue.obs["previous_label"].value_counts())


# =========================================================
# 28. keep epithelial/supporting candidates only
# always keep previous IPhC/IB/HeC/PC, and rescue other cells with epithelial/supporting marker signal

support_candidate_genes = [
    "Epcam", "Gata3", "Sox2", "Sox9", "Gjb2",
    "Slc1a3", "Matn4", "Smpx", "Epyc",
    "Ceacam16", "Rbp", "Rbp1", "Rbp7", "Fabp3",
    "Lgr5", "Fgf3", "Prox1", "Lgr6", "Ednrb", "Prdm12",
    "Pmch", "Aqp4"
]
support_candidate_genes = [
    g for g in support_candidate_genes
    if g in adata_P14_rescue.var_names
]

support_df = sc.get.obs_df(adata_P14_rescue, keys=support_candidate_genes)
support_marker_n = (support_df[support_candidate_genes].fillna(0) > 0).sum(axis=1)

keep_support = (
    adata_P14_rescue.obs["previous_label"].isin(unresolved_types).values
    | (support_marker_n >= 2).values
)

adata_P14_rescue = adata_P14_rescue[keep_support].copy()
adata_P14_rescue.obs["support_marker_n"] = support_marker_n.loc[adata_P14_rescue.obs_names].values

print(adata_P14_rescue)
print(adata_P14_rescue.obs["previous_label"].value_counts())


# =========================================================
# 29. re-cluster rescued unresolved epithelial/supporting cells

adata_P14_rescue.layers["counts"] = adata_P14_rescue.X.copy()

sc.pp.normalize_total(adata_P14_rescue, target_sum=1e4)
sc.pp.log1p(adata_P14_rescue)

sc.pp.highly_variable_genes(
    adata_P14_rescue,
    n_top_genes=2000,
    flavor="seurat",
)

adata_P14_rescue_hvg = adata_P14_rescue[
    :, adata_P14_rescue.var["highly_variable"]
].copy()

sc.pp.scale(adata_P14_rescue_hvg, max_value=10)
sc.tl.pca(adata_P14_rescue_hvg, svd_solver="arpack")
sc.pp.neighbors(adata_P14_rescue_hvg, n_neighbors=8, n_pcs=15)
sc.tl.umap(adata_P14_rescue_hvg)
sc.tl.leiden(
    adata_P14_rescue_hvg,
    resolution=0.5,
    key_added="P14_rescue_leiden"
)

adata_P14_rescue.obs["P14_rescue_leiden"] = (
    adata_P14_rescue_hvg.obs["P14_rescue_leiden"].values
)
adata_P14_rescue.obsm["X_umap"] = adata_P14_rescue_hvg.obsm["X_umap"].copy()

print(adata_P14_rescue.obs["P14_rescue_leiden"].value_counts().sort_index())
print(pd.crosstab(
    adata_P14_rescue.obs["P14_rescue_leiden"],
    adata_P14_rescue.obs["previous_label"]
))


# Notebook cell 51

# =========================================================
# 26. P14 strict rescue re-clustering
# rescue only high-confidence epithelial/supporting candidates

sc.settings.verbosity = 0

accepted_types = ["OHC", "IHC", "DC", "OSC", "ISC"]
unresolved_types = ["IPhC", "IB", "HeC", "PC"]

accepted_cells = adata_P14_rich.obs_names[
    adata_P14_rich.obs["final_celltype"].isin(accepted_types)
]

adata_P14_rescue2 = adata_p14_full[
    ~adata_p14_full.obs_names.isin(accepted_cells)
].copy()

previous_label = pd.Series("Other_rescue", index=adata_P14_rescue2.obs_names)
known_labels = adata_P14_rich.obs["final_celltype"].astype(str)
overlap_cells = previous_label.index.intersection(known_labels.index)
previous_label.loc[overlap_cells] = known_labels.loc[overlap_cells]
adata_P14_rescue2.obs["previous_label"] = previous_label.values


# =========================================================
# 27. strict exclusion and epithelial/supporting rescue filter

strict_exclude_genes = [g for g in ["Oc90", "Otx2"] if g in adata_P14_rescue2.var_names]
if len(strict_exclude_genes) > 0:
    strict_df = sc.get.obs_df(adata_P14_rescue2, keys=strict_exclude_genes)
    keep_strict = (strict_df[strict_exclude_genes].fillna(0) <= 0).all(axis=1)
    adata_P14_rescue2 = adata_P14_rescue2[keep_strict.values].copy()

support_candidate_genes = [
    "Epcam", "Gata3", "Sox2", "Sox9", "Gjb2",
    "Slc1a3", "Matn4", "Smpx", "Epyc",
    "Ceacam16", "Rbp", "Rbp1", "Rbp7", "Fabp3",
    "Lgr5", "Fgf3", "Prox1", "Lgr6", "Ednrb", "Prdm12",
    "Pmch", "Aqp4"
]
support_candidate_genes = [g for g in support_candidate_genes if g in adata_P14_rescue2.var_names]

core_epi_genes = [
    g for g in ["Epcam", "Gata3", "Sox2", "Sox9", "Gjb2", "Epyc"]
    if g in adata_P14_rescue2.var_names
]

contam_genes = [
    "Neurod1", "Neurog1", "Snap25", "Tubb3", "Elavl3",
    "Ptprc", "Lyz2", "Cd74",
    "Sox10", "Mbp", "Mpz", "Plp1",
    "Pecam1", "Kdr"
]
contam_genes = [g for g in contam_genes if g in adata_P14_rescue2.var_names]

support_df = sc.get.obs_df(adata_P14_rescue2, keys=support_candidate_genes)
core_df = sc.get.obs_df(adata_P14_rescue2, keys=core_epi_genes)
contam_df = sc.get.obs_df(adata_P14_rescue2, keys=contam_genes)

support_marker_n = (support_df[support_candidate_genes].fillna(0) > 0).sum(axis=1)
core_marker_n = (core_df[core_epi_genes].fillna(0) > 0).sum(axis=1)
contam_marker_n = (contam_df[contam_genes].fillna(0) > 0).sum(axis=1)

adata_P14_rescue2.obs["support_marker_n"] = support_marker_n.loc[adata_P14_rescue2.obs_names].values
adata_P14_rescue2.obs["core_marker_n"] = core_marker_n.loc[adata_P14_rescue2.obs_names].values
adata_P14_rescue2.obs["contam_marker_n"] = contam_marker_n.loc[adata_P14_rescue2.obs_names].values

keep_previous_unresolved = adata_P14_rescue2.obs["previous_label"].isin(unresolved_types).values

keep_other_rescue = (
    (adata_P14_rescue2.obs["previous_label"].astype(str) == "Other_rescue").values
    & (adata_P14_rescue2.obs["support_marker_n"].values >= 5)
    & (adata_P14_rescue2.obs["core_marker_n"].values >= 2)
    & (adata_P14_rescue2.obs["contam_marker_n"].values <= 1)
)

adata_P14_rescue2 = adata_P14_rescue2[
    keep_previous_unresolved | keep_other_rescue
].copy()

print(adata_P14_rescue2.obs["previous_label"].value_counts())


# =========================================================
# 28. re-cluster strict rescued unresolved epithelial/supporting cells

adata_P14_rescue2.layers["counts"] = adata_P14_rescue2.X.copy()

sc.pp.normalize_total(adata_P14_rescue2, target_sum=1e4)
sc.pp.log1p(adata_P14_rescue2)

sc.pp.highly_variable_genes(
    adata_P14_rescue2,
    n_top_genes=2000,
    flavor="seurat",
)

adata_P14_rescue2_hvg = adata_P14_rescue2[
    :, adata_P14_rescue2.var["highly_variable"]
].copy()

sc.pp.scale(adata_P14_rescue2_hvg, max_value=10)
sc.tl.pca(adata_P14_rescue2_hvg, svd_solver="arpack")
sc.pp.neighbors(adata_P14_rescue2_hvg, n_neighbors=10, n_pcs=15)
sc.tl.umap(adata_P14_rescue2_hvg)
sc.tl.leiden(
    adata_P14_rescue2_hvg,
    resolution=0.35,
    key_added="P14_rescue2_leiden"
)

adata_P14_rescue2.obs["P14_rescue2_leiden"] = (
    adata_P14_rescue2_hvg.obs["P14_rescue2_leiden"].values
)
adata_P14_rescue2.obsm["X_umap"] = adata_P14_rescue2_hvg.obsm["X_umap"].copy()

print(adata_P14_rescue2.obs["P14_rescue2_leiden"].value_counts().sort_index())
print(pd.crosstab(
    adata_P14_rescue2.obs["P14_rescue2_leiden"],
    adata_P14_rescue2.obs["previous_label"]
))


# Notebook cell 52

# =========================================================
# 29. P14 strict rescue UMAP check

sc.pl.umap(
    adata_P14_rescue2,
    color=[
        "previous_label",
        "P14_rescue2_leiden",
        "support_marker_n",
        "core_marker_n",
        "contam_marker_n"
    ],
    title=[
        "Previous labels",
        "P14 strict rescue re-clustering",
        "Support marker count",
        "Core epithelial marker count",
        "Contamination marker count"
    ],
    size=35,
    wspace=0.35,
    show=True
)


# =========================================================
# 30. P14 strict rescue marker dotPlot

p14_rescue2_marker_dict = {
    "Context_check": [
        "Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"
    ],
    "IPhC": [
        "Sox2", "Gata3", "Slc1a3", "Matn4", "Smpx", "Epyc"
    ],
    "IB": [
        "Sox2", "Gata3", "Smpx", "Epyc"
    ],
    "HeC": [
        "Sox2", "Gata3", "Pmch", "Smpx", "Epyc",
        "Fst", "Nupr1", "Fam159b", "Egfl6"
    ],
    "PC": [
        "Sox2", "Gjb2", "Lgr6", "Ednrb", "Prdm12", "Prox1"
    ],
    "DC_check": [
        "Ceacam16", "Rbp", "Rbp1", "Rbp7", "Fabp3", "Fgf3"
    ],
    "Sulcus_check": [
        "Gjb2", "Aqp4", "Epyc"
    ],
    "Exclude": [
        "Oc90", "Otx2",
        "Neurod1", "Neurog1", "Snap25", "Tubb3", "Elavl3",
        "Ptprc", "Lyz2", "Cd74", "Hba-a1", "Hbb-bs",
        "Sox10", "Mbp", "Mpz", "Plp1", "Pecam1", "Kdr"
    ],
}

p14_rescue2_marker_dict = {
    group: [gene for gene in genes if gene in adata_P14_rescue2.var_names]
    for group, genes in p14_rescue2_marker_dict.items()
}
p14_rescue2_marker_dict = {
    group: genes
    for group, genes in p14_rescue2_marker_dict.items()
    if len(genes) > 0
}

sc.pl.dotplot(
    adata_P14_rescue2,
    var_names=p14_rescue2_marker_dict,
    groupby="P14_rescue2_leiden",
    standard_scale="var",
    figsize=(22, 6),
    show=True
)


# Notebook cell 53

# =========================================================
# 31. P14 strict rescue: re-run Leiden at higher resolution
# keep the same filtered object and neighbor graph

sc.tl.leiden(
    adata_P14_rescue2_hvg,
    resolution=0.5,
    key_added="P14_rescue2_leiden_r05"
)

adata_P14_rescue2.obs["P14_rescue2_leiden_r05"] = (
    adata_P14_rescue2_hvg.obs["P14_rescue2_leiden_r05"].values
)

print(adata_P14_rescue2.obs["P14_rescue2_leiden_r05"].value_counts().sort_index())
print(pd.crosstab(
    adata_P14_rescue2.obs["P14_rescue2_leiden_r05"],
    adata_P14_rescue2.obs["previous_label"]
))

# =========================================================
# 32. P14 strict rescue r0.5 UMAP check

sc.pl.umap(
    adata_P14_rescue2,
    color=["previous_label", "P14_rescue2_leiden_r05"],
    title=["Previous labels", "P14 strict rescue re-clustering r0.5"],
    size=35,
    wspace=0.35,
    show=True
)


# =========================================================
# 33. P14 strict rescue r0.5 marker dotPlot

sc.pl.dotplot(
    adata_P14_rescue2,
    var_names=p14_rescue2_marker_dict,
    groupby="P14_rescue2_leiden_r05",
    standard_scale="var",
    figsize=(22, 6),
    show=True
)


# Notebook cell 54

# =========================================================
# 34. P14: merge accepted cell types with rescue r0.5 labels
# keep accepted cells, replace unresolved labels by rescue r0.5 annotation

sc.settings.verbosity = 0

accepted_types = ["OHC", "IHC", "DC", "OSC", "ISC"]

rescue_label_map = {
    "0": "IPhC",
    "1": "IB",
    "2": "HeC",
    "3": "PC",
}

accepted_cells = adata_P14_rich.obs_names[
    adata_P14_rich.obs["final_celltype"].isin(accepted_types)
]

adata_P14_final2 = adata_P14_rich[accepted_cells].copy()

rescue_keep = adata_P14_rescue2.obs["P14_rescue2_leiden_r05"].astype(str).isin(
    rescue_label_map.keys()
)
adata_P14_rescue_keep = adata_P14_rescue2[rescue_keep].copy()

adata_P14_rescue_keep.obs["final_celltype"] = (
    adata_P14_rescue_keep.obs["P14_rescue2_leiden_r05"]
    .astype(str)
    .map(rescue_label_map)
)

adata_P14_final2 = ad.concat(
    [adata_P14_final2, adata_P14_rescue_keep],
    join="outer",
    merge="same",
)

celltype_order = ["OHC", "IHC", "IPhC", "IB", "DC", "PC", "HeC", "OSC", "ISC"]

adata_P14_final2.obs["final_celltype"] = pd.Categorical(
    adata_P14_final2.obs["final_celltype"].astype(str),
    categories=celltype_order,
    ordered=True
)

adata_P14_final2.obs["stage"] = "P14"
adata_P14_final2.obs["source_object"] = "P14_final_rescue_r05"

print(adata_P14_final2)
print(adata_P14_final2.obs["final_celltype"].value_counts().reindex(celltype_order))

# =========================================================
# 35. P14 final2: recompute final UMAP

adata_P14_final2.layers["counts"] = adata_P14_final2.X.copy()

sc.pp.normalize_total(adata_P14_final2, target_sum=1e4)
sc.pp.log1p(adata_P14_final2)
adata_P14_final2.raw = adata_P14_final2

sc.pp.highly_variable_genes(
    adata_P14_final2,
    n_top_genes=2000,
    flavor="seurat",
)

adata_P14_final2_hvg = adata_P14_final2[
    :, adata_P14_final2.var["highly_variable"]
].copy()

sc.pp.scale(adata_P14_final2_hvg, max_value=10)
sc.tl.pca(adata_P14_final2_hvg, svd_solver="arpack")
sc.pp.neighbors(adata_P14_final2_hvg, n_neighbors=10, n_pcs=20)
sc.tl.umap(adata_P14_final2_hvg)
sc.tl.tsne(adata_P14_final2_hvg, n_pcs=20)

adata_P14_final2.obsm["X_umap"] = adata_P14_final2_hvg.obsm["X_umap"].copy()
adata_P14_final2.obsm["X_tsne"] = adata_P14_final2_hvg.obsm["X_tsne"].copy()
adata_P14_final2.uns["umap"] = adata_P14_final2_hvg.uns["umap"].copy()


# Notebook cell 55

# =========================================================
# 36. P14 final2 UMAP validation

sc.settings.set_figure_params(
    dpi=120,
    dpi_save=1201,
    facecolor="white",
    fontsize=9
)

sc.pl.umap(
    adata_P14_final2,
    color="final_celltype",
    legend_loc="right margin",
    title="P14 final 9 cell types, rescue r0.5",
    size=25,
    show=True
)


# =========================================================
# 37. P14 final2 dotPlot validation

p14_final2_marker_dict = {
    "Context_check": [
        "Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"
    ],
    "HC_common": [
        "Myo6"
    ],
    "OHC": [
        "Slc26a5", "Ocm", "Pcp4", "Lmo7"
    ],
    "IHC": [
        "Slc17a8", "Otof", "Calb2", "Acbd7", "Calm2", "Calm1", "Fus"
    ],
    "IPhC": [
        "Sox2", "Gata3", "Slc1a3", "Matn4", "Smpx", "Epyc"
    ],
    "IB": [
        "Sox2", "Gata3", "Smpx", "Epyc"
    ],
    "DC": [
        "Ceacam16", "Rbp", "Rbp1", "Rbp7", "Fabp3",
        "Sox2", "Gjb2", "Lgr5", "Fgf3", "Prox1",
        "S100a1"
    ],
    "PC": [
        "Sox2", "Gjb2", "Lgr6", "Ednrb", "Prdm12", "Prox1"
    ],
    "HeC": [
        "Sox2", "Gata3", "Pmch", "Smpx", "Epyc",
        "Fst", "Nupr1", "Fam159b", "Egfl6"
    ],
    "OSC": [
        "Gjb2", "Aqp4", "Epyc"
    ],
    "ISC": [
        "Gjb2", "Gata3", "Aqp4", "Epyc"
    ],
    "Exclude": [
        "Oc90", "Otx2",
        "Neurod1", "Neurog1", "Snap25", "Tubb3", "Elavl3",
        "Ptprc", "Lyz2", "Cd74", "Hba-a1", "Hbb-bs",
        "Sox10", "Mbp", "Mpz", "Plp1", "Pecam1", "Kdr"
    ],
}

p14_final2_marker_dict = {
    group: [gene for gene in genes if gene in adata_P14_final2.var_names]
    for group, genes in p14_final2_marker_dict.items()
}
p14_final2_marker_dict = {
    group: genes
    for group, genes in p14_final2_marker_dict.items()
    if len(genes) > 0
}

sc.pl.dotplot(
    adata_P14_final2,
    var_names=p14_final2_marker_dict,
    groupby="final_celltype",
    standard_scale="var",
    figsize=(26, 8),
    show=True
)


# Notebook cell 56

# =========================================================
# 38. P14 DC/PC focused check
# test whether current DC and PC labels are swapped or mixed

sc.settings.verbosity = 0

adata_P14_dcpc = adata_P14_final2[
    adata_P14_final2.obs["final_celltype"].isin(["DC", "PC"])
].copy()

adata_P14_dcpc.X = adata_P14_dcpc.layers["counts"].copy()

sc.pp.normalize_total(adata_P14_dcpc, target_sum=1e4)
sc.pp.log1p(adata_P14_dcpc)


def keep_present(adata_in, genes):
    return [g for g in genes if g in adata_in.var_names]


dc_score_genes = keep_present(
    adata_P14_dcpc,
    ["Ceacam16", "Rbp", "Rbp1", "Rbp7", "Fabp3", "Lgr5", "Fgf3"]
)

pc_score_genes = keep_present(
    adata_P14_dcpc,
    ["Lgr6", "Ednrb", "Prdm12", "Prox1"]
)

if len(dc_score_genes) >= 2:
    sc.tl.score_genes(
        adata_P14_dcpc,
        dc_score_genes,
        score_name="DC_score",
        use_raw=False
    )

if len(pc_score_genes) >= 2:
    sc.tl.score_genes(
        adata_P14_dcpc,
        pc_score_genes,
        score_name="PC_score",
        use_raw=False
    )

# display(
    # adata_P14_dcpc.obs
    # .groupby("final_celltype", observed=True)[["DC_score", "PC_score"]]
    # .mean()
    # .round(3)
# )


# =========================================================
# 39. P14 DC/PC focused re-clustering

sc.pp.highly_variable_genes(
    adata_P14_dcpc,
    n_top_genes=1000,
    flavor="seurat",
)

adata_P14_dcpc_hvg = adata_P14_dcpc[
    :, adata_P14_dcpc.var["highly_variable"]
].copy()

sc.pp.scale(adata_P14_dcpc_hvg, max_value=10)
sc.tl.pca(adata_P14_dcpc_hvg, svd_solver="arpack")
sc.pp.neighbors(adata_P14_dcpc_hvg, n_neighbors=5, n_pcs=10)
sc.tl.umap(adata_P14_dcpc_hvg)
sc.tl.leiden(
    adata_P14_dcpc_hvg,
    resolution=0.35,
    key_added="P14_dcpc_leiden"
)

adata_P14_dcpc.obs["P14_dcpc_leiden"] = (
    adata_P14_dcpc_hvg.obs["P14_dcpc_leiden"].values
)
adata_P14_dcpc.obsm["X_umap"] = adata_P14_dcpc_hvg.obsm["X_umap"].copy()

print(adata_P14_dcpc.obs["P14_dcpc_leiden"].value_counts().sort_index())
print(pd.crosstab(
    adata_P14_dcpc.obs["P14_dcpc_leiden"],
    adata_P14_dcpc.obs["final_celltype"]
))

# display(
    # adata_P14_dcpc.obs
    # .groupby("P14_dcpc_leiden", observed=True)[["DC_score", "PC_score"]]
    # .mean()
    # .round(3)
# )


# Notebook cell 57

# =========================================================
# 40. P14 DC/PC focused UMAP and dotPlot

sc.pl.umap(
    adata_P14_dcpc,
    color=["final_celltype", "P14_dcpc_leiden", "DC_score", "PC_score"],
    title=["Current labels", "DC/PC focused Leiden", "DC score", "PC score"],
    size=45,
    wspace=0.35,
    show=True
)

p14_dcpc_marker_dict = {
    "Context_check": [
        "Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"
    ],
    "DC": [
        "Ceacam16", "Rbp", "Rbp1", "Rbp7", "Fabp3",
        "Lgr5", "Fgf3", "S100a1"
    ],
    "PC": [
        "Sox2", "Gjb2", "Lgr6", "Ednrb", "Prdm12", "Prox1"
    ],
    "Exclude": [
        "Oc90", "Otx2",
        "Neurod1", "Neurog1", "Snap25", "Tubb3", "Elavl3",
        "Ptprc", "Lyz2", "Cd74",
        "Sox10", "Mbp", "Mpz", "Plp1", "Pecam1", "Kdr"
    ],
}

p14_dcpc_marker_dict = {
    group: [gene for gene in genes if gene in adata_P14_dcpc.var_names]
    for group, genes in p14_dcpc_marker_dict.items()
}
p14_dcpc_marker_dict = {
    group: genes
    for group, genes in p14_dcpc_marker_dict.items()
    if len(genes) > 0
}

sc.pl.dotplot(
    adata_P14_dcpc,
    var_names=p14_dcpc_marker_dict,
    groupby="P14_dcpc_leiden",
    standard_scale="var",
    figsize=(12, 4),
    show=True
)

sc.pl.dotplot(
    adata_P14_dcpc,
    var_names=p14_dcpc_marker_dict,
    groupby="final_celltype",
    standard_scale=None,
    figsize=(12, 3),
    show=True
)


# Notebook cell 58

# ======================== Save P14 final figures and integration h5ad ========================

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad

base_dir = REPO_ROOT

p14_fig_dir = OUTPUT_DIR
p14_harmony_dir = RESULTS_ROOT / "01_stage_clustering" / "harmony_inputs" / "p14"
p14_scanvi_dir = STAGE_OBJECT_DIR

for save_dir in [p14_fig_dir, p14_harmony_dir, p14_scanvi_dir]:
    save_dir.mkdir(parents=True, exist_ok=True)

p14_celltype_order = ["OHC", "IHC", "IPhC", "IB", "DC", "PC", "HeC", "OSC", "ISC"]

p14_expected_counts = pd.Series(
    {
        "OHC": 181,
        "IHC": 47,
        "IPhC": 71,
        "IB": 68,
        "DC": 43,
        "PC": 10,
        "HeC": 61,
        "OSC": 170,
        "ISC": 27,
    }
).reindex(p14_celltype_order)

if "adata_P14_final2" not in globals():
    raise NameError("adata_P14_final2 is not found. Run the P14 final rescue r0.5 code first.")

if "adata_p14_full" not in globals():
    raise NameError("adata_p14_full is not found. This raw full-gene P14 object is needed for raw h5ad saving.")

if "final_celltype" not in adata_P14_final2.obs.columns:
    raise KeyError("final_celltype is not found in adata_P14_final2.obs.")

if "X_umap" not in adata_P14_final2.obsm:
    raise KeyError("X_umap is not found in adata_P14_final2.obsm.")

missing_cells = adata_P14_final2.obs_names.difference(adata_p14_full.obs_names)
if len(missing_cells) > 0:
    raise ValueError(f"{len(missing_cells)} P14 final cells are not found in adata_p14_full.obs_names.")

adata_P14_final2.obs["P14_final_celltype"] = pd.Categorical(
    adata_P14_final2.obs["final_celltype"].astype(str),
    categories=p14_celltype_order,
    ordered=True,
)
adata_P14_final2.obs["final_celltype"] = pd.Categorical(
    adata_P14_final2.obs["P14_final_celltype"].astype(str),
    categories=p14_celltype_order,
    ordered=True,
)

p14_counts = (
    adata_P14_final2.obs["P14_final_celltype"]
    .value_counts()
    .reindex(p14_celltype_order)
    .fillna(0)
    .astype(int)
)

p14_count_check = pd.DataFrame(
    {
        "observed": p14_counts,
        "expected": p14_expected_counts.astype(int),
    }
)

if not (p14_count_check["observed"] == p14_count_check["expected"]).all():
    # display(p14_count_check)
    raise ValueError("P14 final counts do not match the checked P14 result. Stop before saving.")

if int(p14_counts.sum()) != adata_P14_final2.n_obs:
    raise ValueError("P14 count sum does not match adata_P14_final2.n_obs.")

p14_confidence_map = {
    "OHC": "high",
    "IHC": "high",
    "IPhC": "medium",
    "IB": "medium",
    "DC": "medium",
    "PC": "low",
    "HeC": "medium",
    "OSC": "high",
    "ISC": "high",
}

adata_P14_final2.obs["P14_final_confidence"] = (
    adata_P14_final2.obs["P14_final_celltype"].astype(str).map(p14_confidence_map).astype(str)
)

p14_label_n_map = {
    ct: f"{ct} (n={p14_counts.loc[ct]})"
    for ct in p14_celltype_order
}
adata_P14_final2.obs["P14_final_celltype_label_n"] = (
    adata_P14_final2.obs["P14_final_celltype"].astype(str).map(p14_label_n_map)
)
adata_P14_final2.obs["P14_final_celltype_label_n"] = pd.Categorical(
    adata_P14_final2.obs["P14_final_celltype_label_n"].astype(str),
    categories=[p14_label_n_map[ct] for ct in p14_celltype_order],
    ordered=True,
)

adata_P14_final2.obs["stage"] = "P14"
adata_P14_final2.obs["source_object"] = "P14_final_rescue_r05_keep_PC"

# ======================== Save P14 UMAP ========================

sc.settings.set_figure_params(
    dpi=120,
    dpi_save=1201,
    facecolor="white",
    fontsize=9,
)

sc.pl.umap(
    adata_P14_final2,
    color="P14_final_celltype_label_n",
    legend_loc="right margin",
    title="P14 final celltype",
    size=25,
    frameon=False,
    show=False,
)
plt.savefig(p14_fig_dir / "P14_umap.png", dpi=SAVE_DPI, bbox_inches="tight")
plt.close("all")

sc.pl.umap(
    adata_P14_final2,
    color="P14_final_celltype",
    legend_loc="on data",
    title="P14 final celltype",
    size=25,
    frameon=False,
    show=False,
)
plt.savefig(p14_fig_dir / "P14_umap_label.png", dpi=SAVE_DPI, bbox_inches="tight")
plt.close("all")

# ======================== Save P14 Dotplot ========================

p14_dot_marker_dict = {
    "Core": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "HC_common": ["Myo6"],
    "OHC": ["Slc26a5", "Ocm", "Pcp4", "Lmo7"],
    "IHC": ["Slc17a8", "Otof", "Calb2", "Acbd7", "Calm2", "Calm1", "Fus"],
    "IPhC": ["Sox2", "Gata3", "Slc1a3", "Matn4", "Smpx", "Epyc"],
    "IB": ["Sox2", "Gata3", "Smpx", "Epyc"],
    "DC": ["Ceacam16", "Rbp", "Rbp1", "Rbp7", "Fabp3", "Sox2", "Gjb2", "Lgr5", "Fgf3", "Prox1", "S100a1"],
    "PC": ["Sox2", "Gjb2", "Lgr6", "Ednrb", "Prdm12", "Prox1"],
    "HeC": ["Sox2", "Gata3", "Pmch", "Smpx", "Epyc", "Fst", "Nupr1", "Fam159b", "Egfl6"],
    "OSC": ["Gjb2", "Aqp4", "Epyc"],
    "ISC": ["Gjb2", "Gata3", "Aqp4", "Epyc"],
    "Exclude": ["Oc90", "Otx2", "Neurod1", "Neurog1", "Snap25", "Tubb3", "Elavl3", "Ptprc", "Lyz2", "Cd74", "Hba-a1", "Hbb-bs", "Sox10", "Mbp", "Mpz", "Plp1", "Pecam1", "Kdr"],
}

p14_dot_marker_dict = {
    group: [gene for gene in genes if gene in adata_P14_final2.var_names]
    for group, genes in p14_dot_marker_dict.items()
}
p14_dot_marker_dict = {
    group: genes
    for group, genes in p14_dot_marker_dict.items()
    if len(genes) > 0
}

missing_p14_core_genes = [
    gene for gene in ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"]
    if gene not in adata_P14_final2.var_names
]
assert len(missing_p14_core_genes) == 0, (
    f"P14 dotplot missing Core genes: {missing_p14_core_genes}"
)

def save_p14_dotplot(marker_dict, save_path, figsize):
    dp = sc.pl.dotplot(
        adata_P14_final2,
        var_names=marker_dict,
        groupby="P14_final_celltype",
        categories_order=p14_celltype_order,
        standard_scale="var",
        cmap="Reds",
        figsize=figsize,
        return_fig=True,
        show=False,
    )
    dp.savefig(save_path, dpi=SAVE_DPI)
    plt.close("all")

p14_marker_groups = list(p14_dot_marker_dict.keys())
p14_split_idx = int(np.ceil(len(p14_marker_groups) / 2))

p14_dotplot_part1_dict = {
    group: p14_dot_marker_dict[group]
    for group in p14_marker_groups[:p14_split_idx]
}
p14_dotplot_part2_dict = {
    group: p14_dot_marker_dict[group]
    for group in p14_marker_groups[p14_split_idx:]
}

save_p14_dotplot(
    p14_dot_marker_dict,
    p14_fig_dir / "P14_dotplot.png",
    figsize=(26, 8),
)
save_p14_dotplot(
    p14_dotplot_part1_dict,
    p14_fig_dir / "P14_dotplot_part1.png",
    figsize=(14, 8),
)
save_p14_dotplot(
    p14_dotplot_part2_dict,
    p14_fig_dir / "P14_dotplot_part2.png",
    figsize=(14, 8),
)

# ======================== Save P14 Harmony and scANVI h5ad ========================

def make_p14_integration_adata(raw_source, final_obj, normalized):
    obs = raw_source.obs.loc[final_obj.obs_names].copy()

    obs["stage"] = "P14"
    obs["batch"] = obs["sample"].astype(str) if "sample" in obs.columns else "P14"
    obs["celltype"] = final_obj.obs["P14_final_celltype"].astype(str).values
    obs["final_celltype"] = pd.Categorical(
        obs["celltype"],
        categories=p14_celltype_order,
        ordered=True,
    )
    obs["P14_final_celltype"] = pd.Categorical(
        obs["celltype"],
        categories=p14_celltype_order,
        ordered=True,
    )
    obs["P14_final_confidence"] = final_obj.obs["P14_final_confidence"].astype(str).values
    obs["P14_final_celltype_label_n"] = final_obj.obs["P14_final_celltype_label_n"].astype(str).values
    obs["source_object"] = "P14_final_rescue_r05_keep_PC"

    keep_obs_cols = [
        "sample",
        "batch",
        "age",
        "stage",
        "celltype",
        "final_celltype",
        "P14_final_celltype",
        "P14_final_confidence",
        "P14_final_celltype_label_n",
        "source_object",
        "n_genes_by_counts",
        "total_counts",
        "pct_counts_mt",
    ]
    keep_obs_cols = [col for col in keep_obs_cols if col in obs.columns]

    adata_out = ad.AnnData(
        X=raw_source[final_obj.obs_names, :].X.copy(),
        obs=obs[keep_obs_cols].copy(),
        var=pd.DataFrame(index=raw_source.var_names.copy()),
    )
    adata_out.obs_names = final_obj.obs_names.copy()
    adata_out.var_names_make_unique()
    adata_out.raw = None
    adata_out.uns = {}

    if normalized:
        sc.pp.normalize_total(adata_out, target_sum=1e4)
        sc.pp.log1p(adata_out)

    return adata_out

adata_P14_raw_save = make_p14_integration_adata(
    adata_p14_full,
    adata_P14_final2,
    normalized=False,
)
adata_P14_norm_save = make_p14_integration_adata(
    adata_p14_full,
    adata_P14_final2,
    normalized=True,
)

p14_harmony_norm_path = p14_harmony_dir / "P14_for_Harmony_normalized.h5ad"
p14_harmony_raw_path = p14_harmony_dir / "P14_for_Harmony_raw.h5ad"
p14_scanvi_norm_path = p14_scanvi_dir / "P14_for_scnvi_normalized.h5ad"
p14_scanvi_raw_path = p14_scanvi_dir / "P14_for_scnvi_raw.h5ad"

adata_P14_norm_save.write_h5ad(p14_harmony_norm_path, compression="gzip")
adata_P14_raw_save.write_h5ad(p14_harmony_raw_path, compression="gzip")
adata_P14_norm_save.write_h5ad(p14_scanvi_norm_path, compression="gzip")
adata_P14_raw_save.write_h5ad(p14_scanvi_raw_path, compression="gzip")
