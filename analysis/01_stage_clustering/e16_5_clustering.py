#1. ===========Purpose and reproducibility settings
"""Preprocess, cluster, annotate, and export the E16.5 integration inputs."""

from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("COCHLEA_DATA_DIR", REPO_ROOT / "Data")).expanduser().resolve()
RESULTS_ROOT = Path(os.environ.get("COCHLEA_RESULTS_DIR", REPO_ROOT / "results")).expanduser().resolve()
STAGE_OBJECT_DIR = RESULTS_ROOT / "01_stage_clustering" / "h5ad_for_integration"
OUTPUT_DIR = RESULTS_ROOT / "01_stage_clustering" / "e16_5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STAGE_OBJECT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 0
SAVE_DPI = 1201
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_STATE))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Notebook cell 44

# =========================================================
# 1. E16.5 read matrix + QC / exclusion precheck


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

sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=SAVE_DPI, facecolor="white")
sc.set_figure_params(figsize=(6, 5))

base_dir = REPO_ROOT
count_file = DATA_ROOT / "raw" / "e16_5" / "E16_Normalized_Counts.txt"
figure_dir = OUTPUT_DIR

figure_dir.mkdir(parents=True, exist_ok=True)
os.chdir(base_dir)

# =========================================================
# 2. read normalized matrix

df_E16 = pd.read_csv(count_file, sep="\t", index_col=0)
df_E16.index = df_E16.index.astype(str)
df_E16.columns = df_E16.columns.astype(str)
df_E16 = df_E16.loc[~df_E16.index.duplicated()].copy()
df_E16 = df_E16.apply(pd.to_numeric, errors="coerce").fillna(0)

adata_E16 = ad.AnnData(X=df_E16.T.copy())
adata_E16.obs_names = df_E16.columns.astype(str)
adata_E16.var_names = df_E16.index.astype(str)
adata_E16.obs_names_make_unique()
adata_E16.var_names_make_unique()

# E16 data are normalized values from the paper, so do not normalize_total/log1p again.
adata_E16.var["mt"] = adata_E16.var_names.str.startswith(("mt-", "Mt-", "MT-"))

sc.pp.calculate_qc_metrics(
    adata_E16,
    qc_vars=["mt"],
    percent_top=None,
    log1p=False,
    inplace=True,
)

sc.pp.filter_genes(adata_E16, min_cells=10)

# =========================================================
# 3. basic QC filter

adata_E16_qc = adata_E16[
    (adata_E16.obs["n_genes_by_counts"] >= 200)
    & (adata_E16.obs["pct_counts_mt"] <= 5),
    :
].copy()

print("E16.5 cells before QC:", adata_E16.n_obs)
print("E16.5 cells after QC :", adata_E16_qc.n_obs)
print("genes after filter   :", adata_E16_qc.n_vars)
print("max expression value :", float(np.max(adata_E16_qc.X)))

# =========================================================
# 4. helper

def get_expr_vector(adata_obj: ad.AnnData, gene: str) -> np.ndarray:
    if gene not in adata_obj.var_names:
        return np.zeros(adata_obj.n_obs, dtype=float)

    x = adata_obj[:, gene].X
    if sparse.issparse(x):
        return x.toarray().ravel()
    return np.asarray(x).ravel()


# =========================================================
# 5. Oc90 / Otx2 exclusion

adata_E16_qc.obs["Oc90_expr"] = get_expr_vector(adata_E16_qc, "Oc90")
adata_E16_qc.obs["Otx2_expr"] = get_expr_vector(adata_E16_qc, "Otx2")

adata_E16_clean = adata_E16_qc[
    (adata_E16_qc.obs["Oc90_expr"] == 0)
    & (adata_E16_qc.obs["Otx2_expr"] == 0),
    :
].copy()

print("Oc90 positive cells:", int((adata_E16_qc.obs["Oc90_expr"] > 0).sum()))
print("Otx2 positive cells:", int((adata_E16_qc.obs["Otx2_expr"] > 0).sum()))
print("cells after removing Oc90/Otx2 positive:", adata_E16_clean.n_obs)

# =========================================================
# 6. contamination marker precheck

contamination_markers = [
    "Neurod1", "Neurog1", "Prph", "Insm1", "Myt1", "Nhlh2",
    "Sox10", "Plp1", "Mpz",
    "Ptprc", "Lyz2", "Cd74",
    "Hbb-bs", "Hbb-bt", "Hba-a1", "Hba-a2",
    "Pecam1", "Cdh5", "Kdr",
    "Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1",
]
contamination_markers = [g for g in contamination_markers if g in adata_E16_clean.var_names]

contam_summary = {}
for gene in contamination_markers:
    x = get_expr_vector(adata_E16_clean, gene)
    contam_summary[gene] = {
        "mean": round(float(np.mean(x)), 4),
        "pct_positive": round(float(np.mean(x > 0) * 100), 2),
    }

pd.DataFrame(contam_summary).T.sort_values("pct_positive", ascending=False).head(20)


# Notebook cell 45

# =========================================================
# 7. QC plots, no save

sc.pl.violin(
    adata_E16_qc,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.3,
    multi_panel=True,
    show=False,
)
plt.gcf().set_size_inches(8, 3)
plt.close("all")
plt.close()

sc.pl.scatter(
    adata_E16_qc,
    x="total_counts",
    y="n_genes_by_counts",
    color="pct_counts_mt",
    size=8,
    show=False,
)
plt.gcf().set_size_inches(4.5, 4)
plt.close("all")
plt.close()


# Notebook cell 46

# =========================================================
# 8. remove blood / immune cells and mitochondrial genes

blood_genes = [g for g in ["Hbb-bs", "Hbb-bt", "Hba-a1", "Hba-a2"] if g in adata_E16_clean.var_names]
immune_genes = [g for g in ["Ptprc", "Lyz2", "Cd74"] if g in adata_E16_clean.var_names]

adata_E16_clean.obs["blood_score"] = 0.0
for gene in blood_genes:
    adata_E16_clean.obs["blood_score"] += get_expr_vector(adata_E16_clean, gene)

adata_E16_clean.obs["immune_score"] = 0.0
for gene in immune_genes:
    adata_E16_clean.obs["immune_score"] += get_expr_vector(adata_E16_clean, gene)

blood_mask = adata_E16_clean.obs["blood_score"] > 0
immune_mask = adata_E16_clean.obs["immune_score"] > 0

adata_E16_pre = adata_E16_clean[
    (~blood_mask) & (~immune_mask),
    :
].copy()

if "mt" in adata_E16_pre.var.columns:
    adata_E16_pre = adata_E16_pre[:, ~adata_E16_pre.var["mt"].values].copy()

print("cells before blood/immune removal:", adata_E16_clean.n_obs)
print("blood-positive cells removed     :", int(blood_mask.sum()))
print("immune-positive cells removed    :", int(immune_mask.sum()))
print("cells used for clustering        :", adata_E16_pre.n_obs)
print("genes used for clustering        :", adata_E16_pre.n_vars)


# Notebook cell 47

# =========================================================
# 9. helper for normalized-matrix clustering

def run_norm_clustering(
    adata_in: ad.AnnData,
    leiden_key: str,
    n_top_genes: int = 3000,
    n_neighbors: int = 15,
    n_pcs: int = 30,
    resolution: float = 0.5,
    min_dist: float = 0.3,
) -> ad.AnnData:
    adata_work = adata_in.copy()

    sc.pp.highly_variable_genes(
        adata_work,
        flavor="seurat",
        n_top_genes=min(n_top_genes, adata_work.n_vars),
    )

    block_genes = set(["Oc90", "Otx2"] + blood_genes + immune_genes)
    hvg_mask = adata_work.var["highly_variable"].values.copy()
    hvg_mask &= ~adata_work.var_names.isin(block_genes)

    adata_hvg = adata_work[:, hvg_mask].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, svd_solver="arpack", random_state=0)
    sc.pp.neighbors(adata_hvg, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata_hvg, min_dist=min_dist, random_state=0)
    sc.tl.leiden(adata_hvg, resolution=resolution, key_added=leiden_key)

    adata_work.obs[leiden_key] = adata_hvg.obs[leiden_key].astype(str).values
    adata_work.obsm["X_umap"] = adata_hvg.obsm["X_umap"].copy()

    return adata_work


adata_E16_all = run_norm_clustering(
    adata_E16_pre,
    leiden_key="E16_all_leiden",
    n_top_genes=3000,
    n_neighbors=15,
    n_pcs=30,
    resolution=0.5,
    min_dist=0.3,
)

print(adata_E16_all.obs["E16_all_leiden"].value_counts().sort_index())


# Notebook cell 48

# =========================================================
# 10. first-pass marker check: identify clear HC clusters first

hc_marker_dict = {
    "HC_core": ["Ccer2", "Pou4f3", "Pcp4", "Selenom", "Atoh1", "Cib2", "Pvalb"],
    "IHC": ["Acbd7", "Fgf8", "Ccl21a", "Smpx"],
    "OHC_iOHC": ["Lhfpl5", "Hes6", "Rprm", "Grp", "Tmem255b", "Dlk2"],
    "SC_check": ["Sox2", "Sox9", "Gata3", "Lgr5", "Isl1"],
    "Exclude_check": ["Oc90", "Otx2", "Ptprc", "Lyz2", "Cd74", "Hbb-bs", "Hba-a1"],
}
hc_marker_dict = {
    k: [g for g in v if g in adata_E16_all.var_names]
    for k, v in hc_marker_dict.items()
}
hc_marker_dict = {k: v for k, v in hc_marker_dict.items() if len(v) > 0}

hc_check_genes = list(dict.fromkeys(sum(hc_marker_dict.values(), [])))

cluster_mean = (
    sc.get.obs_df(
        adata_E16_all,
        keys=["E16_all_leiden"] + hc_check_genes,
    )
    .groupby("E16_all_leiden")[hc_check_genes]
    .mean()
    .round(3)
)

cluster_pct = (
    sc.get.obs_df(
        adata_E16_all,
        keys=["E16_all_leiden"] + hc_check_genes,
    )
    .assign(**{
        gene: lambda x, gene=gene: x[gene] > 0
        for gene in hc_check_genes
    })
    .groupby("E16_all_leiden")[hc_check_genes]
    .mean()
    .mul(100)
    .round(1)
)

print("Mean expression by cluster:")
# display(cluster_mean)

print("Percent positive by cluster:")
# display(cluster_pct)

sc.pl.umap(
    adata_E16_all,
    color="E16_all_leiden",
    legend_loc="right margin",
    size=8,
    frameon=True,
    show=False,
)
plt.gcf().set_size_inches(5, 5)
plt.close("all")
plt.close()

sc.pl.dotplot(
    adata_E16_all,
    var_names=hc_marker_dict,
    groupby="E16_all_leiden",
    standard_scale="var",
    dot_max=0.9,
    dot_min=0.05,
    figsize=(12, 4),
    dendrogram=False,
    show=False,
)
plt.close("all")
plt.close()


# Notebook cell 49

# =========================================================
# 11. HC subclustering from clear HC parent cluster

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
sc.settings.verbosity = 1

hc_parent_clusters = ["4"]

adata_E16_HC = adata_E16_all[
    adata_E16_all.obs["E16_all_leiden"].astype(str).isin(hc_parent_clusters),
    :
].copy()

print("E16 HC candidate cells:", adata_E16_HC.n_obs)

# =========================================================
# 12. re-cluster HC candidate cells

adata_E16_HC_work = adata_E16_HC.copy()

sc.pp.highly_variable_genes(
    adata_E16_HC_work,
    flavor="seurat",
    n_top_genes=min(1500, adata_E16_HC_work.n_vars),
)

block_genes = ["Oc90", "Otx2", "Ptprc", "Lyz2", "Cd74", "Hbb-bs", "Hbb-bt", "Hba-a1", "Hba-a2"]
hvg_mask = adata_E16_HC_work.var["highly_variable"].values.copy()
hvg_mask &= ~adata_E16_HC_work.var_names.isin(block_genes)

adata_E16_HC_hvg = adata_E16_HC_work[:, hvg_mask].copy()

sc.pp.scale(adata_E16_HC_hvg, max_value=10)
sc.tl.pca(adata_E16_HC_hvg, svd_solver="arpack", random_state=0)
sc.pp.neighbors(adata_E16_HC_hvg, n_neighbors=8, n_pcs=15)
sc.tl.umap(adata_E16_HC_hvg, min_dist=0.25, random_state=0)
sc.tl.leiden(adata_E16_HC_hvg, resolution=0.45, key_added="E16_HC_leiden")

adata_E16_HC.obs["E16_HC_leiden"] = adata_E16_HC_hvg.obs["E16_HC_leiden"].astype(str).values
adata_E16_HC.obsm["X_umap"] = adata_E16_HC_hvg.obsm["X_umap"].copy()

print(adata_E16_HC.obs["E16_HC_leiden"].value_counts().sort_index())

# =========================================================
# 13. HC subtype marker inspection

hc_marker_dict = {
    "HC_core": ["Ccer2", "Pou4f3", "Pcp4", "Selenom", "Atoh1", "Cib2", "Pvalb"],
    "IHC": ["Acbd7", "Fgf8", "Ccl21a", "Smpx"],
    "iOHC": ["Hes6", "Rprm", "Grp", "Dlk2", "Tmem255b", "Psph"],
    "OHC": ["Lhfpl5", "Gng8", "Grxcr1", "Lmo1", "Evl"],
    "SC_exclude": ["Sox2", "Sox9", "Gata3", "Lgr5", "Isl1"],
    "Exclude_check": ["Oc90", "Otx2", "Ptprc", "Cd74", "Hbb-bs", "Hba-a1"],
}
hc_marker_dict = {
    k: [g for g in v if g in adata_E16_HC.var_names]
    for k, v in hc_marker_dict.items()
}
hc_marker_dict = {k: v for k, v in hc_marker_dict.items() if len(v) > 0}

hc_check_genes = list(dict.fromkeys(sum(hc_marker_dict.values(), [])))

hc_mean = (
    sc.get.obs_df(
        adata_E16_HC,
        keys=["E16_HC_leiden"] + hc_check_genes,
    )
    .groupby("E16_HC_leiden")[hc_check_genes]
    .mean()
    .round(3)
)

hc_pct = (
    sc.get.obs_df(
        adata_E16_HC,
        keys=["E16_HC_leiden"] + hc_check_genes,
    )
    .assign(**{
        gene: lambda x, gene=gene: x[gene] > 0
        for gene in hc_check_genes
    })
    .groupby("E16_HC_leiden")[hc_check_genes]
    .mean()
    .mul(100)
    .round(1)
)

# display(hc_mean)
# display(hc_pct)

# =========================================================
# 14. score check for tentative HC subtype identity

hc_score_sets = {
    "IHC_score": ["Acbd7", "Fgf8", "Ccl21a", "Smpx"],
    "iOHC_score": ["Hes6", "Rprm", "Grp", "Dlk2", "Tmem255b", "Psph"],
    "OHC_score": ["Lhfpl5", "Gng8", "Grxcr1", "Lmo1", "Evl"],
}

for score_name, genes in hc_score_sets.items():
    genes = [g for g in genes if g in adata_E16_HC.var_names]
    if len(genes) == 0:
        adata_E16_HC.obs[score_name] = 0.0
        continue

    score_matrix = adata_E16_HC[:, genes].X
    if sparse.issparse(score_matrix):
        score_matrix = score_matrix.toarray()
    adata_E16_HC.obs[score_name] = np.asarray(score_matrix).mean(axis=1)

hc_score_mean = (
    adata_E16_HC.obs[
        ["E16_HC_leiden", "IHC_score", "iOHC_score", "OHC_score"]
    ]
    .groupby("E16_HC_leiden")
    .mean()
    .round(3)
)
# display(hc_score_mean)

# =========================================================
# 15. HC UMAP and dotplot, no save

sc.pl.umap(
    adata_E16_HC,
    color="E16_HC_leiden",
    legend_loc="right margin",
    size=24,
    frameon=True,
    show=False,
)
plt.gcf().set_size_inches(5, 5)
plt.close("all")
plt.close()

sc.pl.umap(
    adata_E16_HC,
    color=["IHC_score", "iOHC_score", "OHC_score"],
    cmap="Reds",
    size=24,
    frameon=True,
    show=False,
)
plt.gcf().set_size_inches(12, 4)
plt.close("all")
plt.close()

sc.pl.dotplot(
    adata_E16_HC,
    var_names=hc_marker_dict,
    groupby="E16_HC_leiden",
    standard_scale="var",
    dot_max=0.95,
    dot_min=0.05,
    figsize=(13, 4),
    dendrogram=False,
    show=False,
)
plt.close("all")
plt.close()


# Notebook cell 50

# =========================================================
# 16. write confirmed HC labels back, then recluster non-HC cells

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
sc.settings.verbosity = 1

# =========================================================
# 17. confirmed HC annotation

adata_E16_all.obs["E16_step_label"] = "Unassigned"

hc_final_map = {
    "2": "IHC",
    "1": "iOHC",
    "0": "OHC",
    "3": "OHC",
}

adata_E16_HC.obs["E16_step_label"] = (
    adata_E16_HC.obs["E16_HC_leiden"].astype(str).map(hc_final_map)
)

adata_E16_all.obs.loc[
    adata_E16_HC.obs_names,
    "E16_step_label",
] = adata_E16_HC.obs["E16_step_label"].astype(str).values

print("Confirmed HC labels:")
print(adata_E16_all.obs["E16_step_label"].value_counts())

# =========================================================
# 18. remove confirmed HC and recluster remaining cells

adata_E16_nonHC = adata_E16_all[
    adata_E16_all.obs["E16_step_label"].astype(str) == "Unassigned",
    :
].copy()

print("Non-HC cells for next round:", adata_E16_nonHC.n_obs)

adata_E16_nonHC_work = adata_E16_nonHC.copy()

sc.pp.highly_variable_genes(
    adata_E16_nonHC_work,
    flavor="seurat",
    n_top_genes=min(3000, adata_E16_nonHC_work.n_vars),
)

block_genes = [
    "Oc90", "Otx2",
    "Ptprc", "Lyz2", "Cd74",
    "Hbb-bs", "Hbb-bt", "Hba-a1", "Hba-a2",
    "Ccer2", "Pou4f3", "Pcp4", "Selenom", "Atoh1", "Cib2", "Pvalb",
]
hvg_mask = adata_E16_nonHC_work.var["highly_variable"].values.copy()
hvg_mask &= ~adata_E16_nonHC_work.var_names.isin(block_genes)

adata_E16_nonHC_hvg = adata_E16_nonHC_work[:, hvg_mask].copy()

sc.pp.scale(adata_E16_nonHC_hvg, max_value=10)
sc.tl.pca(adata_E16_nonHC_hvg, svd_solver="arpack", random_state=0)
sc.pp.neighbors(adata_E16_nonHC_hvg, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata_E16_nonHC_hvg, min_dist=0.3, random_state=0)
sc.tl.leiden(adata_E16_nonHC_hvg, resolution=0.65, key_added="E16_nonHC_leiden")

adata_E16_nonHC.obs["E16_nonHC_leiden"] = adata_E16_nonHC_hvg.obs["E16_nonHC_leiden"].astype(str).values
adata_E16_nonHC.obsm["X_umap"] = adata_E16_nonHC_hvg.obsm["X_umap"].copy()

print("Non-HC cluster counts:")
print(adata_E16_nonHC.obs["E16_nonHC_leiden"].value_counts().sort_index())

# =========================================================
# 19. marker score summary for non-HC clusters

nonhc_score_sets = {
    "residual_HC_score": ["Ccer2", "Pou4f3", "Pcp4", "Selenom", "Atoh1", "Cib2", "Pvalb"],
    "L_PsC_score": ["Igfbp3", "S100a1", "Socs2", "Tectb", "Fam159b", "Lockd", "Lfng", "Hes5", "Nrcam", "Fzd9"],
    "M_PsC_score": ["Crym", "Ntf3", "Hmga2", "Tectb", "Lockd", "Fgf20", "Sox2", "Plet1", "Anxa5"],
    "IPhC_score": ["Hes5", "Matn4", "Anxa5", "Fabp7", "Gjb2", "Socs2", "Prss23", "S100a1"],
    "IPC_score": ["Npy", "Gm45645", "S100b", "Fam159b", "Tectb", "Ngfr", "Fgfr3", "Cryab", "Emid1"],
    "HeC_score": ["Kazald1", "Pmch", "Fabp7", "Fam159b", "Gata2", "Nupr1", "Egfl6"],
    "GER_score": ["Crabp1", "Epyc", "Calb1", "Slc12a2", "Adgrg2", "Dcn"],
    "L_GER_score": ["Clu", "Gjb2", "Tecta", "Ctgf", "Col9a1"],
    "ISC_score": ["Igf1", "Smoc2", "Meg3", "Col11a1", "Tm4sf1", "Rgcc", "Matn1"],
    "LER_score": ["Gata2", "Igfbp5", "Lgals1", "Prss23", "Sostdc1", "Bmp4", "Tac1", "Fst"],
    "CC_OSC_score": ["Apoe", "Npnt", "Fbln2", "Bmp4"],
    "IdC_score": ["Cdkn1c", "H19", "Otoa", "Tpm2", "Smoc2", "Foxq1"],
}

for score_name, genes in nonhc_score_sets.items():
    genes = [g for g in genes if g in adata_E16_nonHC.var_names]
    if len(genes) == 0:
        adata_E16_nonHC.obs[score_name] = 0.0
        continue

    x = adata_E16_nonHC[:, genes].X
    if sparse.issparse(x):
        x = x.toarray()
    adata_E16_nonHC.obs[score_name] = np.asarray(x).mean(axis=1)

score_cols = list(nonhc_score_sets.keys())
nonhc_score_mean = (
    adata_E16_nonHC.obs[["E16_nonHC_leiden"] + score_cols]
    .groupby("E16_nonHC_leiden")[score_cols]
    .mean()
    .round(3)
)

# display(nonhc_score_mean)

# =========================================================
# 20. non-HC marker dotplot and UMAP, no save

nonhc_marker_dict = {
    "shared_check": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "L.PsC": ["Igfbp3", "S100a1", "Socs2", "Tectb", "Fam159b", "Lockd", "Lfng", "Hes5", "Nrcam", "Fzd9"],
    "M.PsC": ["Crym", "Ntf3", "Hmga2", "Fgf20", "Plet1", "Anxa5"],
    "IPhC": ["Hes5", "Matn4", "Anxa5", "Fabp7", "Gjb2", "Prss23"],
    "IPC": ["Npy", "Gm45645", "S100b", "Ngfr", "Fgfr3", "Cryab", "Emid1"],
    "HeC": ["Kazald1", "Pmch", "Gata2", "Nupr1", "Egfl6"],
    "GER": ["Crabp1", "Epyc", "Calb1", "Slc12a2", "Adgrg2", "Dcn"],
    "L.GER": ["Clu", "Gjb2", "Tecta", "Ctgf", "Col9a1"],
    "ISC": ["Igf1", "Smoc2", "Meg3", "Col11a1", "Tm4sf1", "Rgcc", "Matn1"],
    "LER_CC_IdC": ["Bmp4", "Tac1", "Fst", "Apoe", "Npnt", "Fbln2", "Cdkn1c", "H19", "Otoa", "Tpm2", "Foxq1"],
    "Exclude_check": ["Oc90", "Otx2", "Ptprc", "Lyz2", "Cd74", "Hbb-bs", "Hba-a1", "Ccer2", "Pou4f3", "Pcp4"],
}
nonhc_marker_dict = {
    k: [g for g in v if g in adata_E16_nonHC.var_names]
    for k, v in nonhc_marker_dict.items()
}
nonhc_marker_dict = {k: v for k, v in nonhc_marker_dict.items() if len(v) > 0}

sc.pl.umap(
    adata_E16_nonHC,
    color="E16_nonHC_leiden",
    legend_loc="right margin",
    size=8,
    frameon=True,
    show=False,
)
plt.gcf().set_size_inches(5, 5)
plt.close("all")
plt.close()

sc.pl.dotplot(
    adata_E16_nonHC,
    var_names=nonhc_marker_dict,
    groupby="E16_nonHC_leiden",
    standard_scale="var",
    dot_max=0.9,
    dot_min=0.05,
    figsize=(21, 6),
    dendrogram=False,
    show=False,
)
plt.close("all")
plt.close()


# Notebook cell 51

# =========================================================
# 21. targeted reclustering: PsC / IPC / IPhC candidate branch
# parent clusters:
#   3 = mixed PsC / IPC / IPhC signal
#   8 = strong IPhC-like reference branch

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
sc.settings.verbosity = 1

psc_ipc_parent_clusters = ["3", "8"]

adata_E16_PsC_IPC = adata_E16_nonHC[
    adata_E16_nonHC.obs["E16_nonHC_leiden"].astype(str).isin(psc_ipc_parent_clusters),
    :
].copy()

print("PsC / IPC / IPhC candidate cells:", adata_E16_PsC_IPC.n_obs)
print("parent nonHC clusters:")
print(adata_E16_PsC_IPC.obs["E16_nonHC_leiden"].value_counts().sort_index())

# =========================================================
# 22. recluster candidate branch

adata_E16_PsC_IPC_work = adata_E16_PsC_IPC.copy()

sc.pp.highly_variable_genes(
    adata_E16_PsC_IPC_work,
    flavor="seurat",
    n_top_genes=min(2000, adata_E16_PsC_IPC_work.n_vars),
)

block_genes = [
    "Oc90", "Otx2",
    "Ptprc", "Lyz2", "Cd74",
    "Hbb-bs", "Hbb-bt", "Hba-a1", "Hba-a2",
    "Ccer2", "Pou4f3", "Pcp4", "Selenom", "Atoh1", "Cib2", "Pvalb",
]
hvg_mask = adata_E16_PsC_IPC_work.var["highly_variable"].values.copy()
hvg_mask &= ~adata_E16_PsC_IPC_work.var_names.isin(block_genes)

adata_E16_PsC_IPC_hvg = adata_E16_PsC_IPC_work[:, hvg_mask].copy()

sc.pp.scale(adata_E16_PsC_IPC_hvg, max_value=10)
sc.tl.pca(adata_E16_PsC_IPC_hvg, svd_solver="arpack", random_state=0)
sc.pp.neighbors(adata_E16_PsC_IPC_hvg, n_neighbors=8, n_pcs=20)
sc.tl.umap(adata_E16_PsC_IPC_hvg, min_dist=1.05, random_state=0)
sc.tl.leiden(adata_E16_PsC_IPC_hvg, resolution=0.65, key_added="E16_PsC_IPC_leiden")

adata_E16_PsC_IPC.obs["E16_PsC_IPC_leiden"] = (
    adata_E16_PsC_IPC_hvg.obs["E16_PsC_IPC_leiden"].astype(str).values
)
adata_E16_PsC_IPC.obsm["X_umap"] = adata_E16_PsC_IPC_hvg.obsm["X_umap"].copy()

print("subcluster counts:")
print(adata_E16_PsC_IPC.obs["E16_PsC_IPC_leiden"].value_counts().sort_index())

print("subcluster x parent:")
print(pd.crosstab(
    adata_E16_PsC_IPC.obs["E16_PsC_IPC_leiden"],
    adata_E16_PsC_IPC.obs["E16_nonHC_leiden"],
))

# =========================================================
# 23. marker score summary

psc_ipc_score_sets = {
    "L_PsC_score": ["Igfbp3", "S100a1", "Socs2", "Tectb", "Fam159b", "Lockd", "Lfng", "Hes5", "Nrcam", "Fzd9"],
    "M_PsC_score": ["Crym", "Ntf3", "Hmga2", "Tectb", "Lockd", "Fgf20", "Sox2", "Plet1", "Anxa5"],
    "IPC_score": ["Npy", "Gm45645", "S100b", "Fam159b", "Tectb", "Ngfr", "Fgfr3", "Cryab", "Emid1"],
    "IPhC_score": ["Hes5", "Matn4", "Anxa5", "Fabp7", "Gjb2", "Socs2", "Prss23", "S100a1"],
    "HeC_score": ["Kazald1", "Pmch", "Fabp7", "Fam159b", "Gata2", "Nupr1", "Egfl6"],
    "GER_LGER_score": ["Crabp1", "Epyc", "Calb1", "Clu", "Gjb2", "Tecta", "Ctgf", "Col9a1"],
    "LER_CC_score": ["Gata2", "Igfbp5", "Lgals1", "Prss23", "Sostdc1", "Bmp4", "Apoe", "Npnt", "Fbln2"],
    "residual_HC_score": ["Ccer2", "Pou4f3", "Pcp4", "Selenom", "Atoh1", "Cib2", "Pvalb"],
}

for score_name, genes in psc_ipc_score_sets.items():
    genes = [g for g in genes if g in adata_E16_PsC_IPC.var_names]
    if len(genes) == 0:
        adata_E16_PsC_IPC.obs[score_name] = 0.0
        continue

    x = adata_E16_PsC_IPC[:, genes].X
    if sparse.issparse(x):
        x = x.toarray()
    adata_E16_PsC_IPC.obs[score_name] = np.asarray(x).mean(axis=1)

score_cols = list(psc_ipc_score_sets.keys())
psc_ipc_score_mean = (
    adata_E16_PsC_IPC.obs[["E16_PsC_IPC_leiden"] + score_cols]
    .groupby("E16_PsC_IPC_leiden")[score_cols]
    .mean()
    .round(3)
)

# display(psc_ipc_score_mean)

# =========================================================
# 24. marker dotplot and UMAP, no save

psc_ipc_marker_dict = {
    "shared_check": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "L.PsC": ["Igfbp3", "S100a1", "Socs2", "Tectb", "Fam159b", "Lockd", "Lfng", "Hes5", "Nrcam", "Fzd9"],
    "M.PsC": ["Crym", "Ntf3", "Hmga2", "Fgf20", "Plet1", "Anxa5"],
    "IPC": ["Npy", "Gm45645", "S100b", "Ngfr", "Fgfr3", "Cryab", "Emid1"],
    "IPhC": ["Hes5", "Matn4", "Anxa5", "Fabp7", "Gjb2", "Prss23"],
    "HeC_check": ["Kazald1", "Pmch", "Gata2", "Nupr1", "Egfl6"],
    "GER_LGER_check": ["Crabp1", "Epyc", "Calb1", "Clu", "Gjb2", "Tecta", "Ctgf", "Col9a1"],
    "LER_CC_check": ["Bmp4", "Apoe", "Npnt", "Fbln2", "Igfbp5", "Lgals1"],
    "Exclude_check": ["Oc90", "Otx2", "Ptprc", "Cd74", "Hbb-bs", "Hba-a1", "Ccer2", "Pou4f3", "Pcp4"],
}
psc_ipc_marker_dict = {
    k: [g for g in v if g in adata_E16_PsC_IPC.var_names]
    for k, v in psc_ipc_marker_dict.items()
}
psc_ipc_marker_dict = {k: v for k, v in psc_ipc_marker_dict.items() if len(v) > 0}

sc.pl.umap(
    adata_E16_PsC_IPC,
    color="E16_PsC_IPC_leiden",
    legend_loc="right margin",
    size=16,
    frameon=True,
    show=False,
)
plt.gcf().set_size_inches(5, 5)
plt.close("all")
plt.close()

sc.pl.dotplot(
    adata_E16_PsC_IPC,
    var_names=psc_ipc_marker_dict,
    groupby="E16_PsC_IPC_leiden",
    standard_scale="var",
    dot_max=0.9,
    dot_min=0.05,
    figsize=(18, 5),
    dendrogram=False,
    show=False,
)
plt.close("all")
plt.close()


# Notebook cell 52

# =========================================================
# 29. focused reclustering for remaining mixed PsC clusters 0 / 1 / 2

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
sc.settings.verbosity = 1

mixed_lpsc_clusters = ["0", "1", "2"]

adata_E16_LPsC_mix = adata_E16_PsC_IPC[
    adata_E16_PsC_IPC.obs["E16_PsC_IPC_leiden"].astype(str).isin(mixed_lpsc_clusters),
    :
].copy()

print("Remaining mixed PsC cells:", adata_E16_LPsC_mix.n_obs)
print("previous clusters:")
print(adata_E16_LPsC_mix.obs["E16_PsC_IPC_leiden"].value_counts().sort_index())

# =========================================================
# 30. recluster remaining mixed PsC cells

adata_E16_LPsC_mix_work = adata_E16_LPsC_mix.copy()

sc.pp.highly_variable_genes(
    adata_E16_LPsC_mix_work,
    flavor="seurat",
    n_top_genes=min(2200, adata_E16_LPsC_mix_work.n_vars),
)

block_genes = [
    "Oc90", "Otx2",
    "Ptprc", "Lyz2", "Cd74",
    "Hbb-bs", "Hbb-bt", "Hba-a1", "Hba-a2",
    "Ccer2", "Pou4f3", "Pcp4", "Selenom", "Atoh1", "Cib2", "Pvalb",
]
hvg_mask = adata_E16_LPsC_mix_work.var["highly_variable"].values.copy()
hvg_mask &= ~adata_E16_LPsC_mix_work.var_names.isin(block_genes)

adata_E16_LPsC_mix_hvg = adata_E16_LPsC_mix_work[:, hvg_mask].copy()

sc.pp.scale(adata_E16_LPsC_mix_hvg, max_value=10)
sc.tl.pca(adata_E16_LPsC_mix_hvg, svd_solver="arpack", random_state=0)
sc.pp.neighbors(adata_E16_LPsC_mix_hvg, n_neighbors=8, n_pcs=18)
sc.tl.umap(adata_E16_LPsC_mix_hvg, min_dist=0.18, random_state=0)
sc.tl.leiden(adata_E16_LPsC_mix_hvg, resolution=0.85, key_added="E16_LPsC_mix_leiden")

adata_E16_LPsC_mix.obs["E16_LPsC_mix_leiden"] = (
    adata_E16_LPsC_mix_hvg.obs["E16_LPsC_mix_leiden"].astype(str).values
)
adata_E16_LPsC_mix.obsm["X_umap"] = adata_E16_LPsC_mix_hvg.obsm["X_umap"].copy()

print("new subcluster counts:")
print(adata_E16_LPsC_mix.obs["E16_LPsC_mix_leiden"].value_counts().sort_index())

print("new subcluster x previous:")
print(pd.crosstab(
    adata_E16_LPsC_mix.obs["E16_LPsC_mix_leiden"],
    adata_E16_LPsC_mix.obs["E16_PsC_IPC_leiden"],
))

# =========================================================
# 31. marker score summary

lpsc_mix_score_sets = {
    "L_PsC_score": ["Igfbp3", "S100a1", "Socs2", "Tectb", "Fam159b", "Lockd", "Lfng", "Hes5", "Nrcam", "Fzd9"],
    "M_PsC_score": ["Crym", "Ntf3", "Hmga2", "Tectb", "Lockd", "Fgf20", "Sox2", "Plet1", "Anxa5"],
    "IPC_score": ["Npy", "Gm45645", "S100b", "Fam159b", "Tectb", "Ngfr", "Fgfr3", "Cryab", "Emid1"],
    "IPhC_score": ["Hes5", "Matn4", "Anxa5", "Fabp7", "Gjb2", "Socs2", "Prss23", "S100a1"],
    "GER_LGER_score": ["Crabp1", "Epyc", "Calb1", "Clu", "Gjb2", "Tecta", "Ctgf", "Col9a1"],
    "LER_CC_score": ["Gata2", "Igfbp5", "Lgals1", "Prss23", "Sostdc1", "Bmp4", "Apoe", "Npnt", "Fbln2"],
    "residual_HC_score": ["Ccer2", "Pou4f3", "Pcp4", "Selenom", "Atoh1", "Cib2", "Pvalb"],
}

for score_name, genes in lpsc_mix_score_sets.items():
    genes = [g for g in genes if g in adata_E16_LPsC_mix.var_names]
    if len(genes) == 0:
        adata_E16_LPsC_mix.obs[score_name] = 0.0
        continue

    x = adata_E16_LPsC_mix[:, genes].X
    if sparse.issparse(x):
        x = x.toarray()
    adata_E16_LPsC_mix.obs[score_name] = np.asarray(x).mean(axis=1)

score_cols = list(lpsc_mix_score_sets.keys())
lpsc_mix_score_mean = (
    adata_E16_LPsC_mix.obs[["E16_LPsC_mix_leiden"] + score_cols]
    .groupby("E16_LPsC_mix_leiden")[score_cols]
    .mean()
    .round(3)
)

# display(lpsc_mix_score_mean)

# =========================================================
# 32. marker dotplot and UMAP, no save

lpsc_mix_marker_dict = {
    "shared_check": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "L.PsC": ["Igfbp3", "S100a1", "Socs2", "Tectb", "Fam159b", "Lockd", "Lfng", "Hes5", "Nrcam", "Fzd9"],
    "M.PsC": ["Crym", "Ntf3", "Hmga2", "Fgf20", "Plet1", "Anxa5"],
    "IPC": ["Npy", "Gm45645", "S100b", "Ngfr", "Fgfr3", "Cryab", "Emid1"],
    "IPhC_check": ["Hes5", "Matn4", "Anxa5", "Fabp7", "Gjb2", "Prss23"],
    "GER_LGER_check": ["Crabp1", "Epyc", "Calb1", "Clu", "Gjb2", "Tecta", "Ctgf", "Col9a1"],
    "LER_CC_check": ["Bmp4", "Apoe", "Npnt", "Fbln2", "Igfbp5", "Lgals1", "Tac1", "Fst"],
    "Exclude_check": ["Oc90", "Otx2", "Ptprc", "Cd74", "Hbb-bs", "Hba-a1", "Ccer2", "Pou4f3", "Pcp4"],
}
lpsc_mix_marker_dict = {
    k: [g for g in v if g in adata_E16_LPsC_mix.var_names]
    for k, v in lpsc_mix_marker_dict.items()
}
lpsc_mix_marker_dict = {k: v for k, v in lpsc_mix_marker_dict.items() if len(v) > 0}

sc.pl.umap(
    adata_E16_LPsC_mix,
    color="E16_LPsC_mix_leiden",
    legend_loc="right margin",
    size=18,
    frameon=True,
    show=False,
)
plt.gcf().set_size_inches(5, 5)
plt.close("all")
plt.close()

sc.pl.dotplot(
    adata_E16_LPsC_mix,
    var_names=lpsc_mix_marker_dict,
    groupby="E16_LPsC_mix_leiden",
    standard_scale="var",
    dot_max=0.9,
    dot_min=0.05,
    figsize=(18, 5),
    dendrogram=False,
    show=False,
)
plt.close("all")
plt.close()


# Notebook cell 53

# =========================================================
# 33. write confirmed PsC / IPC / IPhC labels back, then recluster remaining non-HC cells

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
sc.settings.verbosity = 1

# Make sure current working label exists on the non-HC object.
adata_E16_nonHC.obs["E16_step_label"] = "Unassigned"

# =========================================================
# 34. confirmed labels from targeted PsC / IPC / IPhC analyses

# Clear IPhC from first PsC/IPC subclustering.
iphc_names = adata_E16_PsC_IPC.obs_names[
    adata_E16_PsC_IPC.obs["E16_PsC_IPC_leiden"].astype(str).isin(["3"])
]

# M.PsC and first IPC from higher-res PsC/IPC result.
mpsc_names = adata_E16_PsC_IPC.obs_names[
    adata_E16_PsC_IPC.obs["E16_PsC_IPC_leiden"].astype(str).isin(["4"])
]
ipc_names_main = adata_E16_PsC_IPC.obs_names[
    adata_E16_PsC_IPC.obs["E16_PsC_IPC_leiden"].astype(str).isin(["5"])
]

# L.PsC and residual IPC from focused L.PsC-mix result.
lpsc_names = adata_E16_LPsC_mix.obs_names[
    adata_E16_LPsC_mix.obs["E16_LPsC_mix_leiden"].astype(str).isin(["0", "1", "2", "3", "4"])
]
ipc_names_rescue = adata_E16_LPsC_mix.obs_names[
    adata_E16_LPsC_mix.obs["E16_LPsC_mix_leiden"].astype(str).isin(["5"])
]

# Cluster 6 from L.PsC-mix is GER/L.GER-like, leave it unassigned for the next round.
for names, label in [
    (iphc_names, "IPhC"),
    (mpsc_names, "M.PsC"),
    (ipc_names_main, "IPC"),
    (lpsc_names, "L.PsC"),
    (ipc_names_rescue, "IPC"),
]:
    names = [n for n in names if n in adata_E16_nonHC.obs_names]
    adata_E16_nonHC.obs.loc[names, "E16_step_label"] = label

print("Confirmed non-HC labels:")
print(adata_E16_nonHC.obs["E16_step_label"].value_counts())

# =========================================================
# 35. remaining non-HC cells for GER / L.GER / ISC / LER / CC_OSC / IdC / HeC

adata_E16_epith_rest = adata_E16_nonHC[
    adata_E16_nonHC.obs["E16_step_label"].astype(str) == "Unassigned",
    :
].copy()

print("Remaining cells:", adata_E16_epith_rest.n_obs)
print("parent nonHC clusters:")
print(adata_E16_epith_rest.obs["E16_nonHC_leiden"].value_counts().sort_index())

# =========================================================
# 36. recluster remaining epithelial cells

adata_E16_epith_rest_work = adata_E16_epith_rest.copy()

sc.pp.highly_variable_genes(
    adata_E16_epith_rest_work,
    flavor="seurat",
    n_top_genes=min(3000, adata_E16_epith_rest_work.n_vars),
)

block_genes = [
    "Oc90", "Otx2",
    "Ptprc", "Lyz2", "Cd74",
    "Hbb-bs", "Hbb-bt", "Hba-a1", "Hba-a2",
    "Ccer2", "Pou4f3", "Pcp4", "Selenom", "Atoh1", "Cib2", "Pvalb",
]
hvg_mask = adata_E16_epith_rest_work.var["highly_variable"].values.copy()
hvg_mask &= ~adata_E16_epith_rest_work.var_names.isin(block_genes)

adata_E16_epith_rest_hvg = adata_E16_epith_rest_work[:, hvg_mask].copy()

sc.pp.scale(adata_E16_epith_rest_hvg, max_value=10)
sc.tl.pca(adata_E16_epith_rest_hvg, svd_solver="arpack", random_state=0)
sc.pp.neighbors(adata_E16_epith_rest_hvg, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata_E16_epith_rest_hvg, min_dist=0.3, random_state=0)
sc.tl.leiden(adata_E16_epith_rest_hvg, resolution=0.7, key_added="E16_rest_leiden")

adata_E16_epith_rest.obs["E16_rest_leiden"] = (
    adata_E16_epith_rest_hvg.obs["E16_rest_leiden"].astype(str).values
)
adata_E16_epith_rest.obsm["X_umap"] = adata_E16_epith_rest_hvg.obsm["X_umap"].copy()

print("Rest cluster counts:")
print(adata_E16_epith_rest.obs["E16_rest_leiden"].value_counts().sort_index())

print("rest cluster x original nonHC:")
print(pd.crosstab(
    adata_E16_epith_rest.obs["E16_rest_leiden"],
    adata_E16_epith_rest.obs["E16_nonHC_leiden"],
))

# =========================================================
# 37. score summary for remaining epithelial classes

rest_score_sets = {
    "HeC_score": ["Kazald1", "Pmch", "Fabp7", "Fam159b", "Gata2", "Nupr1", "Egfl6"],
    "GER_score": ["Crabp1", "Epyc", "Calb1", "Slc12a2", "Adgrg2", "Dcn"],
    "L_GER_score": ["Clu", "Gjb2", "Tecta", "Ctgf", "Col9a1"],
    "ISC_score": ["Igf1", "Smoc2", "Meg3", "Col11a1", "Tm4sf1", "Rgcc", "Matn1"],
    "LER_score": ["Gata2", "Igfbp5", "Lgals1", "Prss23", "Sostdc1", "Bmp4", "Tac1", "Fst"],
    "CC_OSC_score": ["Apoe", "Npnt", "Fbln2", "Bmp4"],
    "IdC_score": ["Cdkn1c", "H19", "Otoa", "Tpm2", "Smoc2", "Foxq1"],
    "residual_PsC_score": ["Igfbp3", "S100a1", "Socs2", "Tectb", "Crym", "Ntf3", "Hmga2"],
    "residual_HC_score": ["Ccer2", "Pou4f3", "Pcp4", "Selenom", "Atoh1", "Cib2", "Pvalb"],
}

for score_name, genes in rest_score_sets.items():
    genes = [g for g in genes if g in adata_E16_epith_rest.var_names]
    if len(genes) == 0:
        adata_E16_epith_rest.obs[score_name] = 0.0
        continue

    x = adata_E16_epith_rest[:, genes].X
    if sparse.issparse(x):
        x = x.toarray()
    adata_E16_epith_rest.obs[score_name] = np.asarray(x).mean(axis=1)

score_cols = list(rest_score_sets.keys())
rest_score_mean = (
    adata_E16_epith_rest.obs[["E16_rest_leiden"] + score_cols]
    .groupby("E16_rest_leiden")[score_cols]
    .mean()
    .round(3)
)

# display(rest_score_mean)

# =========================================================
# 38. UMAP and dotplot for remaining epithelial classes, no save

rest_marker_dict = {
    "shared_check": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "HeC": ["Kazald1", "Pmch", "Fabp7", "Fam159b", "Gata2", "Nupr1", "Egfl6"],
    "GER": ["Crabp1", "Epyc", "Calb1", "Slc12a2", "Adgrg2", "Dcn"],
    "L.GER": ["Clu", "Gjb2", "Tecta", "Ctgf", "Col9a1"],
    "ISC": ["Igf1", "Smoc2", "Meg3", "Col11a1", "Tm4sf1", "Rgcc", "Matn1"],
    "LER": ["Gata2", "Igfbp5", "Lgals1", "Prss23", "Sostdc1", "Bmp4", "Tac1", "Fst"],
    "CC_OSC": ["Apoe", "Npnt", "Fbln2", "Bmp4"],
    "IdC": ["Cdkn1c", "H19", "Otoa", "Tpm2", "Smoc2", "Foxq1"],
    "Residual_check": ["Igfbp3", "S100a1", "Socs2", "Tectb", "Crym", "Ntf3", "Hmga2"],
    "Exclude_check": ["Oc90", "Otx2", "Ptprc", "Cd74", "Hbb-bs", "Hba-a1", "Ccer2", "Pou4f3", "Pcp4"],
}
rest_marker_dict = {
    k: [g for g in v if g in adata_E16_epith_rest.var_names]
    for k, v in rest_marker_dict.items()
}
rest_marker_dict = {k: v for k, v in rest_marker_dict.items() if len(v) > 0}

sc.pl.umap(
    adata_E16_epith_rest,
    color="E16_rest_leiden",
    legend_loc="right margin",
    size=8,
    frameon=True,
    show=False,
)
plt.gcf().set_size_inches(5, 5)
plt.close("all")
plt.close()

sc.pl.dotplot(
    adata_E16_epith_rest,
    var_names=rest_marker_dict,
    groupby="E16_rest_leiden",
    standard_scale="var",
    dot_max=0.9,
    dot_min=0.05,
    figsize=(20, 6),
    dendrogram=False,
    show=False,
)
plt.close("all")
plt.close()


# Notebook cell 54

# =========================================================
# 39. targeted reclustering for LER / CC_OSC / IdC branches

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
sc.settings.verbosity = 1

target_rest_clusters = ["2", "3", "4"]

adata_E16_LER_IdC = adata_E16_epith_rest[
    adata_E16_epith_rest.obs["E16_rest_leiden"].astype(str).isin(target_rest_clusters),
    :
].copy()

print("LER / CC_OSC / IdC candidate cells:", adata_E16_LER_IdC.n_obs)
print("parent rest clusters:")
print(adata_E16_LER_IdC.obs["E16_rest_leiden"].value_counts().sort_index())

# =========================================================
# 40. recluster LER / CC_OSC / IdC candidate cells

adata_E16_LER_IdC_work = adata_E16_LER_IdC.copy()

sc.pp.highly_variable_genes(
    adata_E16_LER_IdC_work,
    flavor="seurat",
    n_top_genes=min(2500, adata_E16_LER_IdC_work.n_vars),
)

block_genes = [
    "Oc90", "Otx2",
    "Ptprc", "Lyz2", "Cd74",
    "Hbb-bs", "Hbb-bt", "Hba-a1", "Hba-a2",
    "Ccer2", "Pou4f3", "Pcp4", "Selenom", "Atoh1", "Cib2", "Pvalb",
]
hvg_mask = adata_E16_LER_IdC_work.var["highly_variable"].values.copy()
hvg_mask &= ~adata_E16_LER_IdC_work.var_names.isin(block_genes)

adata_E16_LER_IdC_hvg = adata_E16_LER_IdC_work[:, hvg_mask].copy()

sc.pp.scale(adata_E16_LER_IdC_hvg, max_value=10)
sc.tl.pca(adata_E16_LER_IdC_hvg, svd_solver="arpack", random_state=0)
sc.pp.neighbors(adata_E16_LER_IdC_hvg, n_neighbors=10, n_pcs=20)
sc.tl.umap(adata_E16_LER_IdC_hvg, min_dist=0.22, random_state=0)
sc.tl.leiden(adata_E16_LER_IdC_hvg, resolution=0.75, key_added="E16_LER_IdC_leiden")

adata_E16_LER_IdC.obs["E16_LER_IdC_leiden"] = (
    adata_E16_LER_IdC_hvg.obs["E16_LER_IdC_leiden"].astype(str).values
)
adata_E16_LER_IdC.obsm["X_umap"] = adata_E16_LER_IdC_hvg.obsm["X_umap"].copy()

print("new subcluster counts:")
print(adata_E16_LER_IdC.obs["E16_LER_IdC_leiden"].value_counts().sort_index())

print("new subcluster x parent rest:")
print(pd.crosstab(
    adata_E16_LER_IdC.obs["E16_LER_IdC_leiden"],
    adata_E16_LER_IdC.obs["E16_rest_leiden"],
))

# =========================================================
# 41. score summary

ler_idc_score_sets = {
    "ISC_score": ["Igf1", "Smoc2", "Meg3", "Col11a1", "Tm4sf1", "Rgcc", "Matn1"],
    "IdC_score": ["Cdkn1c", "H19", "Otoa", "Tpm2", "Smoc2", "Foxq1"],
    "LER_score": ["Gata2", "Igfbp5", "Lgals1", "Prss23", "Sostdc1", "Bmp4", "Tac1", "Fst"],
    "CC_OSC_score": ["Apoe", "Npnt", "Fbln2", "Bmp4"],
    "GER_score": ["Crabp1", "Epyc", "Calb1", "Slc12a2", "Adgrg2", "Dcn"],
    "L_GER_score": ["Clu", "Gjb2", "Tecta", "Ctgf", "Col9a1"],
    "HeC_score": ["Kazald1", "Pmch", "Fabp7", "Fam159b", "Gata2", "Nupr1", "Egfl6"],
    "residual_PsC_score": ["Igfbp3", "S100a1", "Socs2", "Tectb", "Crym", "Ntf3", "Hmga2"],
}

for score_name, genes in ler_idc_score_sets.items():
    genes = [g for g in genes if g in adata_E16_LER_IdC.var_names]
    if len(genes) == 0:
        adata_E16_LER_IdC.obs[score_name] = 0.0
        continue

    x = adata_E16_LER_IdC[:, genes].X
    if sparse.issparse(x):
        x = x.toarray()
    adata_E16_LER_IdC.obs[score_name] = np.asarray(x).mean(axis=1)

score_cols = list(ler_idc_score_sets.keys())
ler_idc_score_mean = (
    adata_E16_LER_IdC.obs[["E16_LER_IdC_leiden"] + score_cols]
    .groupby("E16_LER_IdC_leiden")[score_cols]
    .mean()
    .round(3)
)

# display(ler_idc_score_mean)

# =========================================================
# 42. UMAP and dotplot, no save

ler_idc_marker_dict = {
    "shared_check": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "ISC": ["Igf1", "Smoc2", "Meg3", "Col11a1", "Tm4sf1", "Rgcc", "Matn1"],
    "IdC": ["Cdkn1c", "H19", "Otoa", "Tpm2", "Smoc2", "Foxq1"],
    "LER": ["Gata2", "Igfbp5", "Lgals1", "Prss23", "Sostdc1", "Bmp4", "Tac1", "Fst"],
    "CC_OSC": ["Apoe", "Npnt", "Fbln2", "Bmp4"],
    "GER_LGER_check": ["Crabp1", "Epyc", "Calb1", "Clu", "Gjb2", "Tecta", "Ctgf", "Col9a1"],
    "HeC_check": ["Kazald1", "Pmch", "Fabp7", "Fam159b", "Nupr1", "Egfl6"],
    "Residual_check": ["Igfbp3", "S100a1", "Socs2", "Tectb", "Crym", "Ntf3", "Hmga2"],
    "Exclude_check": ["Oc90", "Otx2", "Ptprc", "Cd74", "Hbb-bs", "Hba-a1", "Ccer2", "Pou4f3", "Pcp4"],
}
ler_idc_marker_dict = {
    k: [g for g in v if g in adata_E16_LER_IdC.var_names]
    for k, v in ler_idc_marker_dict.items()
}
ler_idc_marker_dict = {k: v for k, v in ler_idc_marker_dict.items() if len(v) > 0}

sc.pl.umap(
    adata_E16_LER_IdC,
    color="E16_LER_IdC_leiden",
    legend_loc="right margin",
    size=10,
    frameon=True,
    show=False,
)
plt.gcf().set_size_inches(5, 5)
plt.close("all")
plt.close()

sc.pl.dotplot(
    adata_E16_LER_IdC,
    var_names=ler_idc_marker_dict,
    groupby="E16_LER_IdC_leiden",
    standard_scale="var",
    dot_max=0.9,
    dot_min=0.05,
    figsize=(18, 5),
    dendrogram=False,
    show=False,
)
plt.close("all")
plt.close()


# Notebook cell 55

# =========================================================
# 43. build final E16.5 annotation for visual validation only
# no file / figure saving in this block

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
sc.settings.verbosity = 1

# =========================================================
# 44. initialize final object from cleaned, filtered all-cell object

adata_E16_final_check = adata_E16_all.copy()
adata_E16_final_check.obs["final_celltype"] = "Unassigned"

# =========================================================
# 45. write confirmed HC labels

for label in ["IHC", "iOHC", "OHC"]:
    names = adata_E16_HC.obs_names[
        adata_E16_HC.obs["E16_step_label"].astype(str) == label
    ]
    adata_E16_final_check.obs.loc[names, "final_celltype"] = label

# =========================================================
# 46. write confirmed PsC / IPC / IPhC labels

confirmed_nonhc_map = {
    "IPhC": iphc_names,
    "M.PsC": mpsc_names,
    "IPC": ipc_names_main.union(ipc_names_rescue),
    "L.PsC": lpsc_names,
}

for label, names in confirmed_nonhc_map.items():
    names = [n for n in names if n in adata_E16_final_check.obs_names]
    adata_E16_final_check.obs.loc[names, "final_celltype"] = label

# =========================================================
# 47. write GER / L.GER / HeC labels from rest clustering

rest_label_map = {
    "0": "GER",
    "1": "GER",
    "5": "L.GER",
    "6": "L.GER",
    "7": "L.GER",
    "8": "HeC",
}

for cluster_id, label in rest_label_map.items():
    names = adata_E16_epith_rest.obs_names[
        adata_E16_epith_rest.obs["E16_rest_leiden"].astype(str) == cluster_id
    ]
    names = [n for n in names if n in adata_E16_final_check.obs_names]
    adata_E16_final_check.obs.loc[names, "final_celltype"] = label

# =========================================================
# 48. write ISC / IdC / LER / CC_OSC labels from targeted clustering

ler_idc_label_map = {
    "0": "ISC",
    "2": "ISC",
    "5": "ISC",
    "1": "IdC",
    "3": "LER",
    "6": "LER",
    "4": "CC/OSC",
}

for cluster_id, label in ler_idc_label_map.items():
    names = adata_E16_LER_IdC.obs_names[
        adata_E16_LER_IdC.obs["E16_LER_IdC_leiden"].astype(str) == cluster_id
    ]
    names = [n for n in names if n in adata_E16_final_check.obs_names]
    adata_E16_final_check.obs.loc[names, "final_celltype"] = label

# =========================================================
# 49. final filtered object for validation

final_order = [
    "IHC",
    "iOHC",
    "OHC",
    "L.PsC",
    "M.PsC",
    "IPhC",
    "IPC",
    "HeC",
    "GER",
    "L.GER",
    "ISC",
    "LER",
    "CC/OSC",
    "IdC",
]

adata_E16_plot = adata_E16_final_check[
    adata_E16_final_check.obs["final_celltype"].astype(str).isin(final_order),
    :
].copy()

adata_E16_plot.obs["final_celltype"] = pd.Categorical(
    adata_E16_plot.obs["final_celltype"].astype(str),
    categories=final_order,
    ordered=True,
)

print("Final E16.5 label counts:")
print(adata_E16_plot.obs["final_celltype"].value_counts().reindex(final_order))

print("Unassigned remaining in cleaned object:")
print((adata_E16_final_check.obs["final_celltype"].astype(str) == "Unassigned").sum())

# =========================================================
# 50. recompute UMAP on final labelled object

adata_E16_plot_work = adata_E16_plot.copy()

sc.pp.highly_variable_genes(
    adata_E16_plot_work,
    flavor="seurat",
    n_top_genes=min(3000, adata_E16_plot_work.n_vars),
)

block_genes = [
    "Oc90", "Otx2",
    "Ptprc", "Lyz2", "Cd74",
    "Hbb-bs", "Hbb-bt", "Hba-a1", "Hba-a2",
]
hvg_mask = adata_E16_plot_work.var["highly_variable"].values.copy()
hvg_mask &= ~adata_E16_plot_work.var_names.isin(block_genes)

adata_E16_plot_hvg = adata_E16_plot_work[:, hvg_mask].copy()

sc.pp.scale(adata_E16_plot_hvg, max_value=10)
sc.tl.pca(adata_E16_plot_hvg, svd_solver="arpack", random_state=0)
sc.pp.neighbors(adata_E16_plot_hvg, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata_E16_plot_hvg, min_dist=0.3, random_state=0)

adata_E16_plot.obsm["X_umap"] = adata_E16_plot_hvg.obsm["X_umap"].copy()

# =========================================================
# 51. final validation UMAP and dotplot, no save

final_marker_dict = {
    "shared_check": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "IHC": ["Acbd7", "Fgf8", "Pvalb", "Ccl21a", "Smpx"],
    "iOHC": ["Hes6", "Rprm", "Grp", "Dlk2", "Tmem255b", "Psph"],
    "OHC": ["Lhfpl5", "Gng8", "Grxcr1", "Lmo1", "Evl", "Pou4f3", "Ccer2"],
    "L.PsC": ["Igfbp3", "S100a1", "Socs2", "Tectb", "Fam159b", "Lockd", "Lfng", "Hes5"],
    "M.PsC": ["Crym", "Ntf3", "Hmga2", "Fgf20", "Plet1", "Anxa5"],
    "IPhC": ["Hes5", "Matn4", "Anxa5", "Fabp7", "Gjb2", "Prss23"],
    "IPC": ["Npy", "Gm45645", "S100b", "Ngfr", "Fgfr3", "Cryab", "Emid1"],
    "HeC": ["Kazald1", "Pmch", "Gata2", "Nupr1", "Egfl6"],
    "GER": ["Crabp1", "Epyc", "Calb1", "Slc12a2", "Adgrg2", "Dcn"],
    "L.GER": ["Clu", "Gjb2", "Tecta", "Ctgf", "Col9a1"],
    "ISC": ["Igf1", "Smoc2", "Meg3", "Col11a1", "Tm4sf1", "Rgcc", "Matn1"],
    "LER": ["Gata2", "Igfbp5", "Lgals1", "Prss23", "Sostdc1", "Bmp4", "Tac1", "Fst"],
    "CC/OSC": ["Apoe", "Npnt", "Fbln2", "Bmp4"],
    "IdC": ["Cdkn1c", "H19", "Otoa", "Tpm2", "Smoc2", "Foxq1"],
    "Exclude_check": [
        "Oc90", "Otx2",
        "Neurod1", "Neurog1", "Prph",
        "Ptprc", "Lyz2", "Cd74",
        "Hbb-bs", "Hbb-bt", "Hba-a1", "Hba-a2",
    ],
}
final_marker_dict = {
    k: [g for g in v if g in adata_E16_plot.var_names]
    for k, v in final_marker_dict.items()
}
final_marker_dict = {k: v for k, v in final_marker_dict.items() if len(v) > 0}

sc.pl.umap(
    adata_E16_plot,
    color="final_celltype",
    legend_loc="right margin",
    size=8,
    frameon=True,
    show=False,
)
plt.gcf().set_size_inches(6, 5)
plt.close("all")
plt.close()

sc.pl.dotplot(
    adata_E16_plot,
    var_names=final_marker_dict,
    groupby="final_celltype",
    standard_scale="var",
    dot_max=0.9,
    dot_min=0.05,
    figsize=(24, 7),
    dendrogram=False,
    show=False,
)
plt.close("all")
plt.close()


# Notebook cell 56

# =========================================================
# 52. save E16.5 final integration h5ad files

harmony_dir = RESULTS_ROOT / "01_stage_clustering" / "harmony_inputs" / "e16_5"
scnvi_dir = STAGE_OBJECT_DIR
harmony_dir.mkdir(parents=True, exist_ok=True)
scnvi_dir.mkdir(parents=True, exist_ok=True)

adata_E16_plot.obs["stage"] = "E16.5"
adata_E16_plot.obs["source_object"] = "E16.5_final_strict_Oc90_Otx2_negative_no_LER2"
adata_E16_plot.obs["final_celltype"] = adata_E16_plot.obs["final_celltype"].astype(str)

obs_cols = [
    "stage",
    "final_celltype",
    "source_object",
    "E16_all_leiden",
]

obs_keep = adata_E16_plot.obs[
    [c for c in obs_cols if c in adata_E16_plot.obs.columns]
].copy()
var_keep = pd.DataFrame(index=adata_E16_plot.var_names.copy())

adata_E16_normalized = ad.AnnData(
    X=adata_E16_plot.X.copy(),
    obs=obs_keep.copy(),
    var=var_keep.copy(),
)
adata_E16_normalized.obs_names = adata_E16_plot.obs_names.copy()
adata_E16_normalized.var_names = adata_E16_plot.var_names.copy()

harmony_normalized_path = harmony_dir / "E16.5_for_Harmony_normalized.h5ad"
scnvi_normalized_path = scnvi_dir / "E16.5_for_scnvi_normalized.h5ad"

adata_E16_normalized.write(harmony_normalized_path)
adata_E16_normalized.write(scnvi_normalized_path)


# Notebook cell 57

# =========================================================
# 53. save E16.5 final UMAP and dotplot

figure_dir = OUTPUT_DIR
figure_dir.mkdir(parents=True, exist_ok=True)

umap_path = figure_dir / "E16.5_final_celltype_umap_strict_no_LER2.png"
dotplot_path = figure_dir / "E16.5_final_celltype_dotplot_strict_no_LER2.png"

sc.pl.umap(
    adata_E16_plot,
    color="final_celltype",
    legend_loc="right margin",
    size=8,
    frameon=True,
    show=False,
)
plt.gcf().set_size_inches(6, 5)
plt.savefig(umap_path, dpi=SAVE_DPI, bbox_inches="tight")
plt.close()

dp = sc.pl.dotplot(
    adata_E16_plot,
    var_names=final_marker_dict,
    groupby="final_celltype",
    standard_scale="var",
    dot_max=0.9,
    dot_min=0.05,
    figsize=(24, 7),
    dendrogram=False,
    return_fig=True,
)
dp.savefig(dotplot_path, dpi=SAVE_DPI, bbox_inches="tight")
plt.close("all")

print("Saved E16.5 final figures.")
print(umap_path)
print(dotplot_path)
