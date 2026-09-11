#1. ===========Purpose and reproducibility settings
"""Preprocess, cluster, annotate, and export the E14.5 integration inputs."""

from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("COCHLEA_DATA_DIR", REPO_ROOT / "Data")).expanduser().resolve()
RESULTS_ROOT = Path(os.environ.get("COCHLEA_RESULTS_DIR", REPO_ROOT / "results")).expanduser().resolve()
STAGE_OBJECT_DIR = RESULTS_ROOT / "01_stage_clustering" / "h5ad_for_integration"
OUTPUT_DIR = RESULTS_ROOT / "01_stage_clustering" / "e14_5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STAGE_OBJECT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 0
SAVE_DPI = 1201
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_STATE))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Notebook cell 39

from pathlib import Path

import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=SAVE_DPI, facecolor="white")
sc.set_figure_params(figsize=(8, 8))

# =========================================================
# 1. paths

base_dir = REPO_ROOT
count_file = DATA_ROOT / "raw" / "e14_5" / "E14_Normalized_Counts.txt"
figure_dir = OUTPUT_DIR

figure_dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# 2. read matrix

df_E14 = pd.read_csv(count_file, sep="\t", index_col=0)

print("\n==================================================")
print("E14.5 matrix basic information")
print("==================================================")
print("matrix shape (genes x cells):", df_E14.shape)
print("total genes:", df_E14.shape[0])
print("total cells:", df_E14.shape[1])

# =========================================================
# 3. clean matrix and build AnnData

df_E14.index = df_E14.index.astype(str)
df_E14.columns = df_E14.columns.astype(str)
df_E14 = df_E14.loc[~df_E14.index.duplicated()].copy()
df_E14 = df_E14.apply(pd.to_numeric, errors="coerce").fillna(0)

X = df_E14.T
adata_E14 = ad.AnnData(X)
adata_E14.obs_names = X.index
adata_E14.var_names = X.columns
adata_E14.obs_names_make_unique()
adata_E14.var_names_make_unique()

adata_E14.obs["sample"] = (
    adata_E14.obs_names.to_series()
    .str.rsplit("_", n=1)
    .str[0]
    .astype("category")
)

print("\n==================================================")
print("AnnData basic information")
print("==================================================")
print(adata_E14)
print(adata_E14.obs["sample"].value_counts().sort_index())

# =========================================================
# 4. basic QC and exclusion filtering

def gene_expr(adata_obj, gene):
    if gene not in adata_obj.var_names:
        return np.zeros(adata_obj.n_obs)
    return np.asarray(adata_obj[:, gene].X).reshape(-1)

def dedupe_marker_dict(marker_dict, var_names):
    marker_dict_clean = {}
    used_genes = set()

    for group, genes in marker_dict.items():
        clean_genes = []
        for gene in genes:
            if gene in var_names and gene not in used_genes:
                clean_genes.append(gene)
                used_genes.add(gene)

        if len(clean_genes) > 0:
            marker_dict_clean[group] = clean_genes

    return marker_dict_clean

def add_mean_scores(adata_obj, score_dict):
    score_cols = []
    for score_name, genes in score_dict.items():
        genes = [g for g in genes if g in adata_obj.var_names]
        if len(genes) > 0:
            adata_obj.obs[score_name] = np.asarray(adata_obj[:, genes].X).mean(axis=1).reshape(-1)
            score_cols.append(score_name)
    return score_cols

adata_E14.var["mt"] = adata_E14.var_names.str.startswith(("mt-", "Mt-", "MT-"))

sc.pp.calculate_qc_metrics(
    adata_E14,
    qc_vars=["mt"],
    percent_top=None,
    log1p=False,
    inplace=True,
)

hb_genes = [
    g for g in ["Hba-a1", "Hba-a2", "Hbb-bs", "Hbb-bt", "Hbb-y", "Hbb-bh1", "Hbb-bh2"]
    if g in adata_E14.var_names
]
immune_genes = [
    g for g in ["Ptprc", "Lyz2", "C1qa", "C1qb", "C1qc", "Tyrobp", "Fcer1g", "Csf1r", "Aif1"]
    if g in adata_E14.var_names
]

adata_E14.obs["Oc90_expr"] = gene_expr(adata_E14, "Oc90")
adata_E14.obs["Otx2_expr"] = gene_expr(adata_E14, "Otx2")
adata_E14.obs["hb_expr"] = (
    np.asarray(adata_E14[:, hb_genes].X).sum(axis=1).reshape(-1)
    if len(hb_genes) > 0 else np.zeros(adata_E14.n_obs)
)
adata_E14.obs["immune_expr"] = (
    np.asarray(adata_E14[:, immune_genes].X).sum(axis=1).reshape(-1)
    if len(immune_genes) > 0 else np.zeros(adata_E14.n_obs)
)

raw_n = adata_E14.n_obs

basic_qc_mask = (
    (adata_E14.obs["n_genes_by_counts"] > 200) &
    (adata_E14.obs["pct_counts_mt"] < 5)
)
after_basic_qc = int(basic_qc_mask.sum())
adata_E14 = adata_E14[basic_qc_mask, :].copy()

otx_oc_mask = (
    (adata_E14.obs["Oc90_expr"] <= 0) &
    (adata_E14.obs["Otx2_expr"] <= 0)
)
after_otx_oc = int(otx_oc_mask.sum())
adata_E14 = adata_E14[otx_oc_mask, :].copy()

blood_immune_mask = (
    (adata_E14.obs["hb_expr"] <= 0) &
    (adata_E14.obs["immune_expr"] <= 0)
)
after_blood_immune = int(blood_immune_mask.sum())
adata_E14 = adata_E14[blood_immune_mask, :].copy()

qc_summary = pd.DataFrame(
    {
        "n_cells": [raw_n, after_basic_qc, after_otx_oc, after_blood_immune],
        "removed_from_previous": [
            0,
            raw_n - after_basic_qc,
            after_basic_qc - after_otx_oc,
            after_otx_oc - after_blood_immune,
        ],
    },
    index=["raw", "after_basic_QC", "after_Oc90_Otx2_filter", "after_blood_immune_filter"],
)

print("\n==================================================")
print("E14.5 QC summary")
print("==================================================")
print(qc_summary)

print("\nCells kept by sample after filtering:")
print(adata_E14.obs["sample"].value_counts().sort_index())

print("\nPositive cells after filtering:")
print({
    "Oc90_positive": int((adata_E14.obs["Oc90_expr"] > 0).sum()),
    "Otx2_positive": int((adata_E14.obs["Otx2_expr"] > 0).sum()),
    "hb_positive": int((adata_E14.obs["hb_expr"] > 0).sum()),
    "immune_positive": int((adata_E14.obs["immune_expr"] > 0).sum()),
})

print("\nQC summary after filtering:")
print(
    adata_E14.obs[
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
    ].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
)


# Notebook cell 40

# =========================================================
# 5. initial clustering
# E14 is already normalized/log-like, so only normalize/log1p if xmax > 30.

adata_E14_proc = adata_E14.copy()
xmax = float(np.max(adata_E14_proc.X))
print("max expression value:", xmax)

if xmax > 30:
    sc.pp.normalize_total(adata_E14_proc, target_sum=1e4)
    sc.pp.log1p(adata_E14_proc)

sc.pp.highly_variable_genes(
    adata_E14_proc,
    min_mean=0.0125,
    max_mean=3,
    min_disp=0.5,
)
adata_E14_proc = adata_E14_proc[:, adata_E14_proc.var["highly_variable"]].copy()
sc.pp.scale(adata_E14_proc, max_value=10)
sc.tl.pca(adata_E14_proc, svd_solver="arpack")
sc.pp.neighbors(adata_E14_proc, n_neighbors=15, n_pcs=20)
sc.tl.umap(adata_E14_proc)
sc.tl.leiden(adata_E14_proc, resolution=0.5)

adata_E14.obs["leiden"] = adata_E14_proc.obs["leiden"].copy()
adata_E14.obsm["X_umap"] = adata_E14_proc.obsm["X_umap"].copy()

print("\nall_leiden")
print(adata_E14.obs["leiden"].value_counts().sort_index())

print("\ncluster x sample:")
print(pd.crosstab(adata_E14.obs["leiden"], adata_E14.obs["sample"]))

# =========================================================
# 6. recluster strict epithelial candidates

adata_E14_epi = adata_E14.copy()

adata_E14_epi_proc = adata_E14_epi.copy()
xmax = float(np.max(adata_E14_epi_proc.X))
print("max expression value:", xmax)

if xmax > 30:
    sc.pp.normalize_total(adata_E14_epi_proc, target_sum=1e4)
    sc.pp.log1p(adata_E14_epi_proc)

sc.pp.highly_variable_genes(
    adata_E14_epi_proc,
    min_mean=0.0125,
    max_mean=3,
    min_disp=0.5,
)
adata_E14_epi_proc = adata_E14_epi_proc[:, adata_E14_epi_proc.var["highly_variable"]].copy()
sc.pp.scale(adata_E14_epi_proc, max_value=10)
sc.tl.pca(adata_E14_epi_proc, svd_solver="arpack")
sc.pp.neighbors(adata_E14_epi_proc, n_neighbors=15, n_pcs=20)
sc.tl.umap(adata_E14_epi_proc)
sc.tl.leiden(adata_E14_epi_proc, resolution=0.8)

adata_E14_epi.obs["epi_leiden"] = adata_E14_epi_proc.obs["leiden"].copy()
adata_E14_epi.obsm["X_umap"] = adata_E14_epi_proc.obsm["X_umap"].copy()

print("\nepi_leiden")
print(adata_E14_epi.obs["epi_leiden"].value_counts().sort_index())

print("\nepi_leiden x sample:")
print(pd.crosstab(adata_E14_epi.obs["epi_leiden"], adata_E14_epi.obs["sample"]))

score_markers_E14 = {
    "GER_KO_score": ["Fgf10", "Prdm16", "Tecta", "Epyc", "Dcn", "Calb1", "Otoa", "Crabp1"],
    "M_PsC_score": ["Fgf20", "Ebf1", "Anxa5", "Sox2", "Eya1", "Cdkn1b", "Jag1", "Isl1", "Socs2", "S100a1"],
    "L_PsC_score": ["Fgfr3", "Prox1", "Bmp2", "Tgfbr1", "Ngfr", "Nrcam", "Camta1", "Lockd"],
    "LER_score": ["Bmp4", "Lmo3", "Zbtb20", "Tectb", "Clu", "Gata2", "Sfrp1"],
    "HC_score": ["Ccer2", "Pou4f3", "Pcp4", "Hes6", "Rprm", "Selenom", "S100a1"],
    "IdC_score": ["Gsn", "Cdkn1c", "Foxq1", "Smoc2", "Ptn"],
    "GER_Hmgn2_score": ["Pclaf", "Ccnb2", "Cenpa", "Pttg1", "Stmn1", "Hmgb2", "H2afz"],
    "exclude_score": ["Otx2", "Oc90", "Neurod1", "Neurog1", "Prph", "Erbb3", "Ednrb", "Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1", "Kdr", "Cdh5", "Egfl7", "Ptprc", "Lyz2", "C1qa", "Tyrobp", "Fcer1g"],
}
score_cols = add_mean_scores(adata_E14_epi, score_markers_E14)

print("\nepi_leiden marker score:")
print(
    adata_E14_epi.obs
    .groupby("epi_leiden")[score_cols]
    .mean()
    .round(3)
)

# =========================================================
# 7. HC local check

adata_E14_hc = adata_E14_epi[
    adata_E14_epi.obs["epi_leiden"].isin(["7"]),
    :
].copy()

adata_E14_hc_proc = adata_E14_hc.copy()
sc.pp.highly_variable_genes(
    adata_E14_hc_proc,
    min_mean=0.0125,
    max_mean=3,
    min_disp=0.5,
)
adata_E14_hc_proc = adata_E14_hc_proc[:, adata_E14_hc_proc.var["highly_variable"]].copy()
sc.pp.scale(adata_E14_hc_proc, max_value=10)
sc.tl.pca(adata_E14_hc_proc, svd_solver="arpack")
sc.pp.neighbors(adata_E14_hc_proc, n_neighbors=5, n_pcs=10)
sc.tl.umap(adata_E14_hc_proc)
sc.tl.leiden(adata_E14_hc_proc, resolution=0.3)

adata_E14_hc.obs["hc_leiden"] = adata_E14_hc_proc.obs["leiden"].copy()
adata_E14_hc.obsm["X_umap"] = adata_E14_hc_proc.obsm["X_umap"].copy()

print("\nhc_leiden")
print(adata_E14_hc.obs["hc_leiden"].value_counts().sort_index())

hc_score_genes = {
    "IHC_score": ["Selenom", "S100a1", "Pcp4"],
    "OHC_score": ["Hes6", "Rprm", "Pcp4"],
    "HC_score": ["Ccer2", "Pou4f3"],
    "exclude_score": ["Otx2", "Oc90"],
}
hc_score_cols = add_mean_scores(adata_E14_hc, hc_score_genes)

print("\nhc_leiden marker means:")
print(
    adata_E14_hc.obs
    .groupby("hc_leiden")[hc_score_cols]
    .mean()
    .round(3)
)

# =========================================================
# 8. non-HC epithelial fine clustering

adata_E14_nonhc = adata_E14_epi[
    ~adata_E14_epi.obs["epi_leiden"].isin(["7"]),
    :
].copy()

adata_E14_nonhc_proc = adata_E14_nonhc.copy()
xmax = float(np.max(adata_E14_nonhc_proc.X))
print("max expression value:", xmax)

if xmax > 30:
    sc.pp.normalize_total(adata_E14_nonhc_proc, target_sum=1e4)
    sc.pp.log1p(adata_E14_nonhc_proc)

sc.pp.highly_variable_genes(
    adata_E14_nonhc_proc,
    min_mean=0.0125,
    max_mean=3,
    min_disp=0.5,
)
adata_E14_nonhc_proc = adata_E14_nonhc_proc[:, adata_E14_nonhc_proc.var["highly_variable"]].copy()
sc.pp.scale(adata_E14_nonhc_proc, max_value=10)
sc.tl.pca(adata_E14_nonhc_proc, svd_solver="arpack")
sc.pp.neighbors(adata_E14_nonhc_proc, n_neighbors=15, n_pcs=20)
sc.tl.umap(adata_E14_nonhc_proc)
sc.tl.leiden(adata_E14_nonhc_proc, resolution=1.0)

adata_E14_nonhc.obs["nonhc_leiden"] = adata_E14_nonhc_proc.obs["leiden"].copy()
adata_E14_nonhc.obsm["X_umap"] = adata_E14_nonhc_proc.obsm["X_umap"].copy()

print("\nnonhc_leiden")
print(adata_E14_nonhc.obs["nonhc_leiden"].value_counts().sort_index())

print("\nnonhc_leiden x sample:")
print(pd.crosstab(adata_E14_nonhc.obs["nonhc_leiden"], adata_E14_nonhc.obs["sample"]))

print("\nnonhc_leiden x original epi_leiden:")
print(pd.crosstab(adata_E14_nonhc.obs["nonhc_leiden"], adata_E14_nonhc.obs["epi_leiden"]))

nonhc_score_genes = {
    "GER_KO_score": ["Fgf10", "Prdm16", "Tecta", "Epyc", "Dcn", "Calb1", "Otoa", "Crabp1"],
    "M_PsC_score": ["Fgf20", "Ebf1", "Anxa5", "Sox2", "Eya1", "Cdkn1b", "Jag1", "Isl1", "Socs2", "S100a1"],
    "L_PsC_score": ["Fgfr3", "Prox1", "Bmp2", "Tgfbr1", "Ngfr", "Nrcam", "Camta1", "Lockd"],
    "LER_score": ["Bmp4", "Lmo3", "Zbtb20", "Tectb", "Clu", "Gata2", "Sfrp1"],
    "IdC_score": ["Gsn", "Cdkn1c", "Foxq1", "Smoc2", "Ptn"],
    "GER_Hmgn2_score": ["Pclaf", "Ccnb2", "Cenpa", "Pttg1", "Stmn1", "Hmgb2", "H2afz"],
    "exclude_score": ["Otx2", "Oc90", "Neurod1", "Neurog1", "Prph", "Erbb3", "Ednrb", "Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1", "Kdr", "Cdh5", "Egfl7", "Ptprc", "Lyz2", "C1qa", "Tyrobp", "Fcer1g"],
}
nonhc_score_cols = add_mean_scores(adata_E14_nonhc, nonhc_score_genes)

print("\nnonhc_leiden marker score:")
print(
    adata_E14_nonhc.obs
    .groupby("nonhc_leiden")[nonhc_score_cols]
    .mean()
    .round(3)
)

# =========================================================
# 9. separate confirmed and uncertain E14.5 non-HC clusters

confirmed_nonhc_map = {
    "0": "GER/KO",
    "1": "M.PsC",
    "2": "IdC",
    "3": "LER",
    "5": "GER/Hmgn2",
    "7": "GER/Hmgn2",
}
uncertain_nonhc_clusters = ["4", "6", "8"]

adata_E14_confirmed = adata_E14_nonhc[
    adata_E14_nonhc.obs["nonhc_leiden"].astype(str).isin(list(confirmed_nonhc_map.keys())),
    :
].copy()
adata_E14_uncertain = adata_E14_nonhc[
    adata_E14_nonhc.obs["nonhc_leiden"].astype(str).isin(uncertain_nonhc_clusters),
    :
].copy()

adata_E14_confirmed.obs["prelim_celltype"] = (
    adata_E14_confirmed.obs["nonhc_leiden"].astype(str).map(confirmed_nonhc_map).astype("category")
)

print("\nconfirmed non-HC celltypes")
print(adata_E14_confirmed.obs["prelim_celltype"].value_counts())

print("\nuncertain original clusters")
print(adata_E14_uncertain.obs["nonhc_leiden"].value_counts().sort_index())

print("\nuncertain original cluster x sample")
print(pd.crosstab(adata_E14_uncertain.obs["nonhc_leiden"], adata_E14_uncertain.obs["sample"]))

adata_E14_uncertain_proc = adata_E14_uncertain.copy()
xmax = float(np.max(adata_E14_uncertain_proc.X))
print("max expression value:", xmax)

if xmax > 30:
    sc.pp.normalize_total(adata_E14_uncertain_proc, target_sum=1e4)
    sc.pp.log1p(adata_E14_uncertain_proc)

sc.pp.highly_variable_genes(
    adata_E14_uncertain_proc,
    min_mean=0.0125,
    max_mean=3,
    min_disp=0.5,
)
adata_E14_uncertain_proc = adata_E14_uncertain_proc[:, adata_E14_uncertain_proc.var["highly_variable"]].copy()
sc.pp.scale(adata_E14_uncertain_proc, max_value=10)
sc.tl.pca(adata_E14_uncertain_proc, svd_solver="arpack")
sc.pp.neighbors(adata_E14_uncertain_proc, n_neighbors=10, n_pcs=15)
sc.tl.umap(adata_E14_uncertain_proc)
sc.tl.leiden(adata_E14_uncertain_proc, resolution=0.8)

adata_E14_uncertain.obs["uncertain_leiden"] = adata_E14_uncertain_proc.obs["leiden"].copy()
adata_E14_uncertain.obsm["X_umap"] = adata_E14_uncertain_proc.obsm["X_umap"].copy()

print("\nuncertain_leiden")
print(adata_E14_uncertain.obs["uncertain_leiden"].value_counts().sort_index())

print("\nuncertain_leiden x original nonhc_leiden")
print(pd.crosstab(adata_E14_uncertain.obs["uncertain_leiden"], adata_E14_uncertain.obs["nonhc_leiden"]))

print("\nuncertain_leiden x sample")
print(pd.crosstab(adata_E14_uncertain.obs["uncertain_leiden"], adata_E14_uncertain.obs["sample"]))

uncertain_score_genes = {
    "GER_KO_score": ["Fgf10", "Prdm16", "Tecta", "Epyc", "Dcn", "Calb1", "Otoa", "Crabp1"],
    "M_PsC_score": ["Fgf20", "Ebf1", "Anxa5", "Sox2", "Eya1", "Cdkn1b", "Jag1", "Isl1", "Socs2", "S100a1"],
    "L_PsC_score": ["Fgfr3", "Prox1", "Bmp2", "Tgfbr1", "Ngfr", "Nrcam", "Camta1", "Lockd"],
    "LER_score": ["Bmp4", "Lmo3", "Zbtb20", "Tectb", "Clu", "Gata2", "Sfrp1"],
    "IdC_score": ["Gsn", "Cdkn1c", "Foxq1", "Smoc2", "Ptn"],
    "GER_Hmgn2_score": ["Pclaf", "Ccnb2", "Cenpa", "Pttg1", "Stmn1", "Hmgb2", "H2afz"],
    "exclude_score": ["Otx2", "Oc90", "Neurod1", "Neurog1", "Prph", "Erbb3", "Ednrb", "Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1", "Kdr", "Cdh5", "Egfl7", "Ptprc", "Lyz2", "C1qa", "Tyrobp", "Fcer1g"],
}
uncertain_score_cols = add_mean_scores(adata_E14_uncertain, uncertain_score_genes)

print("\nuncertain_leiden marker score:")
print(
    adata_E14_uncertain.obs
    .groupby("uncertain_leiden")[uncertain_score_cols]
    .mean()
    .round(3)
)

# =========================================================
# 10. high-resolution L.PsC check in existing non-HC graph

lpsc_genes = [
    g for g in ["Fgfr3", "Prox1", "Bmp2", "Tgfbr1", "Ngfr", "Nrcam", "Camta1", "Lockd", "Fzd9", "Lsamp", "Elmo1"]
    if g in adata_E14_epi.var_names
]
mpsc_genes = [
    g for g in ["Fgf20", "Ebf1", "Anxa5", "Sox2", "Eya1", "Cdkn1b", "Jag1", "Isl1", "Socs2", "S100a1"]
    if g in adata_E14_epi.var_names
]
ler_genes = [
    g for g in ["Bmp4", "Lmo3", "Zbtb20", "Tectb", "Clu", "Gata2", "Sfrp1"]
    if g in adata_E14_epi.var_names
]
exclude_genes = [
    g for g in ["Otx2", "Oc90", "Neurod1", "Neurog1", "Prph", "Erbb3", "Ednrb", "Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1", "Kdr", "Cdh5", "Egfl7", "Ptprc", "Lyz2", "C1qa", "Tyrobp", "Fcer1g"]
    if g in adata_E14_epi.var_names
]

adata_E14_epi.obs["L_PsC_score2"] = np.asarray(adata_E14_epi[:, lpsc_genes].X).mean(axis=1).reshape(-1)
adata_E14_epi.obs["M_PsC_score2"] = np.asarray(adata_E14_epi[:, mpsc_genes].X).mean(axis=1).reshape(-1)
adata_E14_epi.obs["LER_score2"] = np.asarray(adata_E14_epi[:, ler_genes].X).mean(axis=1).reshape(-1)
adata_E14_epi.obs["exclude_score2"] = np.asarray(adata_E14_epi[:, exclude_genes].X).mean(axis=1).reshape(-1)

print("\nL.PsC score by epi_leiden:")
print(
    adata_E14_epi.obs
    .groupby("epi_leiden")[["L_PsC_score2", "M_PsC_score2", "LER_score2", "exclude_score2"]]
    .mean()
    .round(3)
)

sc.tl.leiden(
    adata_E14_nonhc_proc,
    resolution=1.6,
    random_state=0,
    key_added="nonhc_lpsc_leiden",
)

adata_E14_nonhc.obs["nonhc_lpsc_leiden"] = adata_E14_nonhc_proc.obs["nonhc_lpsc_leiden"].copy()

for score_name, genes in {
    "L_PsC_score2": lpsc_genes,
    "M_PsC_score2": mpsc_genes,
    "LER_score2": ler_genes,
    "exclude_score2": exclude_genes,
}.items():
    adata_E14_nonhc.obs[score_name] = np.asarray(adata_E14_nonhc[:, genes].X).mean(axis=1).reshape(-1)

print("\nnonhc_lpsc_leiden")
print(adata_E14_nonhc.obs["nonhc_lpsc_leiden"].value_counts().sort_index())

print("\nnonhc_lpsc_leiden x original nonhc_leiden")
print(pd.crosstab(adata_E14_nonhc.obs["nonhc_lpsc_leiden"], adata_E14_nonhc.obs["nonhc_leiden"]))

print("\nnonhc_lpsc_leiden x sample")
print(pd.crosstab(adata_E14_nonhc.obs["nonhc_lpsc_leiden"], adata_E14_nonhc.obs["sample"]))

print("\nL.PsC score by nonhc_lpsc_leiden:")
print(
    adata_E14_nonhc.obs
    .groupby("nonhc_lpsc_leiden")[["L_PsC_score2", "M_PsC_score2", "LER_score2", "exclude_score2"]]
    .mean()
    .round(3)
    .sort_values("L_PsC_score2", ascending=False)
)


# Notebook cell 41

# =========================================================
# 11. initial UMAP and marker check

adata_E14.obs["leiden_label"] = adata_E14.obs["leiden"].astype(str)
leiden_counts = adata_E14.obs["leiden_label"].value_counts().to_dict()
adata_E14.obs["leiden_label"] = adata_E14.obs["leiden_label"].map(
    lambda x: f"{x} (n={leiden_counts[x]})"
).astype("category")

sample_counts = adata_E14.obs["sample"].value_counts().to_dict()
adata_E14.obs["sample_label"] = adata_E14.obs["sample"].astype(str).map(
    lambda x: f"{x} (n={sample_counts[x]})"
).astype("category")

with plt.rc_context({"figure.figsize": (8, 7)}):
    sc.pl.umap(
        adata_E14,
        color="leiden_label",
        title="E14.5 all cells",
        size=8,
        frameon=True,
        legend_loc="right margin",
    )

with plt.rc_context({"figure.figsize": (8, 7)}):
    sc.pl.umap(
        adata_E14,
        color="sample_label",
        title="E14.5 sample check",
        size=8,
        frameon=True,
        legend_loc="right margin",
    )

marker_dict_E14_check = {
    "shared_check": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "HC": ["Ccer2", "Pou4f3"],
    "IHC": ["Selenom", "S100a1", "Pcp4"],
    "OHC": ["Hes6", "Rprm"],
    "GER_KO": ["Fgf10", "Prdm16", "Tecta", "Epyc", "Dcn", "Calb1", "Otoa", "Crabp1"],
    "M.PsC": ["Fgf20", "Ebf1", "Anxa5", "Sox2", "Eya1", "Cdkn1b", "Jag1"],
    "L.PsC": ["Fgfr3", "Prox1", "Bmp2", "Tgfbr1", "Ngfr", "Nrcam", "Camta1", "Lockd"],
    "LER": ["Bmp4", "Lmo3", "Zbtb20", "Tectb", "Clu", "Gata2", "Sfrp1"],
    "neural_check": ["Neurod1", "Neurog1", "Prph", "Insm1", "Myt1", "Nhlh2"],
    "glial_check": ["Sox10", "Plp1", "Mpz", "Erbb3", "Ednrb"],
    "mesenchyme_check": ["Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1"],
    "endothelial_check": ["Pecam1", "Kdr", "Cdh5", "Egfl7"],
    "immune_check": ["Ptprc", "Lyz2", "C1qa", "Tyrobp", "Fcer1g"],
    "exclude_check": ["Otx2", "Oc90"],
}
marker_dict_E14_check = dedupe_marker_dict(marker_dict_E14_check, adata_E14.var_names)

with plt.rc_context({"figure.figsize": (22, 7)}):
    dp = sc.pl.dotplot(
        adata_E14,
        marker_dict_E14_check,
        groupby="leiden",
        standard_scale="var",
        dot_max=0.9,
        dot_min=0.05,
        cmap="Reds",
        return_fig=True,
    )
    dp.style(dot_edge_color="gray", dot_edge_lw=0.4)
    dp.show()

# =========================================================
# 12. epithelial candidate marker check

adata_E14_epi.obs["epi_leiden_label"] = adata_E14_epi.obs["epi_leiden"].astype(str)
epi_counts = adata_E14_epi.obs["epi_leiden_label"].value_counts().to_dict()
adata_E14_epi.obs["epi_leiden_label"] = adata_E14_epi.obs["epi_leiden_label"].map(
    lambda x: f"{x} (n={epi_counts[x]})"
).astype("category")

with plt.rc_context({"figure.figsize": (8, 7)}):
    sc.pl.umap(
        adata_E14_epi,
        color="epi_leiden_label",
        title="E14.5 epithelial candidates",
        size=10,
        frameon=True,
        legend_loc="right margin",
    )

marker_dict_E14_epi = {
    "shared_check": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "HC": ["Ccer2", "Pou4f3"],
    "IHC": ["Selenom", "S100a1", "Pcp4"],
    "OHC": ["Hes6", "Rprm"],
    "GER_KO": ["Fgf10", "Prdm16", "Tecta", "Epyc", "Dcn", "Calb1", "Otoa", "Crabp1"],
    "M.PsC": ["Fgf20", "Ebf1", "Anxa5", "Eya1", "Cdkn1b", "Jag1"],
    "L.PsC": ["Fgfr3", "Prox1", "Bmp2", "Tgfbr1", "Ngfr", "Nrcam", "Camta1", "Lockd"],
    "LER": ["Bmp4", "Lmo3", "Zbtb20", "Tectb", "Clu", "Gata2", "Sfrp1"],
    "neural_check": ["Neurod1", "Neurog1", "Prph", "Insm1", "Myt1", "Nhlh2"],
    "glial_check": ["Sox10", "Plp1", "Mpz", "Erbb3", "Ednrb"],
    "mesenchyme_check": ["Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1"],
    "endothelial_check": ["Pecam1", "Kdr", "Cdh5", "Egfl7"],
    "immune_check": ["Ptprc", "Lyz2", "C1qa", "Tyrobp", "Fcer1g"],
    "exclude_check": ["Otx2", "Oc90"],
}
marker_dict_E14_epi = dedupe_marker_dict(marker_dict_E14_epi, adata_E14_epi.var_names)

with plt.rc_context({"figure.figsize": (22, 7)}):
    dp = sc.pl.dotplot(
        adata_E14_epi,
        marker_dict_E14_epi,
        groupby="epi_leiden",
        standard_scale="var",
        dot_max=0.9,
        dot_min=0.05,
        cmap="Reds",
        return_fig=True,
    )
    dp.style(dot_edge_color="gray", dot_edge_lw=0.4)
    dp.show()

with plt.rc_context({"figure.figsize": (18, 8)}):
    sc.pl.umap(
        adata_E14_epi,
        color=score_cols,
        cmap="Reds",
        size=10,
        frameon=True,
        ncols=4,
    )

# =========================================================
# 13. HC, non-HC and uncertain marker checks

adata_E14_hc.obs["hc_leiden_label"] = adata_E14_hc.obs["hc_leiden"].astype(str)
hc_counts = adata_E14_hc.obs["hc_leiden_label"].value_counts().to_dict()
adata_E14_hc.obs["hc_leiden_label"] = adata_E14_hc.obs["hc_leiden_label"].map(
    lambda x: f"{x} (n={hc_counts[x]})"
).astype("category")

with plt.rc_context({"figure.figsize": (6, 5)}):
    sc.pl.umap(
        adata_E14_hc,
        color="hc_leiden_label",
        title="E14.5 HC local check",
        size=35,
        frameon=True,
        legend_loc="right margin",
    )

hc_marker_dict = {
    "HC": ["Ccer2", "Pou4f3"],
    "IHC": ["Selenom", "S100a1", "Pcp4"],
    "OHC": ["Hes6", "Rprm"],
    "shared_check": ["Epcam", "Gata3", "Sox2", "Sox9", "Isl1"],
    "exclude_check": ["Otx2", "Oc90"],
}
hc_marker_dict = dedupe_marker_dict(hc_marker_dict, adata_E14_hc.var_names)

with plt.rc_context({"figure.figsize": (8, 4)}):
    dp = sc.pl.dotplot(
        adata_E14_hc,
        hc_marker_dict,
        groupby="hc_leiden",
        standard_scale="var",
        dot_max=0.9,
        dot_min=0.05,
        cmap="Reds",
        return_fig=True,
    )
    dp.style(dot_edge_color="gray", dot_edge_lw=0.4)
    dp.show()

adata_E14_nonhc.obs["nonhc_leiden_label"] = adata_E14_nonhc.obs["nonhc_leiden"].astype(str)
nonhc_counts = adata_E14_nonhc.obs["nonhc_leiden_label"].value_counts().to_dict()
adata_E14_nonhc.obs["nonhc_leiden_label"] = adata_E14_nonhc.obs["nonhc_leiden_label"].map(
    lambda x: f"{x} (n={nonhc_counts[x]})"
).astype("category")

with plt.rc_context({"figure.figsize": (8, 7)}):
    sc.pl.umap(
        adata_E14_nonhc,
        color="nonhc_leiden_label",
        title="E14.5 non-HC epithelial fine clustering",
        size=9,
        frameon=True,
        legend_loc="right margin",
    )

nonhc_marker_dict = {
    "shared_check": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "GER_KO": ["Fgf10", "Prdm16", "Tecta", "Epyc", "Dcn", "Calb1", "Otoa", "Crabp1"],
    "M.PsC": ["Fgf20", "Ebf1", "Anxa5", "Eya1", "Cdkn1b", "Jag1", "Socs2", "S100a1"],
    "L.PsC": ["Fgfr3", "Prox1", "Bmp2", "Tgfbr1", "Ngfr", "Nrcam", "Camta1", "Lockd"],
    "LER": ["Bmp4", "Lmo3", "Zbtb20", "Tectb", "Clu", "Gata2", "Sfrp1"],
    "IdC": ["Gsn", "Cdkn1c", "Foxq1", "Smoc2", "Ptn"],
    "GER_Hmgn2": ["Pclaf", "Ccnb2", "Cenpa", "Pttg1", "Stmn1", "Hmgb2", "H2afz"],
    "neural_check": ["Neurod1", "Neurog1", "Prph", "Insm1", "Myt1", "Nhlh2"],
    "glial_check": ["Sox10", "Plp1", "Mpz", "Erbb3", "Ednrb"],
    "mesenchyme_check": ["Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1"],
    "endothelial_check": ["Pecam1", "Kdr", "Cdh5", "Egfl7"],
    "immune_check": ["Ptprc", "Lyz2", "C1qa", "Tyrobp", "Fcer1g"],
    "exclude_check": ["Otx2", "Oc90"],
}
nonhc_marker_dict = dedupe_marker_dict(nonhc_marker_dict, adata_E14_nonhc.var_names)

with plt.rc_context({"figure.figsize": (22, 7)}):
    dp = sc.pl.dotplot(
        adata_E14_nonhc,
        nonhc_marker_dict,
        groupby="nonhc_leiden",
        standard_scale="var",
        dot_max=0.9,
        dot_min=0.05,
        cmap="Reds",
        return_fig=True,
    )
    dp.style(dot_edge_color="gray", dot_edge_lw=0.4)
    dp.show()

adata_E14_uncertain.obs["uncertain_leiden_label"] = adata_E14_uncertain.obs["uncertain_leiden"].astype(str)
uncertain_counts = adata_E14_uncertain.obs["uncertain_leiden_label"].value_counts().to_dict()
adata_E14_uncertain.obs["uncertain_leiden_label"] = adata_E14_uncertain.obs["uncertain_leiden_label"].map(
    lambda x: f"{x} (n={uncertain_counts[x]})"
).astype("category")

with plt.rc_context({"figure.figsize": (7, 6)}):
    sc.pl.umap(
        adata_E14_uncertain,
        color="uncertain_leiden_label",
        title="E14.5 uncertain non-HC reclustering",
        size=16,
        frameon=True,
        legend_loc="right margin",
    )

uncertain_marker_dict = {
    "shared_check": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "GER_KO": ["Fgf10", "Prdm16", "Tecta", "Epyc", "Dcn", "Calb1", "Otoa", "Crabp1"],
    "M.PsC": ["Fgf20", "Ebf1", "Anxa5", "Eya1", "Cdkn1b", "Jag1", "Socs2", "S100a1"],
    "L.PsC": ["Fgfr3", "Prox1", "Bmp2", "Tgfbr1", "Ngfr", "Nrcam", "Camta1", "Lockd"],
    "LER": ["Bmp4", "Lmo3", "Zbtb20", "Tectb", "Clu", "Gata2", "Sfrp1"],
    "IdC": ["Gsn", "Cdkn1c", "Foxq1", "Smoc2", "Ptn"],
    "GER_Hmgn2": ["Pclaf", "Ccnb2", "Cenpa", "Pttg1", "Stmn1", "Hmgb2", "H2afz"],
    "exclude_check": ["Otx2", "Oc90", "Neurod1", "Neurog1", "Prph", "Erbb3", "Ednrb", "Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1", "Kdr", "Cdh5", "Egfl7", "Ptprc", "Lyz2", "C1qa", "Tyrobp", "Fcer1g"],
}
uncertain_marker_dict = dedupe_marker_dict(uncertain_marker_dict, adata_E14_uncertain.var_names)

with plt.rc_context({"figure.figsize": (18, 6)}):
    dp = sc.pl.dotplot(
        adata_E14_uncertain,
        uncertain_marker_dict,
        groupby="uncertain_leiden",
        standard_scale="var",
        dot_max=0.9,
        dot_min=0.05,
        cmap="Reds",
        return_fig=True,
    )
    dp.style(dot_edge_color="gray", dot_edge_lw=0.4)
    dp.show()

# =========================================================
# 14. L.PsC high-resolution marker check

adata_E14_nonhc.obs["nonhc_lpsc_leiden_label"] = adata_E14_nonhc.obs["nonhc_lpsc_leiden"].astype(str)
nonhc_lpsc_counts = adata_E14_nonhc.obs["nonhc_lpsc_leiden_label"].value_counts().to_dict()
adata_E14_nonhc.obs["nonhc_lpsc_leiden_label"] = adata_E14_nonhc.obs["nonhc_lpsc_leiden_label"].map(
    lambda x: f"{x} (n={nonhc_lpsc_counts[x]})"
).astype("category")

with plt.rc_context({"figure.figsize": (8, 7)}):
    sc.pl.umap(
        adata_E14_nonhc,
        color="nonhc_lpsc_leiden_label",
        title="E14.5 non-HC L.PsC high-resolution check",
        size=9,
        frameon=True,
        legend_loc="right margin",
    )

with plt.rc_context({"figure.figsize": (18, 5)}):
    sc.pl.umap(
        adata_E14_epi,
        color=["L_PsC_score2", "M_PsC_score2", "LER_score2"],
        cmap="Reds",
        size=9,
        frameon=True,
        ncols=3,
    )

lpsc_high_marker_dict = {
    "shared_check": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "M.PsC": ["Fgf20", "Ebf1", "Anxa5", "Eya1", "Cdkn1b", "Jag1", "Socs2", "S100a1"],
    "L.PsC": ["Fgfr3", "Prox1", "Bmp2", "Tgfbr1", "Ngfr", "Nrcam", "Camta1", "Lockd", "Fzd9", "Lsamp", "Elmo1"],
    "LER": ["Bmp4", "Lmo3", "Zbtb20", "Tectb", "Clu", "Gata2", "Sfrp1"],
    "GER_KO": ["Fgf10", "Prdm16", "Tecta", "Epyc", "Dcn", "Calb1", "Otoa", "Crabp1"],
    "IdC": ["Gsn", "Cdkn1c", "Foxq1", "Smoc2", "Ptn"],
    "GER_Hmgn2": ["Pclaf", "Ccnb2", "Cenpa", "Pttg1", "Stmn1", "Hmgb2", "H2afz"],
    "exclude_check": ["Otx2", "Oc90", "Neurod1", "Neurog1", "Prph", "Erbb3", "Ednrb", "Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1", "Kdr", "Cdh5", "Egfl7", "Ptprc", "Lyz2", "C1qa", "Tyrobp", "Fcer1g"],
}
lpsc_high_marker_dict = dedupe_marker_dict(lpsc_high_marker_dict, adata_E14_nonhc.var_names)

with plt.rc_context({"figure.figsize": (24, 7)}):
    dp = sc.pl.dotplot(
        adata_E14_nonhc,
        lpsc_high_marker_dict,
        groupby="nonhc_lpsc_leiden",
        standard_scale="var",
        dot_max=0.9,
        dot_min=0.05,
        cmap="Reds",
        return_fig=True,
    )
    dp.style(dot_edge_color="gray", dot_edge_lw=0.4)
    dp.show()

# =========================================================
# 15. final E14.5 epithelial annotation check

adata_E14_final = adata_E14_epi.copy()
adata_E14_final.obs["final_celltype"] = "Unassigned"

hc_map = {
    "0": "IHC",
    "1": "OHC",
}
hc_labels = adata_E14_hc.obs["hc_leiden"].astype(str).map(hc_map)
adata_E14_final.obs.loc[hc_labels.index, "final_celltype"] = hc_labels

confirmed_labels = adata_E14_nonhc.obs["nonhc_leiden"].astype(str).map(confirmed_nonhc_map)
confirmed_labels = confirmed_labels.dropna()
adata_E14_final.obs.loc[confirmed_labels.index, "final_celltype"] = confirmed_labels

uncertain_map = {
    "0": "GER/Hmgn2",
    "1": "GER/KO",
    "2": "LER",
    "3": "LER",
    "4": "GER/KO",
}
uncertain_labels = adata_E14_uncertain.obs["uncertain_leiden"].astype(str).map(uncertain_map)
uncertain_labels = uncertain_labels.dropna()
adata_E14_final.obs.loc[uncertain_labels.index, "final_celltype"] = uncertain_labels

lpsc_names = adata_E14_nonhc.obs_names[
    adata_E14_nonhc.obs["nonhc_lpsc_leiden"].astype(str).isin(["12"])
]
adata_E14_final.obs.loc[lpsc_names, "final_celltype"] = "L.PsC"

final_order = [
    "GER/KO",
    "GER/Hmgn2",
    "IdC",
    "M.PsC",
    "L.PsC",
    "LER",
    "IHC",
    "OHC",
]

print("\nUnassigned cells:")
print((adata_E14_final.obs["final_celltype"] == "Unassigned").sum())

adata_E14_final = adata_E14_final[
    adata_E14_final.obs["final_celltype"].isin(final_order),
    :
].copy()

adata_E14_final.obs["stage"] = "E14.5"
adata_E14_final.obs["final_celltype"] = pd.Categorical(
    adata_E14_final.obs["final_celltype"],
    categories=final_order,
    ordered=True,
)

print("\nfinal_celltype")
print(adata_E14_final.obs["final_celltype"].value_counts().reindex(final_order))

print("\nfinal_celltype x sample:")
print(pd.crosstab(adata_E14_final.obs["final_celltype"], adata_E14_final.obs["sample"]))

print("\nPositive cells after final annotation:")
print({
    "Oc90_positive": int((adata_E14_final.obs["Oc90_expr"] > 0).sum()),
    "Otx2_positive": int((adata_E14_final.obs["Otx2_expr"] > 0).sum()),
    "hb_positive": int((adata_E14_final.obs["hb_expr"] > 0).sum()),
    "immune_positive": int((adata_E14_final.obs["immune_expr"] > 0).sum()),
})

final_counts = adata_E14_final.obs["final_celltype"].value_counts().to_dict()
label_order = [
    f"{celltype} (n={final_counts[celltype]})"
    for celltype in final_order
    if celltype in final_counts
]

adata_E14_final.obs["final_celltype_label"] = adata_E14_final.obs["final_celltype"].astype(str).map(
    lambda x: f"{x} (n={final_counts[x]})"
)
adata_E14_final.obs["final_celltype_label"] = pd.Categorical(
    adata_E14_final.obs["final_celltype_label"],
    categories=label_order,
    ordered=True,
)

with plt.rc_context({"figure.figsize": (8, 7)}):
    sc.pl.umap(
        adata_E14_final,
        color="final_celltype_label",
        title="E14.5 final epithelial cell types",
        size=9,
        frameon=True,
        legend_loc="right margin",
    )

final_marker_dict_E14 = {
    "shared_check": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "HC": ["Ccer2", "Pou4f3"],
    "IHC": ["Selenom", "S100a1", "Pcp4"],
    "OHC": ["Hes6", "Rprm"],
    "GER_KO": ["Fgf10", "Prdm16", "Tecta", "Epyc", "Dcn", "Calb1", "Otoa", "Crabp1"],
    "M.PsC": ["Fgf20", "Ebf1", "Anxa5", "Eya1", "Cdkn1b", "Jag1", "Socs2", "S100a1"],
    "L.PsC": ["Fgfr3", "Prox1", "Bmp2", "Tgfbr1", "Ngfr", "Nrcam", "Camta1", "Lockd", "Fzd9", "Lsamp", "Elmo1"],
    "LER": ["Bmp4", "Lmo3", "Zbtb20", "Tectb", "Clu", "Gata2", "Sfrp1"],
    "IdC": ["Gsn", "Cdkn1c", "Foxq1", "Smoc2", "Ptn"],
    "GER_Hmgn2": ["Pclaf", "Ccnb2", "Cenpa", "Pttg1", "Stmn1", "Hmgb2", "H2afz"],
    "neural_check": ["Neurod1", "Neurog1", "Prph", "Insm1", "Myt1", "Nhlh2"],
    "glial_check": ["Sox10", "Plp1", "Mpz", "Erbb3", "Ednrb"],
    "mesenchyme_check": ["Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1"],
    "endothelial_check": ["Pecam1", "Kdr", "Cdh5", "Egfl7"],
    "immune_check": ["Ptprc", "Lyz2", "C1qa", "Tyrobp", "Fcer1g"],
    "exclude_check": ["Otx2", "Oc90"],
}
final_marker_dict_E14 = dedupe_marker_dict(final_marker_dict_E14, adata_E14_final.var_names)

with plt.rc_context({"figure.figsize": (28, 5.5)}):
    dp = sc.pl.dotplot(
        adata_E14_final,
        final_marker_dict_E14,
        groupby="final_celltype",
        categories_order=final_order,
        standard_scale="var",
        dot_max=0.9,
        dot_min=0.05,
        cmap="Reds",
        return_fig=True,
    )
    dp.style(dot_edge_color="gray", dot_edge_lw=0.4)
    dp.show()


# Notebook cell 42

# =========================================================
# 16. save final E14.5 epithelial cells for integration

harmony_dir = RESULTS_ROOT / "01_stage_clustering" / "harmony_inputs" / "e14_5"
scnvi_dir = STAGE_OBJECT_DIR

figure_dir.mkdir(parents=True, exist_ok=True)
harmony_dir.mkdir(parents=True, exist_ok=True)
scnvi_dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# save normalized-only h5ad for integration

def make_normalized_integration_obj(adata_in):
    obs_cols = [
        c for c in ["sample", "stage", "final_celltype"]
        if c in adata_in.obs.columns
    ]
    adata_out = ad.AnnData(
        X=adata_in.X.copy(),
        obs=adata_in.obs[obs_cols].copy(),
        var=pd.DataFrame(index=adata_in.var_names.copy()),
    )
    adata_out.obs_names = adata_in.obs_names.copy()
    adata_out.var_names = adata_in.var_names.copy()
    adata_out.raw = None
    adata_out.layers.clear()
    adata_out.obsm.clear()
    adata_out.obsp.clear()
    adata_out.varm.clear()
    adata_out.varp.clear()
    adata_out.uns.clear()
    return adata_out

adata_E14_integration = make_normalized_integration_obj(adata_E14_final)

harmony_file = harmony_dir / "E14.5_for_Harmony_normalized.h5ad"
scnvi_file = scnvi_dir / "E14.5_for_scnvi_normalized.h5ad"

adata_E14_integration.write_h5ad(harmony_file)
adata_E14_integration.write_h5ad(scnvi_file)
# save final review figures

with plt.rc_context({"figure.figsize": (8, 7)}):
    sc.pl.umap(
        adata_E14_final,
        color="final_celltype_label",
        title="E14.5 final epithelial cell types",
        size=9,
        frameon=True,
        legend_loc="right margin",
        show=False,
    )
    plt.savefig(
        figure_dir / "E14.5_final_epithelial_celltypes_umap.png",
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )
    plt.close()

with plt.rc_context({"figure.figsize": (28, 5.5)}):
    dp = sc.pl.dotplot(
        adata_E14_final,
        final_marker_dict_E14,
        groupby="final_celltype",
        categories_order=final_order,
        standard_scale="var",
        dot_max=0.9,
        dot_min=0.05,
        cmap="Reds",
        return_fig=True,
    )
    dp.style(dot_edge_color="gray", dot_edge_lw=0.4)
    dp.savefig(
        figure_dir / "E14.5_final_epithelial_celltypes_dotplot.png",
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )
    plt.close()

# =========================================================
