#1. ===========Purpose and reproducibility settings
"""Preprocess, cluster, annotate, and export the E11.5 integration inputs."""

from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("COCHLEA_DATA_DIR", REPO_ROOT / "Data")).expanduser().resolve()
RESULTS_ROOT = Path(os.environ.get("COCHLEA_RESULTS_DIR", REPO_ROOT / "results")).expanduser().resolve()
STAGE_OBJECT_DIR = RESULTS_ROOT / "01_stage_clustering" / "h5ad_for_integration"
OUTPUT_DIR = RESULTS_ROOT / "01_stage_clustering" / "e11_5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STAGE_OBJECT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 0
SAVE_DPI = 1201
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_STATE))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Notebook cell 16


# =========================================================
# 0. imports

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


# =========================================================
# 1. settings

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=SAVE_DPI, facecolor="white")
sc.set_figure_params(figsize=(8, 8))


# =========================================================
# 2. paths

base_dir = REPO_ROOT
figure_dir = OUTPUT_DIR
data_dir = DATA_ROOT / "raw" / "e11_5"
marker_file = REPO_ROOT / "config" / "cell_type_markers.xlsx"

count_file_1 = data_dir / "GSM5401073_E11_5_1_raw_count.csv"
count_file_2 = data_dir / "GSM5401074_E11_5_2_raw_count.csv"
count_file_3 = data_dir / "GSM5401075_E11_5_3_raw_count.csv"

figure_dir.mkdir(parents=True, exist_ok=True)

os.chdir(base_dir)


# =========================================================
# 3. read raw count files

df_E11_1 = pd.read_csv(count_file_1)
df_E11_2 = pd.read_csv(count_file_2)
df_E11_3 = pd.read_csv(count_file_3)


# =========================================================
# 4. check gene consistency before merge

gene_col_1 = df_E11_1.columns[0]
gene_col_2 = df_E11_2.columns[0]
gene_col_3 = df_E11_3.columns[0]

genes_1 = df_E11_1[gene_col_1].astype(str).reset_index(drop=True)
genes_2 = df_E11_2[gene_col_2].astype(str).reset_index(drop=True)
genes_3 = df_E11_3[gene_col_3].astype(str).reset_index(drop=True)

print("\n==================================================")
print("Gene consistency check across 3 files")
print("==================================================")
print("sample1 == sample2 gene order:", genes_1.equals(genes_2))
print("sample1 == sample3 gene order:", genes_1.equals(genes_3))
print("sample2 == sample3 gene order:", genes_2.equals(genes_3))

if not (genes_1.equals(genes_2) and genes_1.equals(genes_3)):
    raise ValueError("Gene order is not fully consistent across the 3 files. Stop here.")


# =========================================================
# 5. build AnnData for each sample

def build_adata_from_raw_count(df: pd.DataFrame, sample_name: str) -> ad.AnnData:
    gene_col = df.columns[0]
    gene_names = df[gene_col].astype(str).values
    expr = df.drop(columns=[gene_col]).copy()

    X = expr.T
    X.index = X.index.astype(str)
    X.columns = gene_names

    adata = ad.AnnData(X=sparse.csr_matrix(X.values))
    adata.obs_names = X.index
    adata.var_names = X.columns.astype(str)
    adata.obs["sample"] = sample_name
    return adata


adata_E11_1 = build_adata_from_raw_count(df_E11_1, "E11.5_sample1")
adata_E11_2 = build_adata_from_raw_count(df_E11_2, "E11.5_sample2")
adata_E11_3 = build_adata_from_raw_count(df_E11_3, "E11.5_sample3")


# =========================================================
# 6. merge 3 samples

adata_E11 = ad.concat(
    [adata_E11_1, adata_E11_2, adata_E11_3],
    join="outer",
    label="batch",
    keys=["sample1", "sample2", "sample3"],
    index_unique=None,
)

adata_E11.obs_names_make_unique()

print("\n==================================================")
print("Merged E11.5 AnnData basic information")
print("==================================================")
print(adata_E11)
print("total cells:", adata_E11.n_obs)
print("total genes:", adata_E11.n_vars)
print("\nsample distribution:")
print(adata_E11.obs["sample"].value_counts())


# =========================================================
# 7. QC metrics

adata_E11.var["mt"] = adata_E11.var_names.str.startswith(("mt-", "Mt-", "MT-"))

sc.pp.calculate_qc_metrics(
    adata_E11,
    qc_vars=["mt"],
    percent_top=[20],
    log1p=False,
    inplace=True,
)

print("\n==================================================")
print("QC metric summary")
print("==================================================")
print(adata_E11.obs[["total_counts", "n_genes_by_counts", "pct_counts_mt"]].describe())


# =========================================================
# 8. important QC plot before filtering

fig, axes = plt.subplots(2, 3, figsize=(12, 6))

axes[0, 0].hist(adata_E11.obs["total_counts"], bins=80)
axes[0, 0].set_title("total_counts", fontsize=9)
axes[0, 0].set_xlabel("total_counts", fontsize=8)
axes[0, 0].set_ylabel("Cell number", fontsize=8)

axes[0, 1].hist(adata_E11.obs["n_genes_by_counts"], bins=80)
axes[0, 1].set_title("n_genes_by_counts", fontsize=9)
axes[0, 1].set_xlabel("n_genes_by_counts", fontsize=8)
axes[0, 1].set_ylabel("Cell number", fontsize=8)

axes[0, 2].hist(adata_E11.obs["pct_counts_mt"], bins=80)
axes[0, 2].set_title("pct_counts_mt", fontsize=9)
axes[0, 2].set_xlabel("pct_counts_mt", fontsize=8)
axes[0, 2].set_ylabel("Cell number", fontsize=8)

axes[1, 0].scatter(
    adata_E11.obs["total_counts"],
    adata_E11.obs["n_genes_by_counts"],
    s=2,
    alpha=0.4,
)
axes[1, 0].set_title("total_counts vs n_genes", fontsize=9)
axes[1, 0].set_xlabel("total_counts", fontsize=8)
axes[1, 0].set_ylabel("n_genes_by_counts", fontsize=8)

axes[1, 1].scatter(
    adata_E11.obs["total_counts"],
    adata_E11.obs["pct_counts_mt"],
    s=2,
    alpha=0.4,
)
axes[1, 1].set_title("total_counts vs mt%", fontsize=9)
axes[1, 1].set_xlabel("total_counts", fontsize=8)
axes[1, 1].set_ylabel("pct_counts_mt", fontsize=8)

axes[1, 2].scatter(
    adata_E11.obs["n_genes_by_counts"],
    adata_E11.obs["pct_counts_mt"],
    s=2,
    alpha=0.4,
)
axes[1, 2].set_title("n_genes vs mt%", fontsize=9)
axes[1, 2].set_xlabel("n_genes_by_counts", fontsize=8)
axes[1, 2].set_ylabel("pct_counts_mt", fontsize=8)

for ax in axes.flatten():
    ax.tick_params(axis="both", labelsize=7)

plt.tight_layout()
plt.savefig(figure_dir / "E11.5_QC_plot_combined.png", dpi=SAVE_DPI, bbox_inches="tight")
plt.close("all")
plt.close()


# =========================================================
# 9. QC filtering

adata_E11_filt = adata_E11.copy()

print("\n==================================================")
print("Before filtering")
print("==================================================")
print("cells:", adata_E11_filt.n_obs)
print("genes:", adata_E11_filt.n_vars)

sc.pp.filter_genes(adata_E11_filt, min_cells=3)
adata_E11_filt = adata_E11_filt[adata_E11_filt.obs["n_genes_by_counts"] >= 1000].copy()
adata_E11_filt = adata_E11_filt[adata_E11_filt.obs["n_genes_by_counts"] <= 7000].copy()
adata_E11_filt = adata_E11_filt[adata_E11_filt.obs["total_counts"] >= 1000].copy()
adata_E11_filt = adata_E11_filt[adata_E11_filt.obs["total_counts"] <= 80000].copy()
adata_E11_filt = adata_E11_filt[adata_E11_filt.obs["pct_counts_mt"] < 12].copy()

print("\n==================================================")
print("After filtering")
print("==================================================")
print("cells:", adata_E11_filt.n_obs)
print("genes:", adata_E11_filt.n_vars)

print("\nFiltered QC summary:")
print(adata_E11_filt.obs[["total_counts", "n_genes_by_counts", "pct_counts_mt"]].describe())

print("\nremoved cell number:", adata_E11.n_obs - adata_E11_filt.n_obs)
print("retained cell number:", adata_E11_filt.n_obs)
print("retained proportion:", adata_E11_filt.n_obs / adata_E11.n_obs)


# =========================================================
# 10. important QC plot after filtering

fig, axes = plt.subplots(2, 3, figsize=(12, 6))

axes[0, 0].hist(adata_E11_filt.obs["total_counts"], bins=80)
axes[0, 0].set_title("filtered total_counts", fontsize=9)
axes[0, 0].set_xlabel("total_counts", fontsize=8)
axes[0, 0].set_ylabel("Cell number", fontsize=8)

axes[0, 1].hist(adata_E11_filt.obs["n_genes_by_counts"], bins=80)
axes[0, 1].set_title("filtered n_genes_by_counts", fontsize=9)
axes[0, 1].set_xlabel("n_genes_by_counts", fontsize=8)
axes[0, 1].set_ylabel("Cell number", fontsize=8)

axes[0, 2].hist(adata_E11_filt.obs["pct_counts_mt"], bins=80)
axes[0, 2].set_title("filtered pct_counts_mt", fontsize=9)
axes[0, 2].set_xlabel("pct_counts_mt", fontsize=8)
axes[0, 2].set_ylabel("Cell number", fontsize=8)

axes[1, 0].scatter(
    adata_E11_filt.obs["total_counts"],
    adata_E11_filt.obs["n_genes_by_counts"],
    s=2,
    alpha=0.4,
)
axes[1, 0].set_title("filtered total_counts vs n_genes", fontsize=9)
axes[1, 0].set_xlabel("total_counts", fontsize=8)
axes[1, 0].set_ylabel("n_genes_by_counts", fontsize=8)

axes[1, 1].scatter(
    adata_E11_filt.obs["total_counts"],
    adata_E11_filt.obs["pct_counts_mt"],
    s=2,
    alpha=0.4,
)
axes[1, 1].set_title("filtered total_counts vs mt%", fontsize=9)
axes[1, 1].set_xlabel("total_counts", fontsize=8)
axes[1, 1].set_ylabel("pct_counts_mt", fontsize=8)

axes[1, 2].scatter(
    adata_E11_filt.obs["n_genes_by_counts"],
    adata_E11_filt.obs["pct_counts_mt"],
    s=2,
    alpha=0.4,
)
axes[1, 2].set_title("filtered n_genes vs mt%", fontsize=9)
axes[1, 2].set_xlabel("n_genes_by_counts", fontsize=8)
axes[1, 2].set_ylabel("pct_counts_mt", fontsize=8)

for ax in axes.flatten():
    ax.tick_params(axis="both", labelsize=7)

plt.tight_layout()
plt.savefig(figure_dir / "E11.5_QC_plot_after_filtering.png", dpi=SAVE_DPI, bbox_inches="tight")
plt.close("all")
plt.close()


# =========================================================
# 11. normalize + log1p before marker thresholding

adata_E11_filt.layers["counts"] = adata_E11_filt.X.copy()
sc.pp.normalize_total(adata_E11_filt, target_sum=1e4)
sc.pp.log1p(adata_E11_filt)
adata_E11_filt.raw = adata_E11_filt


# =========================================================
# 12. inspect Oc90 / Otx2 distribution after QC

check_genes = [g for g in ["Oc90", "Otx2", "Fbxo2", "Epcam", "Pax2"] if g in adata_E11_filt.var_names]

if len(check_genes) > 0:
    sc.pl.violin(
        adata_E11_filt,
        check_genes,
        stripplot=False,
        jitter=0,
        multi_panel=True,
        show=False
    )
    plt.gcf().set_size_inches(max(8, len(check_genes) * 1.6), 3.2)
    plt.tight_layout()
    plt.close("all")
    plt.close()


def get_expr_vector(adata_obj: ad.AnnData, gene_name: str) -> np.ndarray:
    x = adata_obj[:, gene_name].X
    if sparse.issparse(x):
        return x.toarray().flatten()
    return np.asarray(x).flatten()


adata_E11_filt.obs["Oc90_expr"] = get_expr_vector(adata_E11_filt, "Oc90") if "Oc90" in adata_E11_filt.var_names else 0
adata_E11_filt.obs["Otx2_expr"] = get_expr_vector(adata_E11_filt, "Otx2") if "Otx2" in adata_E11_filt.var_names else 0

print("\n==================================================")
print("Oc90 / Otx2 expression summary after QC")
print("==================================================")
print(adata_E11_filt.obs[["Oc90_expr", "Otx2_expr"]].describe())

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.hist(adata_E11_filt.obs["Oc90_expr"], bins=50)
plt.title("Oc90_expr")
plt.subplot(1, 2, 2)
plt.hist(adata_E11_filt.obs["Otx2_expr"], bins=50)
plt.title("Otx2_expr")
plt.tight_layout()
plt.close("all")
plt.close()


# Notebook cell 17

# =========================================================
# 13. remove Oc90+ / Otx2+ cells by threshold
# adjust these two thresholds after you inspect the distributions

oc90_cutoff = 0.05
otx2_cutoff = 0.05

mask_keep = (
    (adata_E11_filt.obs["Oc90_expr"] <= oc90_cutoff) &
    (adata_E11_filt.obs["Otx2_expr"] <= otx2_cutoff)
)

adata_E11_clean = adata_E11_filt[mask_keep].copy()

print("\n==================================================")
print("After removing Oc90+ / Otx2+ cells")
print("==================================================")
print("cells before:", adata_E11_filt.n_obs)
print("cells after:", adata_E11_clean.n_obs)
print("removed cells:", adata_E11_filt.n_obs - adata_E11_clean.n_obs)

# =========================================================
# 14. relaxed epithelial candidate extraction

def keep_present(adata_obj, genes):
    return [g for g in genes if g in adata_obj.var_names]

def get_expr_vector(adata_obj, gene_name):
    if gene_name not in adata_obj.var_names:
        return np.zeros(adata_obj.n_obs, dtype=float)
    x = adata_obj[:, gene_name].X
    if sparse.issparse(x):
        return x.toarray().ravel()
    return np.asarray(x).ravel()

def gene_positive_mask(adata_obj, gene, cutoff=0):
    return get_expr_vector(adata_obj, gene) > cutoff

def gene_raw_positive_mask(adata_obj, gene):
    if gene not in adata_obj.var_names:
        return np.zeros(adata_obj.n_obs, dtype=bool)

    if "counts" in adata_obj.layers:
        x = adata_obj[:, gene].layers["counts"]
    else:
        x = adata_obj[:, gene].X

    if sparse.issparse(x):
        x = x.toarray().ravel()
    else:
        x = np.asarray(x).ravel()

    return x > 0

# ---------------------------------------------------------
# marker sets

ov_epi_markers = keep_present(
    adata_E11_clean,
    ["Epcam", "Pax2", "Fbxo2", "Sox2", "Wfdc2"]
)

neural_markers = keep_present(
    adata_E11_clean,
    ["Insm1", "Neurog1", "Neurod1", "Myt1", "Phox2a", "Phox2b", "Prph", "Tubb3", "Isl1", "Nhlh2"]
)

mesenchyme_markers = keep_present(
    adata_E11_clean,
    ["Prrx1", "Prrx2", "Twist1", "Col1a1", "Col1a2", "Pdgfra", "Foxd1"]
)

endo_markers = keep_present(
    adata_E11_clean,
    ["Cd34", "Kdr", "Cdh5", "Egfl7"]
)

glial_markers = keep_present(
    adata_E11_clean,
    ["Plp1", "Mpz", "Erbb3", "Ednrb", "Sox10"]
)

immune_markers = keep_present(
    adata_E11_clean,
    ["Ptprc", "Lyz2", "Tyrobp", "Aif1", "C1qa", "C1qb", "C1qc", "Cd68", "Lcp1"]
)

blood_markers = keep_present(
    adata_E11_clean,
    ["Hbb-bt", "Hbb-bs", "Hba-a1", "Hba-a2", "Hba-x", "Alas2", "Klf1", "Gypa"]
)

# ---------------------------------------------------------
# scores

sc.tl.score_genes(adata_E11_clean, ov_epi_markers, score_name="OV_epi_score")

if len(neural_markers) > 0:
    sc.tl.score_genes(adata_E11_clean, neural_markers, score_name="Neural_score")
else:
    adata_E11_clean.obs["Neural_score"] = 0.0

if len(mesenchyme_markers) > 0:
    sc.tl.score_genes(adata_E11_clean, mesenchyme_markers, score_name="Mesenchyme_score")
else:
    adata_E11_clean.obs["Mesenchyme_score"] = 0.0

if len(endo_markers) > 0:
    sc.tl.score_genes(adata_E11_clean, endo_markers, score_name="Endo_score")
else:
    adata_E11_clean.obs["Endo_score"] = 0.0

if len(glial_markers) > 0:
    sc.tl.score_genes(adata_E11_clean, glial_markers, score_name="Glial_score")
else:
    adata_E11_clean.obs["Glial_score"] = 0.0

# ---------------------------------------------------------
# hard anchor: any OV epithelial marker positive

mask_epi_anchor = np.zeros(adata_E11_clean.n_obs, dtype=bool)
for g in ["Epcam", "Pax2", "Fbxo2"]:
    if g in adata_E11_clean.var_names:
        mask_epi_anchor |= gene_positive_mask(adata_E11_clean, g)

# hard exclusion: roof markers only

mask_roof = np.zeros(adata_E11_clean.n_obs, dtype=bool)
for g in ["Oc90", "Otx2"]:
    if g in adata_E11_clean.var_names:
        mask_roof |= gene_positive_mask(adata_E11_clean, g)

# hard exclusion: blood / immune markers

mask_immune = np.zeros(adata_E11_clean.n_obs, dtype=bool)
for g in immune_markers:
    mask_immune |= gene_raw_positive_mask(adata_E11_clean, g)

mask_blood = np.zeros(adata_E11_clean.n_obs, dtype=bool)
for g in blood_markers:
    mask_blood |= gene_raw_positive_mask(adata_E11_clean, g)

print("immune marker-positive cells removed:", int(mask_immune.sum()))
print("blood marker-positive cells removed:", int(mask_blood.sum()))
print("blood/immune marker-positive cells removed:", int((mask_immune | mask_blood).sum()))

# ---------------------------------------------------------
# relaxed keep mask

mask_keep = (
    (adata_E11_clean.obs["OV_epi_score"] > 0.02) &
    (adata_E11_clean.obs["Neural_score"] < 0.25) &
    (adata_E11_clean.obs["Mesenchyme_score"] < 0.18) &
    (adata_E11_clean.obs["Endo_score"] < 0.15) &
    (adata_E11_clean.obs["Glial_score"] < 0.15) &
    mask_epi_anchor &
    (~mask_roof) &
    (~mask_immune) &
    (~mask_blood)
)

adata_E11_epi_relaxed = adata_E11_clean[mask_keep].copy()

print("relaxed epithelial candidates:", adata_E11_epi_relaxed.n_obs)

# =========================================================
# quick clustering for inspection only

sc.pp.highly_variable_genes(
    adata_E11_epi_relaxed,
    n_top_genes=2000,
    flavor="seurat",
    batch_key="sample"
)

adata_E11_epi_hvg = adata_E11_epi_relaxed[:, adata_E11_epi_relaxed.var["highly_variable"]].copy()

sc.pp.scale(adata_E11_epi_hvg, max_value=10)
sc.tl.pca(adata_E11_epi_hvg, svd_solver="arpack")
sc.pp.neighbors(adata_E11_epi_hvg, n_neighbors=12, n_pcs=20)
sc.tl.umap(adata_E11_epi_hvg, min_dist=0.2, spread=1.2, random_state=0)
sc.tl.leiden(adata_E11_epi_hvg, resolution=0.25, random_state=0, key_added="epi_leiden")

adata_E11_epi_relaxed.obs["epi_leiden"] = adata_E11_epi_hvg.obs["epi_leiden"].values
adata_E11_epi_relaxed.obsm["X_umap"] = adata_E11_epi_hvg.obsm["X_umap"].copy()

print(adata_E11_epi_relaxed.obs["epi_leiden"].value_counts().sort_index())

# =========================================================
# final UMAP only

sc.pl.umap(
    adata_E11_epi_relaxed,
    color="epi_leiden",
    legend_loc="right margin",
    size=18,
    show=False
)
plt.close("all")
plt.close()

# =========================================================
# final dotplot only

check_genes = keep_present(
    adata_E11_epi_relaxed,
    [
        "Epcam", "Pax2", "Fbxo2",
        "Sox2", "Fgf10", "Lfng", "Bmp4",
        "Oc90", "Otx2",
        "Neurod1", "Prph"
    ]
)

sc.pl.dotplot(
    adata_E11_epi_relaxed,
    var_names=check_genes,
    groupby="epi_leiden",
    standard_scale="var",
    show=False
)
plt.close("all")
plt.close()


# Notebook cell 18

# =========================================================
# 15. expanded marker inspection

def keep_present(adata_obj, genes):
    return [g for g in genes if g in adata_obj.var_names]

# ---------------------------------------------------------
# marker panels

ov_markers = keep_present(
    adata_E11_epi_relaxed,
    ["Epcam", "Pax2", "Fbxo2", "Foxg1", "Sox2", "Sox9", "Gata3", "Six1", "Eya1", "Tbx1", "Tbx2"]
)

medial_markers = keep_present(
    adata_E11_epi_relaxed,
    ["Fgf10", "Ebf1"]
)

prosensory_markers = keep_present(
    adata_E11_epi_relaxed,
    ["Sox2", "Jag1", "Lfng", "Notch1", "Notch2", "Eya1"]
)

lateral_markers = keep_present(
    adata_E11_epi_relaxed,
    ["Bmp4", "Rorb", "Camta1", "Lockd", "Sema3e"]
)

exclude_markers = keep_present(
    adata_E11_epi_relaxed,
    [
        "Oc90", "Otx2",
        "Neurod1", "Neurog1", "Insm1", "Myt1", "Tubb3", "Prph", "Phox2a", "Phox2b", "Isl1",
        "Col3a1", "Prrx1", "Prrx2", "Twist1", "Pdgfra", "Pdgfrb",
        "Pecam1", "Ptprc", "Lyz2", "Tyrobp", "Aif1", "Hbb-bt", "Hbb-bs"
    ]
)

# ---------------------------------------------------------
# scores for cluster-level inspection only

if len(medial_markers) > 0:
    sc.tl.score_genes(
        adata_E11_epi_relaxed,
        medial_markers,
        score_name="Medial_score"
    )
else:
    adata_E11_epi_relaxed.obs["Medial_score"] = 0.0

if len(prosensory_markers) > 0:
    sc.tl.score_genes(
        adata_E11_epi_relaxed,
        prosensory_markers,
        score_name="Prosensory_score"
    )
else:
    adata_E11_epi_relaxed.obs["Prosensory_score"] = 0.0

if len(lateral_markers) > 0:
    sc.tl.score_genes(
        adata_E11_epi_relaxed,
        lateral_markers,
        score_name="Lateral_score"
    )
else:
    adata_E11_epi_relaxed.obs["Lateral_score"] = 0.0

score_summary = (
    adata_E11_epi_relaxed.obs
    .groupby("epi_leiden")[["Medial_score", "Prosensory_score", "Lateral_score"]]
    .mean()
    .round(3)
)

print("\n==================================================")
print("expanded marker inspection")
print("==================================================")
print("cells per cluster:")
print(adata_E11_epi_relaxed.obs["epi_leiden"].value_counts().sort_index())

print("\nmean scores by cluster:")
print(score_summary)

if "sample" in adata_E11_epi_relaxed.obs:
    print("\nsample x epi_leiden:")
    print(pd.crosstab(adata_E11_epi_relaxed.obs["epi_leiden"], adata_E11_epi_relaxed.obs["sample"]))

# =========================================================
# UMAP for inspection only

sc.pl.umap(
    adata_E11_epi_relaxed,
    color="epi_leiden",
    legend_loc="right margin",
    size=18,
    show=False
)
plt.close("all")
plt.close()

# =========================================================
# identity dotplot for inspection only

marker_dict = {}
if len(ov_markers) > 0:
    marker_dict["OV"] = ov_markers
if len(medial_markers) > 0:
    marker_dict["Medial"] = medial_markers
if len(prosensory_markers) > 0:
    marker_dict["Prosensory"] = prosensory_markers
if len(lateral_markers) > 0:
    marker_dict["Lateral"] = lateral_markers

sc.pl.dotplot(
    adata_E11_epi_relaxed,
    var_names=marker_dict,
    groupby="epi_leiden",
    standard_scale=None,
    cmap="Reds",
    vmin=0,
    vmax=2.5,
    dot_max=0.9,
    show=False
)
plt.gcf().set_size_inches(11, 3)
plt.close("all")
plt.close()

# =========================================================
# exclusion dotplot for inspection only

exclude_dict = {}
if len(exclude_markers) > 0:
    exclude_dict["Exclude_check"] = exclude_markers

sc.pl.dotplot(
    adata_E11_epi_relaxed,
    var_names=exclude_dict,
    groupby="epi_leiden",
    standard_scale=None,
    cmap="Reds",
    vmin=0,
    vmax=2.5,
    dot_max=0.9,
    show=False
)
plt.gcf().set_size_inches(10, 3)
plt.close("all")
plt.close()


# Notebook cell 19

# =========================================================
# 16. remove mesenchymal cluster 3 and re-cluster remaining cells
# Inspection only. Do not save files.

def keep_present(adata_obj, genes):
    return [g for g in genes if g in adata_obj.var_names]

# ---------------------------------------------------------
# remove cluster 3 before re-clustering

drop_clusters = ["3"]

adata_E11_post_rescue = adata_E11_epi_relaxed[
    ~adata_E11_epi_relaxed.obs["epi_leiden"].astype(str).isin(drop_clusters)
].copy()

adata_E11_post_rescue.obs["source_epi_leiden"] = adata_E11_post_rescue.obs["epi_leiden"].astype(str)

print("cells before removing cluster 3:", adata_E11_epi_relaxed.n_obs)
print("cells after removing cluster 3:", adata_E11_post_rescue.n_obs)

if "sample" in adata_E11_post_rescue.obs:
    print("\nsample distribution after removing cluster 3:")
    print(adata_E11_post_rescue.obs["sample"].value_counts().sort_index())

# =========================================================
# re-cluster remaining cells at higher resolution

sc.pp.highly_variable_genes(
    adata_E11_post_rescue,
    n_top_genes=2000,
    flavor="seurat",
    batch_key="sample"
)

adata_E11_post_hvg = adata_E11_post_rescue[
    :, adata_E11_post_rescue.var["highly_variable"]
].copy()

sc.pp.scale(adata_E11_post_hvg, max_value=10)
sc.tl.pca(adata_E11_post_hvg, svd_solver="arpack")
sc.pp.neighbors(adata_E11_post_hvg, n_neighbors=10, n_pcs=20)
sc.tl.umap(adata_E11_post_hvg, min_dist=0.15, spread=1.0, random_state=0)
sc.tl.leiden(
    adata_E11_post_hvg,
    resolution=1.0,
    random_state=0,
    key_added="rescue_view"
)

adata_E11_post_rescue.obs["rescue_view"] = adata_E11_post_hvg.obs["rescue_view"].values
adata_E11_post_rescue.obsm["X_umap"] = adata_E11_post_hvg.obsm["X_umap"].copy()

print("\nre-clustered cell numbers:")
print(adata_E11_post_rescue.obs["rescue_view"].value_counts().sort_index())

print("\nsource epi_leiden x rescue_view:")
print(pd.crosstab(
    adata_E11_post_rescue.obs["source_epi_leiden"],
    adata_E11_post_rescue.obs["rescue_view"]
))

if "sample" in adata_E11_post_rescue.obs:
    print("\nsample x rescue_view:")
    print(pd.crosstab(
        adata_E11_post_rescue.obs["rescue_view"],
        adata_E11_post_rescue.obs["sample"]
    ))

# =========================================================
# UMAP for inspection only

sc.pl.umap(
    adata_E11_post_rescue,
    color="rescue_view",
    legend_loc="right margin",
    size=18,
    show=False
)
plt.gcf().set_size_inches(4.8, 4.2)
plt.close("all")
plt.close()

# =========================================================
# identity dotplot for inspection only

identity_markers = {
    "OV": keep_present(
        adata_E11_post_rescue,
        ["Epcam", "Pax2", "Fbxo2", "Foxg1", "Sox2", "Sox9", "Gata3", "Six1", "Eya1", "Tbx1", "Tbx2"]
    ),
    "Medial": keep_present(
        adata_E11_post_rescue,
        ["Fgf10", "Ebf1"]
    ),
    "Prosensory": keep_present(
        adata_E11_post_rescue,
        ["Sox2", "Jag1", "Lfng", "Notch1", "Notch2", "Eya1"]
    ),
    "Lateral": keep_present(
        adata_E11_post_rescue,
        ["Bmp4", "Rorb", "Camta1", "Lockd", "Sema3e"]
    ),
}
identity_markers = {k: v for k, v in identity_markers.items() if len(v) > 0}

sc.pl.dotplot(
    adata_E11_post_rescue,
    var_names=identity_markers,
    groupby="rescue_view",
    standard_scale=None,
    cmap="Reds",
    vmin=0,
    vmax=2.5,
    dot_max=0.9,
    show=False
)
plt.gcf().set_size_inches(12, 3.2)
plt.close("all")
plt.close()

# =========================================================
# exclusion dotplot for inspection only

exclude_markers = {
    "Roof": keep_present(
        adata_E11_post_rescue,
        ["Oc90", "Otx2"]
    ),
    "Neural": keep_present(
        adata_E11_post_rescue,
        ["Neurod1", "Neurog1", "Insm1", "Myt1", "Tubb3", "Prph", "Phox2a", "Phox2b", "Isl1"]
    ),
    "Mesenchyme": keep_present(
        adata_E11_post_rescue,
        ["Col1a1", "Col1a2", "Col3a1", "Prrx1", "Prrx2", "Twist1", "Pdgfra", "Pdgfrb", "Foxd1"]
    ),
    "Blood/Immune": keep_present(
        adata_E11_post_rescue,
        ["Ptprc", "Lyz2", "Tyrobp", "Aif1", "Hbb-bt", "Hbb-bs"]
    ),
}
exclude_markers = {k: v for k, v in exclude_markers.items() if len(v) > 0}

sc.pl.dotplot(
    adata_E11_post_rescue,
    var_names=exclude_markers,
    groupby="rescue_view",
    standard_scale=None,
    cmap="Reds",
    vmin=0,
    vmax=2.5,
    dot_max=0.9,
    show=False
)
plt.gcf().set_size_inches(12, 3.2)
plt.close("all")
plt.close()


# Notebook cell 20

# =========================================================
# 17. keep clean epithelial clusters after high-resolution re-clustering
# Inspection only. Do not save files.

def keep_present(adata_obj, genes):
    return [g for g in genes if g in adata_obj.var_names]

# ---------------------------------------------------------
# keep clean epithelial clusters

keep_clusters = ["2", "3"]

adata_E11_final_check = adata_E11_post_rescue[
    adata_E11_post_rescue.obs["rescue_view"].astype(str).isin(keep_clusters)
].copy()

adata_E11_final_check.obs["final_celltype"] = "E11.5 sensory epithelial cells"

print("cells kept for final check:", adata_E11_final_check.n_obs)
print("\nkept rescue_view counts:")
print(adata_E11_final_check.obs["rescue_view"].value_counts().sort_index())

if "sample" in adata_E11_final_check.obs:
    print("\nsample distribution:")
    print(adata_E11_final_check.obs["sample"].value_counts().sort_index())

    print("\nsample x rescue_view:")
    print(pd.crosstab(
        adata_E11_final_check.obs["rescue_view"],
        adata_E11_final_check.obs["sample"]
    ))

# =========================================================
# final-check UMAP only

sc.pl.umap(
    adata_E11_final_check,
    color="rescue_view",
    legend_loc="right margin",
    size=18,
    show=False
)
plt.gcf().set_size_inches(4.2, 3.8)
plt.close("all")
plt.close()

# =========================================================
# final-check identity dotplot

identity_markers = {
    "OV": keep_present(
        adata_E11_final_check,
        ["Epcam", "Pax2", "Fbxo2", "Foxg1", "Sox2", "Sox9", "Gata3", "Six1", "Eya1", "Tbx1", "Tbx2"]
    ),
    "Medial_check": keep_present(
        adata_E11_final_check,
        ["Fgf10", "Ebf1"]
    ),
    "Prosensory_check": keep_present(
        adata_E11_final_check,
        ["Sox2", "Jag1", "Lfng", "Notch1", "Notch2", "Eya1"]
    ),
    "Lateral_check": keep_present(
        adata_E11_final_check,
        ["Bmp4", "Rorb", "Camta1", "Lockd", "Sema3e"]
    ),
}
identity_markers = {k: v for k, v in identity_markers.items() if len(v) > 0}

sc.pl.dotplot(
    adata_E11_final_check,
    var_names=identity_markers,
    groupby="rescue_view",
    categories_order=keep_clusters,
    standard_scale=None,
    cmap="Reds",
    vmin=0,
    vmax=2.5,
    dot_max=0.9,
    show=False
)
plt.gcf().set_size_inches(11, 2.5)
plt.close("all")
plt.close()

# =========================================================
# final-check exclusion dotplot

exclude_markers = {
    "Roof": keep_present(
        adata_E11_final_check,
        ["Oc90", "Otx2"]
    ),
    "Neural": keep_present(
        adata_E11_final_check,
        ["Neurod1", "Neurog1", "Insm1", "Myt1", "Tubb3", "Prph", "Phox2a", "Phox2b", "Isl1"]
    ),
    "Mesenchyme": keep_present(
        adata_E11_final_check,
        ["Col1a1", "Col1a2", "Col3a1", "Prrx1", "Prrx2", "Twist1", "Pdgfra", "Pdgfrb", "Foxd1"]
    ),
    "Blood/Immune": keep_present(
        adata_E11_final_check,
        ["Ptprc", "Lyz2", "Tyrobp", "Aif1", "Hbb-bt", "Hbb-bs"]
    ),
}
exclude_markers = {k: v for k, v in exclude_markers.items() if len(v) > 0}

sc.pl.dotplot(
    adata_E11_final_check,
    var_names=exclude_markers,
    groupby="rescue_view",
    categories_order=keep_clusters,
    standard_scale=None,
    cmap="Reds",
    vmin=0,
    vmax=2.5,
    dot_max=0.9,
    show=False
)
plt.gcf().set_size_inches(11, 2.5)
plt.close("all")
plt.close()


# Notebook cell 21

# =========================================================
# 18. save final E11.5 OV epithelial cells for integration

# ---------------------------------------------------------
# paths

base_dir = REPO_ROOT
figure_dir = OUTPUT_DIR

harmony_dir = RESULTS_ROOT / "01_stage_clustering" / "harmony_inputs" / "e11_5"
scnvi_dir = STAGE_OBJECT_DIR

figure_dir.mkdir(parents=True, exist_ok=True)
harmony_dir.mkdir(parents=True, exist_ok=True)
scnvi_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# final object: merge clusters 2 and 3

keep_clusters = ["2", "3"]

adata_E11_final = adata_E11_post_rescue[
    adata_E11_post_rescue.obs["rescue_view"].astype(str).isin(keep_clusters)
].copy()

adata_E11_final.obs["stage"] = "E11.5"
adata_E11_final.obs["final_celltype"] = "OV epithelial cells"
adata_E11_final.obs["source_object"] = "E11.5_raw_count_OV_epithelial_fullgene"

adata_E11_final.obs["final_celltype"] = pd.Categorical(
    adata_E11_final.obs["final_celltype"],
    categories=["OV epithelial cells"],
    ordered=True
)

print("E11.5 final cells:", adata_E11_final.n_obs)
print(adata_E11_final.obs["final_celltype"].value_counts())

# =========================================================
# save normalized/raw h5ad for integration

if "counts" not in adata_E11_final.layers:
    raise ValueError("counts layer is missing. Cannot save raw-count integration files.")

def copy_matrix(x):
    if sparse.issparse(x):
        return x.copy()
    return np.asarray(x).copy()

def make_integration_obj(adata_in, X):
    obs_cols = [
        c for c in [
            "sample", "batch",
            "stage", "final_celltype", "source_object",
            "rescue_view", "source_epi_leiden",
        ]
        if c in adata_in.obs.columns
    ]
    adata_out = ad.AnnData(
        X=copy_matrix(X),
        obs=adata_in.obs[obs_cols].copy(),
        var=pd.DataFrame(index=adata_in.var_names.copy()),
    )
    adata_out.obs_names = adata_in.obs_names.copy()
    adata_out.var_names = adata_in.var_names.copy()
    adata_out.obs_names_make_unique()
    adata_out.var_names_make_unique()
    return adata_out

adata_E11_normalized = make_integration_obj(adata_E11_final, adata_E11_final.X)
adata_E11_raw = make_integration_obj(adata_E11_final, adata_E11_final.layers["counts"])

adata_E11_normalized.write(harmony_dir / "E11_for_Harmony_normalized.h5ad")
adata_E11_raw.write(harmony_dir / "E11_for_Harmony_raw.h5ad")
adata_E11_normalized.write(scnvi_dir / "E11_for_scnvi_normalized.h5ad")
adata_E11_raw.write(scnvi_dir / "E11_for_scnvi_raw.h5ad")
# save final review figures

identity_markers = {
    "OV": [g for g in ["Epcam", "Pax2", "Fbxo2", "Foxg1", "Sox2", "Sox9", "Gata3", "Six1", "Eya1", "Tbx1", "Tbx2"] if g in adata_E11_final.var_names],
    "Medial_check": [g for g in ["Fgf10", "Ebf1"] if g in adata_E11_final.var_names],
    "Prosensory_check": [g for g in ["Sox2", "Jag1", "Lfng", "Notch1", "Notch2", "Eya1"] if g in adata_E11_final.var_names],
    "Lateral_check": [g for g in ["Bmp4", "Rorb", "Camta1", "Lockd", "Sema3e"] if g in adata_E11_final.var_names],
}
identity_markers = {k: v for k, v in identity_markers.items() if len(v) > 0}

exclude_markers = {
    "Roof": [g for g in ["Oc90", "Otx2"] if g in adata_E11_final.var_names],
    "Neural": [g for g in ["Neurod1", "Neurog1", "Insm1", "Myt1", "Tubb3", "Prph", "Phox2a", "Phox2b", "Isl1"] if g in adata_E11_final.var_names],
    "Mesenchyme": [g for g in ["Col1a1", "Col1a2", "Col3a1", "Prrx1", "Prrx2", "Twist1", "Pdgfra", "Pdgfrb", "Foxd1"] if g in adata_E11_final.var_names],
    "Blood/Immune": [g for g in ["Ptprc", "Lyz2", "Tyrobp", "Aif1", "Hbb-bt", "Hbb-bs"] if g in adata_E11_final.var_names],
}
exclude_markers = {k: v for k, v in exclude_markers.items() if len(v) > 0}

sc.pl.umap(
    adata_E11_final,
    color="rescue_view",
    legend_loc="right margin",
    size=18,
    show=False
)
plt.gcf().set_size_inches(4.2, 3.8)
plt.savefig(figure_dir / "E11.5_final_OV_epithelial_umap.png", dpi=SAVE_DPI, bbox_inches="tight")
plt.close()

sc.pl.dotplot(
    adata_E11_final,
    var_names=identity_markers,
    groupby="rescue_view",
    categories_order=keep_clusters,
    standard_scale=None,
    cmap="Reds",
    vmin=0,
    vmax=2.5,
    dot_max=0.9,
    show=False
)
plt.gcf().set_size_inches(11, 2.5)
plt.savefig(figure_dir / "E11.5_final_OV_epithelial_identity_dotplot.png", dpi=SAVE_DPI, bbox_inches="tight")
plt.close()

sc.pl.dotplot(
    adata_E11_final,
    var_names=exclude_markers,
    groupby="rescue_view",
    categories_order=keep_clusters,
    standard_scale=None,
    cmap="Reds",
    vmin=0,
    vmax=2.5,
    dot_max=0.9,
    show=False
)
plt.gcf().set_size_inches(11, 2.5)
plt.savefig(figure_dir / "E11.5_final_OV_epithelial_exclude_dotplot.png", dpi=SAVE_DPI, bbox_inches="tight")
plt.close()

# =========================================================
