#1. ===========Purpose and reproducibility settings
"""Preprocess, cluster, annotate, and export the E13.5 integration inputs."""

from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("COCHLEA_DATA_DIR", REPO_ROOT / "Data")).expanduser().resolve()
RESULTS_ROOT = Path(os.environ.get("COCHLEA_RESULTS_DIR", REPO_ROOT / "results")).expanduser().resolve()
STAGE_OBJECT_DIR = RESULTS_ROOT / "01_stage_clustering" / "h5ad_for_integration"
OUTPUT_DIR = RESULTS_ROOT / "01_stage_clustering" / "e13_5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STAGE_OBJECT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 0
SAVE_DPI = 1201
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_STATE))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Notebook cell 23


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

sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=SAVE_DPI, facecolor="white")
sc.set_figure_params(figsize=(8, 8))

# =========================================================
# 2. paths

base_dir = REPO_ROOT
figure_dir = OUTPUT_DIR
data_dir = DATA_ROOT / "raw" / "e13_5"

figure_dir.mkdir(parents=True, exist_ok=True)

os.chdir(base_dir)

sample_paths = {
    "E13_5_1": data_dir / "GSM5401076_E13_5_1_raw_count.csv",
    "E13_5_2": data_dir / "GSM5401077_E13_5_2_raw_count.csv",
    "E13_5_3": data_dir / "GSM5401078_E13_5_3_raw_count.csv",
    "E13_5_4": data_dir / "GSM5401079_E13_5_4_raw_count.csv",
}

# =========================================================
# 3. read and merge

sample_dfs = {}
for sample, path in sample_paths.items():
    df = pd.read_csv(path, index_col=0)
    sample_dfs[sample] = df

sample_names = list(sample_dfs.keys())
ref_genes = sample_dfs[sample_names[0]].index

for sample in sample_names[1:]:
    if not ref_genes.equals(sample_dfs[sample].index):
        raise ValueError(f"Gene order/content mismatch in {sample}")

combined_df = pd.concat(sample_dfs.values(), axis=1)

adata = ad.AnnData(X=combined_df.T)
adata.obs_names = combined_df.columns.astype(str)
adata.var_names = combined_df.index.astype(str)

cell_to_sample = []
for sample, df in sample_dfs.items():
    cell_to_sample.extend([sample] * df.shape[1])

adata.obs["sample"] = pd.Categorical(cell_to_sample)
adata.obs_names = [
    f"{sample}_{cell}"
    for sample, cell in zip(adata.obs["sample"].astype(str), adata.obs_names.astype(str))
]
adata.obs_names_make_unique()
adata.var_names_make_unique()

# =========================================================
# 4. QC

adata.var["mt"] = adata.var_names.str.startswith(("mt-", "Mt-", "MT-"))
adata.var["hb"] = adata.var_names.str.match(r"^Hb[ab]")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "hb"], inplace=True)

adata_qc = adata[
    (adata.obs["n_genes_by_counts"] >= 500)
    & (adata.obs["n_genes_by_counts"] <= 7000)
    & (adata.obs["total_counts"] >= 1000)
    & (adata.obs["total_counts"] <= 40000)
    & (adata.obs["pct_counts_mt"] <= 20)
].copy()

print("After QC:", adata_qc.n_obs)
print(adata_qc.obs["sample"].value_counts().sort_index())

# =========================================================
# 5. helper

def get_expr_vector(adata_obj: ad.AnnData, gene: str) -> np.ndarray:
    if gene not in adata_obj.var_names:
        return np.zeros(adata_obj.n_obs, dtype=float)
    x = adata_obj[:, gene].X
    if sparse.issparse(x):
        return x.toarray().ravel()
    return np.asarray(x).ravel()

# =========================================================
# 6. remove Oc90 / Otx2 positive cells before clustering

adata_qc.obs["Oc90_expr"] = get_expr_vector(adata_qc, "Oc90")
adata_qc.obs["Otx2_expr"] = get_expr_vector(adata_qc, "Otx2")

oc90_cutoff = 0.05
otx2_cutoff = 0.05

adata_no_oc90_otx2 = adata_qc[
    (adata_qc.obs["Oc90_expr"] <= oc90_cutoff)
    & (adata_qc.obs["Otx2_expr"] <= otx2_cutoff)
].copy()

print("After removing Oc90+ / Otx2+ cells:", adata_no_oc90_otx2.n_obs)

# =========================================================
# 7. remove blood-like cells before clustering

hb_genes = [
    g for g in ["Hba-a1", "Hba-a2", "Hbb-bs", "Hbb-bt", "Hbb-y"]
    if g in adata_no_oc90_otx2.var_names
]

if len(hb_genes) > 0:
    hb_expr = np.zeros(adata_no_oc90_otx2.n_obs, dtype=float)
    for g in hb_genes:
        hb_expr += get_expr_vector(adata_no_oc90_otx2, g)
    adata_no_oc90_otx2.obs["hb_sum_expr"] = hb_expr
    adata_clean = adata_no_oc90_otx2[adata_no_oc90_otx2.obs["hb_sum_expr"] == 0].copy()
else:
    adata_no_oc90_otx2.obs["hb_sum_expr"] = 0.0
    adata_clean = adata_no_oc90_otx2.copy()

print("After removing blood-like cells:", adata_clean.n_obs)
print(adata_clean.obs["sample"].value_counts().sort_index())


# Notebook cell 24

# =========================================================
# 8. all-cell clustering on cleaned object
# start from existing: adata_clean

adata_all = adata_clean.copy()
adata_all.layers["counts"] = adata_all.X.copy()

sc.pp.normalize_total(adata_all, target_sum=1e4)
sc.pp.log1p(adata_all)
adata_all.raw = adata_all.copy()

sc.pp.highly_variable_genes(
    adata_all,
    flavor="seurat",
    n_top_genes=3000,
    batch_key="sample",
)

adata_all_hvg = adata_all[:, adata_all.var["highly_variable"]].copy()

sc.pp.scale(adata_all_hvg, max_value=10)
sc.tl.pca(adata_all_hvg, svd_solver="arpack")
sc.pp.neighbors(adata_all_hvg, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata_all_hvg, random_state=0)
sc.tl.leiden(adata_all_hvg, resolution=0.3, key_added="all_leiden")

adata_all.obs["all_leiden"] = adata_all_hvg.obs["all_leiden"].astype(str).values
adata_all.obsm["X_umap"] = adata_all_hvg.obsm["X_umap"].copy()

print(adata_all.obs["all_leiden"].value_counts().sort_index())

# =========================================================
# 9. marker check

check_markers = [
    "Epcam", "Sox2", "Jag1", "Lfng", "Fgf10", "Bmp4", "Cdkn1b",
    "Camta1", "Lockd", "Eya1", "Ebf1",
    "Neurod1", "Prph", "Neurog1",
    "Erbb3", "Ednrb",
    "Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1",
    "Otx2", "Oc90",
]
check_markers = [g for g in check_markers if g in adata_all.raw.var_names]

cluster_mean = (
    sc.get.obs_df(
        adata_all,
        keys=["all_leiden"] + check_markers,
        use_raw=True,
    )
    .groupby("all_leiden")[check_markers]
    .mean()
    .round(3)
)
print(cluster_mean)

# =========================================================
# 10. top5 genes for each cluster

sc.tl.rank_genes_groups(
    adata_all,
    groupby="all_leiden",
    method="wilcoxon",
    use_raw=True,
)

res = adata_all.uns["rank_genes_groups"]
groups = res["names"].dtype.names

for g in groups:
    genes = []
    for x in res["names"][g]:
        if x not in genes:
            genes.append(x)
        if len(genes) == 5:
            break
    print(f"{g}: {genes}")

# =========================================================
# 11. final UMAP

sc.pl.umap(
    adata_all,
    color="all_leiden",
    legend_loc="right margin",
    size=10,
    show=False,
)
plt.close("all")
plt.close()

# =========================================================
# 12. final dotplot

marker_dict = {
    "epi_floor": [g for g in ["Epcam", "Sox2", "Jag1", "Lfng", "Fgf10", "Bmp4", "Cdkn1b", "Camta1", "Lockd", "Eya1", "Ebf1"] if g in adata_all.var_names],
    "neural": [g for g in ["Neurod1", "Prph", "Neurog1"] if g in adata_all.var_names],
    "glial": [g for g in ["Erbb3", "Ednrb"] if g in adata_all.var_names],
    "mesenchyme": [g for g in ["Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1"] if g in adata_all.var_names],
    "exclude": [g for g in ["Otx2", "Oc90"] if g in adata_all.var_names],
}
marker_dict = {k: v for k, v in marker_dict.items() if len(v) > 0}

sc.pl.dotplot(
    adata_all,
    var_names=marker_dict,
    groupby="all_leiden",
    standard_scale="var",
    show=False,
)

plt.close("all")
plt.close()


# Notebook cell 25

# =========================================================
# 12.5 sample UMAP check only

adata_all.obs["sample_label"] = adata_all.obs["sample"].astype(str)

sample_counts = adata_all.obs["sample_label"].value_counts().sort_index()
sample_label_map = {k: f"{k} (n={v})" for k, v in sample_counts.items()}

adata_all.obs["sample_label"] = adata_all.obs["sample_label"].map(sample_label_map)
adata_all.obs["sample_label"] = pd.Categorical(
    adata_all.obs["sample_label"],
    categories=[sample_label_map[k] for k in sample_counts.index],
    ordered=True,
)

sc.pl.umap(
    adata_all,
    color="sample_label",
    legend_loc="right margin",
    title="E13.5 all cells by sample",
    size=10,
    show=False,
)

plt.gcf().set_size_inches(5, 5)
plt.close("all")
plt.close()


# Notebook cell 26

# =========================================================
# 13. keep only the main epithelial cluster
# start from existing: adata_all, adata_clean

keep_epi_clusters = ["6", "7", "9", "10", "11"]

epi_names = adata_all.obs_names[
    adata_all.obs["all_leiden"].astype(str).isin(keep_epi_clusters)
]

adata_epi = adata_clean[epi_names].copy()
adata_epi.obs["all_leiden"] = adata_all.obs.loc[epi_names, "all_leiden"].astype(str).values

print(adata_epi.obs["all_leiden"].value_counts().sort_index())

# =========================================================
# 14. re-cluster epithelial core only
# restart from raw-like matrix in adata_clean

adata_epi.layers["counts"] = adata_epi.X.copy()

sc.pp.normalize_total(adata_epi, target_sum=1e4)
sc.pp.log1p(adata_epi)
adata_epi.raw = adata_epi.copy()

sc.pp.highly_variable_genes(
    adata_epi,
    flavor="seurat",
    n_top_genes=2000,
    batch_key="sample",
)

adata_epi_hvg = adata_epi[:, adata_epi.var["highly_variable"]].copy()

sc.pp.scale(adata_epi_hvg, max_value=10)
sc.tl.pca(adata_epi_hvg, svd_solver="arpack")
sc.pp.neighbors(adata_epi_hvg, n_neighbors=20, n_pcs=15)
sc.tl.umap(adata_epi_hvg, min_dist=0.15, spread=1.0, random_state=0)
sc.tl.leiden(adata_epi_hvg, resolution=0.2, key_added="epi_leiden")

adata_epi.obs["epi_leiden"] = adata_epi_hvg.obs["epi_leiden"].astype(str).values
adata_epi.obsm["X_umap"] = adata_epi_hvg.obsm["X_umap"].copy()

print(adata_epi.obs["epi_leiden"].value_counts().sort_index())

# =========================================================
# 15. marker check inside epithelial core

check_markers = [
    "Epcam",
    "Fgf10", "Fgf20", "Ebf1",
    "Sox2", "Sox9", "Gata3", "Lgr5", "Isl1",
    "Cdkn1b", "Jag1", "Lfng", "Notch1", "Notch2", "Eya1",
    "Bmp4", "Camta1", "Lockd", "Fgfr3", "Rorb",
    "Neurod1", "Prph", "Neurog1", "Insm1", "Myt1", "Nhlh2",
    "Erbb3", "Ednrb", "Sox10", "Plp1", "Mpz",
    "Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1",
    "Otx2", "Oc90",
]
check_markers = [g for g in check_markers if g in adata_epi.raw.var_names]

epi_mean = (
    sc.get.obs_df(
        adata_epi,
        keys=["epi_leiden"] + check_markers,
        use_raw=True,
    )
    .groupby("epi_leiden")[check_markers]
    .mean()
    .round(3)
)
print(epi_mean)

# =========================================================
# 16. top5 genes for each epithelial subcluster

sc.tl.rank_genes_groups(
    adata_epi,
    groupby="epi_leiden",
    method="wilcoxon",
    use_raw=True,
)

res = adata_epi.uns["rank_genes_groups"]
groups = res["names"].dtype.names

for g in groups:
    genes = []
    for x in res["names"][g]:
        if x not in genes:
            genes.append(x)
        if len(genes) == 5:
            break
    print(f"{g}: {genes}")

# =========================================================
# 17. final UMAP

adata_epi.obs["epi_leiden_label"] = adata_epi.obs["epi_leiden"].astype(str)
epi_counts = adata_epi.obs["epi_leiden_label"].value_counts().sort_index()
epi_label_map = {k: f"{k} (n={v})" for k, v in epi_counts.items()}

adata_epi.obs["epi_leiden_label"] = adata_epi.obs["epi_leiden_label"].map(epi_label_map)
adata_epi.obs["epi_leiden_label"] = pd.Categorical(
    adata_epi.obs["epi_leiden_label"],
    categories=[epi_label_map[k] for k in epi_counts.index],
    ordered=True,
)

sc.pl.umap(
    adata_epi,
    color="epi_leiden_label",
    legend_loc="right margin",
    size=14,
    show=False,
)
plt.gcf().set_size_inches(6, 5)
plt.close("all")
plt.close()

# =========================================================
# 18. final dotplot

marker_dict = {
    "shared_check": [g for g in ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"] if g in adata_epi.var_names],
    "medial": [g for g in ["Fgf10", "Fgf20", "Ebf1"] if g in adata_epi.var_names],
    "prosensory": [g for g in ["Sox2", "Cdkn1b", "Jag1", "Lfng", "Notch1", "Notch2", "Eya1"] if g in adata_epi.var_names],
    "lateral": [g for g in ["Bmp4", "Camta1", "Lockd", "Fgfr3", "Rorb"] if g in adata_epi.var_names],
    "exclude_check": [g for g in ["Neurod1", "Prph", "Neurog1", "Insm1", "Myt1", "Nhlh2", "Erbb3", "Ednrb", "Sox10", "Plp1", "Mpz", "Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1", "Otx2", "Oc90"] if g in adata_epi.var_names],
}
marker_dict = {k: v for k, v in marker_dict.items() if len(v) > 0}

sc.pl.dotplot(
    adata_epi,
    var_names=marker_dict,
    groupby="epi_leiden",
    standard_scale="var",
    dot_max=0.9,
    figsize=(15, 5),
    show=False,
)

plt.close("all")
plt.close()


# Notebook cell 27

# =========================================================
# 19. keep epithelial candidate subclusters
# start from existing: adata_epi, adata_clean

keep_epi_leiden = ["0", "3", "5"]

epi2_names = adata_epi.obs_names[
    adata_epi.obs["epi_leiden"].astype(str).isin(keep_epi_leiden)
]

adata_epi2 = adata_clean[epi2_names].copy()
adata_epi2.obs["all_leiden"] = adata_all.obs.loc[epi2_names, "all_leiden"].astype(str).values
adata_epi2.obs["epi_leiden"] = adata_epi.obs.loc[epi2_names, "epi_leiden"].astype(str).values

print(adata_epi2.obs["epi_leiden"].value_counts().sort_index())


# Notebook cell 28

# =========================================================
# 20. re-cluster epithelial candidate cells

adata_epi2.layers["counts"] = adata_epi2.X.copy()

sc.pp.normalize_total(adata_epi2, target_sum=1e4)
sc.pp.log1p(adata_epi2)
adata_epi2.raw = adata_epi2.copy()

sc.pp.highly_variable_genes(
    adata_epi2,
    flavor="seurat",
    n_top_genes=2000,
    batch_key="sample",
)

adata_epi2_hvg = adata_epi2[:, adata_epi2.var["highly_variable"]].copy()

sc.pp.scale(adata_epi2_hvg, max_value=10)
sc.tl.pca(adata_epi2_hvg, svd_solver="arpack")
sc.pp.neighbors(adata_epi2_hvg, n_neighbors=10, n_pcs=15)
sc.tl.umap(adata_epi2_hvg, min_dist=0.12, spread=1.0, random_state=0)
sc.tl.leiden(adata_epi2_hvg, resolution=0.45, key_added="epi2_leiden")

adata_epi2.obs["epi2_leiden"] = adata_epi2_hvg.obs["epi2_leiden"].astype(str).values
adata_epi2.obsm["X_umap"] = adata_epi2_hvg.obsm["X_umap"].copy()

print(adata_epi2.obs["epi2_leiden"].value_counts().sort_index())
print(pd.crosstab(adata_epi2.obs["epi2_leiden"], adata_epi2.obs["epi_leiden"]))


# Notebook cell 29

# =========================================================
# 21. marker expression check

marker_check = [
    "Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1",
    "Fgf10", "Fgf20", "Ebf1",
    "Cdkn1b", "Jag1", "Lfng", "Notch1", "Notch2", "Eya1",
    "Bmp4", "Camta1", "Lockd", "Fgfr3", "Rorb",
    "Neurod1", "Prph", "Neurog1", "Insm1", "Myt1", "Nhlh2",
    "Erbb3", "Ednrb", "Sox10", "Plp1", "Mpz",
    "Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1",
    "Otx2", "Oc90",
]

sc.get.obs_df(
    adata_epi2,
    keys=["epi2_leiden"] + marker_check,
    use_raw=True,
).groupby("epi2_leiden").mean().round(3)


# Notebook cell 30

# =========================================================
# 22. top marker genes

sc.tl.rank_genes_groups(
    adata_epi2,
    groupby="epi2_leiden",
    method="wilcoxon",
    use_raw=True,
)

top_genes = pd.DataFrame(adata_epi2.uns["rank_genes_groups"]["names"]).head(5)
for cluster in top_genes.columns:
    print(f"{cluster}: {top_genes[cluster].tolist()}")


# Notebook cell 31

# =========================================================
# 23. UMAP check

epi2_counts = adata_epi2.obs["epi2_leiden"].value_counts().to_dict()
adata_epi2.obs["epi2_leiden_label"] = adata_epi2.obs["epi2_leiden"].map(
    lambda x: f"{x} (n={epi2_counts[x]})"
)

sc.pl.umap(
    adata_epi2,
    color="epi2_leiden_label",
    title="E13.5 epithelial candidate",
    size=8,
    legend_loc="right margin",
    frameon=True,
)


# Notebook cell 32

# =========================================================
# 24. dotplot check

marker_dict_epi2 = {
    "shared_check": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "medial": ["Fgf10", "Fgf20", "Ebf1"],
    "prosensory": ["Cdkn1b", "Jag1", "Lfng", "Notch1", "Notch2", "Eya1"],
    "lateral": ["Bmp4", "Camta1", "Lockd", "Fgfr3", "Rorb"],
    "exclude_check": [
        "Neurod1", "Prph", "Neurog1", "Insm1", "Myt1", "Nhlh2",
        "Erbb3", "Ednrb", "Sox10", "Plp1", "Mpz",
        "Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1",
        "Otx2", "Oc90",
    ],
}

sc.pl.dotplot(
    adata_epi2,
    var_names=marker_dict_epi2,
    groupby="epi2_leiden",
    use_raw=True,
    standard_scale="var",
    dot_max=0.9,
    dot_min=0.05,
    figsize=(17, 5),
    dendrogram=False,
)


# Notebook cell 33

# =========================================================
# 25. keep cleaner epithelial candidates
# start from existing: adata_epi2, adata_clean

keep_epi2_leiden = ["0", "1", "2", "3", "4", "5", "7"]

epi3_names = adata_epi2.obs_names[
    adata_epi2.obs["epi2_leiden"].astype(str).isin(keep_epi2_leiden)
]

adata_epi3 = adata_clean[epi3_names].copy()
adata_epi3.obs["epi2_leiden"] = adata_epi2.obs.loc[epi3_names, "epi2_leiden"].astype(str).values

print(adata_epi3.obs["epi2_leiden"].value_counts().sort_index())

# =========================================================
# 26. re-cluster retained epithelial candidates
# restart from full-gene matrix

adata_epi3.layers["counts"] = adata_epi3.X.copy()

sc.pp.normalize_total(adata_epi3, target_sum=1e4)
sc.pp.log1p(adata_epi3)
adata_epi3.raw = adata_epi3.copy()

sc.pp.highly_variable_genes(
    adata_epi3,
    flavor="seurat",
    n_top_genes=2000,
    batch_key="sample",
)

adata_epi3_hvg = adata_epi3[:, adata_epi3.var["highly_variable"]].copy()

sc.pp.scale(adata_epi3_hvg, max_value=10)
sc.tl.pca(adata_epi3_hvg, svd_solver="arpack")
sc.pp.neighbors(adata_epi3_hvg, n_neighbors=10, n_pcs=15)
sc.tl.umap(adata_epi3_hvg, min_dist=0.12, spread=1.0, random_state=0)
sc.tl.leiden(adata_epi3_hvg, resolution=0.40, key_added="epi3_leiden")

adata_epi3.obs["epi3_leiden"] = adata_epi3_hvg.obs["epi3_leiden"].astype(str).values
adata_epi3.obsm["X_umap"] = adata_epi3_hvg.obsm["X_umap"].copy()

print(adata_epi3.obs["epi3_leiden"].value_counts().sort_index())
print(pd.crosstab(adata_epi3.obs["epi3_leiden"], adata_epi3.obs["epi2_leiden"]))

# =========================================================
# 27. marker check after re-clustering

check_markers = [
    "Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1",
    "Fgf10", "Fgf20", "Ebf1",
    "Cdkn1b", "Jag1", "Lfng", "Notch1", "Notch2", "Eya1",
    "Bmp4", "Camta1", "Lockd", "Fgfr3", "Rorb",
    "Neurod1", "Prph", "Neurog1", "Insm1", "Myt1", "Nhlh2",
    "Erbb3", "Ednrb", "Sox10", "Plp1", "Mpz",
    "Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1",
    "Otx2", "Oc90",
]
check_markers = [g for g in check_markers if g in adata_epi3.raw.var_names]

epi3_mean = (
    sc.get.obs_df(
        adata_epi3,
        keys=["epi3_leiden"] + check_markers,
        use_raw=True,
    )
    .groupby("epi3_leiden")[check_markers]
    .mean()
    .round(3)
)
print(epi3_mean)

# =========================================================
# 28. top5 genes for each cluster

sc.tl.rank_genes_groups(
    adata_epi3,
    groupby="epi3_leiden",
    method="wilcoxon",
    use_raw=True,
)

res = adata_epi3.uns["rank_genes_groups"]
groups = res["names"].dtype.names

for g in groups:
    genes = []
    for x in res["names"][g]:
        if x not in genes:
            genes.append(x)
        if len(genes) == 5:
            break
    print(f"{g}: {genes}")

# =========================================================
# 29. final UMAP

epi3_counts = adata_epi3.obs["epi3_leiden"].value_counts().to_dict()
adata_epi3.obs["epi3_leiden_label"] = adata_epi3.obs["epi3_leiden"].map(
    lambda x: f"{x} (n={epi3_counts[x]})"
)

sc.pl.umap(
    adata_epi3,
    color="epi3_leiden_label",
    title="E13.5 cleaner epithelial candidates",
    legend_loc="right margin",
    size=14,
    frameon=True,
    show=False,
)
plt.close("all")
plt.close()

# =========================================================
# 30. final dotplot

marker_dict = {
    "shared_check": [g for g in ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"] if g in adata_epi3.raw.var_names],
    "medial": [g for g in ["Fgf10", "Fgf20", "Ebf1"] if g in adata_epi3.raw.var_names],
    "prosensory": [g for g in ["Cdkn1b", "Jag1", "Lfng", "Notch1", "Notch2", "Eya1"] if g in adata_epi3.raw.var_names],
    "lateral": [g for g in ["Bmp4", "Camta1", "Lockd", "Fgfr3", "Rorb"] if g in adata_epi3.raw.var_names],
    "exclude_check": [g for g in [
        "Neurod1", "Prph", "Neurog1", "Insm1", "Myt1", "Nhlh2",
        "Erbb3", "Ednrb", "Sox10", "Plp1", "Mpz",
        "Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1",
        "Otx2", "Oc90",
    ] if g in adata_epi3.raw.var_names],
}
marker_dict = {k: v for k, v in marker_dict.items() if len(v) > 0}

sc.pl.dotplot(
    adata_epi3,
    var_names=marker_dict,
    groupby="epi3_leiden",
    use_raw=True,
    standard_scale="var",
    dot_max=0.9,
    dot_min=0.05,
    figsize=(19, 5),
    dendrogram=False,
    show=False,
)
plt.close("all")
plt.close()


# Notebook cell 34

# =========================================================
# 31. build final conservative epithelial candidate object
# start from existing: adata_clean, adata_all, adata_epi, adata_epi2, adata_epi3

final_keep = ["0", "1", "2", "3", "4", "5", "6"]

final_names = adata_epi3.obs_names[
    adata_epi3.obs["epi3_leiden"].astype(str).isin(final_keep)
]

adata_E13_final = adata_clean[final_names].copy()

adata_E13_final.obs["all_leiden"] = adata_all.obs.loc[final_names, "all_leiden"].astype(str).values
adata_E13_final.obs["epi_leiden"] = adata_epi.obs.loc[final_names, "epi_leiden"].astype(str).values
adata_E13_final.obs["epi2_leiden"] = adata_epi2.obs.loc[final_names, "epi2_leiden"].astype(str).values
adata_E13_final.obs["epi3_leiden"] = adata_epi3.obs.loc[final_names, "epi3_leiden"].astype(str).values

print(adata_E13_final.obs["epi3_leiden"].value_counts().sort_index())
print(pd.crosstab(adata_E13_final.obs["epi3_leiden"], adata_E13_final.obs["sample"]))

# =========================================================
# 32. recompute final embedding on filtered final object

adata_E13_final.layers["counts"] = adata_E13_final.X.copy()

sc.pp.normalize_total(adata_E13_final, target_sum=1e4)
sc.pp.log1p(adata_E13_final)
adata_E13_final.raw = adata_E13_final.copy()

sc.pp.highly_variable_genes(
    adata_E13_final,
    flavor="seurat",
    n_top_genes=2000,
    batch_key="sample",
)

adata_E13_final_hvg = adata_E13_final[:, adata_E13_final.var["highly_variable"]].copy()

sc.pp.scale(adata_E13_final_hvg, max_value=10)
sc.tl.pca(adata_E13_final_hvg, svd_solver="arpack")
sc.pp.neighbors(adata_E13_final_hvg, n_neighbors=10, n_pcs=15)
sc.tl.umap(adata_E13_final_hvg, min_dist=0.12, spread=1.0, random_state=0)

adata_E13_final.obsm["X_umap"] = adata_E13_final_hvg.obsm["X_umap"].copy()

# =========================================================
# 33. marker check after final filtering

check_markers = [
    "Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1",
    "Fgf10", "Fgf20", "Ebf1",
    "Cdkn1b", "Jag1", "Lfng", "Notch1", "Notch2", "Eya1",
    "Bmp4", "Camta1", "Lockd", "Fgfr3", "Rorb",
    "Neurod1", "Prph", "Neurog1", "Insm1", "Myt1", "Nhlh2",
    "Erbb3", "Ednrb", "Sox10", "Plp1", "Mpz",
    "Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1",
    "Otx2", "Oc90",
]
check_markers = [g for g in check_markers if g in adata_E13_final.raw.var_names]

final_marker_mean = (
    sc.get.obs_df(
        adata_E13_final,
        keys=["epi3_leiden"] + check_markers,
        use_raw=True,
    )
    .groupby("epi3_leiden")[check_markers]
    .mean()
    .round(3)
)
print(final_marker_mean)

# =========================================================
# 34. final UMAP and dotplot check

epi3_counts = adata_E13_final.obs["epi3_leiden"].value_counts().to_dict()
adata_E13_final.obs["epi3_leiden_label"] = adata_E13_final.obs["epi3_leiden"].map(
    lambda x: f"{x} (n={epi3_counts[x]})"
)

sample_counts = adata_E13_final.obs["sample"].value_counts().to_dict()
adata_E13_final.obs["sample_label"] = adata_E13_final.obs["sample"].map(
    lambda x: f"{x} (n={sample_counts[x]})"
)

sc.pl.umap(
    adata_E13_final,
    color="epi3_leiden_label",
    title="E13.5 final epithelial candidate",
    legend_loc="right margin",
    size=16,
    frameon=True,
    show=False,
)
plt.close("all")
plt.close()

sc.pl.umap(
    adata_E13_final,
    color="sample_label",
    title="E13.5 final sample check",
    legend_loc="right margin",
    size=16,
    frameon=True,
    show=False,
)
plt.close("all")
plt.close()

marker_dict = {
    "shared_check": [g for g in ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"] if g in adata_E13_final.raw.var_names],
    "medial": [g for g in ["Fgf10", "Fgf20", "Ebf1"] if g in adata_E13_final.raw.var_names],
    "prosensory": [g for g in ["Cdkn1b", "Jag1", "Lfng", "Notch1", "Notch2", "Eya1"] if g in adata_E13_final.raw.var_names],
    "lateral": [g for g in ["Bmp4", "Camta1", "Lockd", "Fgfr3", "Rorb"] if g in adata_E13_final.raw.var_names],
    "exclude_check": [g for g in [
        "Neurod1", "Prph", "Neurog1", "Insm1", "Myt1", "Nhlh2",
        "Erbb3", "Ednrb", "Sox10", "Plp1", "Mpz",
        "Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1",
        "Otx2", "Oc90",
    ] if g in adata_E13_final.raw.var_names],
}
marker_dict = {k: v for k, v in marker_dict.items() if len(v) > 0}

sc.pl.dotplot(
    adata_E13_final,
    var_names=marker_dict,
    groupby="epi3_leiden",
    use_raw=True,
    standard_scale="var",
    dot_max=0.9,
    dot_min=0.05,
    figsize=(19, 5),
    dendrogram=False,
    show=False,
)
plt.close("all")
plt.close()


# Notebook cell 35

# =========================================================
# 35. domain score check before final annotation
# start from existing: adata_E13_final

from scipy import sparse

domain_markers = {
    "medial_score": ["Fgf10", "Fgf20", "Ebf1"],
    "prosensory_score": ["Sox2", "Cdkn1b", "Jag1", "Lfng", "Notch1", "Notch2", "Eya1"],
    "lateral_score": ["Bmp4", "Camta1", "Lockd", "Fgfr3", "Rorb"],
    "exclude_score": [
        "Neurod1", "Prph", "Neurog1", "Insm1", "Myt1", "Nhlh2",
        "Erbb3", "Ednrb", "Sox10", "Plp1", "Mpz",
        "Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1",
        "Otx2", "Oc90",
    ],
}

def add_mean_marker_score(adata_in, score_name, genes):
    genes = [g for g in genes if g in adata_in.raw.var_names]
    X = adata_in.raw[:, genes].X
    if sparse.issparse(X):
        X = X.toarray()
    adata_in.obs[score_name] = np.asarray(X).mean(axis=1)

for score_name, genes in domain_markers.items():
    add_mean_marker_score(adata_E13_final, score_name, genes)

score_cols = list(domain_markers.keys())

domain_score_mean = (
    adata_E13_final.obs[["epi3_leiden"] + score_cols]
    .groupby("epi3_leiden")[score_cols]
    .mean()
    .round(3)
)
print(domain_score_mean)

# =========================================================
# 36. domain score UMAP check

sc.pl.umap(
    adata_E13_final,
    color=["medial_score", "prosensory_score", "lateral_score", "exclude_score"],
    cmap="Reds",
    size=16,
    frameon=True,
    show=False,
)
plt.close("all")
plt.close()

# =========================================================
# 37. unscaled final dotplot check

marker_dict = {
    "shared_check": [g for g in ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"] if g in adata_E13_final.raw.var_names],
    "medial": [g for g in ["Fgf10", "Fgf20", "Ebf1"] if g in adata_E13_final.raw.var_names],
    "prosensory": [g for g in ["Cdkn1b", "Jag1", "Lfng", "Notch1", "Notch2", "Eya1"] if g in adata_E13_final.raw.var_names],
    "lateral": [g for g in ["Bmp4", "Camta1", "Lockd", "Fgfr3", "Rorb"] if g in adata_E13_final.raw.var_names],
    "exclude_check": [g for g in [
        "Neurod1", "Prph", "Neurog1", "Insm1", "Myt1", "Nhlh2",
        "Erbb3", "Ednrb", "Sox10", "Plp1", "Mpz",
        "Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1",
        "Otx2", "Oc90",
    ] if g in adata_E13_final.raw.var_names],
}
marker_dict = {k: v for k, v in marker_dict.items() if len(v) > 0}

sc.pl.dotplot(
    adata_E13_final,
    var_names=marker_dict,
    groupby="epi3_leiden",
    use_raw=True,
    dot_max=0.9,
    dot_min=0.05,
    figsize=(20, 5),
    dendrogram=False,
    show=False,
)
plt.close("all")
plt.close()


# Notebook cell 36

# =========================================================
# 38. assign final E13.5 epithelial domains by domain score
# start from existing: adata_E13_final

final_map = {
    "0": "Lateral domain",
    "1": "Medial domain",
    "2": "Medial domain",
    "3": "Prosensory domain",
    "4": "Prosensory domain",
    "5": "Prosensory domain",
    "6": "Medial domain",
}

adata_E13_final.obs["final_celltype"] = (
    adata_E13_final.obs["epi3_leiden"].astype(str).map(final_map)
)

final_order = ["Medial domain", "Prosensory domain", "Lateral domain"]
adata_E13_final.obs["final_celltype"] = pd.Categorical(
    adata_E13_final.obs["final_celltype"],
    categories=final_order,
    ordered=True,
)

print(adata_E13_final.obs["final_celltype"].value_counts().reindex(final_order))
print(pd.crosstab(adata_E13_final.obs["final_celltype"], adata_E13_final.obs["sample"]))

# =========================================================
# 39. final celltype UMAP and dotplot check

final_counts = adata_E13_final.obs["final_celltype"].value_counts().to_dict()
adata_E13_final.obs["final_celltype_label"] = adata_E13_final.obs["final_celltype"].map(
    lambda x: f"{x} (n={final_counts[x]})"
)

sc.pl.umap(
    adata_E13_final,
    color="final_celltype_label",
    title="E13.5 final epithelial domains",
    legend_loc="right margin",
    size=16,
    frameon=True,
    show=False,
)
plt.close("all")
plt.close()

marker_dict = {
    "shared_check": [g for g in ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"] if g in adata_E13_final.raw.var_names],
    "medial": [g for g in ["Fgf10", "Fgf20", "Ebf1"] if g in adata_E13_final.raw.var_names],
    "prosensory": [g for g in ["Cdkn1b", "Jag1", "Lfng", "Notch1", "Notch2", "Eya1"] if g in adata_E13_final.raw.var_names],
    "lateral": [g for g in ["Bmp4", "Camta1", "Lockd", "Fgfr3", "Rorb"] if g in adata_E13_final.raw.var_names],
    "exclude_check": [g for g in [
        "Neurod1", "Prph", "Neurog1", "Insm1", "Myt1", "Nhlh2",
        "Erbb3", "Ednrb", "Sox10", "Plp1", "Mpz",
        "Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1",
        "Otx2", "Oc90",
    ] if g in adata_E13_final.raw.var_names],
}
marker_dict = {k: v for k, v in marker_dict.items() if len(v) > 0}

sc.pl.dotplot(
    adata_E13_final,
    var_names=marker_dict,
    groupby="final_celltype",
    use_raw=True,
    dot_max=0.9,
    dot_min=0.05,
    figsize=(19, 4),
    dendrogram=False,
    show=False,
)
plt.close("all")
plt.close()


# Notebook cell 37

# =========================================================
# 40. save final figures and integration h5ad files
# start from existing: adata_E13_final
from pathlib import Path
import anndata as ad

base_dir = REPO_ROOT
figure_dir = OUTPUT_DIR
harmony_dir = RESULTS_ROOT / "01_stage_clustering" / "harmony_inputs" / "e13_5"
scnvi_dir = STAGE_OBJECT_DIR

figure_dir.mkdir(parents=True, exist_ok=True)
harmony_dir.mkdir(parents=True, exist_ok=True)
scnvi_dir.mkdir(parents=True, exist_ok=True)

umap_path = figure_dir / "E13.5_final_epithelial_domains_umap.png"
dotplot_path = figure_dir / "E13.5_final_epithelial_domains_dotplot.png"

sc.pl.umap(
    adata_E13_final,
    color="final_celltype_label",
    title="E13.5 final epithelial domains",
    legend_loc="right margin",
    size=16,
    frameon=True,
    show=False,
)
plt.savefig(umap_path, dpi=SAVE_DPI, bbox_inches="tight")
plt.close("all")
plt.close()

marker_dict = {
    "shared_check": [g for g in ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"] if g in adata_E13_final.raw.var_names],
    "medial": [g for g in ["Fgf10", "Fgf20", "Ebf1"] if g in adata_E13_final.raw.var_names],
    "prosensory": [g for g in ["Cdkn1b", "Jag1", "Lfng", "Notch1", "Notch2", "Eya1"] if g in adata_E13_final.raw.var_names],
    "lateral": [g for g in ["Bmp4", "Camta1", "Lockd", "Fgfr3", "Rorb"] if g in adata_E13_final.raw.var_names],
    "exclude_check": [g for g in [
        "Neurod1", "Prph", "Neurog1", "Insm1", "Myt1", "Nhlh2",
        "Erbb3", "Ednrb", "Sox10", "Plp1", "Mpz",
        "Col1a1", "Col1a2", "Pdgfra", "Foxd1", "Twist1",
        "Otx2", "Oc90",
    ] if g in adata_E13_final.raw.var_names],
}
marker_dict = {k: v for k, v in marker_dict.items() if len(v) > 0}

dp = sc.pl.dotplot(
    adata_E13_final,
    var_names=marker_dict,
    groupby="final_celltype",
    use_raw=True,
    dot_max=0.9,
    dot_min=0.05,
    figsize=(19, 4),
    dendrogram=False,
    return_fig=True,
)
dp.savefig(dotplot_path, dpi=SAVE_DPI, bbox_inches="tight")
plt.close("all")

adata_E13_final.obs["stage"] = "E13.5"
adata_E13_final.obs["final_celltype"] = adata_E13_final.obs["final_celltype"].astype(str)
adata_E13_final.obs["source_object"] = "adata_E13_final_epithelial_domains"

obs_cols = [
    "sample",
    "stage",
    "final_celltype",
    "source_object",
    "all_leiden",
    "epi_leiden",
    "epi2_leiden",
    "epi3_leiden",
]

obs_keep = adata_E13_final.obs[[c for c in obs_cols if c in adata_E13_final.obs.columns]].copy()
var_keep = pd.DataFrame(index=adata_E13_final.var_names.copy())

adata_E13_normalized = ad.AnnData(
    X=adata_E13_final.X.copy(),
    obs=obs_keep.copy(),
    var=var_keep.copy(),
)
adata_E13_normalized.obs_names = adata_E13_final.obs_names.copy()
adata_E13_normalized.var_names = adata_E13_final.var_names.copy()

adata_E13_raw = ad.AnnData(
    X=adata_E13_final.layers["counts"].copy(),
    obs=obs_keep.copy(),
    var=var_keep.copy(),
)
adata_E13_raw.obs_names = adata_E13_final.obs_names.copy()
adata_E13_raw.var_names = adata_E13_final.var_names.copy()

harmony_normalized_path = harmony_dir / "E13.5_for_Harmony_normalized.h5ad"
harmony_raw_path = harmony_dir / "E13.5_for_Harmony_raw.h5ad"
scnvi_normalized_path = scnvi_dir / "E13.5_for_scnvi_normalized.h5ad"
scnvi_raw_path = scnvi_dir / "E13.5_for_scnvi_raw.h5ad"

adata_E13_normalized.write(harmony_normalized_path)
adata_E13_raw.write(harmony_raw_path)
adata_E13_normalized.write(scnvi_normalized_path)
adata_E13_raw.write(scnvi_raw_path)
