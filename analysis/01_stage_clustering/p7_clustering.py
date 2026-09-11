#1. ===========Purpose and reproducibility settings
"""Preprocess, cluster, annotate, and export the P7 integration inputs."""

from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("COCHLEA_DATA_DIR", REPO_ROOT / "Data")).expanduser().resolve()
RESULTS_ROOT = Path(os.environ.get("COCHLEA_RESULTS_DIR", REPO_ROOT / "results")).expanduser().resolve()
STAGE_OBJECT_DIR = RESULTS_ROOT / "01_stage_clustering" / "h5ad_for_integration"
OUTPUT_DIR = RESULTS_ROOT / "01_stage_clustering" / "p7"
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

# Notebook cell 22

# ======================== 1. P7 setup and build AnnData ========================

import os
import io
import warnings
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=120, facecolor="white", figsize=(5, 4))

base_dir = REPO_ROOT
data_path = DATA_ROOT / "raw" / "p7" / "P7_Normalized_Counts.txt"
os.chdir(base_dir)

df = pd.read_csv(data_path, sep="\t", index_col=0)

adata_P7 = ad.AnnData(X=df.T.copy())
adata_P7.obs_names = df.columns.astype(str)
adata_P7.var_names = df.index.astype(str)
adata_P7.obs_names_make_unique()
adata_P7.var_names_make_unique()

obs_names = adata_P7.obs_names.to_series()
adata_P7.obs["batch"] = (
    obs_names.str.extract(r"^(se_\d+)_P7", expand=False)
    .fillna("unknown")
    .astype("category")
)

print(adata_P7)
print(adata_P7.obs["batch"].value_counts())


# ======================== 2. normalized QC precheck ========================
adata_P7.var["mt"] = adata_P7.var_names.str.startswith(("mt-", "Mt-", "MT-"))
adata_P7.var["ribo"] = adata_P7.var_names.str.startswith(("Rps", "Rpl"))

stress_genes = [
    "Fos", "Fosb", "Jun", "Junb", "Jund",
    "Egr1", "Dusp1", "Atf3",
    "Hspa1a", "Hspa1b", "Hsp90aa1", "Hsp90ab1", "Hsph1",
]
adata_P7.var["stress"] = adata_P7.var_names.isin(stress_genes)

total_expr = np.asarray(adata_P7.X.sum(axis=1)).ravel()
total_expr_safe = np.where(total_expr > 0, total_expr, np.nan)

adata_P7.obs["total_expr_norm"] = total_expr
adata_P7.obs["n_genes_norm"] = np.asarray((adata_P7.X > 0).sum(axis=1)).ravel()

for key, col in [
    ("pct_expr_mt_norm", "mt"),
    ("pct_expr_ribo_norm", "ribo"),
    ("pct_expr_stress_norm", "stress"),
]:
    mask = adata_P7.var[col].to_numpy()
    part = np.asarray(adata_P7[:, mask].X.sum(axis=1)).ravel() if mask.sum() > 0 else 0
    adata_P7.obs[key] = np.nan_to_num(part / total_expr_safe * 100)

print(
    adata_P7.obs[
        ["n_genes_norm", "total_expr_norm", "pct_expr_mt_norm", "pct_expr_stress_norm"]
    ].describe().round(2)
)

sc.pl.violin(
    adata_P7,
    keys=["n_genes_norm", "total_expr_norm", "pct_expr_mt_norm", "pct_expr_stress_norm"],
    groupby="batch",
    jitter=0.25,
    rotation=0,
    multi_panel=True,
    show=False,
)
plt.gcf().set_size_inches(9, 3)
plt.close("all")
plt.close("all")


# ======================== 3. Harmony global clustering for inspection ========================
sc.pp.highly_variable_genes(
    adata_P7,
    flavor="seurat",
    n_top_genes=3000,
    batch_key="batch",
)

adata_P7_work = adata_P7[:, adata_P7.var["highly_variable"]].copy()
sc.pp.scale(adata_P7_work, max_value=10)
sc.tl.pca(adata_P7_work, svd_solver="arpack")

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    sce.pp.harmony_integrate(
        adata_P7_work,
        key="batch",
        basis="X_pca",
        adjusted_basis="X_pca_harmony",
        max_iter_harmony=20,
    )

sc.pp.neighbors(
    adata_P7_work,
    n_neighbors=15,
    n_pcs=30,
    use_rep="X_pca_harmony",
)
sc.tl.umap(adata_P7_work, random_state=0)
sc.tl.leiden(
    adata_P7_work,
    resolution=0.6,
    key_added="leiden",
    random_state=0,
)

adata_P7.obs["leiden"] = adata_P7_work.obs["leiden"].copy()
adata_P7.obsm["X_umap"] = adata_P7_work.obsm["X_umap"].copy()
adata_P7.obsm["X_pca_harmony"] = adata_P7_work.obsm["X_pca_harmony"].copy()

leiden_order = sorted(adata_P7.obs["leiden"].unique(), key=lambda x: int(x))
adata_P7.obs["leiden"] = pd.Categorical(
    adata_P7.obs["leiden"].astype(str),
    categories=leiden_order,
    ordered=True,
)

print(adata_P7.obs["leiden"].value_counts().sort_index())


# ======================== 4. exclusion UMAP check ========================
umap_check = [
    "leiden", "batch",
    "Oc90", "Otx2",
    "Cnp", "Mbp", "Mpz", "Pmp22",
    "Ptprc", "Lyz2", "Hbb-bs", "Hba-a1",
]
umap_check = [
    x for x in umap_check
    if x in adata_P7.obs.columns or x in adata_P7.var_names
]

sc.pl.umap(
    adata_P7,
    color=umap_check,
    ncols=4,
    size=7,
    frameon=False,
    show=False,
)
plt.gcf().set_size_inches(11, 8)
plt.close("all")
plt.close("all")


# ======================== 5. P7 marker dotplot precheck ========================
p7_marker_dict = {
    "Core": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "HC": ["Myo6", "Myo7a", "Cib2", "Pvalb", "Ccer2", "Acbd7", "Rprm", "Pcp4", "Cd164l2"],
    "IHC": ["Otof", "Atp2a3", "Cabp2", "Fgf8", "Dlk2", "Nefl", "Calb2"],
    "OHC": ["Aqp11", "Ocm", "Calb1", "Six2", "Ighm"],
    "IPhC": ["Igfbp4", "Maff", "Apoe", "Fst", "Cnmd"],
    "IPC": ["Ppp2r2b", "Npy", "Cep41", "Tuba1a", "Tuba1b"],
    "OPC": ["Sapcd2", "Map4", "S100b", "Smagp", "Cep41"],
    "DC": ["Car14", "Mansc4", "Rbp7", "Rflnb", "Selenom"],
    "HeC": ["Arpc1b", "Sostdc1", "Fabp7", "Ednrb", "Plp1"],
    "KO": ["Cldn4", "Upp1", "Ly6h", "Ier5", "Clu", "Epyc", "Grb14", "Fam159b", "Qpct", "Anxa5", "Foxq1", "Igfbp3", "Gsn", "Hspa2"],
    "CC_OS": ["Coch", "Sparcl1", "Otor", "Col3a1", "Ibsp"],
    "Exclude": ["Oc90", "Otx2", "Neurod1", "Neurog1", "Ptprc", "Lyz2", "Hbb-bs", "Hba-a1", "Cnp", "Mbp", "Mpz", "Pmp22"],
}

p7_marker_dict = {
    k: [g for g in v if g in adata_P7.var_names]
    for k, v in p7_marker_dict.items()
}
p7_marker_dict = {k: v for k, v in p7_marker_dict.items() if len(v) > 0}

dp = sc.pl.dotplot(
    adata_P7,
    var_names=p7_marker_dict,
    groupby="leiden",
    standard_scale="var",
    figsize=(20, 6),
    return_fig=True,
)
dp.show()
plt.close("all")


plt.close("all")


# Notebook cell 23

# ======================== 6. remove definite Glia for downstream analysis ========================
logging.getLogger("harmonypy").setLevel(logging.ERROR)

glia_parent_clusters = ["7"]

adata_P7_no_glia = adata_P7[
    ~adata_P7.obs["leiden"].astype(str).isin(glia_parent_clusters)
].copy()

print(adata_P7_no_glia)
print(adata_P7_no_glia.obs["leiden"].value_counts().sort_index())


# ======================== 7. HC subset ========================
hc_parent_clusters = ["6", "9"]

adata_P7_HC = adata_P7_no_glia[
    adata_P7_no_glia.obs["leiden"].astype(str).isin(hc_parent_clusters)
].copy()

adata_P7_HC.obs["parent_leiden"] = adata_P7_HC.obs["leiden"].astype(str).astype("category")

print(adata_P7_HC)
print(pd.crosstab(adata_P7_HC.obs["parent_leiden"], adata_P7_HC.obs["batch"]))


# ======================== 8. HC marker scoring ========================
ihc_score_genes = [
    "Otof", "Atp2a3", "Cabp2", "Fgf8",
    "Dlk2", "Nefl", "Calb2",
]
ohc_score_genes = [
    "Ocm", "Aqp11", "Pcp4", "Cib2",
    "Calb1", "Six2", "Ighm", "Cd164l2",
]

ihc_score_genes = [g for g in ihc_score_genes if g in adata_P7_HC.var_names]
ohc_score_genes = [g for g in ohc_score_genes if g in adata_P7_HC.var_names]

sc.tl.score_genes(
    adata_P7_HC,
    gene_list=ihc_score_genes,
    score_name="IHC_score",
    use_raw=False,
)
sc.tl.score_genes(
    adata_P7_HC,
    gene_list=ohc_score_genes,
    score_name="OHC_score",
    use_raw=False,
)

adata_P7_HC.obs["HC_score_call"] = np.where(
    adata_P7_HC.obs["IHC_score"] > adata_P7_HC.obs["OHC_score"],
    "IHC_score_high",
    "OHC_score_high",
)
adata_P7_HC.obs["HC_score_call"] = adata_P7_HC.obs["HC_score_call"].astype("category")


# ======================== 9. HC Harmony reclustering ========================
sc.pp.highly_variable_genes(
    adata_P7_HC,
    flavor="seurat",
    n_top_genes=2000,
    batch_key="batch",
)

adata_P7_HC_work = adata_P7_HC[:, adata_P7_HC.var["highly_variable"]].copy()
sc.pp.scale(adata_P7_HC_work, max_value=10)
sc.tl.pca(adata_P7_HC_work, svd_solver="arpack")

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    sce.pp.harmony_integrate(
        adata_P7_HC_work,
        key="batch",
        basis="X_pca",
        adjusted_basis="X_pca_harmony",
        max_iter_harmony=20,
    )

sc.pp.neighbors(
    adata_P7_HC_work,
    n_neighbors=12,
    n_pcs=20,
    use_rep="X_pca_harmony",
)
sc.tl.umap(adata_P7_HC_work, random_state=0)
sc.tl.leiden(
    adata_P7_HC_work,
    resolution=0.35,
    key_added="HC_sub",
    random_state=0,
)

adata_P7_HC.obs["HC_sub"] = adata_P7_HC_work.obs["HC_sub"].copy()
adata_P7_HC.obsm["X_umap"] = adata_P7_HC_work.obsm["X_umap"].copy()

hc_sub_order = sorted(adata_P7_HC.obs["HC_sub"].unique(), key=lambda x: int(x))
adata_P7_HC.obs["HC_sub"] = pd.Categorical(
    adata_P7_HC.obs["HC_sub"].astype(str),
    categories=hc_sub_order,
    ordered=True,
)

print(pd.crosstab(adata_P7_HC.obs["parent_leiden"], adata_P7_HC.obs["HC_sub"]))
print(pd.crosstab(adata_P7_HC.obs["HC_sub"], adata_P7_HC.obs["HC_score_call"]))
print(
    adata_P7_HC.obs
    .groupby("HC_sub", observed=True)[["IHC_score", "OHC_score"]]
    .mean()
    .round(3)
)


# ======================== 10. HC UMAP check ========================
sc.pl.umap(
    adata_P7_HC,
    color=["parent_leiden", "HC_sub", "batch", "HC_score_call", "IHC_score", "OHC_score"],
    ncols=3,
    size=18,
    frameon=False,
    show=False,
)
plt.gcf().set_size_inches(9, 6)
plt.close("all")
plt.close("all")


# ======================== 11. HC marker dotplot ========================
hc_marker_dict = {
    "Core": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "HC": ["Myo6", "Myo7a", "Cib2", "Pvalb", "Ccer2", "Acbd7", "Rprm", "Pcp4", "Cd164l2"],
    "IHC": ["Otof", "Atp2a3", "Cabp2", "Fgf8", "Dlk2", "Nefl", "Calb2"],
    "OHC": ["Aqp11", "Ocm", "Calb1", "Six2", "Ighm"],
    "Exclude": ["Oc90", "Otx2", "Neurod1", "Neurog1", "Ptprc", "Lyz2", "Hbb-bs", "Hba-a1", "Cnp", "Mbp", "Mpz", "Pmp22"],
}

hc_marker_dict = {
    k: [g for g in v if g in adata_P7_HC.var_names]
    for k, v in hc_marker_dict.items()
}
hc_marker_dict = {k: v for k, v in hc_marker_dict.items() if len(v) > 0}

dp = sc.pl.dotplot(
    adata_P7_HC,
    var_names=hc_marker_dict,
    groupby="HC_sub",
    standard_scale="var",
    figsize=(12, 3),
    return_fig=True,
)
dp.show()
plt.close("all")


plt.close("all")


# Notebook cell 24

# ======================== 12. fix HC annotation ========================
adata_P7_HC.obs["HC_final"] = "OHC"
adata_P7_HC.obs.loc[
    adata_P7_HC.obs["HC_sub"].astype(str) == "1",
    "HC_final"
] = "IHC"

adata_P7_HC.obs["HC_final"] = pd.Categorical(
    adata_P7_HC.obs["HC_final"],
    categories=["IHC", "OHC"],
    ordered=True,
)

print(pd.crosstab(adata_P7_HC.obs["HC_sub"], adata_P7_HC.obs["HC_final"]))
print(adata_P7_HC.obs["HC_final"].value_counts().reindex(["IHC", "OHC"]))


plt.close("all")


# Notebook cell 25

# ======================== 13. non-HC epithelial subset ========================
hc_cells = adata_P7_HC.obs_names

adata_P7_nonHC = adata_P7_no_glia[
    ~adata_P7_no_glia.obs_names.isin(hc_cells)
].copy()

adata_P7_nonHC.obs["parent_leiden"] = (
    adata_P7_nonHC.obs["leiden"].astype(str).astype("category")
)

print(adata_P7_nonHC)
print(adata_P7_nonHC.obs["parent_leiden"].value_counts().sort_index())


# ======================== 14. non-HC marker module scoring ========================
score_gene_dict = {
    "IPhC_score": ["Igfbp4", "Maff", "Apoe", "Fst", "Cnmd"],
    "IPC_score": ["Ppp2r2b", "Npy", "Cep41", "Tuba1a", "Tuba1b"],
    "OPC_score": ["Sapcd2", "Map4", "S100b", "Smagp", "Cep41"],
    "DC_score": ["Car14", "Mansc4", "Rbp7", "Rflnb", "Selenom"],
    "HeC_score": ["Arpc1b", "Sostdc1", "Fabp7", "Ednrb", "Plp1"],
    "M_KO_score": ["Cldn4", "Upp1", "Ly6h", "Serpina3a", "Ier5"],
    "ML_KO_score": ["Clu", "Epyc", "Grb14", "Fam159b", "Qpct"],
    "L_KO_score": ["Anxa5", "Foxq1", "Igfbp3", "Gsn", "Hspa2"],
    "CC_OS_score": ["Coch", "Sparcl1", "Otor", "Col3a1", "Ibsp"],
    "Glia_score": ["Cnp", "Mbp", "Mpz", "Pmp22"],
}

score_cols = []
for score_name, genes in score_gene_dict.items():
    genes = [g for g in genes if g in adata_P7_nonHC.var_names]
    if len(genes) == 0:
        continue

    sc.tl.score_genes(
        adata_P7_nonHC,
        gene_list=genes,
        score_name=score_name,
        use_raw=False,
    )
    score_cols.append(score_name)

adata_P7_nonHC.obs["module_call"] = (
    adata_P7_nonHC.obs[score_cols]
    .idxmax(axis=1)
    .str.replace("_score", "", regex=False)
    .astype("category")
)

print(adata_P7_nonHC.obs["module_call"].value_counts())


# ======================== 15. non-HC Harmony reclustering ========================
sc.pp.highly_variable_genes(
    adata_P7_nonHC,
    flavor="seurat",
    n_top_genes=3000,
    batch_key="batch",
)

adata_P7_nonHC_work = adata_P7_nonHC[
    :, adata_P7_nonHC.var["highly_variable"]
].copy()

sc.pp.scale(adata_P7_nonHC_work, max_value=10)
sc.tl.pca(adata_P7_nonHC_work, svd_solver="arpack")

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    sce.pp.harmony_integrate(
        adata_P7_nonHC_work,
        key="batch",
        basis="X_pca",
        adjusted_basis="X_pca_harmony",
        max_iter_harmony=20,
    )

sc.pp.neighbors(
    adata_P7_nonHC_work,
    n_neighbors=15,
    n_pcs=25,
    use_rep="X_pca_harmony",
)
sc.tl.umap(adata_P7_nonHC_work, random_state=0)
sc.tl.leiden(
    adata_P7_nonHC_work,
    resolution=0.65,
    key_added="nonHC_sub",
    random_state=0,
)

adata_P7_nonHC.obs["nonHC_sub"] = adata_P7_nonHC_work.obs["nonHC_sub"].copy()
adata_P7_nonHC.obsm["X_umap"] = adata_P7_nonHC_work.obsm["X_umap"].copy()

nonhc_order = sorted(adata_P7_nonHC.obs["nonHC_sub"].unique(), key=lambda x: int(x))
adata_P7_nonHC.obs["nonHC_sub"] = pd.Categorical(
    adata_P7_nonHC.obs["nonHC_sub"].astype(str),
    categories=nonhc_order,
    ordered=True,
)

print(pd.crosstab(adata_P7_nonHC.obs["parent_leiden"], adata_P7_nonHC.obs["nonHC_sub"]))
print(pd.crosstab(adata_P7_nonHC.obs["nonHC_sub"], adata_P7_nonHC.obs["module_call"]))
print(
    adata_P7_nonHC.obs
    .groupby("nonHC_sub", observed=True)[score_cols]
    .mean()
    .round(3)
)


# ======================== 16. non-HC UMAP check ========================
sc.pl.umap(
    adata_P7_nonHC,
    color=[
        "parent_leiden",
        "nonHC_sub",
        "batch",
        "module_call",
        "IPhC_score",
        "IPC_score",
        "DC_score",
        "Glia_score",
    ],
    ncols=4,
    size=8,
    frameon=False,
    show=False,
)
plt.gcf().set_size_inches(12, 7)
plt.close("all")
plt.close("all")


# ======================== 17. non-HC marker dotplot ========================
nonhc_marker_dict = {
    "Core": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "IPhC": ["Igfbp4", "Maff", "Apoe", "Fst", "Cnmd"],
    "IPC": ["Ppp2r2b", "Npy", "Cep41", "Tuba1a", "Tuba1b"],
    "OPC": ["Sapcd2", "Map4", "S100b", "Smagp", "Cep41"],
    "DC": ["Car14", "Mansc4", "Rbp7", "Rflnb", "Selenom"],
    "HeC": ["Arpc1b", "Sostdc1", "Fabp7", "Ednrb", "Plp1"],
    "M_KO": ["Cldn4", "Upp1", "Ly6h", "Serpina3a", "Ier5"],
    "ML_KO": ["Clu", "Epyc", "Grb14", "Fam159b", "Qpct"],
    "L_KO": ["Anxa5", "Foxq1", "Igfbp3", "Gsn", "Hspa2"],
    "CC_OS": ["Coch", "Sparcl1", "Otor", "Col3a1", "Ibsp"],
    "Exclude": ["Oc90", "Otx2", "Neurod1", "Neurog1", "Ptprc", "Lyz2", "Hbb-bs", "Hba-a1", "Cnp", "Mbp", "Mpz", "Pmp22"],
}

nonhc_marker_dict = {
    k: [g for g in v if g in adata_P7_nonHC.var_names]
    for k, v in nonhc_marker_dict.items()
}
nonhc_marker_dict = {k: v for k, v in nonhc_marker_dict.items() if len(v) > 0}

dp = sc.pl.dotplot(
    adata_P7_nonHC,
    var_names=nonhc_marker_dict,
    groupby="nonHC_sub",
    standard_scale="var",
    figsize=(20, 6),
    return_fig=True,
)
dp.show()
plt.close("all")


plt.close("all")


# Notebook cell 26

# ======================== 18. focused reclustering: IPC / OPC / DC branch ========================
ipc_opc_dc_parent = ["7", "9", "10", "11"]

adata_P7_IPC_OPC_DC = adata_P7_nonHC[
    adata_P7_nonHC.obs["nonHC_sub"].astype(str).isin(ipc_opc_dc_parent)
].copy()

print(adata_P7_IPC_OPC_DC)
print(adata_P7_IPC_OPC_DC.obs["nonHC_sub"].value_counts().sort_index())
print(pd.crosstab(adata_P7_IPC_OPC_DC.obs["nonHC_sub"], adata_P7_IPC_OPC_DC.obs["batch"]))


# ======================== 19. IPC / OPC / DC marker scoring ========================
axis_score_dict = {
    "IPC_axis_score": ["Ppp2r2b", "Npy", "Cep41", "Tuba1a", "Tuba1b"],
    "OPC_axis_score": ["Sapcd2", "Map4", "S100b", "Smagp", "Cep41"],
    "DC_axis_score": ["Car14", "Mansc4", "Rbp7", "Rflnb", "Selenom"],
    "IPhC_check_score": ["Igfbp4", "Maff", "Apoe", "Fst", "Cnmd"],
    "KO_check_score": ["Clu", "Epyc", "Grb14", "Fam159b", "Qpct", "Anxa5", "Foxq1", "Igfbp3"],
    "Glia_check_score": ["Cnp", "Mbp", "Mpz", "Pmp22"],
}

axis_score_cols = []
for score_name, genes in axis_score_dict.items():
    genes = [g for g in genes if g in adata_P7_IPC_OPC_DC.var_names]
    if len(genes) == 0:
        continue

    sc.tl.score_genes(
        adata_P7_IPC_OPC_DC,
        gene_list=genes,
        score_name=score_name,
        use_raw=False,
    )
    axis_score_cols.append(score_name)

print(
    adata_P7_IPC_OPC_DC.obs
    .groupby("nonHC_sub", observed=True)[axis_score_cols]
    .mean()
    .round(3)
)


# ======================== 20. IPC / OPC / DC Harmony reclustering ========================
sc.pp.highly_variable_genes(
    adata_P7_IPC_OPC_DC,
    flavor="seurat",
    n_top_genes=1200,
    batch_key="batch",
)

adata_P7_IPC_OPC_DC_work = adata_P7_IPC_OPC_DC[
    :, adata_P7_IPC_OPC_DC.var["highly_variable"]
].copy()

sc.pp.scale(adata_P7_IPC_OPC_DC_work, max_value=10)
sc.tl.pca(adata_P7_IPC_OPC_DC_work, svd_solver="arpack")

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    sce.pp.harmony_integrate(
        adata_P7_IPC_OPC_DC_work,
        key="batch",
        basis="X_pca",
        adjusted_basis="X_pca_harmony",
        max_iter_harmony=20,
    )

sc.pp.neighbors(
    adata_P7_IPC_OPC_DC_work,
    n_neighbors=8,
    n_pcs=12,
    use_rep="X_pca_harmony",
)
sc.tl.umap(adata_P7_IPC_OPC_DC_work, random_state=0)
sc.tl.leiden(
    adata_P7_IPC_OPC_DC_work,
    resolution=0.45,
    key_added="IPC_OPC_DC_sub",
    random_state=0,
)

adata_P7_IPC_OPC_DC.obs["IPC_OPC_DC_sub"] = (
    adata_P7_IPC_OPC_DC_work.obs["IPC_OPC_DC_sub"].copy()
)
adata_P7_IPC_OPC_DC.obsm["X_umap"] = adata_P7_IPC_OPC_DC_work.obsm["X_umap"].copy()

axis_order = sorted(
    adata_P7_IPC_OPC_DC.obs["IPC_OPC_DC_sub"].unique(),
    key=lambda x: int(x),
)
adata_P7_IPC_OPC_DC.obs["IPC_OPC_DC_sub"] = pd.Categorical(
    adata_P7_IPC_OPC_DC.obs["IPC_OPC_DC_sub"].astype(str),
    categories=axis_order,
    ordered=True,
)

print(pd.crosstab(adata_P7_IPC_OPC_DC.obs["nonHC_sub"], adata_P7_IPC_OPC_DC.obs["IPC_OPC_DC_sub"]))
print(
    adata_P7_IPC_OPC_DC.obs
    .groupby("IPC_OPC_DC_sub", observed=True)[axis_score_cols]
    .mean()
    .round(3)
)


# ======================== 21. IPC / OPC / DC UMAP check ========================
sc.pl.umap(
    adata_P7_IPC_OPC_DC,
    color=[
        "nonHC_sub",
        "IPC_OPC_DC_sub",
        "batch",
        "IPC_axis_score",
        "OPC_axis_score",
        "DC_axis_score",
    ],
    ncols=3,
    size=24,
    frameon=False,
    show=False,
)
plt.gcf().set_size_inches(9, 6)
plt.close("all")
plt.close("all")


# ======================== 22. IPC / OPC / DC marker dotplot ========================
axis_marker_dict = {
    "Core": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "IPC": ["Ppp2r2b", "Npy", "Cep41", "Tuba1a", "Tuba1b"],
    "OPC": ["Sapcd2", "Map4", "S100b", "Smagp", "Cep41"],
    "DC": ["Car14", "Mansc4", "Rbp7", "Rflnb", "Selenom"],
    "IPhC_check": ["Igfbp4", "Maff", "Apoe", "Fst", "Cnmd"],
    "KO_check": ["Clu", "Epyc", "Grb14", "Fam159b", "Qpct", "Anxa5", "Foxq1", "Igfbp3"],
    "Exclude": ["Oc90", "Otx2", "Ptprc", "Lyz2", "Hbb-bs", "Hba-a1", "Cnp", "Mbp", "Mpz", "Pmp22"],
}

axis_marker_dict = {
    k: [g for g in v if g in adata_P7_IPC_OPC_DC.var_names]
    for k, v in axis_marker_dict.items()
}
axis_marker_dict = {k: v for k, v in axis_marker_dict.items() if len(v) > 0}

dp = sc.pl.dotplot(
    adata_P7_IPC_OPC_DC,
    var_names=axis_marker_dict,
    groupby="IPC_OPC_DC_sub",
    standard_scale="var",
    figsize=(14, 4),
    return_fig=True,
)
dp.show()
plt.close("all")


plt.close("all")


# Notebook cell 27

# ======================== 23. focused check: IPC / OPC split ========================
ipc_opc_parent = ["1", "2"]

adata_P7_IPC_OPC = adata_P7_IPC_OPC_DC[
    adata_P7_IPC_OPC_DC.obs["IPC_OPC_DC_sub"].astype(str).isin(ipc_opc_parent)
].copy()

print(adata_P7_IPC_OPC)
print(adata_P7_IPC_OPC.obs["IPC_OPC_DC_sub"].value_counts().sort_index())
print(pd.crosstab(adata_P7_IPC_OPC.obs["IPC_OPC_DC_sub"], adata_P7_IPC_OPC.obs["batch"]))


# ======================== 24. strict IPC / OPC marker scores ========================
ipc_main = ["Ppp2r2b", "Npy"]
ipc_aux = ["Cep41", "Tuba1a", "Tuba1b"]

opc_main = ["Sapcd2", "Map4", "Smagp"]
opc_aux = ["S100b", "Cep41"]

dc_check = ["Car14", "Mansc4", "Rbp7", "Rflnb", "Selenom"]
ko_check = ["Clu", "Epyc", "Grb14", "Fam159b", "Qpct", "Anxa5", "Foxq1", "Igfbp3"]
glia_check = ["Cnp", "Mbp", "Mpz", "Pmp22"]

def mean_marker_expr(adata_obj, genes):
    genes = [g for g in genes if g in adata_obj.var_names]
    if len(genes) == 0:
        return np.zeros(adata_obj.n_obs)
    return np.asarray(adata_obj[:, genes].X.mean(axis=1)).ravel()

adata_P7_IPC_OPC.obs["IPC_main_score"] = mean_marker_expr(adata_P7_IPC_OPC, ipc_main)
adata_P7_IPC_OPC.obs["IPC_aux_score"] = mean_marker_expr(adata_P7_IPC_OPC, ipc_aux)
adata_P7_IPC_OPC.obs["OPC_main_score"] = mean_marker_expr(adata_P7_IPC_OPC, opc_main)
adata_P7_IPC_OPC.obs["OPC_aux_score"] = mean_marker_expr(adata_P7_IPC_OPC, opc_aux)
adata_P7_IPC_OPC.obs["DC_check_score"] = mean_marker_expr(adata_P7_IPC_OPC, dc_check)
adata_P7_IPC_OPC.obs["KO_check_score"] = mean_marker_expr(adata_P7_IPC_OPC, ko_check)
adata_P7_IPC_OPC.obs["Glia_check_score"] = mean_marker_expr(adata_P7_IPC_OPC, glia_check)

ipc_opc_score_cols = [
    "IPC_main_score",
    "IPC_aux_score",
    "OPC_main_score",
    "OPC_aux_score",
    "DC_check_score",
    "KO_check_score",
    "Glia_check_score",
]

print(
    adata_P7_IPC_OPC.obs
    .groupby("IPC_OPC_DC_sub", observed=True)[ipc_opc_score_cols]
    .mean()
    .round(3)
)


# ======================== 25. IPC / OPC Harmony reclustering ========================
sc.pp.highly_variable_genes(
    adata_P7_IPC_OPC,
    flavor="seurat",
    n_top_genes=800,
    batch_key="batch",
)

adata_P7_IPC_OPC_work = adata_P7_IPC_OPC[
    :, adata_P7_IPC_OPC.var["highly_variable"]
].copy()

sc.pp.scale(adata_P7_IPC_OPC_work, max_value=10)
sc.tl.pca(adata_P7_IPC_OPC_work, svd_solver="arpack")

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    sce.pp.harmony_integrate(
        adata_P7_IPC_OPC_work,
        key="batch",
        basis="X_pca",
        adjusted_basis="X_pca_harmony",
        max_iter_harmony=20,
    )

sc.pp.neighbors(
    adata_P7_IPC_OPC_work,
    n_neighbors=6,
    n_pcs=8,
    use_rep="X_pca_harmony",
)
sc.tl.umap(adata_P7_IPC_OPC_work, random_state=0)
sc.tl.leiden(
    adata_P7_IPC_OPC_work,
    resolution=0.25,
    key_added="IPC_OPC_sub",
    random_state=0,
)

adata_P7_IPC_OPC.obs["IPC_OPC_sub"] = adata_P7_IPC_OPC_work.obs["IPC_OPC_sub"].copy()
adata_P7_IPC_OPC.obsm["X_umap"] = adata_P7_IPC_OPC_work.obsm["X_umap"].copy()

ipc_opc_order = sorted(
    adata_P7_IPC_OPC.obs["IPC_OPC_sub"].unique(),
    key=lambda x: int(x),
)
adata_P7_IPC_OPC.obs["IPC_OPC_sub"] = pd.Categorical(
    adata_P7_IPC_OPC.obs["IPC_OPC_sub"].astype(str),
    categories=ipc_opc_order,
    ordered=True,
)

print(pd.crosstab(adata_P7_IPC_OPC.obs["IPC_OPC_DC_sub"], adata_P7_IPC_OPC.obs["IPC_OPC_sub"]))
print(
    adata_P7_IPC_OPC.obs
    .groupby("IPC_OPC_sub", observed=True)[ipc_opc_score_cols]
    .mean()
    .round(3)
)


# ======================== 26. IPC / OPC UMAP check ========================
sc.pl.umap(
    adata_P7_IPC_OPC,
    color=[
        "IPC_OPC_DC_sub",
        "IPC_OPC_sub",
        "batch",
        "IPC_main_score",
        "OPC_main_score",
        "DC_check_score",
    ],
    ncols=3,
    size=34,
    frameon=False,
    show=False,
)
plt.gcf().set_size_inches(9, 6)
plt.close("all")
plt.close("all")


# ======================== 27. IPC / OPC marker dotplot ========================
ipc_opc_marker_dict = {
    "Core": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "IPC_main": ["Ppp2r2b", "Npy"],
    "IPC_aux": ["Cep41", "Tuba1a", "Tuba1b"],
    "OPC_main": ["Sapcd2", "Map4", "Smagp"],
    "OPC_aux": ["S100b", "Cep41"],
    "DC_check": ["Car14", "Mansc4", "Rbp7", "Rflnb", "Selenom"],
    "KO_check": ["Clu", "Epyc", "Grb14", "Fam159b", "Qpct", "Anxa5", "Foxq1", "Igfbp3"],
    "Exclude": ["Oc90", "Otx2", "Ptprc", "Lyz2", "Hbb-bs", "Hba-a1", "Cnp", "Mbp", "Mpz", "Pmp22"],
}

ipc_opc_marker_dict = {
    k: [g for g in v if g in adata_P7_IPC_OPC.var_names]
    for k, v in ipc_opc_marker_dict.items()
}
ipc_opc_marker_dict = {k: v for k, v in ipc_opc_marker_dict.items() if len(v) > 0}

dp = sc.pl.dotplot(
    adata_P7_IPC_OPC,
    var_names=ipc_opc_marker_dict,
    groupby="IPC_OPC_sub",
    standard_scale="var",
    figsize=(13, 3.5),
    return_fig=True,
)
dp.show()
plt.close("all")


plt.close("all")


# Notebook cell 28

# ======================== 28. deep check: IPC / OPC mixed core only ========================
adata_P7_IPC_OPC_core = adata_P7_IPC_OPC[
    adata_P7_IPC_OPC.obs["IPC_OPC_sub"].astype(str) == "0"
].copy()

print(adata_P7_IPC_OPC_core)
print(pd.crosstab(adata_P7_IPC_OPC_core.obs["IPC_OPC_DC_sub"], adata_P7_IPC_OPC_core.obs["batch"]))


# ======================== 29. strict IPC / OPC / DC scores for mixed core ========================
def mean_marker_expr(adata_obj, genes):
    genes = [g for g in genes if g in adata_obj.var_names]
    if len(genes) == 0:
        return np.zeros(adata_obj.n_obs)
    return np.asarray(adata_obj[:, genes].X.mean(axis=1)).ravel()

ipc_main = ["Ppp2r2b", "Npy"]
ipc_aux = ["Cep41", "Tuba1a", "Tuba1b"]

opc_main = ["Sapcd2", "Map4", "Smagp"]
opc_aux = ["S100b", "Cep41"]

dc_check = ["Car14", "Mansc4", "Rbp7", "Rflnb", "Selenom"]
ko_check = ["Clu", "Epyc", "Grb14", "Fam159b", "Qpct", "Anxa5", "Foxq1", "Igfbp3"]
glia_check = ["Cnp", "Mbp", "Mpz", "Pmp22"]

adata_P7_IPC_OPC_core.obs["IPC_main_score"] = mean_marker_expr(adata_P7_IPC_OPC_core, ipc_main)
adata_P7_IPC_OPC_core.obs["IPC_aux_score"] = mean_marker_expr(adata_P7_IPC_OPC_core, ipc_aux)
adata_P7_IPC_OPC_core.obs["OPC_main_score"] = mean_marker_expr(adata_P7_IPC_OPC_core, opc_main)
adata_P7_IPC_OPC_core.obs["OPC_aux_score"] = mean_marker_expr(adata_P7_IPC_OPC_core, opc_aux)
adata_P7_IPC_OPC_core.obs["DC_check_score"] = mean_marker_expr(adata_P7_IPC_OPC_core, dc_check)
adata_P7_IPC_OPC_core.obs["KO_check_score"] = mean_marker_expr(adata_P7_IPC_OPC_core, ko_check)
adata_P7_IPC_OPC_core.obs["Glia_check_score"] = mean_marker_expr(adata_P7_IPC_OPC_core, glia_check)
adata_P7_IPC_OPC_core.obs["IPC_minus_OPC"] = (
    adata_P7_IPC_OPC_core.obs["IPC_main_score"]
    - adata_P7_IPC_OPC_core.obs["OPC_main_score"]
)

core_score_cols = [
    "IPC_main_score",
    "IPC_aux_score",
    "OPC_main_score",
    "OPC_aux_score",
    "DC_check_score",
    "KO_check_score",
    "Glia_check_score",
    "IPC_minus_OPC",
]

print(
    adata_P7_IPC_OPC_core.obs[core_score_cols]
    .describe()
    .round(3)
)


# ======================== 30. IPC / OPC mixed core Harmony reclustering ========================
sc.pp.highly_variable_genes(
    adata_P7_IPC_OPC_core,
    flavor="seurat",
    n_top_genes=600,
    batch_key="batch",
)

adata_P7_IPC_OPC_core_work = adata_P7_IPC_OPC_core[
    :, adata_P7_IPC_OPC_core.var["highly_variable"]
].copy()

sc.pp.scale(adata_P7_IPC_OPC_core_work, max_value=10)
sc.tl.pca(adata_P7_IPC_OPC_core_work, svd_solver="arpack")

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    sce.pp.harmony_integrate(
        adata_P7_IPC_OPC_core_work,
        key="batch",
        basis="X_pca",
        adjusted_basis="X_pca_harmony",
        max_iter_harmony=20,
    )

sc.pp.neighbors(
    adata_P7_IPC_OPC_core_work,
    n_neighbors=4,
    n_pcs=6,
    use_rep="X_pca_harmony",
)
sc.tl.umap(adata_P7_IPC_OPC_core_work, random_state=0)
sc.tl.leiden(
    adata_P7_IPC_OPC_core_work,
    resolution=0.5,
    key_added="IPC_OPC_core_sub",
    random_state=0,
)

adata_P7_IPC_OPC_core.obs["IPC_OPC_core_sub"] = (
    adata_P7_IPC_OPC_core_work.obs["IPC_OPC_core_sub"].copy()
)
adata_P7_IPC_OPC_core.obsm["X_umap"] = adata_P7_IPC_OPC_core_work.obsm["X_umap"].copy()

core_sub_order = sorted(
    adata_P7_IPC_OPC_core.obs["IPC_OPC_core_sub"].unique(),
    key=lambda x: int(x),
)
adata_P7_IPC_OPC_core.obs["IPC_OPC_core_sub"] = pd.Categorical(
    adata_P7_IPC_OPC_core.obs["IPC_OPC_core_sub"].astype(str),
    categories=core_sub_order,
    ordered=True,
)

print(pd.crosstab(adata_P7_IPC_OPC_core.obs["IPC_OPC_core_sub"], adata_P7_IPC_OPC_core.obs["batch"]))
print(
    adata_P7_IPC_OPC_core.obs
    .groupby("IPC_OPC_core_sub", observed=True)[core_score_cols]
    .mean()
    .round(3)
)


# ======================== 31. IPC / OPC mixed core UMAP check ========================
sc.pl.umap(
    adata_P7_IPC_OPC_core,
    color=[
        "IPC_OPC_core_sub",
        "batch",
        "IPC_main_score",
        "OPC_main_score",
        "DC_check_score",
        "IPC_minus_OPC",
    ],
    ncols=3,
    size=45,
    frameon=False,
    show=False,
)
plt.gcf().set_size_inches(9, 6)
plt.close("all")
plt.close("all")


# ======================== 32. IPC / OPC mixed core dotplot ========================
ipc_opc_core_marker_dict = {
    "Core": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "IPC_main": ["Ppp2r2b", "Npy"],
    "IPC_aux": ["Cep41", "Tuba1a", "Tuba1b"],
    "OPC_main": ["Sapcd2", "Map4", "Smagp"],
    "OPC_aux": ["S100b", "Cep41"],
    "DC_check": ["Car14", "Mansc4", "Rbp7", "Rflnb", "Selenom"],
    "KO_check": ["Clu", "Epyc", "Grb14", "Fam159b", "Qpct", "Anxa5", "Foxq1", "Igfbp3"],
    "Exclude": ["Oc90", "Otx2", "Ptprc", "Lyz2", "Hbb-bs", "Hba-a1", "Cnp", "Mbp", "Mpz", "Pmp22"],
}

ipc_opc_core_marker_dict = {
    k: [g for g in v if g in adata_P7_IPC_OPC_core.var_names]
    for k, v in ipc_opc_core_marker_dict.items()
}
ipc_opc_core_marker_dict = {
    k: v for k, v in ipc_opc_core_marker_dict.items()
    if len(v) > 0
}

dp = sc.pl.dotplot(
    adata_P7_IPC_OPC_core,
    var_names=ipc_opc_core_marker_dict,
    groupby="IPC_OPC_core_sub",
    standard_scale="var",
    figsize=(13, 3.5),
    return_fig=True,
)
dp.show()
plt.close("all")


plt.close("all")


# Notebook cell 29

# ======================== 33. write fixed labels before residual reclustering ========================
adata_P7_no_glia.obs["P7_step_label"] = "Unassigned"

# HC fixed labels
hc_final = adata_P7_HC.obs["HC_final"].astype(str)
adata_P7_no_glia.obs.loc[hc_final.index, "P7_step_label"] = hc_final

# CC/OS fixed from non-HC broad clustering
ccos_cells = adata_P7_nonHC.obs_names[
    adata_P7_nonHC.obs["nonHC_sub"].astype(str) == "4"
]
adata_P7_no_glia.obs.loc[ccos_cells, "P7_step_label"] = "CC/OS"

# DC fixed from IPC/OPC/DC branch
dc_cells = adata_P7_IPC_OPC_DC.obs_names[
    adata_P7_IPC_OPC_DC.obs["IPC_OPC_DC_sub"].astype(str).isin(["0", "3"])
]
adata_P7_no_glia.obs.loc[dc_cells, "P7_step_label"] = "DC"

# IPC / OPC fixed from focused core
ipc_cells = adata_P7_IPC_OPC_core.obs_names[
    adata_P7_IPC_OPC_core.obs["IPC_OPC_core_sub"].astype(str).isin(["0", "3"])
]
opc_cells = adata_P7_IPC_OPC_core.obs_names[
    adata_P7_IPC_OPC_core.obs["IPC_OPC_core_sub"].astype(str).isin(["1", "2"])
]

adata_P7_no_glia.obs.loc[ipc_cells, "P7_step_label"] = "IPC"
adata_P7_no_glia.obs.loc[opc_cells, "P7_step_label"] = "OPC"

fixed_order = ["IHC", "OHC", "CC/OS", "DC", "IPC", "OPC", "Unassigned"]
adata_P7_no_glia.obs["P7_step_label"] = pd.Categorical(
    adata_P7_no_glia.obs["P7_step_label"].astype(str),
    categories=fixed_order,
    ordered=True,
)

print(adata_P7_no_glia.obs["P7_step_label"].value_counts().reindex(fixed_order))


# ======================== 34. residual subset for KO / IPhC / HeC check ========================
adata_P7_residual = adata_P7_no_glia[
    adata_P7_no_glia.obs["P7_step_label"].astype(str) == "Unassigned"
].copy()

adata_P7_residual.obs["parent_leiden"] = (
    adata_P7_residual.obs["leiden"].astype(str).astype("category")
)

print(adata_P7_residual)
print(adata_P7_residual.obs["parent_leiden"].value_counts().sort_index())


# ======================== 35. residual marker scoring ========================
residual_score_dict = {
    "IPhC_score": ["Igfbp4", "Maff", "Apoe", "Fst", "Cnmd"],
    "HeC_score": ["Arpc1b", "Sostdc1", "Fabp7", "Ednrb", "Plp1"],
    "M_KO_score": ["Cldn4", "Upp1", "Ly6h", "Serpina3a", "Ier5"],
    "ML_KO_score": ["Clu", "Epyc", "Grb14", "Fam159b", "Qpct"],
    "L_KO_score": ["Anxa5", "Foxq1", "Igfbp3", "Gsn", "Hspa2"],
    "CC_OS_check_score": ["Coch", "Sparcl1", "Otor", "Col3a1", "Ibsp"],
    "IPC_OPC_DC_check_score": ["Ppp2r2b", "Npy", "Sapcd2", "Map4", "Car14", "Mansc4", "Rbp7"],
    "Glia_check_score": ["Cnp", "Mbp", "Mpz", "Pmp22"],
}

residual_score_cols = []
for score_name, genes in residual_score_dict.items():
    genes = [g for g in genes if g in adata_P7_residual.var_names]
    if len(genes) == 0:
        continue

    sc.tl.score_genes(
        adata_P7_residual,
        gene_list=genes,
        score_name=score_name,
        use_raw=False,
    )
    residual_score_cols.append(score_name)

adata_P7_residual.obs["residual_module_call"] = (
    adata_P7_residual.obs[residual_score_cols]
    .idxmax(axis=1)
    .str.replace("_score", "", regex=False)
    .astype("category")
)

print(adata_P7_residual.obs["residual_module_call"].value_counts())


# ======================== 36. residual Harmony reclustering ========================
sc.pp.highly_variable_genes(
    adata_P7_residual,
    flavor="seurat",
    n_top_genes=2500,
    batch_key="batch",
)

adata_P7_residual_work = adata_P7_residual[
    :, adata_P7_residual.var["highly_variable"]
].copy()

sc.pp.scale(adata_P7_residual_work, max_value=10)
sc.tl.pca(adata_P7_residual_work, svd_solver="arpack")

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    sce.pp.harmony_integrate(
        adata_P7_residual_work,
        key="batch",
        basis="X_pca",
        adjusted_basis="X_pca_harmony",
        max_iter_harmony=20,
    )

sc.pp.neighbors(
    adata_P7_residual_work,
    n_neighbors=15,
    n_pcs=20,
    use_rep="X_pca_harmony",
)
sc.tl.umap(adata_P7_residual_work, random_state=0)
sc.tl.leiden(
    adata_P7_residual_work,
    resolution=0.6,
    key_added="residual_sub",
    random_state=0,
)

adata_P7_residual.obs["residual_sub"] = adata_P7_residual_work.obs["residual_sub"].copy()
adata_P7_residual.obsm["X_umap"] = adata_P7_residual_work.obsm["X_umap"].copy()

residual_order = sorted(
    adata_P7_residual.obs["residual_sub"].unique(),
    key=lambda x: int(x),
)
adata_P7_residual.obs["residual_sub"] = pd.Categorical(
    adata_P7_residual.obs["residual_sub"].astype(str),
    categories=residual_order,
    ordered=True,
)

print(pd.crosstab(adata_P7_residual.obs["parent_leiden"], adata_P7_residual.obs["residual_sub"]))
print(pd.crosstab(adata_P7_residual.obs["residual_sub"], adata_P7_residual.obs["residual_module_call"]))
print(
    adata_P7_residual.obs
    .groupby("residual_sub", observed=True)[residual_score_cols]
    .mean()
    .round(3)
)


# ======================== 37. residual UMAP check ========================
sc.pl.umap(
    adata_P7_residual,
    color=[
        "parent_leiden",
        "residual_sub",
        "batch",
        "residual_module_call",
        "IPhC_score",
        "HeC_score",
        "Glia_check_score",
        "IPC_OPC_DC_check_score",
    ],
    ncols=4,
    size=8,
    frameon=False,
    show=False,
)
plt.gcf().set_size_inches(12, 7)
plt.close("all")
plt.close("all")


# ======================== 38. residual marker dotplot ========================
residual_marker_dict = {
    "Core": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "IPhC": ["Igfbp4", "Maff", "Apoe", "Fst", "Cnmd"],
    "HeC": ["Arpc1b", "Sostdc1", "Fabp7", "Ednrb", "Plp1"],
    "M_KO": ["Cldn4", "Upp1", "Ly6h", "Serpina3a", "Ier5"],
    "ML_KO": ["Clu", "Epyc", "Grb14", "Fam159b", "Qpct"],
    "L_KO": ["Anxa5", "Foxq1", "Igfbp3", "Gsn", "Hspa2"],
    "CC_OS_check": ["Coch", "Sparcl1", "Otor", "Col3a1", "Ibsp"],
    "IPC_OPC_DC_check": ["Ppp2r2b", "Npy", "Sapcd2", "Map4", "Car14", "Mansc4", "Rbp7"],
    "Exclude": ["Oc90", "Otx2", "Neurod1", "Neurog1", "Ptprc", "Lyz2", "Hbb-bs", "Hba-a1", "Cnp", "Mbp", "Mpz", "Pmp22"],
}

residual_marker_dict = {
    k: [g for g in v if g in adata_P7_residual.var_names]
    for k, v in residual_marker_dict.items()
}
residual_marker_dict = {k: v for k, v in residual_marker_dict.items() if len(v) > 0}

dp = sc.pl.dotplot(
    adata_P7_residual,
    var_names=residual_marker_dict,
    groupby="residual_sub",
    standard_scale="var",
    figsize=(18, 5.5),
    return_fig=True,
)
dp.show()
plt.close("all")


plt.close("all")


# Notebook cell 30

# ======================== 39. fix clear IPhC and prepare KO candidates ========================

adata_P7_no_glia.obs["P7_step_label"] = adata_P7_no_glia.obs["P7_step_label"].astype(str)

clear_iphc_cells = adata_P7_residual.obs_names[
    adata_P7_residual.obs["residual_sub"].astype(str).isin(["8"])
]
adata_P7_no_glia.obs.loc[clear_iphc_cells, "P7_step_label"] = "IPhC"

ko_candidate_sub = ["0", "1", "2", "3", "4", "5", "6"]
adata_P7_KO = adata_P7_residual[
    adata_P7_residual.obs["residual_sub"].astype(str).isin(ko_candidate_sub)
].copy()

adata_P7_KO.layers["plot_expr"] = adata_P7_KO.X.copy()

print(adata_P7_no_glia.obs["P7_step_label"].value_counts())
print(adata_P7_KO)
print(pd.crosstab(adata_P7_KO.obs["residual_sub"], adata_P7_KO.obs["batch"]))


# ======================== 40. KO candidate marker scoring ========================
ko_score_markers = {
    "M_KO": ["Cldn4", "Upp1", "Ly6h"],
    "ML_KO": ["Clu", "Epyc", "Grb14"],
    "L_KO": ["Anxa5", "Foxq1", "Igfbp3"],
    "IPhC_check": ["Igfbp4", "Maff", "Apoe"],
    "HeC_check": ["Arpc1b", "Sostdc1", "Fabp7"],
    "Glia_check": ["Cnp", "Mbp", "Mpz", "Pmp22"],
    "CC_OS_check": ["Coch", "Sparcl1", "Otor"],
    "IPC_OPC_DC_check": ["Npy", "Sapcd2", "Car14", "Mansc4", "Rbp7"],
}

for score_name, genes in ko_score_markers.items():
    genes_use = [gene for gene in genes if gene in adata_P7_KO.var_names]
    if len(genes_use) > 0:
        sc.tl.score_genes(
            adata_P7_KO,
            gene_list=genes_use,
            score_name=f"{score_name}_score",
            ctrl_size=50,
            random_state=0,
        )


# ======================== 41. KO candidate Harmony reclustering ========================
sc.pp.highly_variable_genes(
    adata_P7_KO,
    n_top_genes=3000,
    batch_key="batch",
    subset=False,
)

sc.pp.scale(adata_P7_KO, max_value=10)
sc.tl.pca(
    adata_P7_KO,
    n_comps=30,
    use_highly_variable=True,
    svd_solver="arpack",
    random_state=0,
)

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    sce.pp.harmony_integrate(
        adata_P7_KO,
        key="batch",
        basis="X_pca",
        adjusted_basis="X_pca_harmony",
        max_iter_harmony=20,
    )

adata_P7_KO.X = adata_P7_KO.layers["plot_expr"].copy()

sc.pp.neighbors(
    adata_P7_KO,
    n_neighbors=35,
    n_pcs=25,
    use_rep="X_pca_harmony",
    random_state=0,
)
sc.tl.umap(adata_P7_KO, min_dist=0.35, spread=1.0, random_state=0)
sc.tl.leiden(
    adata_P7_KO,
    resolution=0.55,
    key_added="KO_sub",
    random_state=0,
)

score_cols = [col for col in adata_P7_KO.obs.columns if col.endswith("_score")]

print(adata_P7_KO.obs["KO_sub"].value_counts().sort_index())
print(pd.crosstab(adata_P7_KO.obs["KO_sub"], adata_P7_KO.obs["batch"]))
print(adata_P7_KO.obs.groupby("KO_sub")[score_cols].mean().round(3))


# ======================== 42. KO candidate UMAP check ========================
sc.pl.umap(
    adata_P7_KO,
    color=[
        "residual_sub",
        "KO_sub",
        "batch",
        "M_KO_score",
        "ML_KO_score",
        "L_KO_score",
        "IPhC_check_score",
        "HeC_check_score",
        "Glia_check_score",
    ],
    ncols=3,
    size=8,
    wspace=0.45,
    cmap="viridis",
)


# ======================== 43. KO candidate marker dotplot ========================
ko_dot_markers_all = {
    "Core": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "M_KO": ["Cldn4", "Upp1", "Ly6h", "Serpina3a", "Ier5"],
    "ML_KO": ["Clu", "Epyc", "Grb14", "Fam159b", "Qpct"],
    "L_KO": ["Anxa5", "Foxq1", "Igfbp3", "Gsn", "Hspa2"],
    "IPhC_check": ["Igfbp4", "Maff", "Apoe", "Fst", "Cnmd"],
    "HeC_check": ["Arpc1b", "Sostdc1", "Fabp7", "Ednrb", "Plp1"],
    "Exclude": ["Oc90", "Otx2", "Neurod1", "Neurog1", "Ptprc", "Lyz2", "Hbb-bs", "Hba-a1", "Cnp", "Mbp", "Mpz", "Pmp22"],
}

ko_dot_markers = {
    group: [gene for gene in genes if gene in adata_P7_KO.var_names]
    for group, genes in ko_dot_markers_all.items()
}
ko_dot_markers = {
    group: genes
    for group, genes in ko_dot_markers.items()
    if len(genes) > 0
}

missing_ko_dot_genes = sorted({
    gene
    for genes in ko_dot_markers_all.values()
    for gene in genes
    if gene not in adata_P7_KO.var_names
})
print("Missing genes:", missing_ko_dot_genes)

sc.pl.dotplot(
    adata_P7_KO,
    var_names=ko_dot_markers,
    groupby="KO_sub",
    layer="plot_expr",
    standard_scale="var",
    dot_max=1,
    dot_min=0,
    figsize=(18, 5),
)


plt.close("all")


# Notebook cell 31

# ======================== 44. fix IPhC-like KO_sub and prepare main KO cells ========================

adata_P7_no_glia.obs["P7_step_label"] = adata_P7_no_glia.obs["P7_step_label"].astype(str)

iphc_from_ko_cells = adata_P7_KO.obs_names[
    adata_P7_KO.obs["KO_sub"].astype(str).isin(["2"])
]
adata_P7_no_glia.obs.loc[iphc_from_ko_cells, "P7_step_label"] = "IPhC"

main_ko_sub = ["0", "1", "3", "4"]
adata_P7_KO_main = adata_P7_KO[
    adata_P7_KO.obs["KO_sub"].astype(str).isin(main_ko_sub)
].copy()

if "plot_expr" not in adata_P7_KO_main.layers:
    adata_P7_KO_main.layers["plot_expr"] = adata_P7_KO_main.X.copy()

print(adata_P7_no_glia.obs["P7_step_label"].value_counts())
print(adata_P7_KO_main)
print(pd.crosstab(adata_P7_KO_main.obs["KO_sub"], adata_P7_KO_main.obs["batch"]))


# ======================== 45. main KO marker scoring ========================
main_ko_score_markers = {
    "M_KO": ["Cldn4", "Upp1", "Ly6h"],
    "ML_KO": ["Clu", "Epyc", "Grb14"],
    "L_KO": ["Anxa5", "Foxq1", "Igfbp3"],
    "IPhC_check": ["Igfbp4", "Maff", "Apoe"],
    "HeC_check": ["Arpc1b", "Sostdc1", "Fabp7"],
    "Glia_check": ["Cnp", "Mbp", "Mpz", "Pmp22"],
    "CC_OS_check": ["Coch", "Sparcl1", "Otor"],
    "IPC_OPC_DC_check": ["Npy", "Sapcd2", "Car14", "Mansc4", "Rbp7"],
}

adata_P7_KO_main.X = adata_P7_KO_main.layers["plot_expr"].copy()

for score_name, genes in main_ko_score_markers.items():
    genes_use = [gene for gene in genes if gene in adata_P7_KO_main.var_names]
    if len(genes_use) > 0:
        sc.tl.score_genes(
            adata_P7_KO_main,
            gene_list=genes_use,
            score_name=f"{score_name}_score",
            ctrl_size=50,
            random_state=0,
        )

adata_P7_KO_main.obs["M_over_others"] = (
    adata_P7_KO_main.obs["M_KO_score"]
    - adata_P7_KO_main.obs[["ML_KO_score", "L_KO_score"]].max(axis=1)
)
adata_P7_KO_main.obs["ML_over_others"] = (
    adata_P7_KO_main.obs["ML_KO_score"]
    - adata_P7_KO_main.obs[["M_KO_score", "L_KO_score"]].max(axis=1)
)
adata_P7_KO_main.obs["L_over_others"] = (
    adata_P7_KO_main.obs["L_KO_score"]
    - adata_P7_KO_main.obs[["M_KO_score", "ML_KO_score"]].max(axis=1)
)

adata_P7_KO_main.obs["KO_main_module_call"] = (
    adata_P7_KO_main.obs[["M_KO_score", "ML_KO_score", "L_KO_score"]]
    .idxmax(axis=1)
    .str.replace("_score", "", regex=False)
)


# ======================== 46. main KO Harmony reclustering ========================
sc.pp.highly_variable_genes(
    adata_P7_KO_main,
    n_top_genes=3000,
    batch_key="batch",
    subset=False,
)

sc.pp.scale(adata_P7_KO_main, max_value=10)
sc.tl.pca(
    adata_P7_KO_main,
    n_comps=30,
    use_highly_variable=True,
    svd_solver="arpack",
    random_state=0,
)

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    sce.pp.harmony_integrate(
        adata_P7_KO_main,
        key="batch",
        basis="X_pca",
        adjusted_basis="X_pca_harmony",
        max_iter_harmony=20,
    )

adata_P7_KO_main.X = adata_P7_KO_main.layers["plot_expr"].copy()

sc.pp.neighbors(
    adata_P7_KO_main,
    n_neighbors=30,
    n_pcs=25,
    use_rep="X_pca_harmony",
    random_state=0,
)
sc.tl.umap(adata_P7_KO_main, min_dist=0.35, spread=1.0, random_state=0)
sc.tl.leiden(
    adata_P7_KO_main,
    resolution=0.65,
    key_added="KO_main_sub",
    random_state=0,
)

score_cols = [
    "M_KO_score",
    "ML_KO_score",
    "L_KO_score",
    "M_over_others",
    "ML_over_others",
    "L_over_others",
    "IPhC_check_score",
    "HeC_check_score",
    "Glia_check_score",
    "CC_OS_check_score",
    "IPC_OPC_DC_check_score",
]

print(adata_P7_KO_main.obs["KO_main_sub"].value_counts().sort_index())
print(pd.crosstab(adata_P7_KO_main.obs["KO_main_sub"], adata_P7_KO_main.obs["batch"]))
print(pd.crosstab(adata_P7_KO_main.obs["KO_main_sub"], adata_P7_KO_main.obs["KO_main_module_call"]))
print(adata_P7_KO_main.obs.groupby("KO_main_sub")[score_cols].mean().round(3))


# ======================== 47. main KO UMAP check ========================
sc.pl.umap(
    adata_P7_KO_main,
    color=[
        "KO_sub",
        "KO_main_sub",
        "batch",
        "KO_main_module_call",
        "M_KO_score",
        "ML_KO_score",
        "L_KO_score",
        "M_over_others",
        "ML_over_others",
        "L_over_others",
        "IPhC_check_score",
        "HeC_check_score",
    ],
    ncols=4,
    size=8,
    wspace=0.45,
    cmap="viridis",
)


# ======================== 48. main KO marker dotplot ========================
main_ko_dot_markers_all = {
    "Core": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "M_KO": ["Cldn4", "Upp1", "Ly6h", "Serpina3a", "Ier5"],
    "ML_KO": ["Clu", "Epyc", "Grb14", "Fam159b", "Qpct"],
    "L_KO": ["Anxa5", "Foxq1", "Igfbp3", "Gsn", "Hspa2"],
    "IPhC_check": ["Igfbp4", "Maff", "Apoe", "Fst", "Cnmd"],
    "HeC_check": ["Arpc1b", "Sostdc1", "Fabp7", "Ednrb", "Plp1"],
    "Exclude": ["Oc90", "Otx2", "Neurod1", "Neurog1", "Ptprc", "Lyz2", "Hbb-bs", "Hba-a1", "Cnp", "Mbp", "Mpz", "Pmp22"],
}

main_ko_dot_markers = {
    group: [gene for gene in genes if gene in adata_P7_KO_main.var_names]
    for group, genes in main_ko_dot_markers_all.items()
}
main_ko_dot_markers = {
    group: genes
    for group, genes in main_ko_dot_markers.items()
    if len(genes) > 0
}

missing_main_ko_dot_genes = sorted({
    gene
    for genes in main_ko_dot_markers_all.values()
    for gene in genes
    if gene not in adata_P7_KO_main.var_names
})
print("Missing genes:", missing_main_ko_dot_genes)

sc.pl.dotplot(
    adata_P7_KO_main,
    var_names=main_ko_dot_markers,
    groupby="KO_main_sub",
    layer="plot_expr",
    standard_scale="var",
    dot_max=1,
    dot_min=0,
    figsize=(18, 5),
)


plt.close("all")


# Notebook cell 32

#49. =========== Fix clear KO labels and subset mixed KO ===========
sc.settings.verbosity = 0

adata_P7_no_glia.obs["P7_step_label"] = adata_P7_no_glia.obs["P7_step_label"].astype(str)

clear_l_ko_cells = adata_P7_KO_main.obs_names[
    adata_P7_KO_main.obs["KO_main_sub"].astype(str).isin(["1"])
]
clear_ml_ko_cells = adata_P7_KO_main.obs_names[
    adata_P7_KO_main.obs["KO_main_sub"].astype(str).isin(["2", "3"])
]

adata_P7_no_glia.obs.loc[clear_l_ko_cells, "P7_step_label"] = "L_KO"
adata_P7_no_glia.obs.loc[clear_ml_ko_cells, "P7_step_label"] = "ML_KO"

ko_mixed_sub = ["0", "4", "5", "6"]
adata_P7_KO_mixed = adata_P7_KO_main[
    adata_P7_KO_main.obs["KO_main_sub"].astype(str).isin(ko_mixed_sub)
].copy()

adata_P7_KO_mixed.X = adata_P7_KO_mixed.layers["plot_expr"].copy()
adata_P7_KO_mixed.obs["KO_mixed_module_call"] = adata_P7_KO_mixed.obs["KO_main_module_call"].astype(str)

print(adata_P7_no_glia.obs["P7_step_label"].value_counts())
print(pd.crosstab(adata_P7_KO_mixed.obs["KO_main_sub"], adata_P7_KO_mixed.obs["batch"]))


#50. =========== Recluster mixed KO with Harmony ===========
sc.pp.highly_variable_genes(
    adata_P7_KO_mixed,
    n_top_genes=2500,
    batch_key="batch",
    subset=False,
)

sc.pp.scale(adata_P7_KO_mixed, max_value=10)
sc.tl.pca(
    adata_P7_KO_mixed,
    n_comps=25,
    use_highly_variable=True,
    svd_solver="arpack",
    random_state=0,
)

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    sce.pp.harmony_integrate(
        adata_P7_KO_mixed,
        key="batch",
        basis="X_pca",
        adjusted_basis="X_pca_harmony",
        max_iter_harmony=20,
    )

adata_P7_KO_mixed.X = adata_P7_KO_mixed.layers["plot_expr"].copy()

sc.pp.neighbors(
    adata_P7_KO_mixed,
    n_neighbors=25,
    n_pcs=20,
    use_rep="X_pca_harmony",
    random_state=0,
)
sc.tl.umap(adata_P7_KO_mixed, min_dist=0.35, spread=1.0, random_state=0)
sc.tl.leiden(
    adata_P7_KO_mixed,
    resolution=0.7,
    key_added="KO_mixed_sub",
    random_state=0,
)

mixed_score_cols = [
    "M_KO_score",
    "ML_KO_score",
    "L_KO_score",
    "M_over_others",
    "ML_over_others",
    "L_over_others",
    "IPhC_check_score",
    "HeC_check_score",
    "IPC_OPC_DC_check_score",
    "Glia_check_score",
]

print(adata_P7_KO_mixed.obs["KO_mixed_sub"].value_counts().sort_index())
print(pd.crosstab(adata_P7_KO_mixed.obs["KO_mixed_sub"], adata_P7_KO_mixed.obs["batch"]))
print(adata_P7_KO_mixed.obs.groupby("KO_mixed_sub")[mixed_score_cols].mean().round(3))


#51. =========== Mixed KO UMAP check ===========
sc.pl.umap(
    adata_P7_KO_mixed,
    color=[
        "KO_main_sub",
        "KO_mixed_sub",
        "batch",
        "KO_mixed_module_call",
        "M_over_others",
        "ML_over_others",
        "L_over_others",
        "HeC_check_score",
        "IPC_OPC_DC_check_score",
    ],
    ncols=3,
    size=10,
    wspace=0.45,
    cmap="viridis",
)


#52. =========== Mixed KO marker dotplot ===========
ko_mixed_dot_markers_all = {
    "Core": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "M_KO": ["Cldn4", "Upp1", "Ly6h", "Serpina3a", "Ier5"],
    "ML_KO": ["Clu", "Epyc", "Grb14", "Fam159b", "Qpct"],
    "L_KO": ["Anxa5", "Foxq1", "Igfbp3", "Gsn", "Hspa2"],
    "IPhC_check": ["Igfbp4", "Maff", "Apoe", "Fst", "Cnmd"],
    "HeC_check": ["Arpc1b", "Sostdc1", "Fabp7", "Ednrb", "Plp1"],
    "Exclude": ["Oc90", "Otx2", "Neurod1", "Neurog1", "Ptprc", "Lyz2", "Hbb-bs", "Hba-a1", "Cnp", "Mbp", "Mpz", "Pmp22"],
}

ko_mixed_dot_markers = {
    group: [gene for gene in genes if gene in adata_P7_KO_mixed.var_names]
    for group, genes in ko_mixed_dot_markers_all.items()
}
ko_mixed_dot_markers = {
    group: genes
    for group, genes in ko_mixed_dot_markers.items()
    if len(genes) > 0
}

missing_ko_mixed_dot_genes = sorted({
    gene
    for genes in ko_mixed_dot_markers_all.values()
    for gene in genes
    if gene not in adata_P7_KO_mixed.var_names
})
print("Missing genes:", missing_ko_mixed_dot_genes)

sc.pl.dotplot(
    adata_P7_KO_mixed,
    var_names=ko_mixed_dot_markers,
    groupby="KO_mixed_sub",
    layer="plot_expr",
    standard_scale="var",
    dot_max=1,
    dot_min=0,
    figsize=(17, 5),
)


plt.close("all")


# Notebook cell 33

#53. =========== Fix clear mixed KO labels ===========
adata_P7_no_glia.obs["P7_step_label"] = adata_P7_no_glia.obs["P7_step_label"].astype(str)

clear_l_ko_cells = adata_P7_KO_mixed.obs_names[
    adata_P7_KO_mixed.obs["KO_mixed_sub"].astype(str).isin(["3"])
]
clear_hec_cells = adata_P7_KO_mixed.obs_names[
    adata_P7_KO_mixed.obs["KO_mixed_sub"].astype(str).isin(["4"])
]
clear_ml_ko_cells = adata_P7_KO_mixed.obs_names[
    adata_P7_KO_mixed.obs["KO_mixed_sub"].astype(str).isin(["5", "6"])
]

adata_P7_no_glia.obs.loc[clear_l_ko_cells, "P7_step_label"] = "L_KO"
adata_P7_no_glia.obs.loc[clear_hec_cells, "P7_step_label"] = "HeC"
adata_P7_no_glia.obs.loc[clear_ml_ko_cells, "P7_step_label"] = "ML_KO"

adata_P7_KO_core = adata_P7_KO_mixed[
    adata_P7_KO_mixed.obs["KO_mixed_sub"].astype(str).isin(["0", "1", "2"])
].copy()

adata_P7_KO_core.X = adata_P7_KO_core.layers["plot_expr"].copy()
adata_P7_KO_core.obs["KO_core_module_call"] = adata_P7_KO_core.obs["KO_mixed_module_call"].astype(str)

print(adata_P7_no_glia.obs["P7_step_label"].value_counts())
print(pd.crosstab(adata_P7_KO_core.obs["KO_mixed_sub"], adata_P7_KO_core.obs["batch"]))


#54. =========== Recluster KO core with Harmony ===========
sc.pp.highly_variable_genes(
    adata_P7_KO_core,
    n_top_genes=2500,
    batch_key="batch",
    subset=False,
)

sc.pp.scale(adata_P7_KO_core, max_value=10)
sc.tl.pca(
    adata_P7_KO_core,
    n_comps=25,
    use_highly_variable=True,
    svd_solver="arpack",
    random_state=0,
)

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    sce.pp.harmony_integrate(
        adata_P7_KO_core,
        key="batch",
        basis="X_pca",
        adjusted_basis="X_pca_harmony",
        max_iter_harmony=20,
    )

adata_P7_KO_core.X = adata_P7_KO_core.layers["plot_expr"].copy()

sc.pp.neighbors(
    adata_P7_KO_core,
    n_neighbors=22,
    n_pcs=20,
    use_rep="X_pca_harmony",
    random_state=0,
)
sc.tl.umap(adata_P7_KO_core, min_dist=0.35, spread=1.0, random_state=0)
sc.tl.leiden(
    adata_P7_KO_core,
    resolution=0.75,
    key_added="KO_core_sub",
    random_state=0,
)

core_score_cols = [
    "M_KO_score",
    "ML_KO_score",
    "L_KO_score",
    "M_over_others",
    "ML_over_others",
    "L_over_others",
    "IPhC_check_score",
    "HeC_check_score",
    "IPC_OPC_DC_check_score",
    "Glia_check_score",
]

print(adata_P7_KO_core.obs["KO_core_sub"].value_counts().sort_index())
print(pd.crosstab(adata_P7_KO_core.obs["KO_core_sub"], adata_P7_KO_core.obs["batch"]))
print(adata_P7_KO_core.obs.groupby("KO_core_sub")[core_score_cols].mean().round(3))


#55. =========== KO core UMAP check ===========
sc.pl.umap(
    adata_P7_KO_core,
    color=[
        "KO_mixed_sub",
        "KO_core_sub",
        "batch",
        "KO_core_module_call",
        "M_over_others",
        "ML_over_others",
        "L_over_others",
        "HeC_check_score",
        "IPC_OPC_DC_check_score",
    ],
    ncols=3,
    size=10,
    wspace=0.45,
    cmap="viridis",
)


#56. =========== KO core marker dotplot ===========
ko_core_dot_markers_all = {
    "Core": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "M_KO": ["Cldn4", "Upp1", "Ly6h", "Serpina3a", "Ier5"],
    "ML_KO": ["Clu", "Epyc", "Grb14", "Fam159b", "Qpct"],
    "L_KO": ["Anxa5", "Foxq1", "Igfbp3", "Gsn", "Hspa2"],
    "IPhC_check": ["Igfbp4", "Maff", "Apoe", "Fst", "Cnmd"],
    "HeC_check": ["Arpc1b", "Sostdc1", "Fabp7", "Ednrb", "Plp1"],
    "Exclude": ["Oc90", "Otx2", "Neurod1", "Neurog1", "Ptprc", "Lyz2", "Hbb-bs", "Hba-a1", "Cnp", "Mbp", "Mpz", "Pmp22"],
}

ko_core_dot_markers = {
    group: [gene for gene in genes if gene in adata_P7_KO_core.var_names]
    for group, genes in ko_core_dot_markers_all.items()
}
ko_core_dot_markers = {
    group: genes
    for group, genes in ko_core_dot_markers.items()
    if len(genes) > 0
}

sc.pl.dotplot(
    adata_P7_KO_core,
    var_names=ko_core_dot_markers,
    groupby="KO_core_sub",
    layer="plot_expr",
    standard_scale="var",
    dot_max=1,
    dot_min=0,
    figsize=(17, 5),
)


plt.close("all")


# Notebook cell 34

#57. =========== Fix clear KO core labels ===========
sc.settings.verbosity = 0

adata_P7_no_glia.obs["P7_step_label"] = adata_P7_no_glia.obs["P7_step_label"].astype(str)

clear_m_ko_cells = adata_P7_KO_core.obs_names[
    adata_P7_KO_core.obs["KO_core_sub"].astype(str).isin(["0", "3", "4"])
]
clear_ml_ko_cells = adata_P7_KO_core.obs_names[
    adata_P7_KO_core.obs["KO_core_sub"].astype(str).isin(["1", "2"])
]

adata_P7_no_glia.obs.loc[clear_m_ko_cells, "P7_step_label"] = "M_KO"
adata_P7_no_glia.obs.loc[clear_ml_ko_cells, "P7_step_label"] = "ML_KO"

final_check_names = adata_P7_no_glia.obs_names[
    adata_P7_no_glia.obs["P7_step_label"].astype(str) == "Unassigned"
]

adata_P7_final_check = adata_P7_residual[
    adata_P7_residual.obs_names.isin(final_check_names)
].copy()

adata_P7_final_check.obs["P7_step_label"] = adata_P7_no_glia.obs.loc[
    adata_P7_final_check.obs_names, "P7_step_label"
].astype(str).values
adata_P7_final_check.layers["plot_expr"] = adata_P7_final_check.X.copy()

print(adata_P7_no_glia.obs["P7_step_label"].value_counts())
print(pd.crosstab(adata_P7_final_check.obs["parent_leiden"], adata_P7_final_check.obs["batch"]))


#58. =========== Final small residual marker scoring ===========
final_check_score_markers = {
    "M_KO": ["Cldn4", "Upp1", "Ly6h"],
    "ML_KO": ["Clu", "Epyc", "Grb14"],
    "L_KO": ["Anxa5", "Foxq1", "Igfbp3"],
    "IPhC": ["Igfbp4", "Maff", "Apoe"],
    "HeC": ["Arpc1b", "Sostdc1", "Fabp7"],
    "IPC_OPC_DC": ["Npy", "Sapcd2", "Car14", "Mansc4", "Rbp7"],
    "Glia": ["Cnp", "Mbp", "Mpz", "Pmp22"],
}

for score_name, genes in final_check_score_markers.items():
    genes_use = [gene for gene in genes if gene in adata_P7_final_check.var_names]
    if len(genes_use) > 0:
        sc.tl.score_genes(
            adata_P7_final_check,
            gene_list=genes_use,
            score_name=f"final_{score_name}_score",
            ctrl_size=50,
            random_state=0,
        )


#59. =========== Recluster final small residual with Harmony ===========
sc.pp.highly_variable_genes(
    adata_P7_final_check,
    n_top_genes=1500,
    batch_key="batch",
    subset=False,
)

sc.pp.scale(adata_P7_final_check, max_value=10)
sc.tl.pca(
    adata_P7_final_check,
    n_comps=15,
    use_highly_variable=True,
    svd_solver="arpack",
    random_state=0,
)

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    sce.pp.harmony_integrate(
        adata_P7_final_check,
        key="batch",
        basis="X_pca",
        adjusted_basis="X_pca_harmony",
        max_iter_harmony=20,
    )

adata_P7_final_check.X = adata_P7_final_check.layers["plot_expr"].copy()

sc.pp.neighbors(
    adata_P7_final_check,
    n_neighbors=10,
    n_pcs=12,
    use_rep="X_pca_harmony",
    random_state=0,
)
sc.tl.umap(adata_P7_final_check, min_dist=0.35, spread=1.0, random_state=0)
sc.tl.leiden(
    adata_P7_final_check,
    resolution=0.5,
    key_added="final_check_sub",
    random_state=0,
)

final_check_score_cols = [
    "final_M_KO_score",
    "final_ML_KO_score",
    "final_L_KO_score",
    "final_IPhC_score",
    "final_HeC_score",
    "final_IPC_OPC_DC_score",
    "final_Glia_score",
]

print(adata_P7_final_check.obs["final_check_sub"].value_counts().sort_index())
print(pd.crosstab(adata_P7_final_check.obs["final_check_sub"], adata_P7_final_check.obs["batch"]))
print(adata_P7_final_check.obs.groupby("final_check_sub")[final_check_score_cols].mean().round(3))


#60. =========== Final small residual UMAP check ===========
sc.pl.umap(
    adata_P7_final_check,
    color=[
        "parent_leiden",
        "final_check_sub",
        "batch",
        "final_M_KO_score",
        "final_ML_KO_score",
        "final_L_KO_score",
        "final_IPhC_score",
        "final_HeC_score",
        "final_IPC_OPC_DC_score",
    ],
    ncols=3,
    size=28,
    wspace=0.45,
    cmap="viridis",
)


#61. =========== Final small residual marker dotplot ===========
final_check_dot_markers_all = {
    "Core": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "M_KO": ["Cldn4", "Upp1", "Ly6h", "Serpina3a", "Ier5"],
    "ML_KO": ["Clu", "Epyc", "Grb14", "Fam159b", "Qpct"],
    "L_KO": ["Anxa5", "Foxq1", "Igfbp3", "Gsn", "Hspa2"],
    "IPhC": ["Igfbp4", "Maff", "Apoe", "Fst", "Cnmd"],
    "HeC": ["Arpc1b", "Sostdc1", "Fabp7", "Ednrb", "Plp1"],
    "IPC_OPC_DC": ["Npy", "Sapcd2", "Car14", "Mansc4", "Rbp7"],
    "Exclude": ["Oc90", "Otx2", "Neurod1", "Neurog1", "Ptprc", "Lyz2", "Hbb-bs", "Hba-a1", "Cnp", "Mbp", "Mpz", "Pmp22"],
}

final_check_dot_markers = {
    group: [gene for gene in genes if gene in adata_P7_final_check.var_names]
    for group, genes in final_check_dot_markers_all.items()
}
final_check_dot_markers = {
    group: genes
    for group, genes in final_check_dot_markers.items()
    if len(genes) > 0
}

sc.pl.dotplot(
    adata_P7_final_check,
    var_names=final_check_dot_markers,
    groupby="final_check_sub",
    layer="plot_expr",
    standard_scale="var",
    dot_max=1,
    dot_min=0,
    figsize=(18, 4),
)


plt.close("all")


# Notebook cell 35

#62. =========== Finalize P7 cell type labels ===========
adata_P7_no_glia.obs["P7_step_label"] = adata_P7_no_glia.obs["P7_step_label"].astype(str)

final_ml_ko_cells = adata_P7_final_check.obs_names[
    adata_P7_final_check.obs["final_check_sub"].astype(str).isin(["0"])
]
final_dc_cells = adata_P7_final_check.obs_names[
    adata_P7_final_check.obs["final_check_sub"].astype(str).isin(["1"])
]

adata_P7_no_glia.obs.loc[final_ml_ko_cells, "P7_step_label"] = "ML_KO"
adata_P7_no_glia.obs.loc[final_dc_cells, "P7_step_label"] = "DC"

p7_celltype_order = [
    "IHC", "OHC", "IPhC", "IPC", "OPC", "DC", "HeC",
    "M_KO", "ML_KO", "L_KO", "CC/OS",
]

adata_P7_no_glia.obs["P7_final_celltype"] = pd.Categorical(
    adata_P7_no_glia.obs["P7_step_label"],
    categories=p7_celltype_order,
    ordered=True,
)

p7_final_counts = adata_P7_no_glia.obs["P7_final_celltype"].value_counts().reindex(p7_celltype_order)
adata_P7_no_glia.obs["P7_final_celltype_n"] = adata_P7_no_glia.obs["P7_final_celltype"].astype(str)
for celltype, count in p7_final_counts.dropna().astype(int).items():
    adata_P7_no_glia.obs.loc[
        adata_P7_no_glia.obs["P7_final_celltype"].astype(str) == celltype,
        "P7_final_celltype_n",
    ] = f"{celltype} ({count})"

print(p7_final_counts)


#63. =========== Final P7 UMAP validation ===========
if "plot_expr" not in adata_P7_no_glia.layers:
    if "plot_expr" in adata_P7.layers:
        adata_P7_no_glia.layers["plot_expr"] = adata_P7[
            adata_P7_no_glia.obs_names, :
        ].layers["plot_expr"].copy()
    else:
        adata_P7_no_glia.layers["plot_expr"] = adata_P7_no_glia.X.copy()

sc.pp.neighbors(
    adata_P7_no_glia,
    n_neighbors=30,
    n_pcs=30,
    use_rep="X_pca_harmony",
    random_state=0,
)
sc.tl.umap(adata_P7_no_glia, min_dist=0.35, spread=1.0, random_state=0)

sc.pl.umap(
    adata_P7_no_glia,
    color=["P7_final_celltype_n", "batch"],
    ncols=2,
    size=8,
    wspace=0.45,
    legend_loc="right margin",
)

sc.pl.umap(
    adata_P7_no_glia,
    color="P7_final_celltype",
    size=8,
    legend_loc="on data",
    legend_fontsize=8,
    legend_fontoutline=2,
)


#64. =========== Final P7 marker dotplot validation ===========
p7_final_dot_markers_all = {
    "Core": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "IHC": ["Myo6", "Myo7a", "Cib2", "Pvalb", "Otof", "Atp2a3", "Cabp2", "Fgf8", "Dlk2", "Nefl"],
    "OHC": ["Aqp11", "Pcp4", "Ocm", "Calb1", "Six2", "Ighm"],
    "IPhC": ["Igfbp4", "Maff", "Apoe", "Fst", "Cnmd"],
    "IPC": ["Ppp2r2b", "Npy", "Cep41", "Tuba1a", "Tuba1b"],
    "OPC": ["Sapcd2", "Map4", "S100b", "Smagp", "Cep41"],
    "DC": ["Car14", "Mansc4", "Rbp7", "Rflnb", "Selenom"],
    "HeC": ["Arpc1b", "Sostdc1", "Fabp7", "Ednrb", "Plp1"],
    "M_KO": ["Cldn4", "Upp1", "Ly6h", "Serpina3a", "Ier5"],
    "ML_KO": ["Clu", "Epyc", "Grb14", "Fam159b", "Qpct"],
    "L_KO": ["Anxa5", "Foxq1", "Igfbp3", "Gsn", "Hspa2"],
    "CC/OS": ["Coch", "Sparcl1", "Otor", "Col3a1", "Ibsp"],
    "Exclude": ["Oc90", "Otx2", "Neurod1", "Neurog1", "Ptprc", "Lyz2", "Hbb-bs", "Hba-a1", "Cnp", "Mbp", "Mpz", "Pmp22"],
}

p7_final_dot_markers = {
    group: [gene for gene in genes if gene in adata_P7_no_glia.var_names]
    for group, genes in p7_final_dot_markers_all.items()
}
p7_final_dot_markers = {
    group: genes
    for group, genes in p7_final_dot_markers.items()
    if len(genes) > 0
}

sc.pl.dotplot(
    adata_P7_no_glia,
    var_names=p7_final_dot_markers,
    groupby="P7_final_celltype",
    layer="plot_expr",
    categories_order=p7_celltype_order,
    standard_scale="var",
    dot_max=1,
    dot_min=0,
    figsize=(24, 7),
)


plt.close("all")


# Notebook cell 36

# ======================== 66. Final validation of P7 labels ========================

p7_label_key = "P7_final_celltype"

if p7_label_key in adata_P7_no_glia.obs.columns:
    adata_P7_final_check2 = adata_P7_no_glia.copy()
elif p7_label_key in adata_P7.obs.columns:
    adata_P7_final_check2 = adata_P7[adata_P7.obs[p7_label_key].astype(str) != "Glia"].copy()
else:
    raise KeyError("P7_final_celltype is not found in adata_P7_no_glia or adata_P7.")

p7_celltype_order = [
    "IHC", "OHC", "IPhC", "IPC", "OPC", "DC", "HeC",
    "M_KO", "ML_KO", "L_KO", "CC/OS"
]

adata_P7_final_check2.obs[p7_label_key] = pd.Categorical(
    adata_P7_final_check2.obs[p7_label_key].astype(str),
    categories=p7_celltype_order,
    ordered=True,
)

p7_final_markers = {
    "Core": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "HC_core": ["Myo6", "Myo7a", "Cib2", "Pvalb"],
    "IHC": ["Otof", "Atp2a3", "Cabp2", "Fgf8", "Dlk2", "Nefl", "Calb2"],
    "OHC": ["Aqp11", "Pcp4", "Ocm", "Calb1", "Six2", "Ighm"],
    "IPhC": ["Igfbp4", "Maff", "Apoe", "Fst", "Cnmd"],
    "IPC": ["Ppp2r2b", "Npy", "Cep41", "Tuba1a", "Tuba1b", "Emid1", "Ccbe1"],
    "OPC": ["Sapcd2", "Map4", "S100b", "Smagp", "Cep41"],
    "DC": ["Car14", "Mansc4", "Rbp7", "Rflnb", "Selenom"],
    "HeC": ["Arpc1b", "Sostdc1", "Fabp7", "Ednrb", "Plp1", "Pmch"],
    "M_KO": ["Cldn4", "Upp1", "Ly6h", "Serpina3a", "Ier5"],
    "ML_KO": ["Clu", "Epyc", "Grb14", "Fam159b", "Qpct"],
    "L_KO": ["Anxa5", "Foxq1", "Igfbp3", "Gsn", "Hspa2"],
    "CC/OS": ["Coch", "Sparcl1", "Otor", "Col3a1", "Ibsp"],
    "Exclude": ["Oc90", "Otx2", "Neurod1", "Neurog1", "Ptprc", "Lyz2", "Hbb-bs", "Hba-a1", "Cnp", "Mbp", "Mpz", "Pmp22"],
}

p7_present_markers = {
    k: [g for g in genes if g in adata_P7_final_check2.var_names]
    for k, genes in p7_final_markers.items()
}
p7_present_markers = {k: v for k, v in p7_present_markers.items() if len(v) > 0}

p7_missing_markers = {
    k: [g for g in genes if g not in adata_P7_final_check2.var_names]
    for k, genes in p7_final_markers.items()
}
p7_missing_markers = {k: v for k, v in p7_missing_markers.items() if len(v) > 0}

p7_score_markers = {
    "IPC": ["Ppp2r2b", "Npy", "Emid1", "Ccbe1"],
    "OPC": ["Sapcd2", "Map4", "S100b", "Smagp"],
    "DC": ["Car14", "Mansc4", "Rbp7", "Rflnb", "Selenom"],
    "HeC": ["Arpc1b", "Sostdc1", "Fabp7", "Ednrb", "Plp1", "Pmch"],
    "IPhC": ["Igfbp4", "Maff", "Apoe", "Fst", "Cnmd"],
    "KO": ["Cldn4", "Upp1", "Ly6h", "Clu", "Epyc", "Grb14", "Anxa5", "Foxq1", "Igfbp3"],
    "Glia_exclude": ["Cnp", "Mbp", "Mpz", "Pmp22"],
}

p7_score_cols = []
for score_name, genes in p7_score_markers.items():
    genes_present = [g for g in genes if g in adata_P7_final_check2.var_names]
    if len(genes_present) >= 2:
        score_col = f"final_{score_name}_score"
        sc.tl.score_genes(
            adata_P7_final_check2,
            gene_list=genes_present,
            score_name=score_col,
            ctrl_size=50,
            use_raw=False,
        )
        p7_score_cols.append(score_col)

p7_focus_types = ["IPC", "OPC", "DC", "HeC", "IPhC", "M_KO", "ML_KO", "L_KO"]
adata_P7_lowconf_check = adata_P7_final_check2[
    adata_P7_final_check2.obs[p7_label_key].astype(str).isin(p7_focus_types)
].copy()

print(adata_P7_final_check2.obs[p7_label_key].value_counts().reindex(p7_celltype_order).dropna().astype(int))
print(pd.crosstab(adata_P7_lowconf_check.obs[p7_label_key], adata_P7_lowconf_check.obs["batch"]))
print(
    adata_P7_lowconf_check.obs
    .groupby(p7_label_key, observed=True)[p7_score_cols]
    .mean()
    .round(3)
)
print("Missing marker genes:", p7_missing_markers)

sc.set_figure_params(dpi=100, figsize=(4, 4), fontsize=10)

sc.pl.umap(
    adata_P7_lowconf_check,
    color=[
        p7_label_key,
        "batch",
        "final_IPC_score",
        "final_OPC_score",
        "final_DC_score",
        "final_HeC_score",
        "final_IPhC_score",
        "final_KO_score",
        "final_Glia_exclude_score",
    ],
    size=14,
    ncols=3,
    wspace=0.45,
    frameon=True,
)

p7_dot_kwargs = {}
if "plot_expr" in adata_P7_final_check2.layers:
    p7_dot_kwargs["layer"] = "plot_expr"

sc.pl.dotplot(
    adata_P7_final_check2,
    var_names=p7_present_markers,
    groupby=p7_label_key,
    categories_order=p7_celltype_order,
    standard_scale="var",
    dot_min=0,
    dot_max=1,
    figsize=(24, 7),
    dendrogram=False,
    **p7_dot_kwargs,
)

p7_focus_marker_groups = {
    k: p7_present_markers[k]
    for k in ["Core", "IPhC", "IPC", "OPC", "DC", "HeC", "M_KO", "ML_KO", "L_KO", "Exclude"]
    if k in p7_present_markers
}

sc.pl.dotplot(
    adata_P7_lowconf_check,
    var_names=p7_focus_marker_groups,
    groupby=p7_label_key,
    categories_order=p7_focus_types,
    standard_scale="var",
    dot_min=0,
    dot_max=1,
    figsize=(20, 5),
    dendrogram=False,
    **p7_dot_kwargs,
)


plt.close("all")


# Notebook cell 37

# ======================== 67. Save final P7 files and figures ========================

p7_fig_dir = OUTPUT_DIR
p7_harmony_dir = RESULTS_ROOT / "01_stage_clustering" / "harmony_inputs" / "p7"
p7_scanvi_dir = STAGE_OBJECT_DIR

p7_fig_dir.mkdir(parents=True, exist_ok=True)
p7_harmony_dir.mkdir(parents=True, exist_ok=True)
p7_scanvi_dir.mkdir(parents=True, exist_ok=True)

p7_label_key = "P7_final_celltype"
p7_label_n_key = "P7_final_celltype_n"

p7_celltype_order = [
    "IHC", "OHC", "IPhC", "IPC", "OPC", "DC", "HeC",
    "M_KO", "ML_KO", "L_KO", "CC/OS"
]

p7_expected_counts = {
    "IHC": 81,
    "OHC": 182,
    "IPhC": 482,
    "IPC": 23,
    "OPC": 22,
    "DC": 112,
    "HeC": 42,
    "M_KO": 348,
    "ML_KO": 1133,
    "L_KO": 516,
    "CC/OS": 259
}

if p7_label_key in adata_P7_no_glia.obs.columns:
    adata_P7_final_save = adata_P7_no_glia.copy()
elif p7_label_key in adata_P7.obs.columns:
    adata_P7_final_save = adata_P7[
        adata_P7.obs[p7_label_key].astype(str).isin(p7_celltype_order)
    ].copy()
else:
    raise KeyError(f"{p7_label_key} not found in adata_P7_no_glia or adata_P7.")

if "batch" not in adata_P7_final_save.obs.columns:
    raise KeyError("batch not found in adata_P7_final_save.obs.")

adata_P7_final_save = adata_P7_final_save[
    adata_P7_final_save.obs[p7_label_key].astype(str).isin(p7_celltype_order)
].copy()

adata_P7_final_save.obs[p7_label_key] = pd.Categorical(
    adata_P7_final_save.obs[p7_label_key].astype(str),
    categories=p7_celltype_order,
    ordered=True
)

if adata_P7_final_save.obs[p7_label_key].isna().any():
    raise ValueError("P7_final_celltype contains labels outside p7_celltype_order.")

if (adata_P7_final_save.obs[p7_label_key].astype(str) == "CC/OSC").any():
    raise ValueError("P7 should use CC/OS, not CC/OSC.")

if "plot_expr" not in adata_P7_final_save.layers:
    if "plot_expr" in adata_P7.layers:
        adata_P7_final_save.layers["plot_expr"] = adata_P7[
            adata_P7_final_save.obs_names, :
        ].layers["plot_expr"].copy()
    else:
        adata_P7_final_save.layers["plot_expr"] = adata_P7_final_save.X.copy()

adata_P7_final_save.X = adata_P7_final_save.layers["plot_expr"].copy()

p7_counts = (
    adata_P7_final_save.obs[p7_label_key]
    .value_counts()
    .reindex(p7_celltype_order)
    .fillna(0)
    .astype(int)
)

p7_count_check = pd.DataFrame(
    {
        "observed": p7_counts,
        "expected": pd.Series(p7_expected_counts).reindex(p7_celltype_order)
    }
)

if not (p7_count_check["observed"] == p7_count_check["expected"]).all():
    display(p7_count_check)
    raise ValueError("P7 final counts do not match the checked P7 result. Stop before saving.")

if int(p7_counts.sum()) != adata_P7_final_save.n_obs:
    raise ValueError("P7 count sum does not match adata_P7_final_save.n_obs.")

p7_label_n_map = {ct: f"{ct} (n={p7_counts.loc[ct]})" for ct in p7_celltype_order}
adata_P7_final_save.obs[p7_label_n_key] = (
    adata_P7_final_save.obs[p7_label_key].map(p7_label_n_map).astype(str)
)
adata_P7_final_save.obs[p7_label_n_key] = pd.Categorical(
    adata_P7_final_save.obs[p7_label_n_key],
    categories=[p7_label_n_map[ct] for ct in p7_celltype_order],
    ordered=True
)

p7_dot_markers = {
    "Core": ["Epcam", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1"],
    "HC_core": ["Myo6", "Myo7a", "Cib2", "Pvalb"],
    "IHC": ["Otof", "Atp2a3", "Cabp2", "Fgf8", "Dlk2", "Nefl", "Calb2"],
    "OHC": ["Aqp11", "Pcp4", "Ocm", "Calb1", "Six2", "Ighm"],
    "IPhC": ["Igfbp4", "Maff", "Apoe", "Fst", "Cnmd"],
    "IPC": ["Ppp2r2b", "Npy", "Cep41", "Tuba1a", "Tuba1b", "Emid1", "Ccbe1"],
    "OPC": ["Sapcd2", "Map4", "S100b", "Smagp", "Cep41"],
    "DC": ["Car14", "Mansc4", "Rbp7", "Rflnb", "Selenom"],
    "HeC": ["Arpc1b", "Sostdc1", "Fabp7", "Ednrb", "Plp1", "Pmch"],
    "M_KO": ["Cldn4", "Upp1", "Ly6h", "Serpina3a", "Ier5"],
    "ML_KO": ["Clu", "Epyc", "Grb14", "Fam159b", "Qpct"],
    "L_KO": ["Anxa5", "Foxq1", "Igfbp3", "Gsn", "Hspa2"],
    "CC/OS": ["Coch", "Sparcl1", "Otor", "Col3a1", "Ibsp"],
    "Exclude": ["Oc90", "Otx2", "Neurod1", "Neurog1", "Ptprc", "Lyz2", "Hbb-bs", "Hba-a1", "Cnp", "Mbp", "Mpz", "Pmp22"]
}

p7_dot_markers = {
    group: [gene for gene in genes if gene in adata_P7_final_save.var_names]
    for group, genes in p7_dot_markers.items()
}
p7_dot_markers = {group: genes for group, genes in p7_dot_markers.items() if len(genes) > 0}

p7_dot_part1 = {
    group: p7_dot_markers[group]
    for group in ["Core", "HC_core", "IHC", "OHC", "IPhC", "IPC", "OPC"]
    if group in p7_dot_markers
}
p7_dot_part2 = {
    group: p7_dot_markers[group]
    for group in ["DC", "HeC", "M_KO", "ML_KO", "L_KO", "CC/OS", "Exclude"]
    if group in p7_dot_markers
}

dot_kwargs = {}
if "plot_expr" in adata_P7_final_save.layers:
    dot_kwargs["layer"] = "plot_expr"

sc.set_figure_params(dpi=100, fontsize=10)

fig = sc.pl.umap(
    adata_P7_final_save,
    color=[p7_label_n_key, "batch"],
    size=8,
    ncols=2,
    wspace=0.45,
    frameon=True,
    show=False,
    return_fig=True
)
fig.savefig(p7_fig_dir / "P7_umap.png", dpi=SAVE_DPI, bbox_inches="tight")
plt.close(fig)

fig = sc.pl.umap(
    adata_P7_final_save,
    color=p7_label_key,
    legend_loc="on data",
    legend_fontsize=9,
    legend_fontoutline=2,
    size=8,
    frameon=True,
    show=False,
    return_fig=True
)
fig.savefig(p7_fig_dir / "P7_umap_label.png", dpi=SAVE_DPI, bbox_inches="tight")
plt.close(fig)

dp = sc.pl.dotplot(
    adata_P7_final_save,
    var_names=p7_dot_markers,
    groupby=p7_label_key,
    categories_order=p7_celltype_order,
    standard_scale="var",
    dot_min=0,
    dot_max=1,
    figsize=(24, 7),
    dendrogram=False,
    show=False,
    return_fig=True,
    **dot_kwargs
)
dp.savefig(p7_fig_dir / "P7_dotplot.png", dpi=SAVE_DPI)
plt.close("all")

dp = sc.pl.dotplot(
    adata_P7_final_save,
    var_names=p7_dot_part1,
    groupby=p7_label_key,
    categories_order=p7_celltype_order,
    standard_scale="var",
    dot_min=0,
    dot_max=1,
    figsize=(16, 7),
    dendrogram=False,
    show=False,
    return_fig=True,
    **dot_kwargs
)
dp.savefig(p7_fig_dir / "P7_dotplot_part1.png", dpi=SAVE_DPI)
plt.close("all")

dp = sc.pl.dotplot(
    adata_P7_final_save,
    var_names=p7_dot_part2,
    groupby=p7_label_key,
    categories_order=p7_celltype_order,
    standard_scale="var",
    dot_min=0,
    dot_max=1,
    figsize=(18, 7),
    dendrogram=False,
    show=False,
    return_fig=True,
    **dot_kwargs
)
dp.savefig(p7_fig_dir / "P7_dotplot_part2.png", dpi=SAVE_DPI)
plt.close("all")

adata_P7_harmony_save = adata_P7_final_save.copy()
adata_P7_harmony_save.obs = adata_P7_harmony_save.obs[["batch", p7_label_key]].copy()
adata_P7_harmony_save.obs["stage"] = "P7"
adata_P7_harmony_save.raw = None

for key in list(adata_P7_harmony_save.layers.keys()):
    del adata_P7_harmony_save.layers[key]
for key in list(adata_P7_harmony_save.obsm.keys()):
    del adata_P7_harmony_save.obsm[key]
for key in list(adata_P7_harmony_save.varm.keys()):
    del adata_P7_harmony_save.varm[key]
for key in list(adata_P7_harmony_save.obsp.keys()):
    del adata_P7_harmony_save.obsp[key]
adata_P7_harmony_save.uns = {}

adata_P7_scanvi_save = adata_P7_harmony_save.copy()

p7_harmony_path = p7_harmony_dir / "P7_for_Harmony_normalized.h5ad"
p7_scanvi_path = p7_scanvi_dir / "P7_for_scnvi_normalized.h5ad"

adata_P7_harmony_save.write_h5ad(p7_harmony_path, compression="gzip")
adata_P7_scanvi_save.write_h5ad(p7_scanvi_path, compression="gzip")



plt.close("all")
