#1. ===========Purpose and reproducibility settings
"""Preprocess, cluster, annotate, and export the P28 integration inputs."""

from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("COCHLEA_DATA_DIR", REPO_ROOT / "Data")).expanduser().resolve()
RESULTS_ROOT = Path(os.environ.get("COCHLEA_RESULTS_DIR", REPO_ROOT / "results")).expanduser().resolve()
STAGE_OBJECT_DIR = RESULTS_ROOT / "01_stage_clustering" / "h5ad_for_integration"
OUTPUT_DIR = RESULTS_ROOT / "01_stage_clustering" / "p28"
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

# Notebook cell 61

#1. =========== Import and random seed ===========

import os
import random
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.io import mmread
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 0
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

sc.settings.verbosity = 0


#2. =========== Load P28 raw count data ===========

data_dir = str(DATA_ROOT / "raw" / "p28")

def read_p28_sample(sample):
    matrix_path = os.path.join(data_dir, f"{sample}_matrix.mtx")
    barcode_path = os.path.join(data_dir, f"{sample}_barcodes.tsv")
    feature_path = os.path.join(data_dir, f"{sample}_features.tsv")

    X = mmread(matrix_path).T.tocsr()
    barcodes = pd.read_csv(barcode_path, sep="\t", header=None)[0].astype(str).values
    features = pd.read_csv(feature_path, sep="\t", header=None)

    gene_names = features[1].astype(str).values if features.shape[1] > 1 else features[0].astype(str).values

    adata = sc.AnnData(X)
    adata.obs_names = [f"{sample}_{barcode}" for barcode in barcodes]
    adata.var_names = gene_names
    adata.var_names_make_unique()
    adata.obs["sample"] = sample
    adata.layers["counts"] = adata.X.copy()

    return adata

adata_P28_1 = read_p28_sample("P28_1")
adata_P28_2 = read_p28_sample("P28_2")
adata_P28_raw = sc.concat(
    [adata_P28_1, adata_P28_2],
    join="outer",
    label=None,
    index_unique=None
)
adata_P28_raw.X = adata_P28_raw.X.tocsr()
adata_P28_raw.layers["counts"] = adata_P28_raw.X.copy()


#3. =========== Calculate QC metrics ===========

adata_P28_raw.var["mt"] = adata_P28_raw.var_names.str.startswith(("mt-", "Mt-"))
adata_P28_raw.var["ribo"] = adata_P28_raw.var_names.str.startswith(("Rpl", "Rps"))
adata_P28_raw.var["hb"] = adata_P28_raw.var_names.str.startswith(("Hba", "Hbb"))

sc.pp.calculate_qc_metrics(
    adata_P28_raw,
    qc_vars=["mt", "ribo", "hb"],
    percent_top=None,
    inplace=True
)


#4. =========== QC filtering ===========

adata_P28 = adata_P28_raw[
    (adata_P28_raw.obs["n_genes_by_counts"] >= 200) &
    (adata_P28_raw.obs["n_genes_by_counts"] <= 3000) &
    (adata_P28_raw.obs["total_counts"] <= 15000) &
    (adata_P28_raw.obs["pct_counts_mt"] <= 15)
].copy()

adata_P28.layers["counts"] = adata_P28.X.copy()


#5. =========== QC summary table ===========

def qc_summary(adata, label):
    summary = adata.obs.groupby("sample").agg(
        cells=("sample", "size"),
        median_genes=("n_genes_by_counts", "median"),
        median_counts=("total_counts", "median"),
        median_mt=("pct_counts_mt", "median"),
        median_hb=("pct_counts_hb", "median"),
        median_ribo=("pct_counts_ribo", "median")
    )
    summary.insert(0, "QC", label)
    return summary

qc_table = pd.concat([
    qc_summary(adata_P28_raw, "Before_QC"),
    qc_summary(adata_P28, "After_QC")
])

display(qc_table.round(2))


#6. =========== QC plots ===========

qc_plot_raw = adata_P28_raw.obs.copy()
qc_plot_raw["QC"] = "Before QC"

qc_plot_filtered = adata_P28.obs.copy()
qc_plot_filtered["QC"] = "After QC"

qc_plot_df = pd.concat([qc_plot_raw, qc_plot_filtered], axis=0)

qc_metrics = [
    "n_genes_by_counts",
    "total_counts",
    "pct_counts_mt",
    "pct_counts_hb"
]

fig, axes = plt.subplots(2, 4, figsize=(15, 6))

for row_idx, qc_label in enumerate(["Before QC", "After QC"]):
    plot_df = qc_plot_df[qc_plot_df["QC"] == qc_label]

    for col_idx, metric in enumerate(qc_metrics):
        ax = axes[row_idx, col_idx]
        sns.violinplot(
            data=plot_df,
            x="sample",
            y=metric,
            ax=ax,
            inner=None,
            linewidth=0.8,
            cut=0
        )
        sns.stripplot(
            data=plot_df,
            x="sample",
            y=metric,
            ax=ax,
            color="black",
            size=0.4,
            alpha=0.45,
            jitter=0.25
        )
        ax.set_title(f"{qc_label}: {metric}")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.close("all")

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

sns.scatterplot(
    data=adata_P28_raw.obs,
    x="total_counts",
    y="pct_counts_mt",
    hue="sample",
    s=2,
    linewidth=0,
    ax=axes[0]
)
axes[0].set_title("Before QC")

sns.scatterplot(
    data=adata_P28.obs,
    x="total_counts",
    y="pct_counts_mt",
    hue="sample",
    s=2,
    linewidth=0,
    ax=axes[1]
)
axes[1].set_title("After QC")

plt.tight_layout()
plt.close("all")


plt.close("all")


# Notebook cell 62

#1. =========== Harmony import and random seed ===========

import scanpy.external as sce
import warnings

RANDOM_STATE = 0
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
sc.settings.verbosity = 0


#2. =========== Prepare broad clustering object ===========

adata_p28_full = adata_P28.copy()
adata_p28_full.layers["counts"] = adata_p28_full.X.copy()

adata_p28_work = adata_P28.copy()
adata_p28_work.layers["counts"] = adata_p28_work.X.copy()

sc.pp.normalize_total(adata_p28_work, target_sum=1e4)
sc.pp.log1p(adata_p28_work)

sc.pp.highly_variable_genes(
    adata_p28_work,
    n_top_genes=2000,
    batch_key="sample",
    flavor="seurat"
)

tech_gene_mask = adata_p28_work.var_names.str.startswith(("mt-", "Mt-", "Rpl", "Rps", "Hba", "Hbb"))
adata_p28_work.var.loc[tech_gene_mask, "highly_variable"] = False
adata_p28_work = adata_p28_work[:, adata_p28_work.var["highly_variable"]].copy()


#3. =========== PCA and Harmony integration ===========

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="zero-centering a sparse array/matrix densifies it.")
    sc.pp.scale(adata_p28_work, max_value=10)

sc.tl.pca(
    adata_p28_work,
    n_comps=30,
    svd_solver="arpack",
    random_state=RANDOM_STATE
)

sce.pp.harmony_integrate(
    adata_p28_work,
    key="sample",
    basis="X_pca",
    adjusted_basis="X_pca_harmony",
    max_iter_harmony=20,
    random_state=RANDOM_STATE,
    verbose=False
)


#4. =========== Broad neighbors, UMAP and Leiden ===========

adata_p28_work.obsm["X_pca_harmony_20"] = adata_p28_work.obsm["X_pca_harmony"][:, :20].copy()

sc.pp.neighbors(
    adata_p28_work,
    n_neighbors=12,
    use_rep="X_pca_harmony_20"
)

sc.tl.umap(
    adata_p28_work,
    min_dist=0.35,
    spread=1.0,
    random_state=RANDOM_STATE
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    sc.tl.leiden(
        adata_p28_work,
        resolution=0.4,
        key_added="broad_leiden",
        random_state=RANDOM_STATE
    )

adata_p28_full.obs["broad_leiden"] = adata_p28_work.obs["broad_leiden"].astype(str).values
adata_p28_full.obsm["X_umap"] = adata_p28_work.obsm["X_umap"].copy()

broad_cluster_order = sorted(
    adata_p28_full.obs["broad_leiden"].unique(),
    key=lambda x: int(x) if str(x).isdigit() else str(x)
)
adata_p28_full.obs["broad_leiden"] = pd.Categorical(
    adata_p28_full.obs["broad_leiden"],
    categories=broad_cluster_order,
    ordered=True
)

broad_counts = adata_p28_full.obs["broad_leiden"].value_counts().reindex(broad_cluster_order)
broad_label_map = {
    cluster: f"{cluster} (n={int(broad_counts.loc[cluster])})"
    for cluster in broad_cluster_order
}
adata_p28_full.obs["broad_leiden_n"] = adata_p28_full.obs["broad_leiden"].astype(str).map(broad_label_map)
adata_p28_full.obs["broad_leiden_n"] = pd.Categorical(
    adata_p28_full.obs["broad_leiden_n"],
    categories=[broad_label_map[cluster] for cluster in broad_cluster_order],
    ordered=True
)


#5. =========== Broad cluster QC tables ===========

broad_sample_table = pd.crosstab(adata_p28_full.obs["broad_leiden"], adata_p28_full.obs["sample"])
for sample in ["P28_1", "P28_2"]:
    if sample not in broad_sample_table.columns:
        broad_sample_table[sample] = 0

broad_sample_table["n_cells"] = broad_sample_table["P28_1"] + broad_sample_table["P28_2"]
broad_sample_table["P28_2_frac"] = (broad_sample_table["P28_2"] / broad_sample_table["n_cells"]).round(3)
display(broad_sample_table[["P28_1", "P28_2", "n_cells", "P28_2_frac"]])

broad_qc_table = adata_p28_full.obs.groupby("broad_leiden", observed=True).agg(
    cells=("broad_leiden", "size"),
    median_genes=("n_genes_by_counts", "median"),
    median_counts=("total_counts", "median"),
    median_mt=("pct_counts_mt", "median"),
    median_hb=("pct_counts_hb", "median")
)
display(broad_qc_table.round(2))


#6. =========== Prepare marker plot object ===========

adata_p28_plot = adata_p28_full.copy()
sc.pp.normalize_total(adata_p28_plot, target_sum=1e4)
sc.pp.log1p(adata_p28_plot)

adata_p28_plot.obs["broad_leiden"] = adata_p28_full.obs["broad_leiden"].copy()
adata_p28_plot.obs["broad_leiden_n"] = adata_p28_full.obs["broad_leiden_n"].copy()
adata_p28_plot.obsm["X_umap"] = adata_p28_full.obsm["X_umap"].copy()


#7. =========== Broad UMAP check ===========

sc.pl.umap(
    adata_p28_plot,
    color="broad_leiden_n",
    legend_loc="right margin",
    legend_fontsize=8,
    size=8,
    frameon=False,
    title="Broad leiden",
    show=False
)

sc.pl.umap(
    adata_p28_plot,
    color="broad_leiden",
    legend_loc="on data",
    legend_fontsize=8,
    legend_fontoutline=2,
    size=8,
    frameon=False,
    title="Broad leiden",
    show=False
)

sc.pl.umap(
    adata_p28_plot,
    color="sample",
    legend_loc="right margin",
    size=8,
    frameon=False,
    title="Sample",
    show=False
)


#8. =========== Broad epithelial marker dotplot ===========

def filter_marker_dict(marker_dict, var_names):
    return {
        group: [gene for gene in genes if gene in var_names]
        for group, genes in marker_dict.items()
        if any(gene in var_names for gene in genes)
    }

broad_marker_dict = {
    "Epithelial_core": ["Epcam", "Cdh1", "Krt8", "Krt18", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1", "Gjb2"],
    "HC_OHC_IHC": ["Myo6", "Myo7a", "Pcp4", "Otof", "Slc26a5", "Ocm", "Slc17a8", "Calb2", "Acbd7"],
    "IPhC_IB": ["Slc1a3", "Matn4", "Smpx", "Epyc"],
    "DC_PC": ["Ceacam16", "Rbp1", "Rbp7", "Fabp3", "Prox1", "Lgr6", "Ednrb", "Prdm12"],
    "HeC_OSC_ISC": ["Pmch", "Fst", "Bmp4", "Nupr1", "Gata2", "Gjb6", "Aqp4", "Slc26a4", "Fxyd3", "Dnase1"]
}

sc.pl.dotplot(
    adata_p28_plot,
    filter_marker_dict(broad_marker_dict, adata_p28_plot.var_names),
    groupby="broad_leiden",
    categories_order=broad_cluster_order,
    standard_scale="var",
    cmap="Reds",
    dot_max=0.8,
    figsize=(18, 5),
    show=False
)


#9. =========== Broad exclusion marker dotplot ===========

exclude_marker_dict = {
    "Must_negative": ["Oc90", "Otx2", "Neurod1", "Neurog1"],
    "Immune": ["Ptprc", "C1qa", "C1qb", "Tyrobp", "Aif1", "Lyz2", "Cd74"],
    "Glia": ["Plp1", "Mpz", "Mbp", "Pmp22", "Sox10"],
    "Blood": ["Hba-a1", "Hba-a2", "Hbb-bs", "Hbb-bt", "Hbb-bh1", "Alas2"],
    "Mesenchymal_Endothelial": ["Pdgfra", "Pdgfrb", "Col1a1", "Col3a1", "Dcn", "Pecam1", "Kdr", "Cldn5"],
    "Neuronal": ["Tubb3", "Elavl3", "Elavl4", "Prph", "Snap25"]
}

sc.pl.dotplot(
    adata_p28_plot,
    filter_marker_dict(exclude_marker_dict, adata_p28_plot.var_names),
    groupby="broad_leiden",
    categories_order=broad_cluster_order,
    cmap="Reds",
    dot_max=0.8,
    vmin=0,
    vmax=3,
    figsize=(17, 5),
    show=False
)


plt.close("all")


# Notebook cell 63

#1. =========== Remove broad contaminant clusters ===========

RANDOM_STATE = 0
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
sc.settings.verbosity = 0

drop_broad_clusters = ["0", "3", "5", "11", "12", "13", "14", "15", "16"]
drop_reason = {
    "0": "Mesenchymal_Endothelial",
    "3": "Otx2_positive",
    "5": "Immune",
    "11": "Neuronal",
    "12": "Blood",
    "13": "Glia",
    "14": "Immune",
    "15": "Neuronal",
    "16": "Mesenchymal_Endothelial"
}

drop_table = pd.crosstab(adata_p28_full.obs["broad_leiden"], adata_p28_full.obs["sample"])
for sample in ["P28_1", "P28_2"]:
    if sample not in drop_table.columns:
        drop_table[sample] = 0

drop_table["n_cells"] = drop_table["P28_1"] + drop_table["P28_2"]
drop_table["P28_2_frac"] = (drop_table["P28_2"] / drop_table["n_cells"]).round(3)
drop_table["drop_reason"] = drop_table.index.astype(str).map(drop_reason)
display(drop_table.loc[drop_broad_clusters, ["P28_1", "P28_2", "n_cells", "P28_2_frac", "drop_reason"]])

adata_p28_epi = adata_p28_full[
    ~adata_p28_full.obs["broad_leiden"].astype(str).isin(drop_broad_clusters)
].copy()
adata_p28_epi.layers["counts"] = adata_p28_epi.X.copy()


#2. =========== Normalize and select HVGs ===========

adata_p28_epi_work = adata_p28_epi.copy()
sc.pp.normalize_total(adata_p28_epi_work, target_sum=1e4)
sc.pp.log1p(adata_p28_epi_work)

sc.pp.highly_variable_genes(
    adata_p28_epi_work,
    n_top_genes=2000,
    batch_key="sample",
    flavor="seurat"
)

tech_gene_mask = adata_p28_epi_work.var_names.str.startswith(("mt-", "Mt-", "Rpl", "Rps", "Hba", "Hbb"))
adata_p28_epi_work.var.loc[tech_gene_mask, "highly_variable"] = False
adata_p28_epi_work = adata_p28_epi_work[:, adata_p28_epi_work.var["highly_variable"]].copy()


#3. =========== PCA and Harmony integration ===========

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="zero-centering a sparse array/matrix densifies it.")
    sc.pp.scale(adata_p28_epi_work, max_value=10)

sc.tl.pca(
    adata_p28_epi_work,
    n_comps=30,
    svd_solver="arpack",
    random_state=RANDOM_STATE
)

sce.pp.harmony_integrate(
    adata_p28_epi_work,
    key="sample",
    basis="X_pca",
    adjusted_basis="X_pca_harmony",
    max_iter_harmony=20,
    random_state=RANDOM_STATE,
    verbose=False
)


#4. =========== Compare clustering parameters ===========

def raw_pct_by_cluster(adata, groupby, genes):
    genes = [gene for gene in genes if gene in adata.var_names]
    rows = []

    cluster_order = sorted(
        adata.obs[groupby].astype(str).unique(),
        key=lambda x: int(x) if str(x).isdigit() else str(x)
    )

    for cluster in cluster_order:
        cell_mask = (adata.obs[groupby].astype(str) == cluster).values
        row = {"cluster": cluster, "n_cells": int(cell_mask.sum())}

        if genes:
            X = adata[cell_mask, genes].layers["counts"] if "counts" in adata.layers else adata[cell_mask, genes].X
            X = X.toarray() if sp.issparse(X) else np.asarray(X)

            for i, gene in enumerate(genes):
                row[f"{gene}_pct"] = round(float((X[:, i] > 0).mean() * 100), 2)

        rows.append(row)

    return pd.DataFrame(rows).set_index("cluster")

param_sets = [
    ("p20_n15_r045", 20, 15, 0.45),
    ("p20_n20_r045", 20, 20, 0.45),
    ("p20_n25_r045", 20, 25, 0.45),
    ("p20_n20_r035", 20, 20, 0.35),
    ("p30_n20_r045", 30, 20, 0.45)
]

forbidden_genes = ["Oc90", "Otx2", "Neurod1", "Neurog1"]
param_summary = []
param_plot_keys = []

for tag, n_pcs, n_neighbors, resolution in param_sets:
    rep_key = f"X_pca_harmony_{n_pcs}"
    neighbors_key = f"neighbors_{tag}"
    cluster_key = f"leiden_{tag}"
    umap_key = f"X_umap_{tag}"

    adata_p28_epi_work.obsm[rep_key] = adata_p28_epi_work.obsm["X_pca_harmony"][:, :n_pcs].copy()

    sc.pp.neighbors(
        adata_p28_epi_work,
        n_neighbors=n_neighbors,
        use_rep=rep_key,
        key_added=neighbors_key,
        random_state=RANDOM_STATE
    )

    sc.tl.umap(
        adata_p28_epi_work,
        neighbors_key=neighbors_key,
        min_dist=0.35,
        spread=1.0,
        random_state=RANDOM_STATE
    )
    adata_p28_epi_work.obsm[umap_key] = adata_p28_epi_work.obsm["X_umap"].copy()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        sc.tl.leiden(
            adata_p28_epi_work,
            resolution=resolution,
            neighbors_key=neighbors_key,
            key_added=cluster_key,
            random_state=RANDOM_STATE
        )

    adata_p28_epi.obs[cluster_key] = adata_p28_epi_work.obs[cluster_key].astype(str).values

    sample_table = pd.crosstab(adata_p28_epi.obs[cluster_key], adata_p28_epi.obs["sample"])
    for sample in ["P28_1", "P28_2"]:
        if sample not in sample_table.columns:
            sample_table[sample] = 0

    sample_table["n_cells"] = sample_table["P28_1"] + sample_table["P28_2"]
    sample_table["P28_2_frac"] = sample_table["P28_2"] / sample_table["n_cells"]

    forbidden_table = raw_pct_by_cluster(adata_p28_epi, cluster_key, forbidden_genes)
    forbidden_cols = [col for col in forbidden_table.columns if col.endswith("_pct")]
    max_forbidden_pct = forbidden_table[forbidden_cols].max().max() if forbidden_cols else np.nan

    param_summary.append({
        "setting": tag,
        "n_pcs": n_pcs,
        "n_neighbors": n_neighbors,
        "resolution": resolution,
        "n_clusters": sample_table.shape[0],
        "min_cells": int(sample_table["n_cells"].min()),
        "clusters_lt30": int((sample_table["n_cells"] < 30).sum()),
        "sample_skewed_clusters": int(((sample_table["P28_2_frac"] < 0.20) | (sample_table["P28_2_frac"] > 0.80)).sum()),
        "max_forbidden_pct": round(float(max_forbidden_pct), 2)
    })

    param_plot_keys.append((tag, cluster_key, umap_key))

display(pd.DataFrame(param_summary))


#5. =========== Parameter UMAP check ===========

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes = axes.ravel()

for ax, (tag, cluster_key, umap_key) in zip(axes, param_plot_keys):
    sc.pl.embedding(
        adata_p28_epi_work,
        basis=umap_key.replace("X_", ""),
        color=cluster_key,
        ax=ax,
        show=False,
        frameon=False,
        size=8,
        legend_loc="on data",
        legend_fontsize=8,
        legend_fontoutline=2,
        title=tag
    )

for ax in axes[len(param_plot_keys):]:
    ax.axis("off")

plt.tight_layout()
plt.close("all")


plt.close("all")


# Notebook cell 64

#1. =========== Select candidate parameters for marker check ===========

candidate_tags = ["p20_n20_r045", "p20_n25_r045"]
candidate_cluster_keys = [f"leiden_{tag}" for tag in candidate_tags]
candidate_umap_keys = [f"X_umap_{tag}" for tag in candidate_tags]

for cluster_key in candidate_cluster_keys:
    if cluster_key not in adata_p28_epi.obs.columns:
        raise ValueError(f"{cluster_key} is missing from adata_p28_epi.obs. Run the parameter comparison code first.")

for umap_key in candidate_umap_keys:
    if umap_key not in adata_p28_epi_work.obsm:
        raise ValueError(f"{umap_key} is missing from adata_p28_epi_work.obsm. Run the parameter comparison code first.")


#2. =========== Prepare marker plot object ===========

adata_p28_epi_marker_plot = adata_p28_epi.copy()
sc.pp.normalize_total(adata_p28_epi_marker_plot, target_sum=1e4)
sc.pp.log1p(adata_p28_epi_marker_plot)

candidate_summary = []

for tag, cluster_key in zip(candidate_tags, candidate_cluster_keys):
    cluster_order = sorted(
        adata_p28_epi.obs[cluster_key].astype(str).unique(),
        key=lambda x: int(x) if str(x).isdigit() else str(x)
    )

    adata_p28_epi.obs[cluster_key] = pd.Categorical(
        adata_p28_epi.obs[cluster_key].astype(str),
        categories=cluster_order,
        ordered=True
    )
    adata_p28_epi_marker_plot.obs[cluster_key] = adata_p28_epi.obs[cluster_key].copy()

    cluster_counts = adata_p28_epi.obs[cluster_key].value_counts().reindex(cluster_order)
    cluster_label_map = {
        cluster: f"{cluster} (n={int(cluster_counts.loc[cluster])})"
        for cluster in cluster_order
    }

    label_key = f"{cluster_key}_n"
    adata_p28_epi.obs[label_key] = adata_p28_epi.obs[cluster_key].astype(str).map(cluster_label_map)
    adata_p28_epi_marker_plot.obs[label_key] = adata_p28_epi.obs[label_key].copy()

    sample_table = pd.crosstab(adata_p28_epi.obs[cluster_key], adata_p28_epi.obs["sample"])
    for sample in ["P28_1", "P28_2"]:
        if sample not in sample_table.columns:
            sample_table[sample] = 0

    sample_table["n_cells"] = sample_table["P28_1"] + sample_table["P28_2"]
    sample_table["P28_2_frac"] = (sample_table["P28_2"] / sample_table["n_cells"]).round(3)
    sample_table.insert(0, "setting", tag)
    candidate_summary.append(sample_table[["setting", "P28_1", "P28_2", "n_cells", "P28_2_frac"]])

display(pd.concat(candidate_summary))


#3. =========== Marker dictionaries ===========

p28_marker_dict = {
    "Epithelial_core": ["Epcam", "Cdh1", "Krt8", "Krt18", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1", "Gjb2"],
    "HC_OHC_IHC": ["Myo6", "Myo7a", "Pcp4", "Otof", "Slc26a5", "Ocm", "Slc17a8", "Calb2", "Acbd7"],
    "IPhC_IB": ["Slc1a3", "Matn4", "Smpx", "Epyc"],
    "DC_PC": ["Ceacam16", "Rbp1", "Rbp7", "Fabp3", "Prox1", "Lgr6", "Ednrb", "Prdm12"],
    "HeC_OSC_ISC": ["Pmch", "Fst", "Bmp4", "Nupr1", "Gata2", "Gjb6", "Aqp4", "Slc26a4", "Fxyd3", "Dnase1"]
}

exclude_marker_dict = {
    "Must_negative": ["Oc90", "Otx2", "Neurod1", "Neurog1"],
    "Immune": ["Ptprc", "C1qa", "C1qb", "Tyrobp", "Aif1", "Lyz2", "Cd74"],
    "Glia": ["Plp1", "Mpz", "Mbp", "Pmp22", "Sox10"],
    "Blood": ["Hba-a1", "Hba-a2", "Hbb-bs", "Hbb-bt", "Hbb-bh1", "Alas2"],
    "Mesenchymal_Endothelial": ["Pdgfra", "Pdgfrb", "Col1a1", "Col3a1", "Dcn", "Pecam1", "Kdr", "Cldn5"],
    "Neuronal": ["Tubb3", "Elavl3", "Elavl4", "Prph", "Snap25"]
}


#4. =========== UMAP and dotplot check ===========

for tag, cluster_key, umap_key in zip(candidate_tags, candidate_cluster_keys, candidate_umap_keys):
    cluster_order = list(adata_p28_epi.obs[cluster_key].cat.categories)
    label_key = f"{cluster_key}_n"

    adata_p28_epi_marker_plot.obsm["X_umap"] = adata_p28_epi_work[
        adata_p28_epi_marker_plot.obs_names, :
    ].obsm[umap_key].copy()

    sc.pl.umap(
        adata_p28_epi_marker_plot,
        color=label_key,
        legend_loc="right margin",
        legend_fontsize=8,
        size=8,
        frameon=False,
        title=f"{tag}: clusters",
        show=False
    )

    sc.pl.umap(
        adata_p28_epi_marker_plot,
        color=cluster_key,
        legend_loc="on data",
        legend_fontsize=8,
        legend_fontoutline=2,
        size=8,
        frameon=False,
        title=f"{tag}: on data",
        show=False
    )

    sc.pl.umap(
        adata_p28_epi_marker_plot,
        color="sample",
        legend_loc="right margin",
        size=8,
        frameon=False,
        title=f"{tag}: sample",
        show=False
    )

    sc.pl.dotplot(
        adata_p28_epi_marker_plot,
        filter_marker_dict(p28_marker_dict, adata_p28_epi_marker_plot.var_names),
        groupby=cluster_key,
        categories_order=cluster_order,
        standard_scale="var",
        cmap="Reds",
        dot_max=0.8,
        figsize=(18, 4.6),
        show=False
    )

    sc.pl.dotplot(
        adata_p28_epi_marker_plot,
        filter_marker_dict(exclude_marker_dict, adata_p28_epi_marker_plot.var_names),
        groupby=cluster_key,
        categories_order=cluster_order,
        cmap="Reds",
        dot_max=0.8,
        vmin=0,
        vmax=3,
        figsize=(17, 4.6),
        show=False
    )


plt.close("all")


# Notebook cell 65

#1. =========== Select p20_n20_r045 candidate ===========

RANDOM_STATE = 0
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
sc.settings.verbosity = 0

selected_tag = "p20_n20_r045"
selected_cluster_key = f"leiden_{selected_tag}"
selected_umap_key = f"X_umap_{selected_tag}"

adata_p28_epi_v2 = adata_p28_epi.copy()
adata_p28_epi_v2.obs["p28_leiden"] = adata_p28_epi.obs[selected_cluster_key].astype(str).values
adata_p28_epi_v2.obsm["X_umap"] = adata_p28_epi_work[adata_p28_epi_v2.obs_names, :].obsm[selected_umap_key].copy()

p28_cluster_order = sorted(
    adata_p28_epi_v2.obs["p28_leiden"].unique(),
    key=lambda x: int(x) if str(x).isdigit() else str(x)
)
adata_p28_epi_v2.obs["p28_leiden"] = pd.Categorical(
    adata_p28_epi_v2.obs["p28_leiden"],
    categories=p28_cluster_order,
    ordered=True
)

cluster_counts = adata_p28_epi_v2.obs["p28_leiden"].value_counts().reindex(p28_cluster_order)
cluster_label_map = {
    cluster: f"{cluster} (n={int(cluster_counts.loc[cluster])})"
    for cluster in p28_cluster_order
}
adata_p28_epi_v2.obs["p28_leiden_n"] = adata_p28_epi_v2.obs["p28_leiden"].astype(str).map(cluster_label_map)
adata_p28_epi_v2.obs["p28_leiden_n"] = pd.Categorical(
    adata_p28_epi_v2.obs["p28_leiden_n"],
    categories=[cluster_label_map[cluster] for cluster in p28_cluster_order],
    ordered=True
)


#2. =========== Prepare marker plot object ===========

adata_p28_epi_v2_plot = adata_p28_epi_v2.copy()
sc.pp.normalize_total(adata_p28_epi_v2_plot, target_sum=1e4)
sc.pp.log1p(adata_p28_epi_v2_plot)

adata_p28_epi_v2_plot.obs["p28_leiden"] = adata_p28_epi_v2.obs["p28_leiden"].copy()
adata_p28_epi_v2_plot.obs["p28_leiden_n"] = adata_p28_epi_v2.obs["p28_leiden_n"].copy()
adata_p28_epi_v2_plot.obsm["X_umap"] = adata_p28_epi_v2.obsm["X_umap"].copy()


#3. =========== Marker score table ===========

score_marker_dict = {
    "OHC": ["Myo7a", "Pcp4", "Otof", "Slc26a5", "Ocm"],
    "IHC": ["Myo6", "Myo7a", "Pcp4", "Otof", "Slc17a8", "Calb2", "Acbd7"],
    "IPhC_IB": ["Sox2", "Gata3", "Slc1a3", "Matn4", "Smpx", "Epyc"],
    "DC": ["Lgr5", "Prox1", "Ceacam16", "Rbp1", "Rbp7", "Fabp3", "Gjb2"],
    "PC": ["Smpx", "Lgr6", "Ednrb", "Prdm12", "Gjb2", "Sox2"],
    "HeC_OSC_ISC": ["Gjb2", "Gjb6", "Aqp4", "Epyc", "Slc26a4", "Fxyd3", "Dnase1", "Pmch", "Fst", "Bmp4", "Nupr1", "Gata2"]
}

score_rows = []

for cluster in p28_cluster_order:
    cell_mask = (adata_p28_epi_v2_plot.obs["p28_leiden"].astype(str) == cluster).values
    row = {"cluster": cluster, "n_cells": int(cell_mask.sum())}

    for marker_group, genes in score_marker_dict.items():
        genes = [gene for gene in genes if gene in adata_p28_epi_v2_plot.var_names]
        if len(genes) == 0:
            row[marker_group] = np.nan
            continue

        X = adata_p28_epi_v2_plot[cell_mask, genes].X
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        row[marker_group] = float(X.mean())

    score_rows.append(row)

score_table = pd.DataFrame(score_rows).set_index("cluster")
score_cols = list(score_marker_dict.keys())
score_table["dominant_marker_group"] = score_table[score_cols].idxmax(axis=1)

group_order = {
    "OHC": 0,
    "IHC": 1,
    "IPhC_IB": 2,
    "DC": 3,
    "PC": 4,
    "HeC_OSC_ISC": 5
}
p28_marker_order = sorted(
    p28_cluster_order,
    key=lambda cluster: (
        group_order.get(score_table.loc[cluster, "dominant_marker_group"], 99),
        -score_table.loc[cluster, score_table.loc[cluster, "dominant_marker_group"]]
    )
)

display(score_table.round(2))


#4. =========== Compact UMAP check ===========

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

sc.pl.umap(
    adata_p28_epi_v2_plot,
    color="p28_leiden_n",
    legend_loc="right margin",
    legend_fontsize=8,
    size=8,
    frameon=False,
    title="P28 clusters",
    ax=axes[0],
    show=False
)

sc.pl.umap(
    adata_p28_epi_v2_plot,
    color="p28_leiden",
    legend_loc="on data",
    legend_fontsize=8,
    legend_fontoutline=2,
    size=8,
    frameon=False,
    title="P28 on data",
    ax=axes[1],
    show=False
)

sc.pl.umap(
    adata_p28_epi_v2_plot,
    color="sample",
    legend_loc="right margin",
    size=8,
    frameon=False,
    title="Sample",
    ax=axes[2],
    show=False
)

plt.tight_layout()
plt.close("all")


#5. =========== Focused marker dotplot ===========

p28_marker_dict = {
    "Epithelial_core": ["Epcam", "Cdh1", "Krt8", "Krt18", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1", "Gjb2"],
    "OHC": ["Myo7a", "Pcp4", "Otof", "Slc26a5", "Ocm"],
    "IHC": ["Myo6", "Myo7a", "Pcp4", "Otof", "Slc17a8", "Calb2", "Acbd7"],
    "IPhC_IB": ["Sox2", "Gata3", "Slc1a3", "Matn4", "Smpx", "Epyc"],
    "DC": ["Lgr5", "Prox1", "Ceacam16", "Rbp1", "Rbp7", "Fabp3", "Gjb2"],
    "PC": ["Smpx", "Lgr6", "Ednrb", "Prdm12", "Gjb2", "Sox2"],
    "HeC_OSC_ISC": ["Gjb2", "Gjb6", "Aqp4", "Epyc", "Slc26a4", "Fxyd3", "Dnase1", "Pmch", "Fst", "Bmp4", "Nupr1", "Gata2"]
}

sc.pl.dotplot(
    adata_p28_epi_v2_plot,
    filter_marker_dict(p28_marker_dict, adata_p28_epi_v2_plot.var_names),
    groupby="p28_leiden",
    categories_order=p28_marker_order,
    standard_scale="var",
    cmap="Reds",
    dot_max=0.8,
    figsize=(18, 4.8),
    show=False
)


#6. =========== Exclusion marker dotplot ===========

exclude_marker_dict = {
    "Must_negative": ["Oc90", "Otx2", "Neurod1", "Neurog1"],
    "Immune": ["Ptprc", "C1qa", "C1qb", "Tyrobp", "Aif1", "Lyz2", "Cd74"],
    "Glia": ["Plp1", "Mpz", "Mbp", "Pmp22", "Sox10"],
    "Blood": ["Hba-a1", "Hba-a2", "Hbb-bs", "Hbb-bt", "Hbb-bh1", "Alas2"],
    "Mesenchymal_Endothelial": ["Pdgfra", "Pdgfrb", "Col1a1", "Col3a1", "Dcn", "Pecam1", "Kdr", "Cldn5"],
    "Neuronal": ["Tubb3", "Elavl3", "Elavl4", "Prph", "Snap25"]
}

sc.pl.dotplot(
    adata_p28_epi_v2_plot,
    filter_marker_dict(exclude_marker_dict, adata_p28_epi_v2_plot.var_names),
    groupby="p28_leiden",
    categories_order=p28_marker_order,
    cmap="Reds",
    dot_max=0.8,
    vmin=0,
    vmax=3,
    figsize=(17, 4.8),
    show=False
)


plt.close("all")


# Notebook cell 66

#1. =========== Split HC and supporting epithelial cells ===========

RANDOM_STATE = 0
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
sc.settings.verbosity = 0

hc_clusters = ["5", "10"]

adata_p28_hc = adata_p28_epi_v2[
    adata_p28_epi_v2.obs["p28_leiden"].astype(str).isin(hc_clusters)
].copy()

adata_p28_support = adata_p28_epi_v2[
    ~adata_p28_epi_v2.obs["p28_leiden"].astype(str).isin(hc_clusters)
].copy()

adata_p28_hc.layers["counts"] = adata_p28_hc.X.copy()
adata_p28_support.layers["counts"] = adata_p28_support.X.copy()

split_summary = pd.DataFrame({
    "group": ["HC", "Support"],
    "n_cells": [adata_p28_hc.n_obs, adata_p28_support.n_obs]
})
display(split_summary)


#2. =========== Normalize and select HVGs for support cells ===========

adata_p28_support_work = adata_p28_support.copy()

sc.pp.normalize_total(adata_p28_support_work, target_sum=1e4)
sc.pp.log1p(adata_p28_support_work)

sc.pp.highly_variable_genes(
    adata_p28_support_work,
    n_top_genes=2000,
    batch_key="sample",
    flavor="seurat"
)

tech_gene_mask = adata_p28_support_work.var_names.str.startswith(("mt-", "Mt-", "Rpl", "Rps", "Hba", "Hbb"))
adata_p28_support_work.var.loc[tech_gene_mask, "highly_variable"] = False
adata_p28_support_work = adata_p28_support_work[:, adata_p28_support_work.var["highly_variable"]].copy()


#3. =========== PCA and Harmony for support cells ===========

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="zero-centering a sparse array/matrix densifies it.")
    sc.pp.scale(adata_p28_support_work, max_value=10)

sc.tl.pca(
    adata_p28_support_work,
    n_comps=30,
    svd_solver="arpack",
    random_state=RANDOM_STATE
)

sce.pp.harmony_integrate(
    adata_p28_support_work,
    key="sample",
    basis="X_pca",
    adjusted_basis="X_pca_harmony",
    max_iter_harmony=20,
    random_state=RANDOM_STATE,
    verbose=False
)


#4. =========== Compare support-cell clustering parameters ===========

support_param_sets = [
    ("s20_n15_r035", 20, 15, 0.35),
    ("s20_n20_r035", 20, 20, 0.35),
    ("s20_n20_r045", 20, 20, 0.45),
    ("s20_n25_r035", 20, 25, 0.35),
    ("s30_n20_r035", 30, 20, 0.35)
]

support_forbidden_genes = ["Oc90", "Otx2", "Neurod1", "Neurog1"]
support_param_summary = []
support_plot_keys = []

for tag, n_pcs, n_neighbors, resolution in support_param_sets:
    rep_key = f"X_pca_harmony_{n_pcs}"
    neighbors_key = f"neighbors_{tag}"
    cluster_key = f"support_leiden_{tag}"
    umap_key = f"X_umap_{tag}"

    adata_p28_support_work.obsm[rep_key] = adata_p28_support_work.obsm["X_pca_harmony"][:, :n_pcs].copy()

    sc.pp.neighbors(
        adata_p28_support_work,
        n_neighbors=n_neighbors,
        use_rep=rep_key,
        key_added=neighbors_key,
        random_state=RANDOM_STATE
    )

    sc.tl.umap(
        adata_p28_support_work,
        neighbors_key=neighbors_key,
        min_dist=0.35,
        spread=1.0,
        random_state=RANDOM_STATE
    )
    adata_p28_support_work.obsm[umap_key] = adata_p28_support_work.obsm["X_umap"].copy()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        sc.tl.leiden(
            adata_p28_support_work,
            resolution=resolution,
            neighbors_key=neighbors_key,
            key_added=cluster_key,
            random_state=RANDOM_STATE
        )

    adata_p28_support.obs[cluster_key] = adata_p28_support_work.obs[cluster_key].astype(str).values

    sample_table = pd.crosstab(adata_p28_support.obs[cluster_key], adata_p28_support.obs["sample"])
    for sample in ["P28_1", "P28_2"]:
        if sample not in sample_table.columns:
            sample_table[sample] = 0

    sample_table["n_cells"] = sample_table["P28_1"] + sample_table["P28_2"]
    sample_table["P28_2_frac"] = sample_table["P28_2"] / sample_table["n_cells"]

    forbidden_table = raw_pct_by_cluster(adata_p28_support, cluster_key, support_forbidden_genes)
    forbidden_cols = [col for col in forbidden_table.columns if col.endswith("_pct")]
    max_forbidden_pct = forbidden_table[forbidden_cols].max().max() if forbidden_cols else np.nan

    support_param_summary.append({
        "setting": tag,
        "n_pcs": n_pcs,
        "n_neighbors": n_neighbors,
        "resolution": resolution,
        "n_clusters": sample_table.shape[0],
        "min_cells": int(sample_table["n_cells"].min()),
        "clusters_lt30": int((sample_table["n_cells"] < 30).sum()),
        "sample_skewed_clusters": int(((sample_table["P28_2_frac"] < 0.20) | (sample_table["P28_2_frac"] > 0.80)).sum()),
        "max_forbidden_pct": round(float(max_forbidden_pct), 2)
    })

    support_plot_keys.append((tag, cluster_key, umap_key))

display(pd.DataFrame(support_param_summary))


#5. =========== Compact support-cell UMAP parameter check ===========

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes = axes.ravel()

for ax, (tag, cluster_key, umap_key) in zip(axes, support_plot_keys):
    sc.pl.embedding(
        adata_p28_support_work,
        basis=umap_key.replace("X_", ""),
        color=cluster_key,
        ax=ax,
        show=False,
        frameon=False,
        size=8,
        legend_loc="on data",
        legend_fontsize=8,
        legend_fontoutline=2,
        title=tag
    )

for ax in axes[len(support_plot_keys):]:
    ax.axis("off")

plt.tight_layout()
plt.close("all")


plt.close("all")


# Notebook cell 67

#1. =========== Select support-cell clustering result ===========

RANDOM_STATE = 0
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
sc.settings.verbosity = 0

support_selected_tag = "s20_n25_r035"
support_cluster_key = f"support_leiden_{support_selected_tag}"
support_umap_key = f"X_umap_{support_selected_tag}"

if support_cluster_key not in adata_p28_support.obs.columns:
    raise ValueError(f"{support_cluster_key} is missing from adata_p28_support.obs. Run the support-cell parameter comparison code first.")

if support_umap_key not in adata_p28_support_work.obsm:
    raise ValueError(f"{support_umap_key} is missing from adata_p28_support_work.obsm. Run the support-cell parameter comparison code first.")

adata_p28_support_v2 = adata_p28_support.copy()
adata_p28_support_v2.obs["support_leiden"] = adata_p28_support.obs[support_cluster_key].astype(str).values
adata_p28_support_v2.obsm["X_umap"] = adata_p28_support_work[
    adata_p28_support_v2.obs_names, :
].obsm[support_umap_key].copy()

support_cluster_order = sorted(
    adata_p28_support_v2.obs["support_leiden"].unique(),
    key=lambda x: int(x) if str(x).isdigit() else str(x)
)
adata_p28_support_v2.obs["support_leiden"] = pd.Categorical(
    adata_p28_support_v2.obs["support_leiden"],
    categories=support_cluster_order,
    ordered=True
)

support_counts = adata_p28_support_v2.obs["support_leiden"].value_counts().reindex(support_cluster_order)
support_label_map = {
    cluster: f"{cluster} (n={int(support_counts.loc[cluster])})"
    for cluster in support_cluster_order
}
adata_p28_support_v2.obs["support_leiden_n"] = adata_p28_support_v2.obs["support_leiden"].astype(str).map(support_label_map)
adata_p28_support_v2.obs["support_leiden_n"] = pd.Categorical(
    adata_p28_support_v2.obs["support_leiden_n"],
    categories=[support_label_map[cluster] for cluster in support_cluster_order],
    ordered=True
)


#2. =========== Prepare support marker plot object ===========

adata_p28_support_v2_plot = adata_p28_support_v2.copy()
sc.pp.normalize_total(adata_p28_support_v2_plot, target_sum=1e4)
sc.pp.log1p(adata_p28_support_v2_plot)

adata_p28_support_v2_plot.obs["support_leiden"] = adata_p28_support_v2.obs["support_leiden"].copy()
adata_p28_support_v2_plot.obs["support_leiden_n"] = adata_p28_support_v2.obs["support_leiden_n"].copy()
adata_p28_support_v2_plot.obsm["X_umap"] = adata_p28_support_v2.obsm["X_umap"].copy()


#3. =========== Support-cell sample table and marker score ===========

support_sample_table = pd.crosstab(
    adata_p28_support_v2.obs["support_leiden"],
    adata_p28_support_v2.obs["sample"]
)
for sample in ["P28_1", "P28_2"]:
    if sample not in support_sample_table.columns:
        support_sample_table[sample] = 0

support_sample_table["n_cells"] = support_sample_table["P28_1"] + support_sample_table["P28_2"]
support_sample_table["P28_2_frac"] = (support_sample_table["P28_2"] / support_sample_table["n_cells"]).round(3)
display(support_sample_table[["P28_1", "P28_2", "n_cells", "P28_2_frac"]])

support_score_marker_dict = {
    "IPhC_IB": ["Sox2", "Gata3", "Slc1a3", "Matn4", "Smpx", "Epyc"],
    "DC": ["Lgr5", "Prox1", "Ceacam16", "Rbp1", "Rbp7", "Fabp3", "Gjb2"],
    "PC": ["Smpx", "Lgr6", "Ednrb", "Prdm12", "Gjb2", "Sox2"],
    "HeC_OSC_ISC": ["Gjb2", "Gjb6", "Aqp4", "Epyc", "Slc26a4", "Fxyd3", "Dnase1", "Pmch", "Fst", "Bmp4", "Nupr1", "Gata2"]
}

support_score_rows = []

for cluster in support_cluster_order:
    cell_mask = (adata_p28_support_v2_plot.obs["support_leiden"].astype(str) == cluster).values
    row = {"cluster": cluster, "n_cells": int(cell_mask.sum())}

    for marker_group, genes in support_score_marker_dict.items():
        genes = [gene for gene in genes if gene in adata_p28_support_v2_plot.var_names]
        if len(genes) == 0:
            row[marker_group] = np.nan
            continue

        X = adata_p28_support_v2_plot[cell_mask, genes].X
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        row[marker_group] = float(X.mean())

    support_score_rows.append(row)

support_score_table = pd.DataFrame(support_score_rows).set_index("cluster")
support_score_cols = list(support_score_marker_dict.keys())
support_score_table["dominant_marker_group"] = support_score_table[support_score_cols].idxmax(axis=1)

support_group_order = {
    "IPhC_IB": 0,
    "DC": 1,
    "PC": 2,
    "HeC_OSC_ISC": 3
}
support_marker_order = sorted(
    support_cluster_order,
    key=lambda cluster: (
        support_group_order.get(support_score_table.loc[cluster, "dominant_marker_group"], 99),
        -support_score_table.loc[cluster, support_score_table.loc[cluster, "dominant_marker_group"]]
    )
)

display(support_score_table.round(2))


#4. =========== Compact support UMAP check ===========

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

sc.pl.umap(
    adata_p28_support_v2_plot,
    color="support_leiden_n",
    legend_loc="right margin",
    legend_fontsize=8,
    size=8,
    frameon=False,
    title="Support clusters",
    ax=axes[0],
    show=False
)

sc.pl.umap(
    adata_p28_support_v2_plot,
    color="support_leiden",
    legend_loc="on data",
    legend_fontsize=8,
    legend_fontoutline=2,
    size=8,
    frameon=False,
    title="Support on data",
    ax=axes[1],
    show=False
)

sc.pl.umap(
    adata_p28_support_v2_plot,
    color="sample",
    legend_loc="right margin",
    size=8,
    frameon=False,
    title="Sample",
    ax=axes[2],
    show=False
)

plt.tight_layout()
plt.close("all")


#5. =========== Support-cell marker dotplot ===========

support_marker_dict = {
    "Epithelial_core": ["Epcam", "Cdh1", "Krt8", "Krt18", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1", "Gjb2"],
    "IPhC_IB": ["Sox2", "Gata3", "Slc1a3", "Matn4", "Smpx", "Epyc"],
    "DC": ["Lgr5", "Prox1", "Ceacam16", "Rbp1", "Rbp7", "Fabp3", "Gjb2"],
    "PC": ["Smpx", "Lgr6", "Ednrb", "Prdm12", "Gjb2", "Sox2"],
    "HeC_OSC_ISC": ["Gjb2", "Gjb6", "Aqp4", "Epyc", "Slc26a4", "Fxyd3", "Dnase1", "Pmch", "Fst", "Bmp4", "Nupr1", "Gata2"],
    "HC_check": ["Myo6", "Myo7a", "Pcp4", "Otof", "Slc26a5", "Ocm", "Slc17a8"]
}

sc.pl.dotplot(
    adata_p28_support_v2_plot,
    filter_marker_dict(support_marker_dict, adata_p28_support_v2_plot.var_names),
    groupby="support_leiden",
    categories_order=support_marker_order,
    standard_scale="var",
    cmap="Reds",
    dot_max=0.8,
    figsize=(17, 4.5),
    show=False
)


#6. =========== Support-cell exclusion marker dotplot ===========

support_exclude_marker_dict = {
    "Must_negative": ["Oc90", "Otx2", "Neurod1", "Neurog1"],
    "Immune": ["Ptprc", "C1qa", "C1qb", "Tyrobp", "Aif1", "Lyz2", "Cd74"],
    "Glia": ["Plp1", "Mpz", "Mbp", "Pmp22", "Sox10"],
    "Blood": ["Hba-a1", "Hba-a2", "Hbb-bs", "Hbb-bt", "Hbb-bh1", "Alas2"],
    "Mesenchymal_Endothelial": ["Pdgfra", "Pdgfrb", "Col1a1", "Col3a1", "Dcn", "Pecam1", "Kdr", "Cldn5"],
    "Neuronal": ["Tubb3", "Elavl3", "Elavl4", "Prph", "Snap25"]
}

sc.pl.dotplot(
    adata_p28_support_v2_plot,
    filter_marker_dict(support_exclude_marker_dict, adata_p28_support_v2_plot.var_names),
    groupby="support_leiden",
    categories_order=support_marker_order,
    cmap="Reds",
    dot_max=0.8,
    vmin=0,
    vmax=3,
    figsize=(16, 4.5),
    show=False
)


plt.close("all")


# Notebook cell 68

#1. =========== Support-cell differential marker check ===========

RANDOM_STATE = 0
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
sc.settings.verbosity = 0

sc.tl.rank_genes_groups(
    adata_p28_support_v2_plot,
    groupby="support_leiden",
    method="wilcoxon",
    n_genes=40,
    use_raw=False
)

support_top_marker_rows = []

for cluster in support_cluster_order:
    marker_df = sc.get.rank_genes_groups_df(
        adata_p28_support_v2_plot,
        group=str(cluster)
    )

    marker_df = marker_df[
        ~marker_df["names"].str.startswith(("mt-", "Mt-", "Rpl", "Rps", "Hba", "Hbb"))
    ].head(8)

    support_top_marker_rows.append({
        "cluster": cluster,
        "top_genes": ", ".join(marker_df["names"].astype(str).tolist())
    })

support_top_marker_table = pd.DataFrame(support_top_marker_rows).set_index("cluster")
display(support_top_marker_table)


#2. =========== Raw marker percentage table ===========

support_key_marker_dict = {
    "IPhC_IB": ["Slc1a3", "Sox2", "Gata3", "Matn4", "Smpx", "Epyc"],
    "DC": ["Lgr5", "Fgfr3", "Prox1", "Ceacam16", "Rbp1", "Rbp7", "Fabp3"],
    "PC": ["Smpx", "Lgr6", "Ednrb", "Prdm12"],
    "HeC_OSC_ISC": ["Gjb2", "Gjb6", "Aqp4", "Slc26a4", "Fxyd3", "Dnase1", "Pmch", "Fst", "Bmp4", "Nupr1", "Gata2"],
    "HC_residual": ["Myo6", "Myo7a", "Pcp4", "Otof", "Slc26a5", "Ocm", "Slc17a8"],
    "Must_negative": ["Oc90", "Otx2", "Neurod1", "Neurog1"]
}

support_key_genes = []
for genes in support_key_marker_dict.values():
    support_key_genes.extend(genes)

support_key_genes = [
    gene for gene in dict.fromkeys(support_key_genes)
    if gene in adata_p28_support_v2.var_names
]

support_raw_pct_rows = []

for cluster in support_cluster_order:
    cell_mask = (adata_p28_support_v2.obs["support_leiden"].astype(str) == str(cluster)).values
    row = {"cluster": cluster, "n_cells": int(cell_mask.sum())}

    X = adata_p28_support_v2[cell_mask, support_key_genes].layers["counts"]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)

    for i, gene in enumerate(support_key_genes):
        row[f"{gene}_pct"] = round(float((X[:, i] > 0).mean() * 100), 1)

    support_raw_pct_rows.append(row)

support_raw_pct_table = pd.DataFrame(support_raw_pct_rows).set_index("cluster")
display(support_raw_pct_table)


#3. =========== Compact support UMAP with original cluster reference ===========

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

sc.pl.umap(
    adata_p28_support_v2_plot,
    color="support_leiden",
    legend_loc="on data",
    legend_fontsize=8,
    legend_fontoutline=2,
    size=8,
    frameon=False,
    title="Support leiden",
    ax=axes[0],
    show=False
)

sc.pl.umap(
    adata_p28_support_v2_plot,
    color="p28_leiden",
    legend_loc="on data",
    legend_fontsize=8,
    legend_fontoutline=2,
    size=8,
    frameon=False,
    title="Original P28 leiden",
    ax=axes[1],
    show=False
)

sc.pl.umap(
    adata_p28_support_v2_plot,
    color="sample",
    legend_loc="right margin",
    size=8,
    frameon=False,
    title="Sample",
    ax=axes[2],
    show=False
)

plt.tight_layout()
plt.close("all")


#4. =========== Focused support marker dotplot ===========

support_focused_marker_dict = {
    "IPhC_IB": ["Slc1a3", "Sox2", "Gata3", "Matn4", "Smpx", "Epyc"],
    "DC": ["Lgr5", "Fgfr3", "Prox1", "Ceacam16", "Rbp1", "Rbp7", "Fabp3"],
    "PC": ["Smpx", "Lgr6", "Ednrb", "Prdm12"],
    "HeC_OSC_ISC": ["Gjb2", "Gjb6", "Aqp4", "Slc26a4", "Fxyd3", "Dnase1", "Pmch", "Fst", "Bmp4", "Nupr1", "Gata2"],
    "HC_residual": ["Myo6", "Myo7a", "Pcp4", "Otof", "Slc26a5", "Ocm", "Slc17a8"],
    "Must_negative": ["Oc90", "Otx2", "Neurod1", "Neurog1"]
}

sc.pl.dotplot(
    adata_p28_support_v2_plot,
    filter_marker_dict(support_focused_marker_dict, adata_p28_support_v2_plot.var_names),
    groupby="support_leiden",
    categories_order=support_marker_order,
    cmap="Reds",
    dot_max=0.8,
    vmin=0,
    vmax=3,
    figsize=(17, 4.5),
    show=False
)


plt.close("all")


# Notebook cell 69

#1. =========== Remove residual non-sensory support clusters ===========

RANDOM_STATE = 0
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
sc.settings.verbosity = 0

drop_support_clusters = ["0", "3", "4"]
drop_support_reason = {
    "0": "Emcn/Cav1/Fn1 positive, vascular/mesenchymal-like",
    "3": "Ogn/Apoe/Apod positive, mesenchymal-like",
    "4": "Spp1/Coch/Trf/Apod positive, non-sensory-like"
}

drop_support_table = support_sample_table.copy()
drop_support_table["drop_reason"] = drop_support_table.index.astype(str).map(drop_support_reason)
display(drop_support_table.loc[drop_support_clusters, ["P28_1", "P28_2", "n_cells", "P28_2_frac", "drop_reason"]])

adata_p28_support_clean = adata_p28_support_v2[
    ~adata_p28_support_v2.obs["support_leiden"].astype(str).isin(drop_support_clusters)
].copy()
adata_p28_support_clean.layers["counts"] = adata_p28_support_clean.X.copy()

adata_p28_hc_clean = adata_p28_hc.copy()
adata_p28_hc_clean.layers["counts"] = adata_p28_hc_clean.X.copy()

adata_p28_hc_clean.obs["p28_major_group"] = adata_p28_hc_clean.obs["p28_leiden"].astype(str).map({
    "5": "OHC",
    "10": "IHC"
})
adata_p28_support_clean.obs["p28_major_group"] = "Support"

adata_p28_sensory_candidate = sc.concat(
    [adata_p28_hc_clean, adata_p28_support_clean],
    join="outer",
    label=None,
    index_unique=None
)
adata_p28_sensory_candidate.layers["counts"] = adata_p28_sensory_candidate.X.copy()


#2. =========== Prepare sensory candidate plot object ===========

adata_p28_sensory_plot = adata_p28_sensory_candidate.copy()
sc.pp.normalize_total(adata_p28_sensory_plot, target_sum=1e4)
sc.pp.log1p(adata_p28_sensory_plot)

candidate_group_table = pd.crosstab(
    adata_p28_sensory_candidate.obs["p28_major_group"],
    adata_p28_sensory_candidate.obs["sample"]
)
for sample in ["P28_1", "P28_2"]:
    if sample not in candidate_group_table.columns:
        candidate_group_table[sample] = 0

candidate_group_table["n_cells"] = candidate_group_table["P28_1"] + candidate_group_table["P28_2"]
candidate_group_table["P28_2_frac"] = (candidate_group_table["P28_2"] / candidate_group_table["n_cells"]).round(3)
display(candidate_group_table[["P28_1", "P28_2", "n_cells", "P28_2_frac"]])


#3. =========== Recalculate UMAP for sensory candidate check ===========

adata_p28_sensory_work = adata_p28_sensory_candidate.copy()

sc.pp.normalize_total(adata_p28_sensory_work, target_sum=1e4)
sc.pp.log1p(adata_p28_sensory_work)

sc.pp.highly_variable_genes(
    adata_p28_sensory_work,
    n_top_genes=2000,
    batch_key="sample",
    flavor="seurat"
)

tech_gene_mask = adata_p28_sensory_work.var_names.str.startswith(("mt-", "Mt-", "Rpl", "Rps", "Hba", "Hbb"))
adata_p28_sensory_work.var.loc[tech_gene_mask, "highly_variable"] = False
adata_p28_sensory_work = adata_p28_sensory_work[:, adata_p28_sensory_work.var["highly_variable"]].copy()

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="zero-centering a sparse array/matrix densifies it.")
    sc.pp.scale(adata_p28_sensory_work, max_value=10)

sc.tl.pca(
    adata_p28_sensory_work,
    n_comps=30,
    svd_solver="arpack",
    random_state=RANDOM_STATE
)

sce.pp.harmony_integrate(
    adata_p28_sensory_work,
    key="sample",
    basis="X_pca",
    adjusted_basis="X_pca_harmony",
    max_iter_harmony=20,
    random_state=RANDOM_STATE,
    verbose=False
)

adata_p28_sensory_work.obsm["X_pca_harmony_20"] = adata_p28_sensory_work.obsm["X_pca_harmony"][:, :20].copy()

sc.pp.neighbors(
    adata_p28_sensory_work,
    n_neighbors=20,
    use_rep="X_pca_harmony_20",
    random_state=RANDOM_STATE
)

sc.tl.umap(
    adata_p28_sensory_work,
    min_dist=0.35,
    spread=1.0,
    random_state=RANDOM_STATE
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    sc.tl.leiden(
        adata_p28_sensory_work,
        resolution=0.35,
        key_added="sensory_leiden",
        random_state=RANDOM_STATE
    )

adata_p28_sensory_candidate.obs["sensory_leiden"] = adata_p28_sensory_work.obs["sensory_leiden"].astype(str).values
adata_p28_sensory_candidate.obsm["X_umap"] = adata_p28_sensory_work.obsm["X_umap"].copy()

adata_p28_sensory_plot.obs["sensory_leiden"] = adata_p28_sensory_candidate.obs["sensory_leiden"].copy()
adata_p28_sensory_plot.obsm["X_umap"] = adata_p28_sensory_candidate.obsm["X_umap"].copy()

sensory_cluster_order = sorted(
    adata_p28_sensory_candidate.obs["sensory_leiden"].unique(),
    key=lambda x: int(x) if str(x).isdigit() else str(x)
)
adata_p28_sensory_candidate.obs["sensory_leiden"] = pd.Categorical(
    adata_p28_sensory_candidate.obs["sensory_leiden"],
    categories=sensory_cluster_order,
    ordered=True
)
adata_p28_sensory_plot.obs["sensory_leiden"] = adata_p28_sensory_candidate.obs["sensory_leiden"].copy()

sensory_counts = adata_p28_sensory_candidate.obs["sensory_leiden"].value_counts().reindex(sensory_cluster_order)
sensory_label_map = {
    cluster: f"{cluster} (n={int(sensory_counts.loc[cluster])})"
    for cluster in sensory_cluster_order
}
adata_p28_sensory_plot.obs["sensory_leiden_n"] = adata_p28_sensory_plot.obs["sensory_leiden"].astype(str).map(sensory_label_map)


#4. =========== Compact sensory candidate UMAP check ===========

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

sc.pl.umap(
    adata_p28_sensory_plot,
    color="sensory_leiden_n",
    legend_loc="right margin",
    legend_fontsize=8,
    size=8,
    frameon=False,
    title="Sensory candidate clusters",
    ax=axes[0],
    show=False
)

sc.pl.umap(
    adata_p28_sensory_plot,
    color="sensory_leiden",
    legend_loc="on data",
    legend_fontsize=8,
    legend_fontoutline=2,
    size=8,
    frameon=False,
    title="Sensory on data",
    ax=axes[1],
    show=False
)

sc.pl.umap(
    adata_p28_sensory_plot,
    color="sample",
    legend_loc="right margin",
    size=8,
    frameon=False,
    title="Sample",
    ax=axes[2],
    show=False
)

plt.tight_layout()
plt.close("all")


#5. =========== Sensory candidate marker dotplot ===========

sensory_marker_dict = {
    "Epithelial_core": ["Epcam", "Cdh1", "Krt8", "Krt18", "Gata3", "Sox2", "Sox9", "Lgr5", "Isl1", "Gjb2"],
    "OHC": ["Myo7a", "Pcp4", "Otof", "Slc26a5", "Ocm"],
    "IHC": ["Myo6", "Myo7a", "Pcp4", "Otof", "Slc17a8", "Calb2", "Acbd7"],
    "IPhC_IB": ["Slc1a3", "Sox2", "Gata3", "Matn4", "Smpx", "Epyc"],
    "DC": ["Lgr5", "Fgfr3", "Prox1", "Ceacam16", "Rbp1", "Rbp7", "Fabp3"],
    "PC": ["Smpx", "Lgr6", "Ednrb", "Prdm12", "Gjb2", "Sox2"],
    "HeC_OSC_ISC": ["Gjb2", "Gjb6", "Aqp4", "Epyc", "Slc26a4", "Fxyd3", "Dnase1", "Pmch", "Fst", "Bmp4", "Nupr1", "Gata2"],
    "Must_negative": ["Oc90", "Otx2", "Neurod1", "Neurog1"]
}

sc.pl.dotplot(
    adata_p28_sensory_plot,
    filter_marker_dict(sensory_marker_dict, adata_p28_sensory_plot.var_names),
    groupby="sensory_leiden",
    categories_order=sensory_cluster_order,
    standard_scale="var",
    cmap="Reds",
    dot_max=0.8,
    figsize=(18, 4.8),
    show=False
)


plt.close("all")


# Notebook cell 70

#1. =========== Define P28 marker sets from marker table ===========

RANDOM_STATE = 0
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
sc.settings.verbosity = 0

p28_leaf_marker_dict = {
    "OHC": {
        "main": ["Slc26a5", "Ocm"],
        "aux": ["Pcp4"]
    },
    "IHC": {
        "main": ["Slc17a8", "Otof"],
        "aux": ["Calb2", "Acbd7"]
    },
    "IPhC": {
        "main": ["Sox2", "Gata3", "Slc1a3"],
        "aux": ["Matn4", "Smpx", "Epyc"]
    },
    "IB": {
        "main": ["Sox2", "Gata3"],
        "aux": ["Smpx", "Epyc"]
    },
    "HeC": {
        "main": ["Sox2", "Gata3", "Pmch", "Smpx", "Epyc"],
        "aux": []
    },
    "DC": {
        "main": ["Lgr5", "Fgfr3", "Prox1", "Ceacam16"],
        "aux": ["Rbp1", "Rbp7", "Fabp3", "Sox2", "Gjb2"]
    },
    "PC": {
        "main": ["Smpx", "Lgr6"],
        "aux": ["Sox2", "Gjb2", "Ednrb", "Prdm12"]
    },
    "OSC": {
        "main": ["Gjb2", "Aqp4", "Epyc"],
        "aux": []
    },
    "ISC": {
        "main": ["Gjb2", "Gata3", "Aqp4", "Epyc"],
        "aux": []
    }
}

p28_check_marker_dict = {
    "HC_broad_check": ["Myo7a", "Strc", "Tmc1", "Pcp4", "Otof", "Slc26a5"],
    "Inner_border_phalangeal_Hensen_check": ["S100a1", "Slc1a3"],
    "Must_negative": ["Oc90", "Otx2", "Neurod1", "Neurog1"]
}

expected_p28_leaf_types = ["OHC", "IHC", "IPhC", "IB", "HeC", "DC", "PC", "OSC", "ISC"]


#2. =========== Calculate cluster-level marker scores ===========

gene_lookup = {
    gene.lower(): gene
    for gene in adata_p28_sensory_plot.var_names
}

def match_genes(genes):
    matched = []
    for gene in genes:
        key = gene.lower()
        if key in gene_lookup:
            matched.append(gene_lookup[key])
    return list(dict.fromkeys(matched))

score_gene_list = []
for marker_info in p28_leaf_marker_dict.values():
    score_gene_list.extend(match_genes(marker_info["main"]))
    score_gene_list.extend(match_genes(marker_info["aux"]))

score_gene_list = list(dict.fromkeys(score_gene_list))

cluster_mean_rows = []

for cluster in sensory_cluster_order:
    cell_mask = (adata_p28_sensory_plot.obs["sensory_leiden"].astype(str) == str(cluster)).values
    X = adata_p28_sensory_plot[cell_mask, score_gene_list].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)

    row = {"cluster": cluster, "n_cells": int(cell_mask.sum())}
    for i, gene in enumerate(score_gene_list):
        row[gene] = float(X[:, i].mean())

    cluster_mean_rows.append(row)

cluster_mean_table = pd.DataFrame(cluster_mean_rows).set_index("cluster")
cluster_gene_mean = cluster_mean_table[score_gene_list]

cluster_gene_z = cluster_gene_mean.copy()
cluster_gene_z = (cluster_gene_z - cluster_gene_z.mean(axis=0)) / cluster_gene_z.std(axis=0).replace(0, np.nan)
cluster_gene_z = cluster_gene_z.fillna(0)

p28_score_table = cluster_mean_table[["n_cells"]].copy()

for cell_type, marker_info in p28_leaf_marker_dict.items():
    main_genes = match_genes(marker_info["main"])
    aux_genes = match_genes(marker_info["aux"])

    score_parts = []
    if len(main_genes) > 0:
        score_parts.append(cluster_gene_z[main_genes].mean(axis=1))
    if len(aux_genes) > 0:
        score_parts.append(cluster_gene_z[aux_genes].mean(axis=1) * 0.5)

    if len(score_parts) == 0:
        p28_score_table[cell_type] = np.nan
    else:
        p28_score_table[cell_type] = pd.concat(score_parts, axis=1).sum(axis=1)

score_cols = expected_p28_leaf_types
p28_score_table["p28_cell_type"] = p28_score_table[score_cols].idxmax(axis=1)
p28_score_table["best_score"] = p28_score_table[score_cols].max(axis=1)

second_type_rows = []
for cluster in p28_score_table.index:
    ordered_scores = p28_score_table.loc[cluster, score_cols].sort_values(ascending=False)
    second_type_rows.append({
        "cluster": cluster,
        "second_type": ordered_scores.index[1],
        "second_score": ordered_scores.iloc[1],
        "score_margin": ordered_scores.iloc[0] - ordered_scores.iloc[1]
    })

second_type_table = pd.DataFrame(second_type_rows).set_index("cluster")
p28_score_table = p28_score_table.join(second_type_table)

display_cols = ["n_cells", "p28_cell_type", "best_score", "second_type", "second_score", "score_margin"] + score_cols
display(p28_score_table[display_cols].round(2))


#3. =========== Apply candidate P28 cell type labels ===========

p28_cluster_to_type = p28_score_table["p28_cell_type"].to_dict()

adata_p28_sensory_candidate.obs["p28_cell_type"] = (
    adata_p28_sensory_candidate.obs["sensory_leiden"].astype(str).map(p28_cluster_to_type)
)
adata_p28_sensory_plot.obs["p28_cell_type"] = (
    adata_p28_sensory_plot.obs["sensory_leiden"].astype(str).map(p28_cluster_to_type)
)

present_type_order = [
    cell_type for cell_type in expected_p28_leaf_types
    if cell_type in adata_p28_sensory_plot.obs["p28_cell_type"].dropna().unique()
]

adata_p28_sensory_candidate.obs["p28_cell_type"] = pd.Categorical(
    adata_p28_sensory_candidate.obs["p28_cell_type"],
    categories=present_type_order,
    ordered=True
)
adata_p28_sensory_plot.obs["p28_cell_type"] = adata_p28_sensory_candidate.obs["p28_cell_type"].copy()

cell_type_counts = adata_p28_sensory_plot.obs["p28_cell_type"].value_counts().reindex(present_type_order)
cell_type_label_map = {
    cell_type: f"{cell_type} (n={int(cell_type_counts.loc[cell_type])})"
    for cell_type in present_type_order
}
adata_p28_sensory_plot.obs["p28_cell_type_n"] = (
    adata_p28_sensory_plot.obs["p28_cell_type"].astype(str).map(cell_type_label_map)
)

cell_type_sample_table = pd.crosstab(
    adata_p28_sensory_candidate.obs["p28_cell_type"],
    adata_p28_sensory_candidate.obs["sample"]
)
for sample in ["P28_1", "P28_2"]:
    if sample not in cell_type_sample_table.columns:
        cell_type_sample_table[sample] = 0

cell_type_sample_table["n_cells"] = cell_type_sample_table["P28_1"] + cell_type_sample_table["P28_2"]
cell_type_sample_table["P28_2_frac"] = (cell_type_sample_table["P28_2"] / cell_type_sample_table["n_cells"]).round(3)
display(cell_type_sample_table[["P28_1", "P28_2", "n_cells", "P28_2_frac"]])

missing_type_table = pd.DataFrame({
    "P28_leaf_type": expected_p28_leaf_types,
    "present": [cell_type in present_type_order for cell_type in expected_p28_leaf_types]
})
display(missing_type_table)


#4. =========== Compact UMAP for P28 cell type check ===========

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

sc.pl.umap(
    adata_p28_sensory_plot,
    color="p28_cell_type_n",
    legend_loc="right margin",
    legend_fontsize=8,
    size=8,
    frameon=False,
    title="P28 cell types",
    ax=axes[0],
    show=False
)

sc.pl.umap(
    adata_p28_sensory_plot,
    color="p28_cell_type",
    legend_loc="on data",
    legend_fontsize=8,
    legend_fontoutline=2,
    size=8,
    frameon=False,
    title="P28 cell types on data",
    ax=axes[1],
    show=False
)

sc.pl.umap(
    adata_p28_sensory_plot,
    color="sample",
    legend_loc="right margin",
    size=8,
    frameon=False,
    title="Sample",
    ax=axes[2],
    show=False
)

plt.tight_layout()
plt.close("all")


#5. =========== Dotplot for P28 table marker validation ===========

p28_final_marker_dict = {
    "HC_broad_check": match_genes(p28_check_marker_dict["HC_broad_check"]),
    "OHC": match_genes(p28_leaf_marker_dict["OHC"]["main"] + p28_leaf_marker_dict["OHC"]["aux"]),
    "IHC": match_genes(p28_leaf_marker_dict["IHC"]["main"] + p28_leaf_marker_dict["IHC"]["aux"]),
    "IPhC": match_genes(p28_leaf_marker_dict["IPhC"]["main"] + p28_leaf_marker_dict["IPhC"]["aux"]),
    "Inner_border_phalangeal_Hensen_check": match_genes(p28_check_marker_dict["Inner_border_phalangeal_Hensen_check"]),
    "HeC": match_genes(p28_leaf_marker_dict["HeC"]["main"] + p28_leaf_marker_dict["HeC"]["aux"]),
    "IB": match_genes(p28_leaf_marker_dict["IB"]["main"] + p28_leaf_marker_dict["IB"]["aux"]),
    "DC": match_genes(p28_leaf_marker_dict["DC"]["main"] + p28_leaf_marker_dict["DC"]["aux"]),
    "PC": match_genes(p28_leaf_marker_dict["PC"]["main"] + p28_leaf_marker_dict["PC"]["aux"]),
    "OSC": match_genes(p28_leaf_marker_dict["OSC"]["main"] + p28_leaf_marker_dict["OSC"]["aux"]),
    "ISC": match_genes(p28_leaf_marker_dict["ISC"]["main"] + p28_leaf_marker_dict["ISC"]["aux"]),
    "Must_negative": match_genes(p28_check_marker_dict["Must_negative"])
}
p28_final_marker_dict = {
    group: genes
    for group, genes in p28_final_marker_dict.items()
    if len(genes) > 0
}

sc.pl.dotplot(
    adata_p28_sensory_plot,
    p28_final_marker_dict,
    groupby="p28_cell_type",
    categories_order=present_type_order,
    standard_scale="var",
    cmap="Reds",
    dot_max=0.8,
    figsize=(18, 4.8),
    show=False
)


plt.close("all")


# Notebook cell 71

#1. =========== Fix confident clusters and select ambiguous support cells ===========

RANDOM_STATE = 0
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
sc.settings.verbosity = 0

confident_sensory_cluster_to_type = {
    "2": "OHC",
    "11": "IHC",
    "1": "PC",
    "9": "DC"
}

ambiguous_sensory_clusters = [
    cluster for cluster in sensory_cluster_order
    if cluster not in confident_sensory_cluster_to_type
]

cluster_status_table = p28_score_table[["n_cells", "p28_cell_type", "second_type", "score_margin"]].copy()
cluster_status_table["fixed_type"] = cluster_status_table.index.astype(str).map(confident_sensory_cluster_to_type)
cluster_status_table["status"] = np.where(
    cluster_status_table["fixed_type"].notna(),
    "fixed",
    "need_subcluster"
)
display(cluster_status_table)

adata_p28_ambiguous_support = adata_p28_sensory_candidate[
    adata_p28_sensory_candidate.obs["sensory_leiden"].astype(str).isin(ambiguous_sensory_clusters)
].copy()
adata_p28_ambiguous_support.layers["counts"] = adata_p28_ambiguous_support.X.copy()


#2. =========== Marker-guided HVG selection ===========

ambiguous_support_marker_dict = {
    "IPhC": ["Sox2", "Gata3", "Slc1a3", "Matn4", "Smpx", "Epyc"],
    "IB": ["Sox2", "Gata3", "Smpx", "Epyc"],
    "HeC": ["Sox2", "Gata3", "Pmch", "Smpx", "Epyc"],
    "OSC": ["Gjb2", "Aqp4", "Epyc"],
    "ISC": ["Gjb2", "Gata3", "Aqp4", "Epyc"],
    "DC_PC_check": ["Lgr5", "Fgfr3", "Prox1", "Ceacam16", "Rbp1", "Fabp3", "Lgr6", "Ednrb", "Prdm12"],
    "HC_check": ["Myo6", "Myo7a", "Pcp4", "Otof", "Slc26a5", "Ocm", "Slc17a8"],
    "Must_negative": ["Oc90", "Otx2", "Neurod1", "Neurog1"]
}

ambiguous_marker_genes = []
for genes in ambiguous_support_marker_dict.values():
    ambiguous_marker_genes.extend(match_genes(genes))

ambiguous_marker_genes = list(dict.fromkeys(ambiguous_marker_genes))

adata_p28_ambiguous_work = adata_p28_ambiguous_support.copy()
sc.pp.normalize_total(adata_p28_ambiguous_work, target_sum=1e4)
sc.pp.log1p(adata_p28_ambiguous_work)

sc.pp.highly_variable_genes(
    adata_p28_ambiguous_work,
    n_top_genes=1500,
    batch_key="sample",
    flavor="seurat"
)

tech_gene_mask = adata_p28_ambiguous_work.var_names.str.startswith(("mt-", "Mt-", "Rpl", "Rps", "Hba", "Hbb"))
marker_gene_mask = adata_p28_ambiguous_work.var_names.isin(ambiguous_marker_genes)

adata_p28_ambiguous_work.var.loc[marker_gene_mask, "highly_variable"] = True
adata_p28_ambiguous_work.var.loc[tech_gene_mask, "highly_variable"] = False
adata_p28_ambiguous_work = adata_p28_ambiguous_work[:, adata_p28_ambiguous_work.var["highly_variable"]].copy()


#3. =========== PCA, Harmony and ambiguous support clustering ===========

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="zero-centering a sparse array/matrix densifies it.")
    sc.pp.scale(adata_p28_ambiguous_work, max_value=10)

sc.tl.pca(
    adata_p28_ambiguous_work,
    n_comps=30,
    svd_solver="arpack",
    random_state=RANDOM_STATE
)

sce.pp.harmony_integrate(
    adata_p28_ambiguous_work,
    key="sample",
    basis="X_pca",
    adjusted_basis="X_pca_harmony",
    max_iter_harmony=20,
    random_state=RANDOM_STATE,
    verbose=False
)

adata_p28_ambiguous_work.obsm["X_pca_harmony_20"] = adata_p28_ambiguous_work.obsm["X_pca_harmony"][:, :20].copy()

sc.pp.neighbors(
    adata_p28_ambiguous_work,
    n_neighbors=15,
    use_rep="X_pca_harmony_20",
    random_state=RANDOM_STATE
)

sc.tl.umap(
    adata_p28_ambiguous_work,
    min_dist=0.35,
    spread=1.0,
    random_state=RANDOM_STATE
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    sc.tl.leiden(
        adata_p28_ambiguous_work,
        resolution=0.6,
        key_added="ambiguous_support_leiden",
        random_state=RANDOM_STATE
    )

adata_p28_ambiguous_support.obs["ambiguous_support_leiden"] = (
    adata_p28_ambiguous_work.obs["ambiguous_support_leiden"].astype(str).values
)
adata_p28_ambiguous_support.obsm["X_umap"] = adata_p28_ambiguous_work.obsm["X_umap"].copy()

ambiguous_cluster_order = sorted(
    adata_p28_ambiguous_support.obs["ambiguous_support_leiden"].unique(),
    key=lambda x: int(x) if str(x).isdigit() else str(x)
)
adata_p28_ambiguous_support.obs["ambiguous_support_leiden"] = pd.Categorical(
    adata_p28_ambiguous_support.obs["ambiguous_support_leiden"],
    categories=ambiguous_cluster_order,
    ordered=True
)

ambiguous_counts = adata_p28_ambiguous_support.obs["ambiguous_support_leiden"].value_counts().reindex(ambiguous_cluster_order)
ambiguous_label_map = {
    cluster: f"{cluster} (n={int(ambiguous_counts.loc[cluster])})"
    for cluster in ambiguous_cluster_order
}
adata_p28_ambiguous_support.obs["ambiguous_support_leiden_n"] = (
    adata_p28_ambiguous_support.obs["ambiguous_support_leiden"].astype(str).map(ambiguous_label_map)
)


#4. =========== Prepare ambiguous support plot object ===========

adata_p28_ambiguous_plot = adata_p28_ambiguous_support.copy()
sc.pp.normalize_total(adata_p28_ambiguous_plot, target_sum=1e4)
sc.pp.log1p(adata_p28_ambiguous_plot)

adata_p28_ambiguous_plot.obs["ambiguous_support_leiden"] = adata_p28_ambiguous_support.obs["ambiguous_support_leiden"].copy()
adata_p28_ambiguous_plot.obs["ambiguous_support_leiden_n"] = adata_p28_ambiguous_support.obs["ambiguous_support_leiden_n"].copy()
adata_p28_ambiguous_plot.obsm["X_umap"] = adata_p28_ambiguous_support.obsm["X_umap"].copy()

ambiguous_sample_table = pd.crosstab(
    adata_p28_ambiguous_support.obs["ambiguous_support_leiden"],
    adata_p28_ambiguous_support.obs["sample"]
)
for sample in ["P28_1", "P28_2"]:
    if sample not in ambiguous_sample_table.columns:
        ambiguous_sample_table[sample] = 0

ambiguous_sample_table["n_cells"] = ambiguous_sample_table["P28_1"] + ambiguous_sample_table["P28_2"]
ambiguous_sample_table["P28_2_frac"] = (ambiguous_sample_table["P28_2"] / ambiguous_sample_table["n_cells"]).round(3)
display(ambiguous_sample_table[["P28_1", "P28_2", "n_cells", "P28_2_frac"]])

source_cluster_table = pd.crosstab(
    adata_p28_ambiguous_support.obs["ambiguous_support_leiden"],
    adata_p28_ambiguous_support.obs["sensory_leiden"]
)
display(source_cluster_table)


#5. =========== Compact ambiguous support UMAP check ===========

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

sc.pl.umap(
    adata_p28_ambiguous_plot,
    color="ambiguous_support_leiden_n",
    legend_loc="right margin",
    legend_fontsize=8,
    size=8,
    frameon=False,
    title="Ambiguous support clusters",
    ax=axes[0],
    show=False
)

sc.pl.umap(
    adata_p28_ambiguous_plot,
    color="ambiguous_support_leiden",
    legend_loc="on data",
    legend_fontsize=8,
    legend_fontoutline=2,
    size=8,
    frameon=False,
    title="Ambiguous support on data",
    ax=axes[1],
    show=False
)

sc.pl.umap(
    adata_p28_ambiguous_plot,
    color="sample",
    legend_loc="right margin",
    size=8,
    frameon=False,
    title="Sample",
    ax=axes[2],
    show=False
)

plt.tight_layout()
plt.close("all")


#6. =========== Ambiguous support marker dotplot ===========

sc.pl.dotplot(
    adata_p28_ambiguous_plot,
    filter_marker_dict(ambiguous_support_marker_dict, adata_p28_ambiguous_plot.var_names),
    groupby="ambiguous_support_leiden",
    categories_order=ambiguous_cluster_order,
    cmap="Reds",
    dot_max=0.8,
    vmin=0,
    vmax=3,
    figsize=(17, 4.8),
    show=False
)


plt.close("all")


# Notebook cell 72

#1. =========== Build full P28 candidate object for rule-assisted annotation ===========

RANDOM_STATE = 0
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
sc.settings.verbosity = 0

adata_p28_final_candidate = adata_p28_sensory_candidate.copy()
adata_p28_final_candidate.obs["p28_cell_type_v2"] = "Unassigned"

fixed_cluster_to_type = {
    "2": "OHC",
    "11": "IHC",
    "1": "PC",
    "9": "DC"
}

for cluster, cell_type in fixed_cluster_to_type.items():
    cell_mask = adata_p28_final_candidate.obs["sensory_leiden"].astype(str) == cluster
    adata_p28_final_candidate.obs.loc[cell_mask, "p28_cell_type_v2"] = cell_type

ambiguous_cluster_to_type = {
    "0": "ISC",
    "1": "ISC",
    "2": "DC",
    "3": "OSC",
    "4": "OSC",
    "5": "HeC",
    "6": "IB",
    "7": "OSC",
    "8": "PC",
    "9": "IPhC",
    "10": "ISC"
}

for cluster, cell_type in ambiguous_cluster_to_type.items():
    cell_names = adata_p28_ambiguous_support.obs_names[
        adata_p28_ambiguous_support.obs["ambiguous_support_leiden"].astype(str) == cluster
    ]
    adata_p28_final_candidate.obs.loc[cell_names, "p28_cell_type_v2"] = cell_type

p28_final_type_order = ["OHC", "IHC", "IPhC", "IB", "HeC", "DC", "PC", "OSC", "ISC"]

adata_p28_final_candidate.obs["p28_cell_type_v2"] = pd.Categorical(
    adata_p28_final_candidate.obs["p28_cell_type_v2"],
    categories=p28_final_type_order,
    ordered=True
)


#2. =========== Prepare final validation plot object ===========

adata_p28_final_plot = adata_p28_final_candidate.copy()
sc.pp.normalize_total(adata_p28_final_plot, target_sum=1e4)
sc.pp.log1p(adata_p28_final_plot)

adata_p28_final_plot.obs["p28_cell_type_v2"] = adata_p28_final_candidate.obs["p28_cell_type_v2"].copy()
adata_p28_final_plot.obsm["X_umap"] = adata_p28_sensory_candidate.obsm["X_umap"].copy()

type_counts = adata_p28_final_plot.obs["p28_cell_type_v2"].value_counts().reindex(p28_final_type_order)
type_label_map = {
    cell_type: f"{cell_type} (n={int(type_counts.loc[cell_type])})"
    for cell_type in p28_final_type_order
}
adata_p28_final_plot.obs["p28_cell_type_v2_n"] = (
    adata_p28_final_plot.obs["p28_cell_type_v2"].astype(str).map(type_label_map)
)

final_sample_table = pd.crosstab(
    adata_p28_final_candidate.obs["p28_cell_type_v2"],
    adata_p28_final_candidate.obs["sample"]
)
for sample in ["P28_1", "P28_2"]:
    if sample not in final_sample_table.columns:
        final_sample_table[sample] = 0

final_sample_table["n_cells"] = final_sample_table["P28_1"] + final_sample_table["P28_2"]
final_sample_table["P28_2_frac"] = (final_sample_table["P28_2"] / final_sample_table["n_cells"]).round(3)
display(final_sample_table[["P28_1", "P28_2", "n_cells", "P28_2_frac"]])


#3. =========== Check source clusters of final labels ===========

final_source_table = pd.crosstab(
    adata_p28_final_candidate.obs["p28_cell_type_v2"],
    adata_p28_final_candidate.obs["sensory_leiden"]
)
display(final_source_table)


#4. =========== Compact final UMAP check ===========

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

sc.pl.umap(
    adata_p28_final_plot,
    color="p28_cell_type_v2_n",
    legend_loc="right margin",
    legend_fontsize=8,
    size=8,
    frameon=False,
    title="P28 cell types v2",
    ax=axes[0],
    show=False
)

sc.pl.umap(
    adata_p28_final_plot,
    color="p28_cell_type_v2",
    legend_loc="on data",
    legend_fontsize=8,
    legend_fontoutline=2,
    size=8,
    frameon=False,
    title="P28 cell types on data",
    ax=axes[1],
    show=False
)

sc.pl.umap(
    adata_p28_final_plot,
    color="sample",
    legend_loc="right margin",
    size=8,
    frameon=False,
    title="Sample",
    ax=axes[2],
    show=False
)

plt.tight_layout()
plt.close("all")


#5. =========== Final P28 marker validation dotplot ===========

p28_final_marker_dict = {
    "HC_broad_check": match_genes(["Myo7a", "Strc", "Tmc1", "Pcp4", "Otof", "Slc26a5"]),
    "OHC": match_genes(["Slc26a5", "Ocm", "Pcp4"]),
    "IHC": match_genes(["Slc17a8", "Otof", "Calb2", "Acbd7"]),
    "IPhC": match_genes(["Sox2", "Gata3", "Slc1a3", "Matn4", "Smpx", "Epyc"]),
    "Inner_border_phalangeal_Hensen_check": match_genes(["S100a1", "Slc1a3"]),
    "HeC": match_genes(["Sox2", "Gata3", "Pmch", "Smpx", "Epyc"]),
    "IB": match_genes(["Sox2", "Gata3", "Smpx", "Epyc"]),
    "DC": match_genes(["Lgr5", "Fgfr3", "Prox1", "Ceacam16", "Rbp1", "Fabp3", "Sox2", "Gjb2"]),
    "PC": match_genes(["Smpx", "Lgr6", "Sox2", "Gjb2", "Ednrb", "Prdm12"]),
    "OSC": match_genes(["Gjb2", "Aqp4", "Epyc"]),
    "ISC": match_genes(["Gjb2", "Gata3", "Aqp4", "Epyc"]),
    "Must_negative": match_genes(["Oc90", "Otx2", "Neurod1", "Neurog1"])
}
p28_final_marker_dict = {
    group: genes
    for group, genes in p28_final_marker_dict.items()
    if len(genes) > 0
}

sc.pl.dotplot(
    adata_p28_final_plot,
    p28_final_marker_dict,
    groupby="p28_cell_type_v2",
    categories_order=p28_final_type_order,
    standard_scale="var",
    cmap="Reds",
    dot_max=0.8,
    figsize=(18, 4.8),
    show=False
)


#6. =========== Final exclusion marker dotplot ===========

p28_final_exclude_marker_dict = {
    "Must_negative": match_genes(["Oc90", "Otx2", "Neurod1", "Neurog1"]),
    "Immune": match_genes(["Ptprc", "C1qa", "C1qb", "Tyrobp", "Aif1", "Lyz2", "Cd74"]),
    "Glia": match_genes(["Plp1", "Mpz", "Mbp", "Pmp22", "Sox10"]),
    "Blood": match_genes(["Hba-a1", "Hba-a2", "Hbb-bs", "Hbb-bt", "Hbb-bh1", "Alas2"]),
    "Mesenchymal_Endothelial": match_genes(["Pdgfra", "Pdgfrb", "Col1a1", "Col3a1", "Dcn", "Pecam1", "Kdr", "Cldn5"]),
    "Neuronal": match_genes(["Tubb3", "Elavl3", "Elavl4", "Prph", "Snap25"])
}
p28_final_exclude_marker_dict = {
    group: genes
    for group, genes in p28_final_exclude_marker_dict.items()
    if len(genes) > 0
}

sc.pl.dotplot(
    adata_p28_final_plot,
    p28_final_exclude_marker_dict,
    groupby="p28_cell_type_v2",
    categories_order=p28_final_type_order,
    cmap="Reds",
    dot_max=0.8,
    vmin=0,
    vmax=3,
    figsize=(16, 4.5),
    show=False
)


plt.close("all")


# Notebook cell 73

#1. =========== Build HC-only object ===========

RANDOM_STATE = 0
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
sc.settings.verbosity = 0

adata_p28_hc_check = adata_p28_final_candidate[
    adata_p28_final_candidate.obs["p28_cell_type_v2"].astype(str).isin(["OHC", "IHC"])
].copy()
adata_p28_hc_check.layers["counts"] = adata_p28_hc_check.X.copy()
adata_p28_hc_check.obs["hc_type"] = adata_p28_hc_check.obs["p28_cell_type_v2"].astype(str)

hc_check_table = pd.crosstab(
    adata_p28_hc_check.obs["hc_type"],
    adata_p28_hc_check.obs["sample"]
)
for sample in ["P28_1", "P28_2"]:
    if sample not in hc_check_table.columns:
        hc_check_table[sample] = 0

hc_check_table["n_cells"] = hc_check_table["P28_1"] + hc_check_table["P28_2"]
hc_check_table["P28_2_frac"] = (hc_check_table["P28_2"] / hc_check_table["n_cells"]).round(3)
display(hc_check_table[["P28_1", "P28_2", "n_cells", "P28_2_frac"]])


#2. =========== HC-only PCA and Harmony ===========

adata_p28_hc_work = adata_p28_hc_check.copy()

sc.pp.normalize_total(adata_p28_hc_work, target_sum=1e4)
sc.pp.log1p(adata_p28_hc_work)

sc.pp.highly_variable_genes(
    adata_p28_hc_work,
    n_top_genes=1000,
    batch_key="sample",
    flavor="seurat"
)

hc_marker_genes = match_genes([
    "Myo6", "Myo7a", "Pcp4", "Otof", "Slc26a5", "Ocm",
    "Slc17a8", "Calb2", "Acbd7", "Strc", "Tmc1"
])
tech_gene_mask = adata_p28_hc_work.var_names.str.startswith(("mt-", "Mt-", "Rpl", "Rps", "Hba", "Hbb"))
marker_gene_mask = adata_p28_hc_work.var_names.isin(hc_marker_genes)

adata_p28_hc_work.var.loc[marker_gene_mask, "highly_variable"] = True
adata_p28_hc_work.var.loc[tech_gene_mask, "highly_variable"] = False
adata_p28_hc_work = adata_p28_hc_work[:, adata_p28_hc_work.var["highly_variable"]].copy()

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="zero-centering a sparse array/matrix densifies it.")
    sc.pp.scale(adata_p28_hc_work, max_value=10)

sc.tl.pca(
    adata_p28_hc_work,
    n_comps=20,
    svd_solver="arpack",
    random_state=RANDOM_STATE
)

sce.pp.harmony_integrate(
    adata_p28_hc_work,
    key="sample",
    basis="X_pca",
    adjusted_basis="X_pca_harmony",
    max_iter_harmony=20,
    random_state=RANDOM_STATE,
    verbose=False
)


#3. =========== HC-only UMAP with two neighbor settings ===========

hc_umap_settings = [
    ("hc_n10", 10),
    ("hc_n20", 20)
]

for tag, n_neighbors in hc_umap_settings:
    neighbors_key = f"neighbors_{tag}"
    umap_key = f"X_umap_{tag}"

    sc.pp.neighbors(
        adata_p28_hc_work,
        n_neighbors=n_neighbors,
        use_rep="X_pca_harmony",
        key_added=neighbors_key,
        random_state=RANDOM_STATE
    )

    sc.tl.umap(
        adata_p28_hc_work,
        neighbors_key=neighbors_key,
        min_dist=0.35,
        spread=1.0,
        random_state=RANDOM_STATE
    )

    adata_p28_hc_work.obsm[umap_key] = adata_p28_hc_work.obsm["X_umap"].copy()


#4. =========== Compact HC-only UMAP check ===========

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes = axes.ravel()

for plot_idx, (tag, n_neighbors) in enumerate(hc_umap_settings):
    adata_p28_hc_check.obsm["X_umap"] = adata_p28_hc_work[
        adata_p28_hc_check.obs_names, :
    ].obsm[f"X_umap_{tag}"].copy()

    adata_p28_hc_plot = adata_p28_hc_check.copy()
    sc.pp.normalize_total(adata_p28_hc_plot, target_sum=1e4)
    sc.pp.log1p(adata_p28_hc_plot)
    adata_p28_hc_plot.obs["hc_type"] = adata_p28_hc_check.obs["hc_type"].copy()
    adata_p28_hc_plot.obsm["X_umap"] = adata_p28_hc_check.obsm["X_umap"].copy()

    sc.pl.umap(
        adata_p28_hc_plot,
        color="hc_type",
        legend_loc="right margin",
        size=18,
        frameon=False,
        title=f"HC-only {tag}: type",
        ax=axes[plot_idx * 3],
        show=False
    )

    sc.pl.umap(
        adata_p28_hc_plot,
        color="sample",
        legend_loc="right margin",
        size=18,
        frameon=False,
        title=f"HC-only {tag}: sample",
        ax=axes[plot_idx * 3 + 1],
        show=False
    )

    sc.pl.umap(
        adata_p28_hc_plot,
        color="sensory_leiden",
        legend_loc="on data",
        legend_fontsize=8,
        legend_fontoutline=2,
        size=18,
        frameon=False,
        title=f"HC-only {tag}: original cluster",
        ax=axes[plot_idx * 3 + 2],
        show=False
    )

plt.tight_layout()
plt.close("all")


#5. =========== HC marker validation dotplot ===========

adata_p28_hc_dotplot = adata_p28_hc_check.copy()
sc.pp.normalize_total(adata_p28_hc_dotplot, target_sum=1e4)
sc.pp.log1p(adata_p28_hc_dotplot)
adata_p28_hc_dotplot.obs["hc_type"] = adata_p28_hc_check.obs["hc_type"].copy()

hc_marker_dict = {
    "HC_broad_check": match_genes(["Myo7a", "Strc", "Tmc1", "Pcp4", "Otof"]),
    "OHC": match_genes(["Slc26a5", "Ocm", "Pcp4"]),
    "IHC": match_genes(["Slc17a8", "Otof", "Calb2", "Acbd7"]),
    "Support_check": match_genes(["Sox2", "Gata3", "Slc1a3", "Gjb2", "Aqp4", "Epyc"]),
    "Must_negative": match_genes(["Oc90", "Otx2", "Neurod1", "Neurog1"])
}
hc_marker_dict = {
    group: genes
    for group, genes in hc_marker_dict.items()
    if len(genes) > 0
}

sc.pl.dotplot(
    adata_p28_hc_dotplot,
    hc_marker_dict,
    groupby="hc_type",
    categories_order=["OHC", "IHC"],
    standard_scale="var",
    cmap="Reds",
    dot_max=0.8,
    figsize=(10, 3.2),
    show=False
)


plt.close("all")


# Notebook cell 74

#1. =========== P28 marker-guided annotation check ===========

adata_p28_final_marker_guided = adata_p28_final_candidate.copy()

if "X_umap" not in adata_p28_final_marker_guided.obsm and "adata_p28_final_plot" in globals():
    adata_p28_final_marker_guided.obsm["X_umap"] = adata_p28_final_plot[
        adata_p28_final_marker_guided.obs_names, :
    ].obsm["X_umap"].copy()

gene_map = {g.upper(): g for g in adata_p28_final_marker_guided.var_names}

def keep_genes(genes):
    out = []
    for gene in genes:
        real_gene = gene_map.get(gene.upper())
        if real_gene is not None and real_gene not in out:
            out.append(real_gene)
    return out

def filter_marker_dict(marker_dict):
    return {key: keep_genes(genes) for key, genes in marker_dict.items() if len(keep_genes(genes)) > 0}

p28_marker_plot_dict = filter_marker_dict({
    "HC_broad_check": ["Myo7a", "Strc", "Tmc1", "Pcp4", "Otof"],
    "OHC": ["Slc26a5", "Ocm", "Pcp4"],
    "IHC": ["Slc17a8", "Otof", "Calb2", "Acbd7"],
    "IPhC": ["Sox2", "Gata3", "Slc1a3", "Matn4", "Smpx", "Epyc"],
    "Inner_border_phalangeal_Hensen_check": ["S100a1", "Slc1a3"],
    "HeC": ["Pmch", "Smpx", "Epyc", "Sox2", "Gata3"],
    "IB": ["Sox2", "Gata3", "Smpx", "Epyc"],
    "DC": ["Lgr5", "Fgfr3", "Prox1", "Ceacam16", "Rbp1", "Rbp7", "Fabp3", "Sox2", "Gjb2"],
    "PC": ["Smpx", "Lgr6", "Sox2", "Gjb2", "Ednrb", "Prdm12"],
    "OSC": ["Gjb2", "Aqp4", "Epyc"],
    "ISC": ["Gjb2", "Gata3", "Aqp4", "Epyc"],
    "Must_negative": ["Oc90", "Otx2", "Neurod1", "Neurog1"],
})

p28_support_score_markers = {
    "IPhC": {"main": ["Slc1a3", "Sox2", "Gata3"], "aux": ["Matn4", "Smpx", "Epyc"]},
    "IB": {"main": ["Sox2", "Gata3"], "aux": ["Smpx", "Epyc"]},
    "HeC": {"main": ["Pmch", "Smpx", "Epyc"], "aux": ["Sox2", "Gata3"]},
    "DC": {"main": ["Lgr5", "Fgfr3", "Prox1", "Ceacam16"], "aux": ["Rbp1", "Rbp7", "Fabp3", "Sox2", "Gjb2"]},
    "PC": {"main": ["Smpx", "Lgr6"], "aux": ["Sox2", "Gjb2", "Ednrb", "Prdm12"]},
    "OSC": {"main": ["Gjb2", "Aqp4", "Epyc"], "aux": ["Sox2"]},
    "ISC": {"main": ["Gjb2", "Gata3", "Aqp4", "Epyc"], "aux": ["Sox2"]},
}

adata_p28_final_marker_guided.obs["p28_marker_unit"] = adata_p28_final_marker_guided.obs["p28_cell_type_v2"].astype(str)

if "ambiguous_support_leiden" in adata_p28_final_marker_guided.obs.columns:
    amb_mask = (
        adata_p28_final_marker_guided.obs["ambiguous_support_leiden"].notna()
        & ~adata_p28_final_marker_guided.obs["p28_cell_type_v2"].isin(["OHC", "IHC"])
    )
    adata_p28_final_marker_guided.obs.loc[amb_mask, "p28_marker_unit"] = (
        "amb_" + adata_p28_final_marker_guided.obs.loc[amb_mask, "ambiguous_support_leiden"].astype(str)
    )

support_mask = ~adata_p28_final_marker_guided.obs["p28_cell_type_v2"].isin(["OHC", "IHC"])
support_units = adata_p28_final_marker_guided.obs.loc[support_mask, "p28_marker_unit"].astype(str)

score_genes = []
for marker_set in p28_support_score_markers.values():
    score_genes.extend(marker_set["main"] + marker_set["aux"])
score_genes.extend(["Oc90", "Otx2", "Neurod1", "Neurog1"])
score_genes = keep_genes(score_genes)

X_score = adata_p28_final_marker_guided[support_mask, score_genes].X
if hasattr(X_score, "toarray"):
    X_score = X_score.toarray()

expr_score = pd.DataFrame(
    X_score,
    index=adata_p28_final_marker_guided.obs_names[support_mask],
    columns=score_genes,
)

mean_by_unit = expr_score.groupby(support_units).mean()
pct_by_unit = (expr_score > 0).groupby(support_units).mean() * 100

z_by_unit = (mean_by_unit - mean_by_unit.mean(axis=0)) / mean_by_unit.std(axis=0).replace(0, np.nan)
z_by_unit = z_by_unit.fillna(0)

support_score = pd.DataFrame(index=mean_by_unit.index)

for cell_type, marker_set in p28_support_score_markers.items():
    main_genes = keep_genes(marker_set["main"])
    aux_genes = keep_genes(marker_set["aux"])

    score = pd.Series(0.0, index=mean_by_unit.index)
    if len(main_genes) > 0:
        score = score + z_by_unit[main_genes].mean(axis=1)
    if len(aux_genes) > 0:
        score = score + 0.5 * z_by_unit[aux_genes].mean(axis=1)

    support_score[cell_type] = score

ranked_types = support_score.apply(lambda row: row.sort_values(ascending=False).index.tolist(), axis=1)
ranked_scores = support_score.apply(lambda row: row.sort_values(ascending=False).values.tolist(), axis=1)

support_score_summary = pd.DataFrame({
    "n_cells": support_units.value_counts().reindex(support_score.index).astype(int),
    "current_majority": adata_p28_final_marker_guided.obs.loc[support_mask]
        .groupby("p28_marker_unit")["p28_cell_type_v2"]
        .agg(lambda x: x.value_counts().index[0])
        .reindex(support_score.index),
    "best_type": ranked_types.str[0],
    "best_score": ranked_scores.str[0],
    "second_type": ranked_types.str[1],
    "second_score": ranked_scores.str[1],
})

support_score_summary["score_margin"] = (
    support_score_summary["best_score"] - support_score_summary["second_score"]
)

sample_table = pd.crosstab(
    adata_p28_final_marker_guided.obs.loc[support_mask, "p28_marker_unit"],
    adata_p28_final_marker_guided.obs.loc[support_mask, "sample"],
)
for sample in ["P28_1", "P28_2"]:
    if sample not in sample_table.columns:
        sample_table[sample] = 0

support_score_summary["P28_1"] = sample_table["P28_1"].reindex(support_score_summary.index).fillna(0).astype(int)
support_score_summary["P28_2"] = sample_table["P28_2"].reindex(support_score_summary.index).fillna(0).astype(int)
support_score_summary["P28_2_frac"] = (
    support_score_summary["P28_2"] / support_score_summary[["P28_1", "P28_2"]].sum(axis=1)
).round(3)

negative_genes = keep_genes(["Oc90", "Otx2", "Neurod1", "Neurog1"])
support_score_summary["max_negative_pct"] = (
    pct_by_unit[negative_genes].max(axis=1).reindex(support_score_summary.index).round(2)
    if len(negative_genes) > 0 else 0
)

support_score_summary["status"] = "candidate"
support_score_summary.loc[support_score_summary["score_margin"] < 0.25, "status"] = "low_margin_check"
support_score_summary.loc[support_score_summary["max_negative_pct"] > 5, "status"] = "negative_marker_check"
support_score_summary.loc[
    (support_score_summary["P28_2_frac"] > 0.85) | (support_score_summary["P28_2_frac"] < 0.15),
    "status"
] = "sample_skew_check"

adata_p28_final_marker_guided.obs["p28_cell_type_v3_marker_guided"] = (
    adata_p28_final_marker_guided.obs["p28_cell_type_v2"].astype(str).astype(object)
)

for unit, row in support_score_summary.iterrows():
    unit_mask = adata_p28_final_marker_guided.obs["p28_marker_unit"].astype(str) == str(unit)
    adata_p28_final_marker_guided.obs.loc[unit_mask, "p28_cell_type_v3_marker_guided"] = row["best_type"]

adata_p28_final_marker_guided.obs.loc[
    adata_p28_final_marker_guided.obs["p28_cell_type_v2"] == "OHC",
    "p28_cell_type_v3_marker_guided"
] = "OHC"

adata_p28_final_marker_guided.obs.loc[
    adata_p28_final_marker_guided.obs["p28_cell_type_v2"] == "IHC",
    "p28_cell_type_v3_marker_guided"
] = "IHC"

p28_type_order = ["OHC", "IHC", "IPhC", "IB", "HeC", "DC", "PC", "OSC", "ISC"]
adata_p28_final_marker_guided.obs["p28_cell_type_v3_marker_guided"] = pd.Categorical(
    adata_p28_final_marker_guided.obs["p28_cell_type_v3_marker_guided"],
    categories=[x for x in p28_type_order if x in adata_p28_final_marker_guided.obs["p28_cell_type_v3_marker_guided"].unique()],
    ordered=True,
)

type_table_v3 = pd.crosstab(
    adata_p28_final_marker_guided.obs["p28_cell_type_v3_marker_guided"],
    adata_p28_final_marker_guided.obs["sample"],
)
for sample in ["P28_1", "P28_2"]:
    if sample not in type_table_v3.columns:
        type_table_v3[sample] = 0

type_table_v3["n_cells"] = type_table_v3[["P28_1", "P28_2"]].sum(axis=1)
type_table_v3["P28_2_frac"] = (type_table_v3["P28_2"] / type_table_v3["n_cells"]).round(3)

display(
    support_score_summary[
        ["n_cells", "current_majority", "best_type", "second_type", "score_margin",
         "P28_1", "P28_2", "P28_2_frac", "max_negative_pct", "status"]
    ].sort_values(["best_type", "status", "score_margin"], ascending=[True, True, False])
)

display(type_table_v3[["P28_1", "P28_2", "n_cells", "P28_2_frac"]])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sc.pl.umap(
    adata_p28_final_marker_guided,
    color="p28_cell_type_v3_marker_guided",
    ax=axes[0],
    show=False,
    size=8,
    legend_loc="right margin",
    frameon=False,
    title="P28 marker-guided cell types",
)
sc.pl.umap(
    adata_p28_final_marker_guided,
    color="p28_cell_type_v2",
    ax=axes[1],
    show=False,
    size=8,
    legend_loc="right margin",
    frameon=False,
    title="P28 previous cell types",
)
sc.pl.umap(
    adata_p28_final_marker_guided,
    color="sample",
    ax=axes[2],
    show=False,
    size=8,
    legend_loc="right margin",
    frameon=False,
    title="Sample",
)
plt.tight_layout()
plt.close("all")

sc.pl.dotplot(
    adata_p28_final_marker_guided,
    p28_marker_plot_dict,
    groupby="p28_cell_type_v3_marker_guided",
    standard_scale="var",
    dendrogram=False,
    dot_max=0.85,
    dot_min=0.02,
    smallest_dot=0,
    color_map="Reds",
    figsize=(22, 5),
    show=False,
)

sc.pl.dotplot(
    adata_p28_final_marker_guided[support_mask].copy(),
    p28_marker_plot_dict,
    groupby="p28_marker_unit",
    standard_scale="var",
    dendrogram=False,
    dot_max=0.85,
    dot_min=0.02,
    smallest_dot=0,
    color_map="Reds",
    figsize=(24, 7),
    show=False,
)


plt.close("all")


# Notebook cell 75

#1. =========== P28 marker-informed support-cell reclustering ===========

RANDOM_STATE = 0

p28_support_marker_source = adata_p28_final_candidate.obs_names[
    ~adata_p28_final_candidate.obs["p28_cell_type_v2"].isin(["OHC", "IHC"])
]

adata_p28_support_marker = adata_P28[p28_support_marker_source, :].copy()

for col in ["p28_cell_type_v2", "sensory_leiden", "ambiguous_support_leiden"]:
    if col in adata_p28_final_candidate.obs.columns:
        adata_p28_support_marker.obs[col] = adata_p28_final_candidate.obs.loc[
            adata_p28_support_marker.obs_names, col
        ].values

gene_map = {g.upper(): g for g in adata_p28_support_marker.var_names}

def keep_genes(genes):
    kept = []
    for gene in genes:
        real_gene = gene_map.get(gene.upper())
        if real_gene is not None and real_gene not in kept:
            kept.append(real_gene)
    return kept

def filter_marker_dict(marker_dict):
    return {
        key: keep_genes(genes)
        for key, genes in marker_dict.items()
        if len(keep_genes(genes)) > 0
    }

p28_support_marker_plot_dict = filter_marker_dict({
    "IPhC": ["Sox2", "Gata3", "Slc1a3", "Matn4", "Smpx", "Epyc"],
    "IB": ["Sox2", "Gata3", "Smpx", "Epyc"],
    "HeC": ["Pmch", "Smpx", "Epyc", "Sox2", "Gata3"],
    "DC": ["Lgr5", "Fgfr3", "Prox1", "Ceacam16", "Rbp1", "Rbp7", "Fabp3", "Sox2", "Gjb2"],
    "PC": ["Smpx", "Lgr6", "Sox2", "Gjb2", "Ednrb", "Prdm12"],
    "OSC": ["Gjb2", "Aqp4", "Epyc"],
    "ISC": ["Gjb2", "Gata3", "Aqp4", "Epyc"],
    "HC_check": ["Myo7a", "Strc", "Tmc1", "Pcp4", "Otof", "Slc26a5", "Ocm", "Slc17a8"],
    "Must_negative": ["Oc90", "Otx2", "Neurod1", "Neurog1"],
})

p28_force_clustering_markers = keep_genes([
    "Slc1a3", "Matn4", "Pmch",
    "Lgr5", "Fgfr3", "Prox1", "Ceacam16", "Rbp1", "Rbp7", "Fabp3",
    "Smpx", "Lgr6", "Ednrb", "Prdm12",
    "Gjb2", "Aqp4", "Epyc", "Sox2", "Gata3",
])

adata_p28_support_marker_norm = adata_p28_support_marker.copy()
sc.pp.normalize_total(adata_p28_support_marker_norm, target_sum=1e4)
sc.pp.log1p(adata_p28_support_marker_norm)

sc.pp.highly_variable_genes(
    adata_p28_support_marker_norm,
    n_top_genes=2500,
    flavor="seurat",
    batch_key="sample",
)

adata_p28_support_marker_norm.var["highly_variable"] = adata_p28_support_marker_norm.var["highly_variable"].fillna(False)

qc_or_negative_genes = (
    adata_p28_support_marker_norm.var_names.str.startswith(("mt-", "Mt-", "Rpl", "Rps"))
    | adata_p28_support_marker_norm.var_names.str.match(r"^Hb[ab]-")
    | adata_p28_support_marker_norm.var_names.isin(keep_genes(["Oc90", "Otx2", "Neurod1", "Neurog1"]))
)

adata_p28_support_marker_norm.var.loc[qc_or_negative_genes, "highly_variable"] = False
adata_p28_support_marker_norm.var.loc[p28_force_clustering_markers, "highly_variable"] = True

p28_marker_features = adata_p28_support_marker_norm.var_names[
    adata_p28_support_marker_norm.var["highly_variable"]
].tolist()

adata_p28_support_marker_work = adata_p28_support_marker_norm[:, p28_marker_features].copy()

sc.pp.scale(adata_p28_support_marker_work, max_value=10)
sc.tl.pca(
    adata_p28_support_marker_work,
    n_comps=35,
    svd_solver="arpack",
    random_state=RANDOM_STATE,
)

sce.pp.harmony_integrate(
    adata_p28_support_marker_work,
    key="sample",
    basis="X_pca",
    adjusted_basis="X_pca_harmony",
    random_state=RANDOM_STATE,
)

p28_support_marker_settings = [
    ("mg20_n15_r035", 20, 15, 0.35),
    ("mg20_n20_r035", 20, 20, 0.35),
    ("mg20_n25_r035", 20, 25, 0.35),
    ("mg20_n20_r045", 20, 20, 0.45),
    ("mg25_n20_r045", 25, 20, 0.45),
    ("mg25_n25_r045", 25, 25, 0.45),
]

p28_support_marker_summary = []

for tag, n_pcs, n_neighbors, resolution in p28_support_marker_settings:
    cluster_key = f"support_marker_leiden_{tag}"
    umap_key = f"X_umap_{tag}"

    sc.pp.neighbors(
        adata_p28_support_marker_work,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep="X_pca_harmony",
        random_state=RANDOM_STATE,
    )
    sc.tl.umap(
        adata_p28_support_marker_work,
        min_dist=0.35,
        spread=1.0,
        random_state=RANDOM_STATE,
    )
    sc.tl.leiden(
        adata_p28_support_marker_work,
        resolution=resolution,
        key_added=cluster_key,
        random_state=RANDOM_STATE,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )

    adata_p28_support_marker_work.obsm[umap_key] = adata_p28_support_marker_work.obsm["X_umap"].copy()

    cluster_counts = adata_p28_support_marker_work.obs[cluster_key].value_counts()
    sample_counts = pd.crosstab(
        adata_p28_support_marker_work.obs[cluster_key],
        adata_p28_support_marker_work.obs["sample"],
    )

    for sample in ["P28_1", "P28_2"]:
        if sample not in sample_counts.columns:
            sample_counts[sample] = 0

    sample_frac = sample_counts["P28_2"] / sample_counts[["P28_1", "P28_2"]].sum(axis=1)

    p28_support_marker_summary.append({
        "setting": tag,
        "n_pcs": n_pcs,
        "n_neighbors": n_neighbors,
        "resolution": resolution,
        "n_clusters": int(cluster_counts.shape[0]),
        "min_cells": int(cluster_counts.min()),
        "clusters_lt30": int((cluster_counts < 30).sum()),
        "sample_skewed_clusters": int(((sample_frac > 0.85) | (sample_frac < 0.15)).sum()),
    })

p28_support_marker_summary = pd.DataFrame(p28_support_marker_summary)
p28_support_marker_summary["rank_score"] = (
    p28_support_marker_summary["sample_skewed_clusters"] * 10
    + p28_support_marker_summary["clusters_lt30"] * 5
    + (p28_support_marker_summary["n_clusters"] - 8).abs()
)

p28_support_marker_choice = p28_support_marker_summary.sort_values(
    ["rank_score", "sample_skewed_clusters", "clusters_lt30", "min_cells"],
    ascending=[True, True, True, False],
).iloc[0]["setting"]

p28_support_marker_key = f"support_marker_leiden_{p28_support_marker_choice}"

adata_p28_support_marker_plot = adata_p28_support_marker_norm.copy()
adata_p28_support_marker_plot.obs[p28_support_marker_key] = adata_p28_support_marker_work.obs[
    p28_support_marker_key
].values
adata_p28_support_marker_plot.obsm["X_umap"] = adata_p28_support_marker_work.obsm[
    f"X_umap_{p28_support_marker_choice}"
].copy()

support_marker_sample_table = pd.crosstab(
    adata_p28_support_marker_plot.obs[p28_support_marker_key],
    adata_p28_support_marker_plot.obs["sample"],
)

for sample in ["P28_1", "P28_2"]:
    if sample not in support_marker_sample_table.columns:
        support_marker_sample_table[sample] = 0

support_marker_sample_table["n_cells"] = support_marker_sample_table[["P28_1", "P28_2"]].sum(axis=1)
support_marker_sample_table["P28_2_frac"] = (
    support_marker_sample_table["P28_2"] / support_marker_sample_table["n_cells"]
).round(3)

score_marker_dict = {
    "IPhC": ["Sox2", "Gata3", "Slc1a3", "Matn4", "Smpx", "Epyc"],
    "IB": ["Sox2", "Gata3", "Smpx", "Epyc"],
    "HeC": ["Pmch", "Smpx", "Epyc", "Sox2", "Gata3"],
    "DC": ["Lgr5", "Fgfr3", "Prox1", "Ceacam16", "Rbp1", "Rbp7", "Fabp3"],
    "PC": ["Smpx", "Lgr6", "Ednrb", "Prdm12", "Gjb2"],
    "OSC": ["Gjb2", "Aqp4", "Epyc"],
    "ISC": ["Gjb2", "Gata3", "Aqp4", "Epyc"],
}

score_genes = keep_genes(sum(score_marker_dict.values(), []))
X_score = adata_p28_support_marker_plot[:, score_genes].X
if hasattr(X_score, "toarray"):
    X_score = X_score.toarray()

expr_score = pd.DataFrame(
    X_score,
    index=adata_p28_support_marker_plot.obs_names,
    columns=score_genes,
)

mean_by_cluster = expr_score.groupby(
    adata_p28_support_marker_plot.obs[p28_support_marker_key].astype(str)
).mean()

z_by_cluster = (mean_by_cluster - mean_by_cluster.mean(axis=0)) / mean_by_cluster.std(axis=0).replace(0, np.nan)
z_by_cluster = z_by_cluster.fillna(0)

support_marker_score = pd.DataFrame(index=mean_by_cluster.index)

for cell_type, genes in score_marker_dict.items():
    genes = keep_genes(genes)
    support_marker_score[cell_type] = z_by_cluster[genes].mean(axis=1)

ranked_types = support_marker_score.apply(lambda row: row.sort_values(ascending=False).index.tolist(), axis=1)
ranked_scores = support_marker_score.apply(lambda row: row.sort_values(ascending=False).values.tolist(), axis=1)

support_marker_score_summary = pd.DataFrame({
    "n_cells": support_marker_sample_table["n_cells"].reindex(support_marker_score.index).astype(int),
    "P28_1": support_marker_sample_table["P28_1"].reindex(support_marker_score.index).astype(int),
    "P28_2": support_marker_sample_table["P28_2"].reindex(support_marker_score.index).astype(int),
    "P28_2_frac": support_marker_sample_table["P28_2_frac"].reindex(support_marker_score.index),
    "best_type": ranked_types.str[0],
    "best_score": ranked_scores.str[0],
    "second_type": ranked_types.str[1],
    "second_score": ranked_scores.str[1],
})

support_marker_score_summary["score_margin"] = (
    support_marker_score_summary["best_score"] - support_marker_score_summary["second_score"]
).round(3)

display(p28_support_marker_summary.drop(columns="rank_score"))
display(pd.DataFrame({"chosen_setting": [p28_support_marker_choice]}))
display(support_marker_score_summary.sort_index())

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

for ax, (tag, n_pcs, n_neighbors, resolution) in zip(axes, p28_support_marker_settings):
    sc.pl.embedding(
        adata_p28_support_marker_work,
        basis=f"umap_{tag}",
        color=f"support_marker_leiden_{tag}",
        ax=ax,
        show=False,
        size=8,
        legend_loc="on data",
        legend_fontsize=8,
        legend_fontoutline=2,
        frameon=False,
        title=tag,
    )

plt.tight_layout()
plt.close("all")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sc.pl.umap(
    adata_p28_support_marker_plot,
    color=p28_support_marker_key,
    ax=axes[0],
    show=False,
    size=8,
    legend_loc="right margin",
    frameon=False,
    title=f"{p28_support_marker_choice}: clusters",
)

sc.pl.umap(
    adata_p28_support_marker_plot,
    color="p28_cell_type_v2",
    ax=axes[1],
    show=False,
    size=8,
    legend_loc="right margin",
    frameon=False,
    title="Previous support labels",
)

sc.pl.umap(
    adata_p28_support_marker_plot,
    color="sample",
    ax=axes[2],
    show=False,
    size=8,
    legend_loc="right margin",
    frameon=False,
    title="Sample",
)

plt.tight_layout()
plt.close("all")

sc.pl.dotplot(
    adata_p28_support_marker_plot,
    p28_support_marker_plot_dict,
    groupby=p28_support_marker_key,
    standard_scale="var",
    dendrogram=False,
    dot_max=0.85,
    dot_min=0.02,
    smallest_dot=0,
    color_map="Reds",
    figsize=(22, 6),
    show=False,
)


plt.close("all")


# Notebook cell 76

#1. =========== P28 local marker-informed reclustering ===========

RANDOM_STATE = 0

p28_local_source = adata_p28_final_candidate.copy()
gene_map_raw = {g.upper(): g for g in adata_P28.var_names}

def keep_genes_raw(genes):
    kept = []
    for gene in genes:
        real_gene = gene_map_raw.get(gene.upper())
        if real_gene is not None and real_gene not in kept:
            kept.append(real_gene)
    return kept

def filter_marker_dict_raw(marker_dict):
    return {
        key: keep_genes_raw(genes)
        for key, genes in marker_dict.items()
        if len(keep_genes_raw(genes)) > 0
    }

def res_tag(module_name, resolution):
    return f"{module_name}_r{int(round(resolution * 100)):03d}"

p28_local_modules = {
    "iphc_ib_hec": {
        "start_types": ["IPhC", "IB", "HeC"],
        "target_types": ["IPhC", "IB", "HeC"],
        "markers": {
            "IPhC": ["Matn4", "Slc1a3", "Sox2", "Gata3", "Smpx", "Epyc"],
            "IB": ["Sox2", "Gata3", "Smpx", "Epyc", "Slc1a3"],
            "HeC": ["Pmch", "Smpx", "Epyc", "Sox2", "Gata3"],
            "DC_PC_check": ["Ceacam16", "Rbp1", "Lgr6", "Ednrb"],
            "OSC_ISC_check": ["Gjb2", "Aqp4"],
            "HC_check": ["Myo7a", "Slc26a5", "Slc17a8"],
            "Must_negative": ["Oc90", "Otx2", "Neurod1", "Neurog1"],
        },
        "force_markers": ["Pmch", "Matn4", "Slc1a3", "S100a1", "Smpx", "Epyc", "Sox2", "Gata3"],
        "n_top_genes": 1200,
        "n_pcs": 20,
        "n_neighbors": 10,
    },
    "dc_pc": {
        "start_types": ["DC", "PC"],
        "target_types": ["DC", "PC"],
        "markers": {
            "DC": ["Ceacam16", "Rbp1", "Rbp7", "Fabp3", "Lgr5", "Fgfr3", "Prox1", "Gjb2"],
            "PC": ["Lgr6", "Ednrb", "Prdm12", "Smpx", "Sox2", "Gjb2"],
            "IPhC_IB_HeC_check": ["Slc1a3", "Matn4", "Pmch", "Epyc"],
            "OSC_ISC_check": ["Aqp4", "Gata3"],
            "HC_check": ["Myo7a", "Slc26a5", "Slc17a8"],
            "Must_negative": ["Oc90", "Otx2", "Neurod1", "Neurog1"],
        },
        "force_markers": [
            "Ceacam16", "Rbp1", "Rbp7", "Fabp3", "Lgr5", "Fgfr3", "Prox1",
            "Lgr6", "Ednrb", "Prdm12", "Smpx", "Sox2", "Gjb2"
        ],
        "n_top_genes": 1400,
        "n_pcs": 20,
        "n_neighbors": 12,
    },
    "osc_isc": {
        "start_types": ["OSC", "ISC"],
        "target_types": ["OSC", "ISC"],
        "markers": {
            "OSC": ["Gjb2", "Aqp4", "Epyc", "Bmp4", "Fst"],
            "ISC": ["Gjb2", "Gata3", "Aqp4", "Epyc"],
            "IPhC_IB_HeC_check": ["Slc1a3", "Matn4", "Pmch", "Smpx"],
            "DC_PC_check": ["Ceacam16", "Rbp1", "Lgr6", "Ednrb"],
            "HC_check": ["Myo7a", "Slc26a5", "Slc17a8"],
            "Must_negative": ["Oc90", "Otx2", "Neurod1", "Neurog1"],
        },
        "force_markers": ["Aqp4", "Gjb2", "Epyc", "Gata3", "Bmp4", "Fst", "Sox2"],
        "n_top_genes": 1400,
        "n_pcs": 20,
        "n_neighbors": 12,
    },
}

p28_local_resolutions = [0.45, 0.60, 0.80]
p28_local_scan_summary = []
p28_local_score_tables = {}
p28_local_plot_objects = {}

for module_name, module_info in p28_local_modules.items():
    module_obs = p28_local_source.obs_names[
        p28_local_source.obs["p28_cell_type_v2"].isin(module_info["start_types"])
    ]

    adata_local_raw = adata_P28[module_obs, :].copy()

    for col in ["p28_cell_type_v2", "sample", "sensory_leiden", "ambiguous_support_leiden"]:
        if col in p28_local_source.obs.columns:
            adata_local_raw.obs[col] = p28_local_source.obs.loc[adata_local_raw.obs_names, col].values

    adata_local_norm = adata_local_raw.copy()
    sc.pp.normalize_total(adata_local_norm, target_sum=1e4)
    sc.pp.log1p(adata_local_norm)

    sc.pp.highly_variable_genes(
        adata_local_norm,
        n_top_genes=module_info["n_top_genes"],
        flavor="seurat",
        batch_key="sample",
    )

    adata_local_norm.var["highly_variable"] = adata_local_norm.var["highly_variable"].fillna(False)

    qc_genes = (
        adata_local_norm.var_names.str.startswith(("mt-", "Mt-", "Rpl", "Rps"))
        | adata_local_norm.var_names.str.match(r"^Hb[ab]-")
        | adata_local_norm.var_names.isin(keep_genes_raw(["Oc90", "Otx2", "Neurod1", "Neurog1"]))
    )
    adata_local_norm.var.loc[qc_genes, "highly_variable"] = False
    adata_local_norm.var.loc[keep_genes_raw(module_info["force_markers"]), "highly_variable"] = True

    local_features = adata_local_norm.var_names[adata_local_norm.var["highly_variable"]].tolist()
    adata_local_work = adata_local_norm[:, local_features].copy()

    sc.pp.scale(adata_local_work, max_value=10)
    sc.tl.pca(
        adata_local_work,
        n_comps=module_info["n_pcs"],
        svd_solver="arpack",
        random_state=RANDOM_STATE,
    )

    sce.pp.harmony_integrate(
        adata_local_work,
        key="sample",
        basis="X_pca",
        adjusted_basis="X_pca_harmony",
        random_state=RANDOM_STATE,
    )

    for resolution in p28_local_resolutions:
        tag = res_tag(module_name, resolution)
        cluster_key = f"local_leiden_{tag}"
        umap_key = f"X_umap_{tag}"

        sc.pp.neighbors(
            adata_local_work,
            n_neighbors=module_info["n_neighbors"],
            n_pcs=module_info["n_pcs"],
            use_rep="X_pca_harmony",
            random_state=RANDOM_STATE,
        )
        sc.tl.umap(
            adata_local_work,
            min_dist=0.35,
            spread=1.0,
            random_state=RANDOM_STATE,
        )
        sc.tl.leiden(
            adata_local_work,
            resolution=resolution,
            key_added=cluster_key,
            random_state=RANDOM_STATE,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )

        adata_local_work.obsm[umap_key] = adata_local_work.obsm["X_umap"].copy()

        cluster_counts = adata_local_work.obs[cluster_key].value_counts()
        sample_counts = pd.crosstab(adata_local_work.obs[cluster_key], adata_local_work.obs["sample"])

        for sample in ["P28_1", "P28_2"]:
            if sample not in sample_counts.columns:
                sample_counts[sample] = 0

        sample_frac = sample_counts["P28_2"] / sample_counts[["P28_1", "P28_2"]].sum(axis=1)

        p28_local_scan_summary.append({
            "module": module_name,
            "setting": tag,
            "resolution": resolution,
            "n_cells": int(adata_local_work.n_obs),
            "n_clusters": int(cluster_counts.shape[0]),
            "min_cells": int(cluster_counts.min()),
            "clusters_lt30": int((cluster_counts < 30).sum()),
            "sample_skewed_clusters": int(((sample_frac > 0.85) | (sample_frac < 0.15)).sum()),
        })

    chosen_tag = res_tag(module_name, 0.60)
    chosen_key = f"local_leiden_{chosen_tag}"

    adata_local_plot = adata_local_norm.copy()
    adata_local_plot.obs[chosen_key] = adata_local_work.obs[chosen_key].values
    adata_local_plot.obsm["X_umap"] = adata_local_work.obsm[f"X_umap_{chosen_tag}"].copy()

    marker_dict = filter_marker_dict_raw(module_info["markers"])
    target_marker_dict = {
        key: marker_dict[key]
        for key in module_info["target_types"]
        if key in marker_dict
    }

    score_genes = keep_genes_raw(sum(target_marker_dict.values(), []))
    X_score = adata_local_plot[:, score_genes].X
    if hasattr(X_score, "toarray"):
        X_score = X_score.toarray()

    expr_score = pd.DataFrame(
        X_score,
        index=adata_local_plot.obs_names,
        columns=score_genes,
    )

    mean_by_cluster = expr_score.groupby(adata_local_plot.obs[chosen_key].astype(str)).mean()
    z_by_cluster = (mean_by_cluster - mean_by_cluster.mean(axis=0)) / mean_by_cluster.std(axis=0).replace(0, np.nan)
    z_by_cluster = z_by_cluster.fillna(0)

    local_score = pd.DataFrame(index=mean_by_cluster.index)

    for cell_type, genes in target_marker_dict.items():
        local_score[cell_type] = z_by_cluster[genes].mean(axis=1)

    ranked_types = local_score.apply(lambda row: row.sort_values(ascending=False).index.tolist(), axis=1)
    ranked_scores = local_score.apply(lambda row: row.sort_values(ascending=False).values.tolist(), axis=1)

    local_sample_table = pd.crosstab(adata_local_plot.obs[chosen_key], adata_local_plot.obs["sample"])
    for sample in ["P28_1", "P28_2"]:
        if sample not in local_sample_table.columns:
            local_sample_table[sample] = 0

    local_sample_table["n_cells"] = local_sample_table[["P28_1", "P28_2"]].sum(axis=1)

    local_score_summary = pd.DataFrame({
        "n_cells": local_sample_table["n_cells"].reindex(local_score.index).astype(int),
        "P28_1": local_sample_table["P28_1"].reindex(local_score.index).astype(int),
        "P28_2": local_sample_table["P28_2"].reindex(local_score.index).astype(int),
        "P28_2_frac": (
            local_sample_table["P28_2"].reindex(local_score.index)
            / local_sample_table["n_cells"].reindex(local_score.index)
        ).round(3),
        "best_type": ranked_types.str[0],
        "best_score": ranked_scores.str[0],
        "second_type": ranked_types.str[1],
        "second_score": ranked_scores.str[1],
    })

    local_score_summary["score_margin"] = (
        local_score_summary["best_score"] - local_score_summary["second_score"]
    ).round(3)

    p28_local_score_tables[module_name] = local_score_summary
    p28_local_plot_objects[module_name] = {
        "plot": adata_local_plot,
        "work": adata_local_work,
        "chosen_key": chosen_key,
        "markers": marker_dict,
    }

p28_local_scan_summary = pd.DataFrame(p28_local_scan_summary)

display(p28_local_scan_summary)

for module_name, score_table in p28_local_score_tables.items():
    display(pd.DataFrame({"module": [module_name], "chosen_resolution_for_dotplot": [0.60]}))
    display(score_table.sort_index())

for module_name, module_info in p28_local_modules.items():
    local_obj = p28_local_plot_objects[module_name]
    adata_local_work = local_obj["work"]
    adata_local_plot = local_obj["plot"]
    chosen_key = local_obj["chosen_key"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, resolution in zip(axes, p28_local_resolutions):
        tag = res_tag(module_name, resolution)
        sc.pl.embedding(
            adata_local_work,
            basis=f"umap_{tag}",
            color=f"local_leiden_{tag}",
            ax=ax,
            show=False,
            size=10,
            legend_loc="on data",
            legend_fontsize=8,
            legend_fontoutline=2,
            frameon=False,
            title=f"{module_name} res={resolution}",
        )

    plt.tight_layout()
    plt.close("all")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    sc.pl.umap(
        adata_local_plot,
        color=chosen_key,
        ax=axes[0],
        show=False,
        size=10,
        legend_loc="right margin",
        frameon=False,
        title=f"{module_name} clusters",
    )

    sc.pl.umap(
        adata_local_plot,
        color="p28_cell_type_v2",
        ax=axes[1],
        show=False,
        size=10,
        legend_loc="right margin",
        frameon=False,
        title="Previous labels",
    )

    sc.pl.umap(
        adata_local_plot,
        color="sample",
        ax=axes[2],
        show=False,
        size=10,
        legend_loc="right margin",
        frameon=False,
        title="Sample",
    )

    plt.tight_layout()
    plt.close("all")

    sc.pl.dotplot(
        adata_local_plot,
        local_obj["markers"],
        groupby=chosen_key,
        standard_scale="var",
        dendrogram=False,
        dot_max=0.85,
        dot_min=0.02,
        smallest_dot=0,
        color_map="Reds",
        figsize=(18, 4.5),
        show=False,
    )


plt.close("all")


# Notebook cell 77

#1. =========== P28 key-marker direct identity check ===========

p28_key_marker_config = {
    "iphc_ib_hec": {
        "key": p28_local_plot_objects["iphc_ib_hec"]["chosen_key"],
        "gene_sets": {
            "HeC_key": ["Pmch"],
            "IPhC_key": ["Matn4", "Slc1a3"],
            "IB_candidate": ["Sox2", "Gata3", "Smpx", "Epyc"],
            "shared_check": ["S100a1", "Gjb2", "Aqp4"],
            "negative": ["Oc90", "Otx2", "Neurod1", "Neurog1"],
        },
        "direct_genes": ["Pmch", "Matn4", "Slc1a3", "S100a1", "Sox2", "Gata3", "Smpx", "Epyc"],
    },
    "dc_pc": {
        "key": p28_local_plot_objects["dc_pc"]["chosen_key"],
        "gene_sets": {
            "DC_key": ["Ceacam16", "Rbp1", "Rbp7", "Fabp3", "Prox1", "Lgr5", "Fgfr3"],
            "PC_key": ["Lgr6", "Ednrb", "Prdm12", "Smpx"],
            "shared_check": ["Sox2", "Gjb2"],
            "other_support_check": ["Slc1a3", "Matn4", "Pmch", "Aqp4", "Gata3"],
            "negative": ["Oc90", "Otx2", "Neurod1", "Neurog1"],
        },
        "direct_genes": ["Ceacam16", "Rbp1", "Rbp7", "Fabp3", "Prox1", "Lgr5", "Fgfr3", "Lgr6", "Ednrb", "Prdm12", "Smpx"],
    },
    "osc_isc": {
        "key": p28_local_plot_objects["osc_isc"]["chosen_key"],
        "gene_sets": {
            "OSC_key": ["Bmp4", "Fst", "Slc26a4"],
            "ISC_key": ["Gata3", "Aqp4", "Gjb2", "Epyc"],
            "IPhC_IB_HeC_check": ["Slc1a3", "Matn4", "Pmch", "Smpx"],
            "DC_PC_check": ["Ceacam16", "Rbp1", "Lgr6", "Ednrb"],
            "negative": ["Oc90", "Otx2", "Neurod1", "Neurog1"],
        },
        "direct_genes": ["Bmp4", "Fst", "Slc26a4", "Gata3", "Aqp4", "Gjb2", "Epyc"],
    },
}

p28_key_identity_tables = {}

for module_name, config in p28_key_marker_config.items():
    adata_local = p28_local_plot_objects[module_name]["plot"]
    cluster_key = config["key"]
    gene_map_local = {g.upper(): g for g in adata_local.var_names}

    def keep_genes_local(genes):
        kept = []
        for gene in genes:
            real_gene = gene_map_local.get(gene.upper())
            if real_gene is not None and real_gene not in kept:
                kept.append(real_gene)
        return kept

    all_genes = keep_genes_local(sum(config["gene_sets"].values(), []))
    X = adata_local[:, all_genes].X
    if hasattr(X, "toarray"):
        X = X.toarray()

    expr = pd.DataFrame(X, index=adata_local.obs_names, columns=all_genes)
    cluster_series = adata_local.obs[cluster_key].astype(str)

    mean_table = expr.groupby(cluster_series).mean()
    pct_table = (expr > 0).groupby(cluster_series).mean() * 100

    z_table = (mean_table - mean_table.mean(axis=0)) / mean_table.std(axis=0).replace(0, np.nan)
    z_table = z_table.fillna(0)

    key_score = pd.DataFrame(index=mean_table.index)

    for set_name, genes in config["gene_sets"].items():
        if set_name == "negative":
            continue
        genes = keep_genes_local(genes)
        if len(genes) > 0:
            key_score[set_name] = z_table[genes].mean(axis=1)

    ranked_sets = key_score.apply(lambda row: row.sort_values(ascending=False).index.tolist(), axis=1)
    ranked_scores = key_score.apply(lambda row: row.sort_values(ascending=False).values.tolist(), axis=1)

    sample_table = pd.crosstab(adata_local.obs[cluster_key].astype(str), adata_local.obs["sample"])
    for sample in ["P28_1", "P28_2"]:
        if sample not in sample_table.columns:
            sample_table[sample] = 0

    sample_table["n_cells"] = sample_table[["P28_1", "P28_2"]].sum(axis=1)

    identity_table = pd.DataFrame({
        "n_cells": sample_table["n_cells"].reindex(mean_table.index).astype(int),
        "P28_1": sample_table["P28_1"].reindex(mean_table.index).astype(int),
        "P28_2": sample_table["P28_2"].reindex(mean_table.index).astype(int),
        "P28_2_frac": (
            sample_table["P28_2"].reindex(mean_table.index)
            / sample_table["n_cells"].reindex(mean_table.index)
        ).round(3),
        "top_key_set": ranked_sets.str[0],
        "top_key_score": ranked_scores.str[0],
        "second_key_set": ranked_sets.str[1],
        "second_key_score": ranked_scores.str[1],
    })

    identity_table["key_margin"] = (
        identity_table["top_key_score"] - identity_table["second_key_score"]
    ).round(3)

    direct_genes = keep_genes_local(config["direct_genes"])
    for gene in direct_genes:
        identity_table[f"{gene}_mean"] = mean_table[gene].round(3)
        identity_table[f"{gene}_pct"] = pct_table[gene].round(1)

    negative_genes = keep_genes_local(["Oc90", "Otx2", "Neurod1", "Neurog1"])
    if len(negative_genes) > 0:
        identity_table["max_negative_pct"] = pct_table[negative_genes].max(axis=1).round(1)
    else:
        identity_table["max_negative_pct"] = 0.0

    identity_table["direct_call"] = "ambiguous"
    identity_table["confidence"] = "check"

    if module_name == "iphc_ib_hec":
        if "Pmch_pct" in identity_table.columns:
            identity_table.loc[
                (identity_table["Pmch_pct"] >= 10) & (identity_table["Pmch_mean"] > 0),
                ["direct_call", "confidence"]
            ] = ["HeC", "high_if_Pmch_specific"]

        iphc_mask = pd.Series(False, index=identity_table.index)
        if "Matn4_pct" in identity_table.columns:
            iphc_mask = iphc_mask | ((identity_table["Matn4_pct"] >= 10) & (identity_table["Matn4_mean"] > 0))
        if "Slc1a3_pct" in identity_table.columns:
            iphc_mask = iphc_mask | ((identity_table["Slc1a3_pct"] >= 35) & (identity_table["Slc1a3_mean"] > 0))

        identity_table.loc[
            (identity_table["direct_call"] == "ambiguous") & iphc_mask,
            ["direct_call", "confidence"]
        ] = ["IPhC", "medium_key_supported"]

        identity_table.loc[
            (identity_table["direct_call"] == "ambiguous")
            & (identity_table["top_key_set"] == "IB_candidate"),
            ["direct_call", "confidence"]
        ] = ["IB", "low_no_unique_marker"]

    if module_name == "dc_pc":
        identity_table.loc[
            identity_table["top_key_set"] == "DC_key",
            ["direct_call", "confidence"]
        ] = ["DC", "high_key_supported"]

        identity_table.loc[
            identity_table["top_key_set"] == "PC_key",
            ["direct_call", "confidence"]
        ] = ["PC", "high_key_supported"]

    if module_name == "osc_isc":
        identity_table.loc[
            identity_table["top_key_set"] == "OSC_key",
            ["direct_call", "confidence"]
        ] = ["OSC", "medium_key_supported"]

        identity_table.loc[
            identity_table["top_key_set"] == "ISC_key",
            ["direct_call", "confidence"]
        ] = ["ISC", "low_shared_marker"]

        if "Bmp4_pct" in identity_table.columns:
            identity_table.loc[
                (identity_table["Bmp4_pct"] >= 15) & (identity_table["Bmp4_mean"] > 0),
                ["direct_call", "confidence"]
            ] = ["OSC", "medium_Bmp4_supported"]

        if "Slc26a4_pct" in identity_table.columns:
            identity_table.loc[
                (identity_table["Slc26a4_pct"] >= 10) & (identity_table["Slc26a4_mean"] > 0),
                ["direct_call", "confidence"]
            ] = ["OSC", "medium_Slc26a4_supported"]

    identity_table.loc[
        identity_table["n_cells"] < 30,
        "confidence"
    ] = identity_table.loc[identity_table["n_cells"] < 30, "confidence"] + "_small_cluster"

    identity_table.loc[
        (identity_table["P28_2_frac"] > 0.85) | (identity_table["P28_2_frac"] < 0.15),
        "confidence"
    ] = identity_table.loc[
        (identity_table["P28_2_frac"] > 0.85) | (identity_table["P28_2_frac"] < 0.15),
        "confidence"
    ] + "_sample_skew"

    identity_table.loc[
        identity_table["max_negative_pct"] > 5,
        "confidence"
    ] = identity_table.loc[identity_table["max_negative_pct"] > 5, "confidence"] + "_negative_check"

    p28_key_identity_tables[module_name] = identity_table

    display(pd.DataFrame({"module": [module_name], "cluster_key": [cluster_key]}))
    display(identity_table)

    key_marker_dict = {
        set_name: keep_genes_local(genes)
        for set_name, genes in config["gene_sets"].items()
        if len(keep_genes_local(genes)) > 0
    }

    sc.pl.dotplot(
        adata_local,
        key_marker_dict,
        groupby=cluster_key,
        standard_scale="var",
        dendrogram=False,
        dot_max=0.85,
        dot_min=0.02,
        smallest_dot=0,
        color_map="Reds",
        figsize=(18, 4.2),
        show=False,
    )


plt.close("all")


# Notebook cell 78

#1. =========== P28 key-marker-guided final candidate labels ===========

adata_p28_key_guided = adata_p28_final_candidate.copy()

if "X_umap" not in adata_p28_key_guided.obsm and "adata_p28_final_plot" in globals():
    adata_p28_key_guided.obsm["X_umap"] = adata_p28_final_plot[
        adata_p28_key_guided.obs_names, :
    ].obsm["X_umap"].copy()

adata_p28_key_guided.obs["p28_cell_type_v4_key_guided"] = adata_p28_key_guided.obs["p28_cell_type_v2"].astype(str)
adata_p28_key_guided.obs["p28_cell_type_v4_confidence"] = "kept_from_v2"

local_key_maps = {
    "iphc_ib_hec": {
        "0": ("IB", "low_no_unique_IB_marker"),
        "1": ("IB", "low_small_negative_check"),
        "2": ("IPhC", "medium_Slc1a3_supported_small"),
        "3": ("IPhC", "medium_Slc1a3_supported"),
        "4": ("HeC", "low_Pmch_absent"),
        "5": ("IPhC", "medium_Slc1a3_supported"),
        "6": ("IPhC", "low_small_sample_skew"),
        "7": ("IPhC", "medium_Slc1a3_supported"),
        "8": ("HeC", "low_Pmch_absent_sample_skew"),
        "9": ("IB", "low_small_cluster"),
    },
    "dc_pc": {
        "0": ("PC", "medium_PC_key_with_shared_markers"),
        "1": ("PC", "medium_PC_key_with_shared_markers"),
        "2": ("DC", "high_Ceacam16_Rbp_supported"),
        "3": ("DC", "high_Ceacam16_Rbp_supported"),
        "4": ("PC", "low_Ednrb_supported_sample_skew"),
        "5": ("PC", "medium_Ednrb_Prmd12_supported"),
    },
    "osc_isc": {
        "0": ("IB", "low_IPhC_IB_HeC_marker_contamination"),
        "1": ("ISC", "medium_Gata3_Gjb2_Epyc_supported"),
        "2": ("OSC", "medium_Bmp4_supported"),
        "3": ("OSC", "low_Bmp4_supported_small_sample_skew"),
        "4": ("DC", "low_DC_PC_marker_contamination"),
        "5": ("OSC", "low_Bmp4_supported_small"),
        "6": ("DC", "low_DC_PC_marker_contamination"),
        "7": ("ISC", "medium_Gata3_Aqp4_Gjb2_supported"),
        "8": ("OSC", "medium_Bmp4_Gjb2_Epyc_supported"),
    },
}

for module_name, cluster_map in local_key_maps.items():
    adata_local = p28_local_plot_objects[module_name]["plot"]
    cluster_key = p28_local_plot_objects[module_name]["chosen_key"]

    local_clusters = adata_local.obs[cluster_key].astype(str)

    for cluster_id, (cell_type, confidence) in cluster_map.items():
        local_obs = adata_local.obs_names[local_clusters == cluster_id]
        local_obs = local_obs[local_obs.isin(adata_p28_key_guided.obs_names)]

        adata_p28_key_guided.obs.loc[local_obs, "p28_cell_type_v4_key_guided"] = cell_type
        adata_p28_key_guided.obs.loc[local_obs, "p28_cell_type_v4_confidence"] = confidence

p28_type_order = ["OHC", "IHC", "IPhC", "IB", "HeC", "DC", "PC", "OSC", "ISC"]
adata_p28_key_guided.obs["p28_cell_type_v4_key_guided"] = pd.Categorical(
    adata_p28_key_guided.obs["p28_cell_type_v4_key_guided"],
    categories=[x for x in p28_type_order if x in adata_p28_key_guided.obs["p28_cell_type_v4_key_guided"].unique()],
    ordered=True,
)

p28_v4_table = pd.crosstab(
    adata_p28_key_guided.obs["p28_cell_type_v4_key_guided"],
    adata_p28_key_guided.obs["sample"],
)

for sample in ["P28_1", "P28_2"]:
    if sample not in p28_v4_table.columns:
        p28_v4_table[sample] = 0

p28_v4_table["n_cells"] = p28_v4_table[["P28_1", "P28_2"]].sum(axis=1)
p28_v4_table["P28_2_frac"] = (p28_v4_table["P28_2"] / p28_v4_table["n_cells"]).round(3)

p28_v4_confidence_table = pd.crosstab(
    adata_p28_key_guided.obs["p28_cell_type_v4_key_guided"],
    adata_p28_key_guided.obs["p28_cell_type_v4_confidence"],
)

display(p28_v4_table[["P28_1", "P28_2", "n_cells", "P28_2_frac"]])
display(p28_v4_confidence_table)

if "sensory_leiden" in adata_p28_key_guided.obs.columns:
    display(pd.crosstab(
        adata_p28_key_guided.obs["p28_cell_type_v4_key_guided"],
        adata_p28_key_guided.obs["sensory_leiden"],
    ))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sc.pl.umap(
    adata_p28_key_guided,
    color="p28_cell_type_v4_key_guided",
    ax=axes[0],
    show=False,
    size=8,
    legend_loc="right margin",
    frameon=False,
    title="P28 key-guided cell types",
)

sc.pl.umap(
    adata_p28_key_guided,
    color="p28_cell_type_v4_confidence",
    ax=axes[1],
    show=False,
    size=8,
    legend_loc="right margin",
    frameon=False,
    title="Confidence",
)

sc.pl.umap(
    adata_p28_key_guided,
    color="sample",
    ax=axes[2],
    show=False,
    size=8,
    legend_loc="right margin",
    frameon=False,
    title="Sample",
)

plt.tight_layout()
plt.close("all")

sc.pl.dotplot(
    adata_p28_key_guided,
    p28_marker_plot_dict,
    groupby="p28_cell_type_v4_key_guided",
    standard_scale="var",
    dendrogram=False,
    dot_max=0.85,
    dot_min=0.02,
    smallest_dot=0,
    color_map="Reds",
    figsize=(22, 5),
    show=False,
)


plt.close("all")


# Notebook cell 79

#1. =========== P28 UMAP with confidence in cell-type labels ===========

adata_p28_key_guided_plot = adata_p28_key_guided.copy()

def simplify_p28_confidence(confidence):
    confidence = str(confidence)

    if confidence == "kept_from_v2":
        return "high"

    if confidence.startswith("high"):
        return "high"

    if confidence.startswith("medium"):
        return "medium"

    if confidence.startswith("low"):
        return "low"

    return "check"

adata_p28_key_guided_plot.obs["p28_confidence_simple"] = (
    adata_p28_key_guided_plot.obs["p28_cell_type_v4_confidence"]
    .map(simplify_p28_confidence)
)

adata_p28_key_guided_plot.obs["p28_cell_type_v4_label"] = (
    adata_p28_key_guided_plot.obs["p28_cell_type_v4_key_guided"].astype(str)
    + " ("
    + adata_p28_key_guided_plot.obs["p28_confidence_simple"].astype(str)
    + ")"
)

p28_label_order = []
for cell_type in ["OHC", "IHC", "IPhC", "IB", "HeC", "DC", "PC", "OSC", "ISC"]:
    for confidence in ["high", "medium", "low", "check"]:
        label = f"{cell_type} ({confidence})"
        if label in adata_p28_key_guided_plot.obs["p28_cell_type_v4_label"].unique():
            p28_label_order.append(label)

adata_p28_key_guided_plot.obs["p28_cell_type_v4_label"] = pd.Categorical(
    adata_p28_key_guided_plot.obs["p28_cell_type_v4_label"],
    categories=p28_label_order,
    ordered=True,
)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

sc.pl.umap(
    adata_p28_key_guided_plot,
    color="p28_cell_type_v4_label",
    ax=axes[0],
    show=False,
    size=8,
    legend_loc="right margin",
    frameon=False,
    title="P28 key-guided cell types",
)

sc.pl.umap(
    adata_p28_key_guided_plot,
    color="sample",
    ax=axes[1],
    show=False,
    size=8,
    legend_loc="right margin",
    frameon=False,
    title="Sample",
)

plt.tight_layout()
plt.close("all")


plt.close("all")


# Notebook cell 80

#1. =========== P28 UMAP overview and zoomed view ===========

adata_p28_key_guided_plot = adata_p28_key_guided_plot.copy()

x = adata_p28_key_guided_plot.obsm["X_umap"][:, 0]
y = adata_p28_key_guided_plot.obsm["X_umap"][:, 1]

support_mask = ~adata_p28_key_guided_plot.obs["p28_cell_type_v4_key_guided"].isin(["OHC", "IHC"])
x_support = x[support_mask]
y_support = y[support_mask]

x_pad = (np.percentile(x_support, 99) - np.percentile(x_support, 1)) * 0.08
y_pad = (np.percentile(y_support, 99) - np.percentile(y_support, 1)) * 0.08

xlim_zoom = (
    np.percentile(x_support, 1) - x_pad,
    np.percentile(x_support, 99) + x_pad,
)
ylim_zoom = (
    np.percentile(y_support, 1) - y_pad,
    np.percentile(y_support, 99) + y_pad,
)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

sc.pl.umap(
    adata_p28_key_guided_plot,
    color="p28_cell_type_v4_label",
    ax=axes[0],
    show=False,
    size=8,
    legend_loc="right margin",
    frameon=False,
    title="P28 full UMAP",
)

sc.pl.umap(
    adata_p28_key_guided_plot,
    color="p28_cell_type_v4_label",
    ax=axes[1],
    show=False,
    size=8,
    legend_loc=None,
    frameon=False,
    title="P28 support-region zoom",
)
axes[1].set_xlim(xlim_zoom)
axes[1].set_ylim(ylim_zoom)

sc.pl.umap(
    adata_p28_key_guided_plot,
    color="sample",
    ax=axes[2],
    show=False,
    size=8,
    legend_loc="right margin",
    frameon=False,
    title="Sample",
)
axes[2].set_xlim(xlim_zoom)
axes[2].set_ylim(ylim_zoom)

plt.tight_layout()
plt.close("all")


plt.close("all")


# Notebook cell 81

#1. =========== Prepare P28 v4 expression for dotplot ===========

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)
sc.settings.verbosity = 0

adata_p28_v4_dotplot = adata_p28_key_guided_plot.copy()

if "counts" in adata_p28_v4_dotplot.layers:
    adata_p28_v4_dotplot.X = adata_p28_v4_dotplot.layers["counts"].copy()

sc.pp.normalize_total(adata_p28_v4_dotplot, target_sum=1e4)
sc.pp.log1p(adata_p28_v4_dotplot)

p28_v4_group_key = "p28_cell_type_v4_label"

p28_v4_group_order = [
    group for group in p28_label_order
    if group in adata_p28_v4_dotplot.obs[p28_v4_group_key].astype(str).unique()
]

adata_p28_v4_dotplot.obs[p28_v4_group_key] = pd.Categorical(
    adata_p28_v4_dotplot.obs[p28_v4_group_key].astype(str),
    categories=p28_v4_group_order,
    ordered=True
)


#2. =========== Read P28 marker genes from Excel ===========

marker_df = pd.read_excel(
    REPO_ROOT / "config" / "cell_type_markers.xlsx",
    sheet_name="Cochlear epithelium"
)

marker_df["Stage_ffill"] = marker_df["Stage"].ffill()

p28_marker_rows = marker_df[
    marker_df["Stage_ffill"].astype(str).str.contains("P28", case=False, na=False)
].copy()

marker_cols = [
    col for col in p28_marker_rows.columns
    if col not in ["Stage", "Stage_ffill", "Cell-type or Region"]
]

gene_lookup = {gene.upper(): gene for gene in adata_p28_v4_dotplot.var_names}

def match_marker_gene(gene):
    gene = str(gene).strip()

    if gene == "" or gene.lower() == "nan":
        return []

    if gene.upper() in gene_lookup:
        return [gene_lookup[gene.upper()]]

    if gene.upper() == "RBP":
        return [
            gene_lookup[alias.upper()]
            for alias in ["Rbp1", "Rbp7"]
            if alias.upper() in gene_lookup
        ]

    return []

def get_p28_marker_group(cell_region):
    cell_region = str(cell_region)

    if cell_region == "HC":
        return "HC_broad_check"
    if "Outer Hair" in cell_region or "OHC" in cell_region:
        return "OHC"
    if "Inner Hari" in cell_region or "IHC" in cell_region:
        return "IHC"
    if "Inner Phalangeal" in cell_region or "IPhC" in cell_region:
        return "IPhC"
    if "Inner_border" in cell_region:
        return "Inner_border_phalangeal_Hensen_check"
    if "Hensen" in cell_region or "HeC" in cell_region:
        return "HeC"
    if "Inner Border" in cell_region or "IB" in cell_region:
        return "IB"
    if "Deiters" in cell_region or "DC" in cell_region:
        return "DC"
    if "Pillar" in cell_region or "PC" in cell_region:
        return "PC"
    if "Outer Sulcus" in cell_region or "OSC" in cell_region:
        return "OSC"
    if "Inner sulcus" in cell_region or "ISC" in cell_region:
        return "ISC"

    return cell_region

p28_excel_marker_dict = {}

for _, row in p28_marker_rows.iterrows():
    marker_group = get_p28_marker_group(row["Cell-type or Region"])
    marker_genes = []

    for col in marker_cols:
        for gene in match_marker_gene(row[col]):
            if gene not in marker_genes:
                marker_genes.append(gene)

    if len(marker_genes) > 0:
        p28_excel_marker_dict[marker_group] = marker_genes

p28_excel_marker_dict["Must_negative"] = [
    gene_lookup[gene.upper()]
    for gene in ["Oc90", "Otx2", "Neurod1", "Neurog1"]
    if gene.upper() in gene_lookup
]

p28_marker_group_order = [
    "HC_broad_check",
    "OHC",
    "IHC",
    "IPhC",
    "Inner_border_phalangeal_Hensen_check",
    "HeC",
    "IB",
    "DC",
    "PC",
    "OSC",
    "ISC",
    "Must_negative",
]

p28_excel_marker_dict = {
    group: p28_excel_marker_dict[group]
    for group in p28_marker_group_order
    if group in p28_excel_marker_dict and len(p28_excel_marker_dict[group]) > 0
}


#3. =========== Plot matched Scanpy-style dotplot ===========

sc.pl.dotplot(
    adata_p28_v4_dotplot,
    p28_excel_marker_dict,
    groupby=p28_v4_group_key,
    categories_order=p28_v4_group_order,
    standard_scale="var",
    cmap="Reds",
    dot_max=0.8,
    figsize=(18, 5.8),
    show=False
)


plt.close("all")


# Notebook cell 82

#1. =========== P28 remove low-confidence groups before final save ===========

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)
sc.settings.verbosity = 0

p28_filter_celltype_key = "p28_cell_type_v4_key_guided"
p28_filter_confidence_key = "p28_cell_type_v4_confidence"
p28_label_key = "p28_cell_type_v4_label"

# Keep the final cell-type names and order consistent with config/cell_type_markers.xlsx.
p28_excel_celltype_order = ["OHC", "IHC", "IPhC", "IB", "HeC", "DC", "PC", "OSC", "ISC"]

# User-requested exception: keep IB even though its current confidence is low.
p28_keep_low_celltypes_user = ["IB"]

if "adata_p28_key_guided_plot" not in globals():
    raise NameError("adata_p28_key_guided_plot is missing. Run the P28 key-guided UMAP code first.")

for required_key in [p28_filter_celltype_key, p28_filter_confidence_key, "sample"]:
    if required_key not in adata_p28_key_guided_plot.obs.columns:
        raise KeyError(f"{required_key} not found in adata_p28_key_guided_plot.obs.")

def p28_confidence_to_simple(confidence):
    confidence = str(confidence)

    if confidence == "kept_from_v2" or confidence.startswith("high"):
        return "high"
    if confidence.startswith("medium"):
        return "medium"
    if confidence.startswith("low"):
        return "low"

    return "check"

adata_p28_filter_source = adata_p28_key_guided_plot.copy()
adata_p28_filter_source.obs["p28_confidence_simple"] = (
    adata_p28_filter_source.obs[p28_filter_confidence_key]
    .map(p28_confidence_to_simple)
    .astype(str)
)

p28_celltype_series = adata_p28_filter_source.obs[p28_filter_celltype_key].astype(str)
p28_confidence_series = adata_p28_filter_source.obs["p28_confidence_simple"].astype(str)

p28_before_counts = (
    p28_celltype_series.value_counts()
    .reindex(p28_excel_celltype_order)
    .fillna(0)
    .astype(int)
)

p28_low_mask = p28_confidence_series.eq("low")
p28_strict_keep_mask = ~(p28_low_mask & ~p28_celltype_series.isin(p28_keep_low_celltypes_user))
p28_strict_present_types = set(p28_celltype_series[p28_strict_keep_mask].unique())

# If strict low removal would erase an Excel-listed type entirely, keep that low type so the
# final UMAP/dotplot still has the same cell-type set as the marker table.
p28_keep_low_to_preserve_types = [
    cell_type for cell_type in p28_excel_celltype_order
    if p28_before_counts.loc[cell_type] > 0 and cell_type not in p28_strict_present_types
]
p28_keep_low_celltypes = [
    cell_type for cell_type in p28_excel_celltype_order
    if cell_type in set(p28_keep_low_celltypes_user + p28_keep_low_to_preserve_types)
]

p28_drop_low_mask = p28_low_mask & ~p28_celltype_series.isin(p28_keep_low_celltypes)
p28_keep_obs = adata_p28_filter_source.obs_names[~p28_drop_low_mask.values]

p28_after_counts = (
    p28_celltype_series.loc[p28_keep_obs].value_counts()
    .reindex(p28_excel_celltype_order)
    .fillna(0)
    .astype(int)
)
p28_strict_after_counts = (
    p28_celltype_series[p28_strict_keep_mask].value_counts()
    .reindex(p28_excel_celltype_order)
    .fillna(0)
    .astype(int)
)

p28_low_filter_summary = pd.DataFrame(
    {
        "before": p28_before_counts,
        "strict_after_drop_low_except_IB": p28_strict_after_counts,
        "final_after_preserve_marker_table_types": p28_after_counts,
        "removed": p28_before_counts - p28_after_counts,
    }
)

p28_low_by_type = pd.crosstab(p28_celltype_series, p28_confidence_series)
p28_low_by_type = p28_low_by_type.reindex(
    index=p28_excel_celltype_order,
    columns=["high", "medium", "low", "check"],
    fill_value=0,
).astype(int)

print("Low-confidence cell types kept:", p28_keep_low_celltypes)
display(p28_low_by_type)
display(p28_low_filter_summary)

adata_p28_key_guided_plot = adata_p28_filter_source[p28_keep_obs, :].copy()

if "adata_p28_key_guided" in globals():
    adata_p28_key_guided = adata_p28_key_guided[p28_keep_obs, :].copy()
else:
    adata_p28_key_guided = adata_p28_key_guided_plot.copy()

def finalize_p28_filtered_labels(adata_obj):
    adata_obj.obs[p28_filter_celltype_key] = pd.Categorical(
        adata_obj.obs[p28_filter_celltype_key].astype(str),
        categories=p28_excel_celltype_order,
        ordered=True,
    )
    adata_obj.obs["p28_confidence_simple"] = (
        adata_obj.obs[p28_filter_confidence_key]
        .map(p28_confidence_to_simple)
        .astype(str)
    )
    adata_obj.obs[p28_label_key] = adata_obj.obs[p28_filter_celltype_key].astype(str)
    adata_obj.obs[p28_label_key] = pd.Categorical(
        adata_obj.obs[p28_label_key].astype(str),
        categories=[
            cell_type for cell_type in p28_excel_celltype_order
            if cell_type in adata_obj.obs[p28_label_key].astype(str).unique()
        ],
        ordered=True,
    )
    return adata_obj

adata_p28_key_guided_plot = finalize_p28_filtered_labels(adata_p28_key_guided_plot)
adata_p28_key_guided = finalize_p28_filtered_labels(adata_p28_key_guided)

p28_v4_group_key = p28_label_key
p28_v4_group_order = [
    cell_type for cell_type in p28_excel_celltype_order
    if cell_type in adata_p28_key_guided_plot.obs[p28_v4_group_key].astype(str).unique()
]
p28_label_order = p28_v4_group_order.copy()

# Let the final save cell validate against the filtered counts instead of the old unfiltered counts.
p28_expected_counts_override = (
    adata_p28_key_guided_plot.obs[p28_filter_celltype_key]
    .astype(str)
    .value_counts()
    .reindex(p28_excel_celltype_order)
    .fillna(0)
    .astype(int)
)

gene_lookup = {gene.upper(): gene for gene in adata_p28_key_guided_plot.var_names}
p28_fabp7_gene = gene_lookup.get("FABP7")

if p28_fabp7_gene is None:
    print("Fabp7 was not found in adata_p28_key_guided_plot.var_names; it will not be added to the dotplot.")
else:
    if "p28_excel_marker_dict" not in globals():
        raise NameError("p28_excel_marker_dict is missing. Run the P28 Excel marker dotplot code first.")

    p28_excel_marker_dict = {
        group: list(genes)
        for group, genes in p28_excel_marker_dict.items()
    }
    p28_excel_marker_dict.setdefault("HeC", [])

    if p28_fabp7_gene not in p28_excel_marker_dict["HeC"]:
        p28_excel_marker_dict["HeC"].append(p28_fabp7_gene)

    print(f"Added {p28_fabp7_gene} to the HeC dotplot marker group.")

adata_p28_v4_dotplot = adata_p28_key_guided_plot.copy()

if "counts" in adata_p28_v4_dotplot.layers:
    adata_p28_v4_dotplot.X = adata_p28_v4_dotplot.layers["counts"].copy()

sc.pp.normalize_total(adata_p28_v4_dotplot, target_sum=1e4)
sc.pp.log1p(adata_p28_v4_dotplot)

adata_p28_v4_dotplot.obs[p28_v4_group_key] = pd.Categorical(
    adata_p28_v4_dotplot.obs[p28_v4_group_key].astype(str),
    categories=p28_v4_group_order,
    ordered=True,
)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

sc.pl.umap(
    adata_p28_key_guided_plot,
    color=p28_filter_celltype_key,
    ax=axes[0],
    show=False,
    size=8,
    legend_loc="right margin",
    frameon=False,
    title="P28 after low-confidence filtering",
)

sc.pl.umap(
    adata_p28_key_guided_plot,
    color="sample",
    ax=axes[1],
    show=False,
    size=8,
    legend_loc="right margin",
    frameon=False,
    title="Sample",
)

plt.tight_layout()
plt.close("all")

sc.pl.dotplot(
    adata_p28_v4_dotplot,
    p28_excel_marker_dict,
    groupby=p28_v4_group_key,
    categories_order=p28_v4_group_order,
    standard_scale="var",
    cmap="Reds",
    dot_max=0.8,
    figsize=(18, 5.2),
    show=False,
)


plt.close("all")


# Notebook cell 83

#1. =========== Rebuild P28 key-guided labels from unfiltered object ===========

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)
sc.settings.verbosity = 0

p28_filter_celltype_key = "p28_cell_type_v4_key_guided"
p28_filter_confidence_key = "p28_cell_type_v4_confidence"
p28_label_key = "p28_cell_type_v4_label"
p28_label_n_key = "p28_cell_type_v4_label_n"

p28_excel_celltype_order = ["OHC", "IHC", "IPhC", "IB", "HeC", "DC", "PC", "OSC", "ISC"]

adata_p28_key_guided = adata_p28_final_candidate.copy()

if "X_umap" not in adata_p28_key_guided.obsm and "adata_p28_final_plot" in globals():
    adata_p28_key_guided.obsm["X_umap"] = adata_p28_final_plot[
        adata_p28_key_guided.obs_names, :
    ].obsm["X_umap"].copy()

adata_p28_key_guided.obs[p28_filter_celltype_key] = (
    adata_p28_key_guided.obs["p28_cell_type_v2"].astype(str)
)
adata_p28_key_guided.obs[p28_filter_confidence_key] = "kept_from_v2"

local_key_maps = {
    "iphc_ib_hec": {
        "0": ("IB", "low_no_unique_IB_marker"),
        "1": ("IB", "low_small_negative_check"),
        "2": ("IPhC", "medium_Slc1a3_supported_small"),
        "3": ("IPhC", "medium_Slc1a3_supported"),
        "4": ("HeC", "low_Pmch_absent"),
        "5": ("IPhC", "medium_Slc1a3_supported"),
        "6": ("IPhC", "low_small_sample_skew"),
        "7": ("IPhC", "medium_Slc1a3_supported"),
        "8": ("HeC", "low_Pmch_absent_sample_skew"),
        "9": ("IB", "low_small_cluster"),
    },
    "dc_pc": {
        "0": ("PC", "medium_PC_key_with_shared_markers"),
        "1": ("PC", "medium_PC_key_with_shared_markers"),
        "2": ("DC", "high_Ceacam16_Rbp_supported"),
        "3": ("DC", "high_Ceacam16_Rbp_supported"),
        "4": ("PC", "low_Ednrb_supported_sample_skew"),
        "5": ("PC", "medium_Ednrb_Prmd12_supported"),
    },
    "osc_isc": {
        "0": ("IB", "low_IPhC_IB_HeC_marker_contamination"),
        "1": ("ISC", "medium_Gata3_Gjb2_Epyc_supported"),
        "2": ("OSC", "medium_Bmp4_supported"),
        "3": ("OSC", "low_Bmp4_supported_small_sample_skew"),
        "4": ("DC", "low_DC_PC_marker_contamination"),
        "5": ("OSC", "low_Bmp4_supported_small"),
        "6": ("DC", "low_DC_PC_marker_contamination"),
        "7": ("ISC", "medium_Gata3_Aqp4_Gjb2_supported"),
        "8": ("OSC", "medium_Bmp4_Gjb2_Epyc_supported"),
    },
}

for module_name, cluster_map in local_key_maps.items():
    adata_local = p28_local_plot_objects[module_name]["plot"]
    cluster_key = p28_local_plot_objects[module_name]["chosen_key"]
    local_clusters = adata_local.obs[cluster_key].astype(str)

    for cluster_id, (cell_type, confidence) in cluster_map.items():
        local_obs = adata_local.obs_names[local_clusters == cluster_id]
        local_obs = local_obs[local_obs.isin(adata_p28_key_guided.obs_names)]

        adata_p28_key_guided.obs.loc[local_obs, p28_filter_celltype_key] = cell_type
        adata_p28_key_guided.obs.loc[local_obs, p28_filter_confidence_key] = confidence


#2. =========== Remove low-confidence cells but keep all marker-table cell types ===========

def p28_confidence_to_simple(confidence):
    confidence = str(confidence)
    if confidence == "kept_from_v2" or confidence.startswith("high"):
        return "high"
    if confidence.startswith("medium"):
        return "medium"
    if confidence.startswith("low"):
        return "low"
    return "check"

adata_p28_key_guided.obs["p28_confidence_simple"] = (
    adata_p28_key_guided.obs[p28_filter_confidence_key]
    .map(p28_confidence_to_simple)
    .astype(str)
)

p28_celltype_series = adata_p28_key_guided.obs[p28_filter_celltype_key].astype(str)
p28_confidence_series = adata_p28_key_guided.obs["p28_confidence_simple"].astype(str)

p28_before_counts = (
    p28_celltype_series.value_counts()
    .reindex(p28_excel_celltype_order)
    .fillna(0)
    .astype(int)
)

p28_low_mask = p28_confidence_series.eq("low")
p28_strict_keep_mask = ~p28_low_mask

p28_keep_low_celltypes = [
    cell_type for cell_type in p28_excel_celltype_order
    if p28_before_counts.loc[cell_type] > 0
    and cell_type not in set(p28_celltype_series[p28_strict_keep_mask].unique())
]

p28_drop_low_mask = (
    p28_low_mask
    & ~p28_celltype_series.isin(p28_keep_low_celltypes)
)

p28_keep_obs = adata_p28_key_guided.obs_names[~p28_drop_low_mask.values]
adata_p28_key_guided = adata_p28_key_guided[p28_keep_obs, :].copy()

p28_after_counts = (
    adata_p28_key_guided.obs[p28_filter_celltype_key]
    .astype(str)
    .value_counts()
    .reindex(p28_excel_celltype_order)
    .fillna(0)
    .astype(int)
)

p28_filter_summary = pd.DataFrame(
    {
        "before": p28_before_counts,
        "after": p28_after_counts,
        "removed": p28_before_counts - p28_after_counts,
    }
)

if (p28_after_counts == 0).any():
    display(p28_filter_summary)
    raise ValueError("At least one final P28 cell type is still missing.")

display(p28_filter_summary)


#3. =========== Rebuild final labels ===========

adata_p28_key_guided.obs[p28_filter_celltype_key] = pd.Categorical(
    adata_p28_key_guided.obs[p28_filter_celltype_key].astype(str),
    categories=p28_excel_celltype_order,
    ordered=True,
)

p28_v4_group_order = [
    cell_type for cell_type in p28_excel_celltype_order
    if p28_after_counts.loc[cell_type] > 0
]

p28_label_order = p28_v4_group_order.copy()
p28_v4_group_key = p28_label_key
p28_expected_counts_override = p28_after_counts.copy()

adata_p28_key_guided.obs[p28_label_key] = pd.Categorical(
    adata_p28_key_guided.obs[p28_filter_celltype_key].astype(str),
    categories=p28_v4_group_order,
    ordered=True,
)

p28_label_count_map = {
    cell_type: f"{cell_type} (n={p28_after_counts.loc[cell_type]})"
    for cell_type in p28_v4_group_order
}

adata_p28_key_guided.obs[p28_label_n_key] = (
    adata_p28_key_guided.obs[p28_filter_celltype_key]
    .astype(str)
    .map(p28_label_count_map)
)

adata_p28_key_guided.obs[p28_label_n_key] = pd.Categorical(
    adata_p28_key_guided.obs[p28_label_n_key].astype(str),
    categories=[p28_label_count_map[cell_type] for cell_type in p28_v4_group_order],
    ordered=True,
)

adata_p28_key_guided_plot = adata_p28_key_guided.copy()


#4. =========== Add Fabp7 to HeC marker group ===========

gene_lookup = {gene.upper(): gene for gene in adata_p28_key_guided_plot.var_names}
p28_fabp7_gene = gene_lookup.get("FABP7")

if p28_fabp7_gene is not None:
    p28_excel_marker_dict = {
        group: list(genes)
        for group, genes in p28_excel_marker_dict.items()
    }
    p28_excel_marker_dict.setdefault("HeC", [])

    if p28_fabp7_gene not in p28_excel_marker_dict["HeC"]:
        p28_excel_marker_dict["HeC"].append(p28_fabp7_gene)


#5. =========== Recompute P28 UMAP with HVG5000 ===========

p28_umap_n_top_genes = 5000
p28_umap_n_pcs = 40
p28_umap_n_neighbors = 25

adata_p28_umap_check = adata_p28_key_guided_plot.copy()

if "counts" in adata_p28_umap_check.layers:
    adata_p28_umap_check.X = adata_p28_umap_check.layers["counts"].copy()

sc.pp.normalize_total(adata_p28_umap_check, target_sum=1e4)
sc.pp.log1p(adata_p28_umap_check)

sc.pp.highly_variable_genes(
    adata_p28_umap_check,
    flavor="seurat",
    n_top_genes=p28_umap_n_top_genes,
)

adata_p28_umap_hvg = adata_p28_umap_check[
    :, adata_p28_umap_check.var["highly_variable"]
].copy()

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="zero-centering a sparse array/matrix densifies it.")
    sc.pp.scale(adata_p28_umap_hvg, max_value=10)

sc.tl.pca(
    adata_p28_umap_hvg,
    n_comps=50,
    svd_solver="arpack",
    random_state=RANDOM_STATE,
)

sc.pp.neighbors(
    adata_p28_umap_hvg,
    n_neighbors=p28_umap_n_neighbors,
    n_pcs=p28_umap_n_pcs,
    random_state=RANDOM_STATE,
)

sc.tl.umap(
    adata_p28_umap_hvg,
    random_state=RANDOM_STATE,
)

adata_p28_umap_check.obsm["X_umap"] = adata_p28_umap_hvg.obsm["X_umap"].copy()
adata_p28_umap_check.uns["umap"] = adata_p28_umap_hvg.uns["umap"].copy()

adata_p28_key_guided_plot.obsm["X_umap"] = adata_p28_umap_check.obsm["X_umap"].copy()
adata_p28_key_guided_plot.uns["umap"] = adata_p28_umap_check.uns["umap"].copy()


#6. =========== Prepare dotplot expression ===========

adata_p28_v4_dotplot = adata_p28_key_guided_plot.copy()

if "counts" in adata_p28_v4_dotplot.layers:
    adata_p28_v4_dotplot.X = adata_p28_v4_dotplot.layers["counts"].copy()

sc.pp.normalize_total(adata_p28_v4_dotplot, target_sum=1e4)
sc.pp.log1p(adata_p28_v4_dotplot)

adata_p28_v4_dotplot.obs[p28_label_key] = pd.Categorical(
    adata_p28_v4_dotplot.obs[p28_label_key].astype(str),
    categories=p28_v4_group_order,
    ordered=True,
)


#7. =========== Plot P28 UMAP and dotplot ===========

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sc.pl.umap(
    adata_p28_key_guided_plot,
    color=p28_label_n_key,
    ax=axes[0],
    show=False,
    size=8,
    legend_loc="right margin",
    frameon=False,
    title="P28 UMAP, HVG5000",
)

sc.pl.umap(
    adata_p28_key_guided_plot,
    color=p28_filter_celltype_key,
    ax=axes[1],
    show=False,
    size=8,
    legend_loc="on data",
    legend_fontsize=8,
    frameon=False,
    title="P28 labels, HVG5000",
)

sc.pl.umap(
    adata_p28_key_guided_plot,
    color="sample",
    ax=axes[2],
    show=False,
    size=8,
    legend_loc="right margin",
    frameon=False,
    title="Sample",
)

plt.tight_layout()
plt.close("all")

sc.pl.dotplot(
    adata_p28_v4_dotplot,
    p28_excel_marker_dict,
    groupby=p28_label_key,
    categories_order=p28_v4_group_order,
    standard_scale="var",
    cmap="Reds",
    dot_max=0.8,
    figsize=(18, 5.4),
    show=False,
)


plt.close("all")


# Notebook cell 84

#1. =========== Set P28 save paths ===========

p28_fig_dir = str(OUTPUT_DIR)
p28_harmony_dir = str(RESULTS_ROOT / "01_stage_clustering" / "harmony_inputs" / "p28")
p28_scanvi_dir = str(STAGE_OBJECT_DIR)

for save_dir in [p28_fig_dir, p28_harmony_dir, p28_scanvi_dir]:
    os.makedirs(save_dir, exist_ok=True)

p28_umap_path = os.path.join(p28_fig_dir, "P28_umap.png")
p28_dotplot_path = os.path.join(p28_fig_dir, "P28_dotplot.png")
p28_dotplot_part1_path = os.path.join(p28_fig_dir, "P28_dotplot_part1.png")
p28_dotplot_part2_path = os.path.join(p28_fig_dir, "P28_dotplot_part2.png")

p28_harmony_norm_path = os.path.join(p28_harmony_dir, "P28_for_Harmony_normalized.h5ad")
p28_harmony_raw_path = os.path.join(p28_harmony_dir, "P28_for_Harmony_raw.h5ad")
p28_scanvi_norm_path = os.path.join(p28_scanvi_dir, "P28_for_scnvi_normalized.h5ad")
p28_scanvi_raw_path = os.path.join(p28_scanvi_dir, "P28_for_scnvi_raw.h5ad")

#2. =========== Prepare final P28 object ===========

p28_celltype_key = "p28_cell_type_v4_key_guided"
p28_confidence_key = "p28_cell_type_v4_confidence"
p28_label_key = "p28_cell_type_v4_label"
p28_label_n_key = "p28_cell_type_v4_label_n"

p28_celltype_order = ["OHC", "IHC", "IPhC", "IB", "HeC", "DC", "PC", "OSC", "ISC"]

adata_p28_save_base = adata_p28_key_guided_plot.copy()

if "batch" not in adata_p28_save_base.obs.columns:
    adata_p28_save_base.obs["batch"] = adata_p28_save_base.obs["sample"].astype(str)

def simplify_p28_confidence(confidence):
    confidence = str(confidence)
    if confidence == "kept_from_v2" or confidence.startswith("high"):
        return "high"
    if confidence.startswith("medium"):
        return "medium"
    if confidence.startswith("low"):
        return "low"
    return "check"

adata_p28_save_base.obs["P28_final_celltype"] = pd.Categorical(
    adata_p28_save_base.obs[p28_celltype_key].astype(str),
    categories=p28_celltype_order,
    ordered=True,
)

adata_p28_save_base.obs["P28_final_celltype_confidence"] = (
    adata_p28_save_base.obs[p28_confidence_key].astype(str)
)

adata_p28_save_base.obs["P28_final_confidence"] = (
    adata_p28_save_base.obs[p28_confidence_key]
    .map(simplify_p28_confidence)
    .astype(str)
)

p28_counts = (
    adata_p28_save_base.obs["P28_final_celltype"]
    .value_counts()
    .reindex(p28_celltype_order)
    .fillna(0)
    .astype(int)
)

p28_group_order = [
    cell_type for cell_type in p28_celltype_order
    if p28_counts.loc[cell_type] > 0
]

p28_label_count_map = {
    cell_type: f"{cell_type} (n={p28_counts.loc[cell_type]})"
    for cell_type in p28_group_order
}

adata_p28_save_base.obs[p28_label_key] = pd.Categorical(
    adata_p28_save_base.obs["P28_final_celltype"].astype(str),
    categories=p28_group_order,
    ordered=True,
)

adata_p28_save_base.obs[p28_label_n_key] = (
    adata_p28_save_base.obs["P28_final_celltype"]
    .astype(str)
    .map(p28_label_count_map)
)

adata_p28_save_base.obs[p28_label_n_key] = pd.Categorical(
    adata_p28_save_base.obs[p28_label_n_key].astype(str),
    categories=[p28_label_count_map[cell_type] for cell_type in p28_group_order],
    ordered=True,
)

adata_p28_save_base.obs["P28_final_celltype_label"] = (
    adata_p28_save_base.obs[p28_label_key].astype(str)
)

adata_p28_save_base.obs["P28_final_celltype_label_n"] = (
    adata_p28_save_base.obs[p28_label_n_key].astype(str)
)

p28_count_check = pd.DataFrame({"n_cells": p28_counts})


#3. =========== Save P28 UMAP ===========

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sc.pl.umap(
    adata_p28_save_base,
    color=p28_label_n_key,
    ax=axes[0],
    show=False,
    size=8,
    legend_loc="right margin",
    frameon=False,
    title="P28 UMAP, HVG5000",
)

sc.pl.umap(
    adata_p28_save_base,
    color=p28_celltype_key,
    ax=axes[1],
    show=False,
    size=8,
    legend_loc="on data",
    legend_fontsize=8,
    frameon=False,
    title="P28 labels, HVG5000",
)

sc.pl.umap(
    adata_p28_save_base,
    color="sample",
    ax=axes[2],
    show=False,
    size=8,
    legend_loc="right margin",
    frameon=False,
    title="Sample",
)

plt.tight_layout()
fig.savefig(p28_umap_path, dpi=SAVE_DPI, bbox_inches="tight")
plt.close(fig)


#4. =========== Save P28 dotplot ===========

adata_p28_v4_dotplot = adata_p28_save_base.copy()
adata_p28_v4_dotplot.X = adata_p28_v4_dotplot.layers["counts"].copy()

sc.pp.normalize_total(adata_p28_v4_dotplot, target_sum=1e4)
sc.pp.log1p(adata_p28_v4_dotplot)

adata_p28_v4_dotplot.obs[p28_label_key] = pd.Categorical(
    adata_p28_v4_dotplot.obs[p28_label_key].astype(str),
    categories=p28_group_order,
    ordered=True,
)

def save_p28_dotplot(marker_dict, save_path, figsize):
    dp = sc.pl.dotplot(
        adata_p28_v4_dotplot,
        marker_dict,
        groupby=p28_label_key,
        categories_order=p28_group_order,
        standard_scale="var",
        cmap="Reds",
        dot_max=0.8,
        figsize=figsize,
        show=False,
        return_fig=True,
    )
    dp.savefig(save_path, dpi=SAVE_DPI)
    plt.close("all")

p28_marker_groups = list(p28_excel_marker_dict.keys())
p28_split_idx = int(np.ceil(len(p28_marker_groups) / 2))

p28_dotplot_part1_dict = {
    group: p28_excel_marker_dict[group]
    for group in p28_marker_groups[:p28_split_idx]
}

p28_dotplot_part2_dict = {
    group: p28_excel_marker_dict[group]
    for group in p28_marker_groups[p28_split_idx:]
}

save_p28_dotplot(
    p28_excel_marker_dict,
    p28_dotplot_path,
    figsize=(18, 5.8),
)

save_p28_dotplot(
    p28_dotplot_part1_dict,
    p28_dotplot_part1_path,
    figsize=(11, 5.8),
)

save_p28_dotplot(
    p28_dotplot_part2_dict,
    p28_dotplot_part2_path,
    figsize=(11, 5.8),
)

#5. =========== Save P28 h5ad files ===========

def prepare_p28_integration_adata(adata):
    adata.obs["stage"] = "P28"
    adata.obs["batch"] = adata.obs["sample"].astype(str)
    adata.obs["celltype"] = adata.obs["P28_final_celltype"].astype(str)

    adata.obs["P28_final_celltype"] = pd.Categorical(
        adata.obs["P28_final_celltype"].astype(str),
        categories=p28_celltype_order,
        ordered=True,
    )
    adata.obs["celltype"] = pd.Categorical(
        adata.obs["celltype"].astype(str),
        categories=p28_celltype_order,
        ordered=True,
    )

    keep_obs_cols = [
        "sample",
        "batch",
        "stage",
        "celltype",
        "P28_final_celltype",
        "P28_final_celltype_confidence",
        "P28_final_confidence",
        "P28_final_celltype_label",
        "P28_final_celltype_label_n",
    ]

    adata.obs = adata.obs[[col for col in keep_obs_cols if col in adata.obs.columns]].copy()
    adata.raw = None

    for key in list(adata.layers.keys()):
        del adata.layers[key]
    for key in list(adata.obsm.keys()):
        del adata.obsm[key]
    for key in list(adata.varm.keys()):
        del adata.varm[key]
    for key in list(adata.obsp.keys()):
        del adata.obsp[key]

    adata.uns = {}
    return adata

adata_p28_raw_save = adata_p28_save_base.copy()
adata_p28_raw_save.X = adata_p28_raw_save.layers["counts"].copy()
adata_p28_raw_save = prepare_p28_integration_adata(adata_p28_raw_save)

adata_p28_norm_save = adata_p28_save_base.copy()
adata_p28_norm_save.X = adata_p28_norm_save.layers["counts"].copy()
sc.pp.normalize_total(adata_p28_norm_save, target_sum=1e4)
sc.pp.log1p(adata_p28_norm_save)
adata_p28_norm_save = prepare_p28_integration_adata(adata_p28_norm_save)

adata_p28_norm_save.write_h5ad(p28_harmony_norm_path, compression="gzip")
adata_p28_raw_save.write_h5ad(p28_harmony_raw_path, compression="gzip")
adata_p28_norm_save.write_h5ad(p28_scanvi_norm_path, compression="gzip")
adata_p28_raw_save.write_h5ad(p28_scanvi_raw_path, compression="gzip")




plt.close("all")
