#1. ===========Purpose and reproducibility settings
"""Preprocess, cluster, annotate, and export the E9.5 integration inputs."""

from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("COCHLEA_DATA_DIR", REPO_ROOT / "Data")).expanduser().resolve()
RESULTS_ROOT = Path(os.environ.get("COCHLEA_RESULTS_DIR", REPO_ROOT / "results")).expanduser().resolve()
STAGE_OBJECT_DIR = RESULTS_ROOT / "01_stage_clustering" / "h5ad_for_integration"
OUTPUT_DIR = RESULTS_ROOT / "01_stage_clustering" / "e9_5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STAGE_OBJECT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 0
SAVE_DPI = 1201
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_STATE))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Notebook cell 1

import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re
from scipy import sparse

sc.settings.verbosity = 3 #Set the prompt information display mode for the terminal/output area
sc.settings.set_figure_params(dpi=SAVE_DPI, facecolor="white") #background color is white


# Notebook cell 2

# path settings
base_dir = REPO_ROOT
count_file = DATA_ROOT / "raw" / "e9_5" / "GSM5401072_E9_5_raw_count.csv"
marker_file = REPO_ROOT / "config" / "cell_type_markers.xlsx"
os.chdir(REPO_ROOT)

# Read file
df_E9 = pd.read_csv(count_file)
# print("=== basic info ===")
# print("shape:", df_E9.shape)

# print("\nThe first 3 lines:")
# print(df_E9.iloc[:3, :3]) # Take the first 5 rows and 5 columns
# #iloc takes by index, while loc takes by label

# print("\ndata type:")
# print(df_E9.dtypes.head(10))

E9_gene_names = df_E9.iloc[:, 0].astype(str)
# print("total gene num:", len(E9_gene_names))
# print("-------------------")
# print("\nfirst 10 gene names:")
# print(E9_gene_names.head(5).tolist())

E9_expr_matrix = df_E9.iloc[:,1:]
# print("E9_expr_matrix shape:", E9_expr_matrix.shape)

E9_expr_matrix_numeric = E9_expr_matrix.apply(pd.to_numeric, errors="coerce")#If conversion fails, it will become NaN
# print("Total num of NaN:", E9_expr_matrix_numeric.isna().sum().sum())#First calculate the sum for each column, then calculate the overall sum
# print("duplicate genes:", E9_gene_names.duplicated().sum())
# print("min num in expr:", E9_expr_matrix_numeric.min().min())
# print("max num in expr:", E9_expr_matrix_numeric.max().max())

vals = E9_expr_matrix_numeric.values
zero_fraction = (vals == 0).sum() / vals.size
# print("0 proposation:", zero_fraction) #> 50% is 0: It can already be said to have a certain degree of sparsity. > 70% is 0: It is usually clearly considered to be relatively sparse. > 90% is 0: It belongs to high sparsity

E9_cell_total_counts = E9_expr_matrix_numeric.sum(axis=0)
E9_cell_n_genes = (E9_expr_matrix_numeric > 0).sum(axis=0)
# print("-------------------")
# print("every cell's total counts description:")
# print(E9_cell_total_counts.describe())
# print("\nevery cell's detected genes description:")
# print(E9_cell_n_genes.describe())

# print("mt- prefixes num:", E9_gene_names.str.startswith("mt-").sum())
# print("Mt- prefixes num:", E9_gene_names.str.startswith("Mt-").sum())
# print("MT- prefixes num:", E9_gene_names.str.startswith("MT-").sum())


# Notebook cell 3

# ======================== 1. QC plot ========================
# total counts distribution
from pathlib import Path

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(E9_cell_total_counts, bins=50)
axes[0].set_xlabel("Total counts per cell")
axes[0].set_ylabel("Number of cells")
axes[0].set_title("Distribution of total counts")

# detected genes istribution
axes[1].hist(E9_cell_n_genes, bins=50)
axes[1].set_xlabel("Detected genes per cell")
axes[1].set_ylabel("Number of cells")
axes[1].set_title("Distribution of detected genes")

# scatter plot
axes[2].scatter(E9_cell_total_counts, E9_cell_n_genes, s=6)
axes[2].set_xlabel("Total counts per cell")
axes[2].set_ylabel("Detected genes per cell")
axes[2].set_title("Counts vs detected genes")

plt.tight_layout()

# Save the QC figure to the organized output directory.
figure_dir = OUTPUT_DIR
os.makedirs(figure_dir, exist_ok=True)

save_path = os.path.join(figure_dir, "E9.5 total counts distribution.png")
fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")

print("Saved to:", save_path)
print("File exists:", os.path.exists(save_path))
plt.close("all")


# Notebook cell 4

os.chdir(REPO_ROOT)
# ======================== 2. build AnnData ========================
#   current table is gene x cell, while AnnData needs cell x gene

# raw count matrix should not have NaN
E9_expr_matrix_numeric = E9_expr_matrix_numeric.fillna(0)
df_E9_indexed = df_E9.set_index(df_E9.columns[0]) #set rownames as index
df_E9_indexed = df_E9_indexed.apply(pd.to_numeric, errors="coerce").fillna(0) #Thus the dataframe is a pure numerical matrix

adata_E9 = ad.AnnData(X=sparse.csr_matrix(df_E9_indexed.T))  
# It not only stores an expression matrix X, but also allows for future additions. 
# adata.obs: information for each cell; 
# adata.var: information for each gene; 
# adata.obsm: dimensionality reduction results; 
# adata.uns: some analysis parameters and results

# AnnData requires the input matrix X to be in the following format:
#   Row = cell (observation)
#   Column = gene (variable, variables)

# AnnData does not automatically know the specific names of these cells, so you need to manually tell it:
adata_E9.obs_names = df_E9_indexed.columns.astype(str)   # cell barcodes
adata_E9.var_names = df_E9_indexed.index.astype(str)     # gene symbols

adata_E9.obs_names_make_unique()
adata_E9.var_names_make_unique() # won't delete

print("\nAnnData object:")
print(adata_E9)


# Notebook cell 5

# ======================== 3. calculate QC metrics ========================
adata_E9.var["mt"] = adata_E9.var_names.str.startswith(("mt-", "Mt-", "MT-"))
adata_E9.var["ribo"] = adata_E9.var_names.str.startswith(("Rps", "Rpl"))
adata_E9.var["hb"] = adata_E9.var_names.str.match(r"^Hb[ab]")

sc.pp.calculate_qc_metrics( #Scanpy will automatically calculate some QC metrics for each cell based on the expression matrix adata_E9.X, and then write the results into obs. including
    adata_E9,
    qc_vars=["mt", "ribo", "hb"],
    percent_top=None, #Whether to calculate the proportion of the top N highly expressed genes to the total counts.
    log1p=False, #Do not apply the log(1+x) transformation to the expression values. Calculate the QC metrics directly based on the original counts.
    inplace=True, #Write the result directly back to adata_E9, rather than returning a new object.
)

print("\nQC metric summary:")
print(adata_E9.obs[["total_counts", "n_genes_by_counts", "pct_counts_mt"]].describe()) # Describe the condition of the cell population

sc.pl.violin(
    adata_E9,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4, #Overlap some scatter points on the violin plot and give these points a slight horizontal jitter. Ensure the points are not clustered on the same vertical line.
    multi_panel=True,
    show=False
)

save_path = os.path.join(figure_dir, "E9 QC violin.png")
plt.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
plt.close("all")
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sc.pl.scatter(adata_E9, x="total_counts", y="n_genes_by_counts", ax=axes[0], show=False)
sc.pl.scatter(adata_E9, x="total_counts", y="pct_counts_mt", ax=axes[1], show=False)
sc.pl.scatter(adata_E9, x="n_genes_by_counts", y="pct_counts_mt", ax=axes[2], show=False)

save_path = os.path.join(figure_dir, "E9 QC scatter plot.png")
fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
plt.tight_layout()
plt.close("all")


# Notebook cell 6

# ======================== 4. cell / gene filtering ========================
adata_E9_final = adata_E9.copy()

sc.pp.filter_genes(adata_E9_final, min_cells=3) #If a gene is only present in 0, 1, or 2 cells, it will be deleted

adata_E9_final = adata_E9_final[adata_E9_final.obs["n_genes_by_counts"] >= 1000, :].copy() #How many genes have been detected in this cell? It refers to the number of genes
adata_E9_final = adata_E9_final[adata_E9_final.obs["pct_counts_mt"] < 12, :].copy()
adata_E9_final = adata_E9_final[adata_E9_final.obs["n_genes_by_counts"] <= 7500, :].copy()
adata_E9_final = adata_E9_final[adata_E9_final.obs["total_counts"] <= 80000, :].copy() #The sum of expression counts of all genes in a single cell. It represents the number of gene reads

# print("\nAfter QC filtering:")
# print(adata_E9_final)

sc.pl.violin(
    adata_E9_final,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
    show=False
)

save_path = os.path.join(figure_dir, "E9 QC violin filtered.png")
plt.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
plt.close("all")
plt.close()


# Notebook cell 7

# ======================== 5. normalize + log1p ========================
adata_E9_final.layers["counts"] = adata_E9_final.X.copy() # Create a layer named "counts" and copy the main matrix .X over

sc.pp.normalize_total(adata_E9_final, target_sum=1e4) #pp stands for preprocessing, while normalize_total refers to normalization based on the total expression level of each cell.
sc.pp.log1p(adata_E9_final) #log(1+x)
# Do not use the Pearson residuals approach unless you find the following: 
# #   Certain clusters are strongly dominated by a very small number of highly expressed genes 
# #   Clustering is unstable 
# #   There are significant differences in total RNA amounts among different clusters

adata_E9_final.raw = adata_E9_final #.raw is a specific location in AnnData used to store a "fixed version expression matrix".
# print("\nAfter normalize/log1p:")
# print(adata_E9_final)


# Notebook cell 8

# ======================== 6. E9.5 marker setup and OV epithelial extraction ========================
base_dir = REPO_ROOT
figure_dir = OUTPUT_DIR
figure_dir.mkdir(parents=True, exist_ok=True)

def keep_present(adata_obj, genes):
    return [g for g in genes if g in adata_obj.var_names]

def expr_positive(adata_obj, gene):
    x = adata_obj[:, gene].X
    x = x.toarray().ravel() if sparse.issparse(x) else np.asarray(x).ravel()
    return x > 0

def add_score(adata_obj, genes, score_name):
    genes = keep_present(adata_obj, genes)
    if len(genes) > 0:
        sc.tl.score_genes(adata_obj, genes, score_name=score_name, random_state=0)
    else:
        adata_obj.obs[score_name] = 0.0
    return genes

ov_parent_markers = keep_present(
    adata_E9_final,
    ["Epcam", "Fbxo2", "Tbx2", "Pax2", "Foxg1", "Six1", "Eya1", "Sox2", "Lmx1a", "Tbx1"]
)

ov_epi_markers = keep_present(
    adata_E9_final,
    ["Tbx1", "Tbx2", "Fbxo2", "Sp5", "Id3", "Epcam", "Pax2"]
)

cvg_markers = ["Insm1", "Tmsb4x", "Hes6", "Dll1", "Miat", "Gadd45g", "Rgs13", "Neurod1", "Neurog1"]
mesenchyme_markers = ["Foxd1", "Col3a1", "Twist1", "Pdgfrb", "Prrx1"]
blood_markers = ["Hbb-bt", "Hbb-bs", "Hba-a1", "Hba-a2", "Alas2", "Car1"]
immune_markers = ["Ptprc", "Lyz2", "C1qa", "C1qb", "C1qc"]
endothelial_markers = ["Pecam1", "Kdr", "Cdh5", "Vwf"]

add_score(adata_E9_final, ov_parent_markers, "OV_parent_score")
add_score(adata_E9_final, cvg_markers, "CVG_score")
add_score(adata_E9_final, mesenchyme_markers, "Mesenchyme_score")
add_score(adata_E9_final, blood_markers, "Blood_score")
add_score(adata_E9_final, immune_markers, "Immune_score")
add_score(adata_E9_final, endothelial_markers, "Endothelial_score")

mask_epi = adata_E9_final.obs["OV_parent_score"] > 0.20

if "Fbxo2" in adata_E9_final.var_names and "Tbx2" in adata_E9_final.var_names:
    mask_epi &= expr_positive(adata_E9_final, "Fbxo2") | expr_positive(adata_E9_final, "Tbx2")

# Remove non-target cells. Use strict removal for definite off-target markers.
strict_exclude_genes = keep_present(
    adata_E9_final,
    ["Otx2", "Oc90", "Neurod1", "Insm1", "Trp63", "Phox2a", "Phox2b", "Neurog2"]
)

for gene in strict_exclude_genes:
    mask_epi &= ~expr_positive(adata_E9_final, gene)

# Neurog1 alone can be low-level/noisy at E9.5, so remove it only when CVG score is also high.
if "Neurog1" in adata_E9_final.var_names:
    mask_neurog1_cvg = expr_positive(adata_E9_final, "Neurog1") & (adata_E9_final.obs["CVG_score"] > 0.10)
    mask_epi &= ~mask_neurog1_cvg

for score in ["Mesenchyme_score", "Blood_score", "Immune_score", "Endothelial_score"]:
    mask_epi &= adata_E9_final.obs[score] < 0.10

adata_E9_epi = adata_E9_final[mask_epi].copy()
adata_E9_epi.obs["stage"] = "E9.5"
adata_E9_epi.obs["final_celltype"] = "OV epithelial cells"
adata_E9_epi.obs["source_object"] = "E9.5_raw_count_OV_epithelial_fullgene"

print("E9.5 cells after QC:", adata_E9_final.n_obs)
print("E9.5 OV epithelial cells kept:", adata_E9_epi.n_obs)


# Notebook cell 9

# ======================== 7. full-gene HVG clustering and inspection only ========================
import warnings
warnings.filterwarnings("ignore", message=".*zero-centering a sparse.*")
warnings.filterwarnings("ignore", message=".*default backend for leiden.*")
warnings.filterwarnings("ignore", message=".*IProgress not found.*")

sc.settings.verbosity = 0
sc.settings.set_figure_params(
    dpi=120,
    dpi_save=1201,
    facecolor="white",
    fontsize=10,
    figsize=(4, 4)
)

adata_E9_work = adata_E9_epi.copy()

sc.pp.highly_variable_genes(
    adata_E9_work,
    n_top_genes=min(2000, adata_E9_work.n_vars),
    flavor="seurat"
)

adata_E9_work = adata_E9_work[:, adata_E9_work.var["highly_variable"]].copy()
sc.pp.scale(adata_E9_work, max_value=10)

n_pcs = min(20, adata_E9_work.n_obs - 1, adata_E9_work.n_vars - 1)
n_neighbors = min(10, max(2, adata_E9_work.n_obs // 3))

sc.tl.pca(adata_E9_work, n_comps=n_pcs, svd_solver="arpack", random_state=0)
sc.pp.neighbors(adata_E9_work, n_neighbors=n_neighbors, n_pcs=n_pcs)
sc.tl.umap(adata_E9_work, random_state=0)
sc.tl.leiden(adata_E9_work, resolution=0.3, key_added="E9_leiden")

adata_E9_epi.obs["E9_leiden"] = adata_E9_work.obs["E9_leiden"].astype(str).values
adata_E9_epi.obsm["X_umap"] = adata_E9_work.obsm["X_umap"].copy()

fig, ax = plt.subplots(figsize=(4, 4))
sc.pl.umap(
    adata_E9_epi,
    color="E9_leiden",
    size=22,
    legend_loc="right margin",
    title="E9.5 OV epithelial",
    ax=ax,
    show=False
)
plt.close("all")
plt.close()

print("E9.5 OV epithelial cells kept:", adata_E9_epi.n_obs)
print(adata_E9_epi.obs["E9_leiden"].value_counts().sort_index())


# Notebook cell 10

# ======================== 8. E9.5 merged OV epithelial dotplot ========================
sc.settings.verbosity = 0
sc.settings.set_figure_params(
    dpi=120,
    dpi_save=1201,
    facecolor="white",
    fontsize=9
)

adata_E9_epi.obs["final_celltype"] = "OV epithelial cells"
adata_E9_epi.obs["final_celltype"] = adata_E9_epi.obs["final_celltype"].astype("category")

e9_dotplot_genes = keep_present(
    adata_E9_epi,
    [
        # OV epithelial markers
        "Epcam", "Fbxo2", "Tbx2", "Pax2",
        "Foxg1", "Six1", "Eya1", "Sox2",
        "Lmx1a", "Tbx1", "Sp5", "Id3",

        # added developmental / sensory-epithelium inspection markers
        "Gata3", "Sox9", "Lgr5", "Isl1",

        # negative-control / exclusion markers
        "Otx2", "Oc90", "Neurod1", "Neurog1"
    ]
)

sc.pl.dotplot(
    adata_E9_epi,
    var_names=e9_dotplot_genes,
    groupby="final_celltype",
    standard_scale=None,
    cmap="Reds",
    vmin=0,
    vmax=3,
    figsize=(8, 1.5),
    dot_max=0.9,
    show=False
)

save_path = figure_dir / "E9_final_OV_epithelial_dotplot.png"
plt.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
plt.close("all")
plt.close()

# ======================== 9. E9.5 contamination marker check only ========================
contamination_genes = keep_present(
    adata_E9_epi,
    ["Col3a1", "Pdgfrb", "Twist1", "Prrx1", "Pecam1", "Ptprc", "Hbb-bt", "Hbb-bs"]
)

sc.pl.dotplot(
    adata_E9_epi,
    var_names=contamination_genes,
    groupby="final_celltype",
    standard_scale="var",
    figsize=(4, 1.4),
    dot_max=0.9,
    show=False
)
plt.close("all")
plt.close()


# Notebook cell 11

# ======================== 9. E9.5 OV epithelial + strict local CVG display version ========================
sc.settings.verbosity = 0
sc.settings.set_figure_params(
    dpi=120,
    dpi_save=1201,
    facecolor="white",
    fontsize=9,
    figsize=(4, 4)
)

figure_dir = OUTPUT_DIR
figure_dir.mkdir(parents=True, exist_ok=True)

cvg_markers = keep_present(
    adata_E9_final,
    ["Insm1", "Tmsb4x", "Hes6", "Dll1", "Miat", "Gadd45g", "Rgs13", "Neurod1", "Neurog1"]
)
ov_epi_markers = keep_present(
    adata_E9_final,
    ["Tbx1", "Tbx2", "Fbxo2", "Sp5", "Id3", "Epcam", "Pax2"]
)

if "CVG_score" not in adata_E9_final.obs.columns:
    add_score(adata_E9_final, cvg_markers, "CVG_score")

if "OV_epi_score" not in adata_E9_final.obs.columns:
    add_score(adata_E9_final, ov_epi_markers, "OV_epi_score")

if "OV_parent_score" not in adata_E9_final.obs.columns:
    add_score(adata_E9_final, ov_parent_markers, "OV_parent_score")

remaining_mask = ~adata_E9_final.obs_names.isin(adata_E9_epi.obs_names)

core_cvg_genes = keep_present(
    adata_E9_final,
    ["Insm1", "Neurod1", "Neurog1", "Dll1", "Hes6", "Gadd45g"]
)

core_cvg_n = np.zeros(adata_E9_final.n_obs, dtype=int)
for gene in core_cvg_genes:
    core_cvg_n += expr_positive(adata_E9_final, gene).astype(int)

mask_cvg = remaining_mask.copy()
mask_cvg &= core_cvg_n >= 2
mask_cvg &= adata_E9_final.obs["CVG_score"] > 0.20
mask_cvg &= adata_E9_final.obs["CVG_score"] > adata_E9_final.obs["OV_epi_score"] + 0.15
mask_cvg &= adata_E9_final.obs["OV_parent_score"] > 0.05

for gene in keep_present(adata_E9_final, ["Otx2", "Oc90", "Trp63", "Phox2a", "Phox2b", "Neurog2"]):
    mask_cvg &= ~expr_positive(adata_E9_final, gene)

for score in ["Mesenchyme_score", "Blood_score", "Immune_score", "Endothelial_score"]:
    if score in adata_E9_final.obs.columns:
        mask_cvg &= adata_E9_final.obs[score] < 0.10

adata_E9_cvg = adata_E9_final[mask_cvg].copy()
adata_E9_cvg.obs["E9_display_celltype"] = "CVG cells"

adata_E9_epi_display = adata_E9_epi.copy()
adata_E9_epi_display.obs["E9_display_celltype"] = "OV epithelial cells"

adata_E9_display = ad.concat(
    [adata_E9_epi_display, adata_E9_cvg],
    join="inner",
    label=None,
    index_unique=None
)

adata_E9_display.obs["E9_display_celltype"] = pd.Categorical(
    adata_E9_display.obs["E9_display_celltype"],
    categories=["OV epithelial cells", "CVG cells"],
    ordered=True
)

print(adata_E9_display.obs["E9_display_celltype"].value_counts())


# Notebook cell 12

# ======================== 10. E9.5 OV epithelial + CVG UMAP ========================
adata_E9_display_work = adata_E9_display.copy()

sc.pp.highly_variable_genes(
    adata_E9_display_work,
    n_top_genes=min(2000, adata_E9_display_work.n_vars),
    flavor="seurat"
)

adata_E9_display_work = adata_E9_display_work[:, adata_E9_display_work.var["highly_variable"]].copy()
sc.pp.scale(adata_E9_display_work, max_value=10)

n_pcs = min(20, adata_E9_display_work.n_obs - 1, adata_E9_display_work.n_vars - 1)
n_neighbors = min(10, max(2, adata_E9_display_work.n_obs // 3))

sc.tl.pca(adata_E9_display_work, n_comps=n_pcs, svd_solver="arpack", random_state=0)
sc.pp.neighbors(adata_E9_display_work, n_neighbors=n_neighbors, n_pcs=n_pcs)
sc.tl.umap(adata_E9_display_work, random_state=0)

adata_E9_display.obsm["X_umap"] = adata_E9_display_work.obsm["X_umap"].copy()

fig, ax = plt.subplots(figsize=(4, 4))
sc.pl.umap(
    adata_E9_display,
    color="E9_display_celltype",
    size=22,
    legend_loc="right margin",
    title="E9.5 OV epithelial + CVG",
    ax=ax,
    show=False
)

save_path = figure_dir / "E9_OV_epithelial_CVG_UMAP.png"
plt.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
plt.close("all")
plt.close()


# Notebook cell 13

# ======================== 11. E9.5 OV epithelial + CVG dotplot ========================
e9_display_dotplot_genes = keep_present(
    adata_E9_display,
    [
        # OV epithelial markers
        "Epcam", "Fbxo2", "Tbx2", "Pax2",
        "Foxg1", "Six1", "Eya1", "Sox2",
        "Lmx1a", "Tbx1", "Sp5", "Id3",
        "Gata3", "Sox9", "Lgr5", "Isl1",

        # CVG markers
        "Insm1", "Neurod1", "Neurog1", "Dll1",
        "Hes6", "Gadd45g", "Miat", "Rgs13", "Tmsb4x",

        # negative-control markers
        "Otx2", "Oc90"
    ]
)

sc.pl.dotplot(
    adata_E9_display,
    var_names=e9_display_dotplot_genes,
    groupby="E9_display_celltype",
    standard_scale="var",
    cmap="Reds",
    dot_max=0.9,
    figsize=(10, 2.1),
    show=False
)

save_path = figure_dir / "E9_OV_epithelial_CVG_dotplot.png"
plt.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
plt.close("all")
plt.close()


# Notebook cell 14

# ======================== 12. save E9.5 integration h5ad files ========================
harmony_dir = RESULTS_ROOT / "01_stage_clustering" / "harmony_inputs" / "e9_5"
scnvi_dir = STAGE_OBJECT_DIR
harmony_dir.mkdir(parents=True, exist_ok=True)
scnvi_dir.mkdir(parents=True, exist_ok=True)

def copy_matrix(x, dtype=np.float32):
    if sparse.issparse(x):
        return x.copy().astype(dtype)
    return np.asarray(x, dtype=dtype).copy()

def make_integration_obj(adata_in, X):
    obs_cols = [
        "stage", "final_celltype", "source_object",
        "n_genes_by_counts", "total_counts", "pct_counts_mt",
    ]
    obs = adata_in.obs[[c for c in obs_cols if c in adata_in.obs.columns]].copy()
    var = pd.DataFrame(index=adata_in.var_names.copy())
    var.index.name = None
    out = ad.AnnData(
        X=copy_matrix(X),
        obs=obs,
        var=var,
    )
    out.obs_names = adata_in.obs_names.copy()
    out.var_names = adata_in.var_names.copy()
    out.raw = None
    return out

adata_E9_normalized = make_integration_obj(adata_E9_epi, adata_E9_epi.X)
adata_E9_raw = make_integration_obj(adata_E9_epi, adata_E9_epi.layers["counts"])

adata_E9_normalized.write(harmony_dir / "E9_for_Harmony_normalized.h5ad")
adata_E9_raw.write(harmony_dir / "E9_for_Harmony_raw.h5ad")
adata_E9_normalized.write(scnvi_dir / "E9_for_scnvi_normalized.h5ad")
adata_E9_raw.write(scnvi_dir / "E9_for_scnvi_raw.h5ad")
