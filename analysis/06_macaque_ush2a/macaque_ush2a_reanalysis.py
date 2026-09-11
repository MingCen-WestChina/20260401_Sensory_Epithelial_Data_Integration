"""Reproduce current Figure 4A-C from the published macaque cochlear h5AD.

Upstream study and code
-----------------------
Chen X, Che Y, Qi J, Cen M, Gao S, Zhu B, et al. Molecular heterogeneity of
the non-human primate cochlea. Nature Communications. 2026;17:1633.
https://doi.org/10.1038/s41467-026-68350-2

This script is a focused extraction of the cell-type UMAP, USH2A feature plot,
and TMC1/MYO7A/USH2A dot-plot logic from the upstream analysis notebook.  The
permanent download URL for the published integrated h5AD was not present in the
local copy of that repository and must be added to the project data README.
Place that file at Data/macaque/scvi_macaque_cell-type_final.h5ad, or supply a
different path with --input-h5ad.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc


EXPECTED_NUCLEI = 36_701
EXPECTED_AGES = {"1-year-old", "5-year-old", "11-year-old"}
CELLTYPE_KEY = "cell_type"
AGE_KEY = "age"
MARKER_GENES = ["TMC1", "MYO7A", "USH2A"]
RANDOM_SEED = 0
PNG_DPI = 1201


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def environment_path(variable: str, default: Path) -> Path:
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else default.resolve()


def expression_gene_names(adata) -> set[str]:
    source = adata.raw if adata.raw is not None else adata
    return set(map(str, source.var_names))


def validate_input(adata, input_h5ad: Path) -> bool:
    missing_obs = [key for key in (CELLTYPE_KEY, AGE_KEY) if key not in adata.obs]
    if missing_obs:
        raise KeyError(f"Missing required obs columns in {input_h5ad}: {missing_obs}")
    if "X_umap" not in adata.obsm:
        raise KeyError("The published integrated h5AD must contain obsm['X_umap'].")
    missing_genes = [gene for gene in MARKER_GENES if gene not in expression_gene_names(adata)]
    if missing_genes:
        raise KeyError(f"Genes absent from the expression matrix: {missing_genes}")
    if adata.n_obs != EXPECTED_NUCLEI:
        warnings.warn(
            f"Expected {EXPECTED_NUCLEI:,} nuclei reported in the manuscript, "
            f"but {adata.n_obs:,} are present in {input_h5ad}.",
            stacklevel=2,
        )
    observed_ages = set(map(str, adata.obs[AGE_KEY].dropna().unique()))
    if observed_ages != EXPECTED_AGES:
        warnings.warn(
            f"Expected ages {sorted(EXPECTED_AGES)}, observed {sorted(observed_ages)}.",
            stacklevel=2,
        )
    if adata.obs[CELLTYPE_KEY].isna().any():
        raise ValueError("The published cell-type annotation contains missing values.")
    return adata.raw is not None


def save_matplotlib_figure(fig, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=PNG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def draw_celltype_umap(adata, output_dir: Path) -> None:
    ax = sc.pl.umap(
        adata,
        color=CELLTYPE_KEY,
        frameon=True,
        legend_loc="right margin",
        legend_fontsize=7,
        size=5,
        show=False,
    )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    save_matplotlib_figure(ax.figure, output_dir, "fig4a_macaque_celltype_umap")


def draw_ush2a_umap(adata, output_dir: Path, use_raw: bool) -> None:
    ax = sc.pl.umap(
        adata,
        color="USH2A",
        color_map="RdPu",
        frameon=True,
        size=5,
        use_raw=use_raw,
        show=False,
    )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    save_matplotlib_figure(ax.figure, output_dir, "fig4b_macaque_ush2a_umap")


def draw_marker_dotplot(adata, output_dir: Path, use_raw: bool) -> None:
    dendrogram_key = f"dendrogram_{CELLTYPE_KEY}"
    if dendrogram_key not in adata.uns:
        sc.tl.dendrogram(adata, groupby=CELLTYPE_KEY, use_raw=use_raw)
    dotplot = sc.pl.dotplot(
        adata,
        var_names=MARKER_GENES,
        groupby=CELLTYPE_KEY,
        dendrogram=True,
        standard_scale="var",
        color_map="RdPu",
        use_raw=use_raw,
        return_fig=True,
        show=False,
    )
    dotplot.make_figure()
    dotplot.savefig(output_dir / "fig4c_macaque_ush2a_dotplot.pdf")
    dotplot.savefig(output_dir / "fig4c_macaque_ush2a_dotplot.png", dpi=PNG_DPI)
    plt.close(dotplot.fig)


def main() -> None:
    root = repository_root()
    data_dir = environment_path("COCHLEA_DATA_DIR", root / "Data")
    results_dir = environment_path("COCHLEA_RESULTS_DIR", root / "results")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5ad",
        type=Path,
        default=data_dir / "macaque/scvi_macaque_cell-type_final.h5ad",
        help="Published integrated macaque cochlear h5AD.",
    )
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="Validate data, annotations, embedding, and marker genes without drawing files.",
    )
    args = parser.parse_args()
    input_h5ad = args.input_h5ad.expanduser().resolve()
    if not input_h5ad.is_file():
        raise FileNotFoundError(f"Published integrated macaque h5AD not found: {input_h5ad}")

    sc.settings.seed = RANDOM_SEED
    sc.settings.set_figure_params(
        dpi=150,
        dpi_save=PNG_DPI,
        color_map="RdPu",
        vector_friendly=True,
        format="pdf",
        fontsize=8,
    )
    adata = sc.read_h5ad(input_h5ad)
    use_raw = validate_input(adata, input_h5ad)
    print(
        f"Validated macaque input: {adata.n_obs:,} nuclei, {adata.n_vars:,} genes, "
        f"{adata.obs[CELLTYPE_KEY].nunique()} cell types; use_raw={use_raw}."
    )
    if args.check_inputs:
        return

    output_dir = results_dir / "figures/fig4"
    output_dir.mkdir(parents=True, exist_ok=True)
    draw_celltype_umap(adata, output_dir)
    draw_ush2a_umap(adata, output_dir, use_raw)
    draw_marker_dotplot(adata, output_dir, use_raw)
    print(f"Saved current Figure 4A-C to {output_dir}")


if __name__ == "__main__":
    main()
