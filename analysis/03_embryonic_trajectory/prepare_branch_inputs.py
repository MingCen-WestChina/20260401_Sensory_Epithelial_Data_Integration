"""Prepare medial and lateral embryonic inputs for the Monocle 2 workflows."""

from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


RANDOM_STATE = 0
STAGE_ORDER = ["E9.5", "E11.5", "E13.5", "E14.5", "E16.5"]
MEDIAL_CELLTYPES = [
    "OV epithelial cells", "Prosensory domain", "Medial domain", "M.PsC",
    "IHC", "IPhC", "IBC", "IPC",
]
LATERAL_CELLTYPES = [
    "OV epithelial cells", "Prosensory domain", "Lateral domain", "L.PsC",
    "OHC", "iOHC", "HeC",
]
IPHC_QC_MARKERS = [
    "Id1", "Id2", "Id3", "Hes1", "Krt18", "Ctgf", "Anxa5",
    "S100a1", "S100b", "Fabp7", "Lgr5",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("COCHLEA_DATA_DIR", REPO_ROOT / "Data")).expanduser().resolve()
RESULTS_ROOT = Path(os.environ.get("COCHLEA_RESULTS_DIR", REPO_ROOT / "results")).expanduser().resolve()
OUTPUT_DIR = RESULTS_ROOT / "03_embryonic_trajectory" / "inputs"


def resolve_integrated_h5ad() -> Path:
    candidates = [
        RESULTS_ROOT / "02_scanvi_integration" / "all_stages" / "all_stages_scanvi.h5ad",
        DATA_ROOT / "integrated" / "all_stages_scanvi.h5ad",
        DATA_ROOT / "all_stages_scanvi.h5ad",
        DATA_ROOT / "Entire period" / "EntirePeriod_scanvi_HVG5000" / "all_retained_scanvi_HVG5000_integrated.h5ad",
        DATA_ROOT / "Entire period" / "all_retained_scanvi_HVG5000_integrated.h5ad",
        DATA_ROOT / "Integration_3types" / "Entire period files" / "all_retained_scanvi_HVG5000_integrated.h5ad",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def expression_vector(adata: ad.AnnData, gene: str) -> np.ndarray | None:
    if gene not in adata.var_names:
        return None
    values = adata[:, gene].X
    return np.asarray(values.toarray()).ravel() if sparse.issparse(values) else np.asarray(values).ravel()


def make_branch(adata: ad.AnnData, name: str, celltypes: list[str]) -> ad.AnnData:
    keep = adata.obs["trajectory_celltype"].astype(str).isin(celltypes)
    branch = adata[keep].copy()
    branch.obs["trajectory_branch"] = name
    branch.obs["stage"] = pd.Categorical(
        branch.obs["stage"].astype(str), categories=STAGE_ORDER, ordered=True
    )
    present = set(branch.obs["trajectory_celltype"].astype(str))
    branch.obs["trajectory_celltype"] = pd.Categorical(
        branch.obs["trajectory_celltype"].astype(str),
        categories=[celltype for celltype in celltypes if celltype in present],
        ordered=True,
    )
    return branch


def main() -> None:
    np.random.seed(RANDOM_STATE)
    input_h5ad = resolve_integrated_h5ad()
    if not input_h5ad.exists():
        raise FileNotFoundError(
            f"Missing all-stage integrated object: {input_h5ad}. Run all_stages_scanvi.py "
            "or place a precomputed all-stage h5AD under the configured data root."
        )

    adata_all = sc.read_h5ad(input_h5ad)
    required_obs = {"stage", "revised_celltype"}
    missing_obs = sorted(required_obs.difference(adata_all.obs.columns))
    if missing_obs:
        raise KeyError(f"Missing obs columns: {missing_obs}")
    if "X_scanvi" not in adata_all.obsm:
        raise KeyError("Missing obsm['X_scanvi'] in the all-stage integrated object.")

    embryo_mask = adata_all.obs["stage"].astype(str).isin(STAGE_ORDER)
    embryo = adata_all[embryo_mask].copy()
    cd74 = expression_vector(embryo, "Cd74")
    if cd74 is not None:
        embryo = embryo[cd74 == 0].copy()

    embryo.obs["stage"] = pd.Categorical(
        embryo.obs["stage"].astype(str), categories=STAGE_ORDER, ordered=True
    )
    embryo.obs["trajectory_celltype"] = embryo.obs["revised_celltype"].astype(str)

    iphc_mask = (
        embryo.obs["stage"].astype(str).eq("E16.5")
        & embryo.obs["revised_celltype"].astype(str).eq("IPhC")
    )
    iphc = embryo[iphc_mask].copy()
    if iphc.n_obs < 20:
        raise ValueError("At least 20 E16.5 IPhC-labelled cells are required for subclustering.")

    sc.pp.neighbors(
        iphc,
        use_rep="X_scanvi",
        n_neighbors=min(20, max(5, iphc.n_obs // 5)),
        random_state=RANDOM_STATE,
    )
    sc.tl.umap(iphc, min_dist=0.35, random_state=RANDOM_STATE)
    sc.tl.leiden(iphc, resolution=0.4, key_added="iphc_sub_leiden", random_state=RANDOM_STATE)
    clusters = set(iphc.obs["iphc_sub_leiden"].astype(str))
    if not {"0", "1"}.issubset(clusters):
        raise ValueError(f"Expected IPhC subclusters 0 and 1, observed: {sorted(clusters)}")

    cluster = iphc.obs["iphc_sub_leiden"].astype(str)
    embryo.obs.loc[cluster.index[cluster == "0"], "trajectory_celltype"] = "IPhC"
    embryo.obs.loc[cluster.index[cluster == "1"], "trajectory_celltype"] = "IBC"

    marker_present = [gene for gene in IPHC_QC_MARKERS if gene in iphc.var_names]
    marker_qc = []
    for cluster_id in sorted(clusters):
        cluster_mask = iphc.obs["iphc_sub_leiden"].astype(str).eq(cluster_id)
        row = {"iphc_sub_leiden": cluster_id}
        for gene in marker_present:
            row[gene] = float(expression_vector(iphc[cluster_mask], gene).mean())
        marker_qc.append(row)
    marker_qc = pd.DataFrame(marker_qc)

    medial = make_branch(embryo, "medial", MEDIAL_CELLTYPES)
    lateral = make_branch(embryo, "lateral", LATERAL_CELLTYPES)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    medial_path = OUTPUT_DIR / "embryonic_medial_monocle2_input.h5ad"
    lateral_path = OUTPUT_DIR / "embryonic_lateral_monocle2_input.h5ad"
    medial.write_h5ad(medial_path, compression="gzip")
    lateral.write_h5ad(lateral_path, compression="gzip")

    print(pd.DataFrame({
        "object": ["embryo_after_cd74_filter", "medial_branch", "lateral_branch"],
        "cells": [embryo.n_obs, medial.n_obs, lateral.n_obs],
        "genes": [embryo.n_vars, medial.n_vars, lateral.n_vars],
    }).to_string(index=False))
    print(pd.crosstab(embryo.obs["trajectory_celltype"], embryo.obs["stage"]).to_string())
    if not marker_qc.empty:
        print("IPhC subcluster marker means")
        print(marker_qc.round(3).to_string(index=False))
    print(f"Saved: {medial_path}")
    print(f"Saved: {lateral_path}")


if __name__ == "__main__":
    main()
