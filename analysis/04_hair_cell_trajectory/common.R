#1. ===========Load packages and set shared paths===========

set.seed(0)
options(stringsAsFactors = FALSE)

required_pkgs <- c(
  "rhdf5", "monocle", "Biobase", "Matrix", "VGAM", "igraph",
  "ggplot2", "ggrepel", "grid", "patchwork", "uwot"
)

missing_pkgs <- required_pkgs[
  !vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_pkgs) > 0) {
  stop("Missing R packages: ", paste(missing_pkgs, collapse = ", "))
}

suppressPackageStartupMessages({
  library(rhdf5)
  library(monocle)
  library(Biobase)
  library(Matrix)
  library(ggplot2)
  library(grid)
})

get_sourced_file <- function() {
  frame_files <- vapply(
    sys.frames(),
    function(frame_env) {
      if (!is.null(frame_env$ofile)) {
        return(frame_env$ofile)
      }
      NA_character_
    },
    character(1)
  )

  frame_files <- frame_files[!is.na(frame_files)]
  if (length(frame_files) > 0) {
    return(normalizePath(frame_files[length(frame_files)], winslash = "/", mustWork = TRUE))
  }

  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd_args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE))
  }

  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

SCRIPT_DIR <- dirname(get_sourced_file())
ORG_ROOT <- normalizePath(file.path(SCRIPT_DIR, ".."), winslash = "/", mustWork = TRUE)
PROJECT_ROOT <- normalizePath(file.path(ORG_ROOT, "..", ".."), winslash = "/", mustWork = TRUE)
DATA_ROOT <- normalizePath(
  Sys.getenv("COCHLEA_DATA_DIR", file.path(PROJECT_ROOT, "Data")),
  winslash = "/",
  mustWork = FALSE
)
RESULTS_ROOT <- normalizePath(
  Sys.getenv("COCHLEA_RESULTS_DIR", file.path(PROJECT_ROOT, "results")),
  winslash = "/",
  mustWork = FALSE
)
OUTPUT_ROOT <- file.path(RESULTS_ROOT, "04_hair_cell_trajectory")
DATA_OUTPUT_DIR <- file.path(OUTPUT_ROOT, "data")
HC_OUTPUT_DIR <- file.path(OUTPUT_ROOT, "Monocle2 trajectory_HCs")
IHC_OUTPUT_DIR <- file.path(OUTPUT_ROOT, "Monocle2 trajectory_IHCs")
OHC_OUTPUT_DIR <- file.path(OUTPUT_ROOT, "Monocle2 trajectory_OHCs")

dir.create(DATA_OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(HC_OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(IHC_OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(OHC_OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

input_candidates <- c(
  file.path(RESULTS_ROOT, "02_scanvi_integration/all_stages/all_stages_scanvi.h5ad"),
  file.path(DATA_ROOT, "integrated/all_stages_scanvi.h5ad"),
  file.path(DATA_ROOT, "all_stages_scanvi.h5ad"),
  file.path(
    DATA_ROOT,
    c(
      "Entire period/EntirePeriod_scanvi_HVG5000/all_retained_scanvi_HVG5000_integrated.h5ad",
      "Entire period/all_retained_scanvi_HVG5000_integrated.h5ad",
      "Integration_3types/Entire period files/all_retained_scanvi_HVG5000_integrated.h5ad",
      "Entire period/AllPeriod_scanvi_HVG5000_integrated.h5ad"
    )
  )
)

input_h5ad <- input_candidates[file.exists(input_candidates)][1]

if (length(input_h5ad) == 0 || is.na(input_h5ad)) {
  stop("No all-period HVG5000 h5AD was found.")
}

fullgene_names <- c(
  "E14.5_for_scnvi_normalized.h5ad",
  "E16.5_for_scnvi_normalized.h5ad",
  "P1_for_scnvi_normalized.h5ad",
  "P7_for_scnvi_normalized.h5ad",
  "P14_for_scnvi_normalized.h5ad",
  "P28_for_scnvi_normalized.h5ad"
)
fullgene_h5ad_by_stage <- vapply(
  fullgene_names,
  function(filename) {
    candidates <- c(
      file.path(RESULTS_ROOT, "01_stage_clustering/h5ad_for_integration", filename),
      file.path(DATA_ROOT, "h5ad_for_Integration", filename)
    )
    existing <- candidates[file.exists(candidates)]
    if (length(existing) > 0) existing[1] else candidates[1]
  },
  character(1)
)
names(fullgene_h5ad_by_stage) <- c("E14.5", "E16.5", "P1", "P7", "P14", "P28")

missing_fullgene_files <- fullgene_h5ad_by_stage[!file.exists(fullgene_h5ad_by_stage)]
if (length(missing_fullgene_files) > 0) {
  stop("Missing full-gene h5AD files: ", paste(missing_fullgene_files, collapse = ", "))
}

hc_cds_rds <- file.path(DATA_OUTPUT_DIR, "hc_monocle2_all_hc_cds.rds")
hc_plot_meta_csv <- file.path(DATA_OUTPUT_DIR, "hc_monocle2_plot_metadata.csv")
hc_trajectory_rds <- file.path(DATA_OUTPUT_DIR, "hc_monocle2_trajectory_data.rds")
hc_bundle_rds <- file.path(DATA_OUTPUT_DIR, "hc_monocle2_data_bundle.rds")
hc_modules_rds <- file.path(DATA_OUTPUT_DIR, "hc_monocle2_lineage_modules.rds")


#2. ===========Set shared constants===========

target_stages <- c("E14.5", "E16.5", "P1", "P7", "P14", "P28")
target_labels <- c("IHC", "OHC")

stage_rank_map <- c(
  "E14.5" = 1,
  "E16.5" = 2,
  "P1" = 3,
  "P7" = 4,
  "P14" = 5,
  "P28" = 6
)

stage_short_map <- c(
  "E14.5" = "E14",
  "E16.5" = "E16",
  "P1" = "P1",
  "P7" = "P7",
  "P14" = "P14",
  "P28" = "P28"
)

stage_colors <- c(
  "E14.5" = "#D5A72F",
  "E16.5" = "#F0D84A",
  "P1" = "#BDBDBD",
  "P7" = "#858585",
  "P14" = "#454545",
  "P28" = "#000000"
)

stage_colors_recolor <- c(
  "E14.5" = "#C79A2E",
  "E16.5" = "#E4BC47",
  "P1" = "#B9C8C8",
  "P7" = "#7F9696",
  "P14" = "#4F5D61",
  "P28" = "#1F272B"
)

stage_celltype_levels <- c(
  "E14_IHC", "E14_OHC",
  "E16_IHC", "E16_iOHC", "E16_OHC",
  "P1_IHC", "P1_OHC",
  "P7_IHC", "P7_OHC",
  "P14_IHC", "P14_OHC",
  "P28_IHC", "P28_OHC"
)

stage_celltype_colors <- c(
  "E14_IHC" = "#C9972B",
  "E14_OHC" = "#E2C64C",
  "E16_IHC" = "#A67C2D",
  "E16_iOHC" = "#E8A93A",
  "E16_OHC" = "#F1D84E",
  "P1_IHC" = "#9E9E9E",
  "P1_OHC" = "#C9C9C9",
  "P7_IHC" = "#6F6F6F",
  "P7_OHC" = "#8F8F8F",
  "P14_IHC" = "#3F3F3F",
  "P14_OHC" = "#5A5A5A",
  "P28_IHC" = "#000000",
  "P28_OHC" = "#1F1F1F"
)

stage_celltype_colors_recolor <- c(
  "E14_IHC" = "#C79A2E",
  "E16_IHC" = "#9F762B",
  "P1_IHC" = "#B9C8C8",
  "P7_IHC" = "#7F9696",
  "P14_IHC" = "#4F5D61",
  "P28_IHC" = "#1F272B",
  "E14_OHC" = "#E0B34A",
  "E16_iOHC" = "#D68C35",
  "E16_OHC" = "#E9C85B",
  "P1_OHC" = "#C8D4D1",
  "P7_OHC" = "#8AA69F",
  "P14_OHC" = "#60706E",
  "P28_OHC" = "#2B3134"
)

pseudotime_colors_recolor <- c(
  "#D39A32", "#E3C65D", "#C5D0CC", "#6F8E96", "#263841"
)

lower_detection_limit <- 0.1
min_cells_gene <- 10
ordering_gene_n <- 1500
img_dpi <- 1300


#3. ===========Read h5AD files with rhdf5===========

h5_path_exists <- function(h5ad_file, h5_path) {
  h5_path <- paste0("/", sub("^/+", "", h5_path))
  h5_table <- rhdf5::h5ls(h5ad_file, recursive = TRUE)
  full_paths <- paste0(h5_table$group, "/", h5_table$name)
  full_paths <- sub("^//", "/", full_paths)
  h5_path %in% full_paths
}

read_h5ad_obs_column <- function(h5ad_file, column_name) {
  base_path <- paste0("obs/", column_name)
  categories_path <- paste0(base_path, "/categories")
  codes_path <- paste0(base_path, "/codes")

  if (h5_path_exists(h5ad_file, categories_path) && h5_path_exists(h5ad_file, codes_path)) {
    categories <- as.character(rhdf5::h5read(h5ad_file, categories_path))
    codes <- as.integer(rhdf5::h5read(h5ad_file, codes_path))
    values <- rep(NA_character_, length(codes))
    valid <- !is.na(codes) & codes >= 0
    values[valid] <- categories[codes[valid] + 1L]
    return(values)
  }

  as.character(rhdf5::h5read(h5ad_file, base_path))
}

read_h5ad_obs <- function(h5ad_file, columns) {
  cell_id <- as.character(rhdf5::h5read(h5ad_file, "obs/_index"))
  meta <- data.frame(row.names = cell_id)

  for (column_name in columns) {
    meta[[column_name]] <- read_h5ad_obs_column(h5ad_file, column_name)
  }

  meta
}

read_h5ad_var_index <- function(h5ad_file) {
  as.character(rhdf5::h5read(h5ad_file, "var/_index"))
}

read_h5ad_sparse_x_gene_by_cell <- function(h5ad_file) {
  cell_id <- as.character(rhdf5::h5read(h5ad_file, "obs/_index"))
  gene_id <- read_h5ad_var_index(h5ad_file)
  x_attributes <- rhdf5::h5readAttributes(h5ad_file, "X")

  if (!identical(as.character(x_attributes[["encoding-type"]]), "csr_matrix")) {
    stop("The h5AD X matrix is not a CSR matrix: ", h5ad_file)
  }

  x_data <- as.numeric(rhdf5::h5read(h5ad_file, "X/data"))
  x_indices <- as.integer(rhdf5::h5read(h5ad_file, "X/indices")) + 1L
  x_indptr <- as.integer(rhdf5::h5read(h5ad_file, "X/indptr"))

  n_cells <- length(cell_id)
  n_genes <- length(gene_id)
  cell_index <- rep.int(seq_len(n_cells), diff(x_indptr))

  Matrix::sparseMatrix(
    i = x_indices,
    j = cell_index,
    x = x_data,
    dims = c(n_genes, n_cells),
    dimnames = list(gene_id, cell_id)
  )
}

read_h5ad_dense_x_rows <- function(h5ad_file, row_index) {
  if (length(row_index) == 0) {
    stop("No row index was supplied for dense h5AD reading.")
  }

  x_attributes <- rhdf5::h5readAttributes(h5ad_file, "X")
  encoding_type <- as.character(x_attributes[["encoding-type"]])

  if (identical(encoding_type, "csr_matrix")) {
    x_sparse <- read_h5ad_sparse_x_gene_by_cell(h5ad_file)
    return(as.matrix(x_sparse[as.integer(row_index), , drop = FALSE]))
  }

  n_cells <- length(rhdf5::h5read(h5ad_file, "obs/_index"))
  x <- rhdf5::h5read(
    h5ad_file,
    "X",
    index = list(as.integer(row_index), seq_len(n_cells))
  )
  if (is.null(dim(x))) {
    x <- matrix(x, nrow = length(row_index), byrow = TRUE)
  }
  x
}

read_h5ad_obsm <- function(h5ad_file, obsm_key) {
  mat <- rhdf5::h5read(h5ad_file, paste0("obsm/", obsm_key))
  cell_id <- as.character(rhdf5::h5read(h5ad_file, "obs/_index"))

  if (ncol(mat) == length(cell_id)) {
    mat <- t(mat)
  }

  rownames(mat) <- cell_id
  colnames(mat) <- paste0(obsm_key, "_", seq_len(ncol(mat)))
  mat
}


#4. ===========Patch Monocle2 for current igraph===========

patch_namespace_function <- function(function_name, function_value, namespace_name) {
  namespace_env <- asNamespace(namespace_name)
  environment(function_value) <- namespace_env

  was_locked <- bindingIsLocked(function_name, namespace_env)
  if (was_locked) unlockBinding(function_name, namespace_env)

  assign(function_name, function_value, envir = namespace_env)

  if (was_locked) lockBinding(function_name, namespace_env)

  invisible(TRUE)
}

extract_ddrtree_ordering_fixed <- function(cds, root_cell, verbose = TRUE) {
  dp <- cellPairwiseDistances(cds)
  dp_mst <- minSpanningTree(cds)

  states <- rep(1, ncol(dp))
  names(states) <- igraph::V(dp_mst)$name

  pseudotimes <- rep(0, ncol(dp))
  names(pseudotimes) <- igraph::V(dp_mst)$name

  parents <- rep(NA, ncol(dp))
  names(parents) <- igraph::V(dp_mst)$name

  mst_traversal <- igraph::dfs(
    dp_mst,
    root = root_cell,
    mode = "all",
    unreachable = FALSE,
    father = TRUE
  )

  mst_traversal$father <- as.numeric(mst_traversal$father)
  curr_state <- 1

  for (i in seq_along(mst_traversal$order)) {
    curr_node <- mst_traversal$order[i]
    curr_node_name <- igraph::V(dp_mst)[curr_node]$name

    if (!is.na(mst_traversal$father[curr_node])) {
      parent_node <- mst_traversal$father[curr_node]
      parent_node_name <- igraph::V(dp_mst)[parent_node]$name
      curr_node_pseudotime <- pseudotimes[parent_node_name] + dp[curr_node_name, parent_node_name]

      if (igraph::degree(dp_mst, v = parent_node_name) > 2) {
        curr_state <- curr_state + 1
      }
    } else {
      parent_node_name <- NA
      curr_node_pseudotime <- 0
    }

    pseudotimes[curr_node_name] <- curr_node_pseudotime
    states[curr_node_name] <- curr_state
    parents[curr_node_name] <- parent_node_name
  }

  ordering_df <- data.frame(
    sample_name = names(states),
    cell_state = factor(states),
    pseudo_time = as.vector(pseudotimes),
    parent = parents
  )

  rownames(ordering_df) <- ordering_df$sample_name
  ordering_df
}

project2MST_fixed <- function(cds, Projection_Method) {
  dp_mst <- minSpanningTree(cds)
  Z <- reducedDimS(cds)
  Y <- reducedDimK(cds)

  cds <- findNearestPointOnMST(cds)
  closest_vertex <- cds@auxOrderingData[["DDRTree"]]$pr_graph_cell_proj_closest_vertex
  closest_vertex_names <- colnames(Y)[closest_vertex]

  closest_vertex_df <- as.matrix(closest_vertex)
  rownames(closest_vertex_df) <- rownames(closest_vertex)
  tip_leaves <- names(which(igraph::degree(dp_mst) == 1))

  if (!is.function(Projection_Method)) {
    P <- Y[, closest_vertex]
  } else {
    P <- matrix(rep(0, length(Z)), nrow = nrow(Z))

    for (i in seq_along(closest_vertex)) {
      neighbors <- igraph::as_ids(
        igraph::neighbors(dp_mst, v = closest_vertex_names[i], mode = "all")
      )

      projection <- NULL
      distance <- NULL
      Z_i <- Z[, i]

      for (neighbor in neighbors) {
        if (closest_vertex_names[i] %in% tip_leaves) {
          tmp <- projPointOnLine(Z_i, Y[, c(closest_vertex_names[i], neighbor)])
        } else {
          tmp <- Projection_Method(Z_i, Y[, c(closest_vertex_names[i], neighbor)])
        }

        projection <- rbind(projection, tmp)
        distance <- c(distance, stats::dist(rbind(Z_i, tmp)))
      }

      if (!is(projection, "matrix")) projection <- as.matrix(projection)
      P[, i] <- projection[which(distance == min(distance))[1], ]
    }
  }

  colnames(P) <- colnames(Z)
  dp <- as.matrix(stats::dist(t(P)))
  min_dist <- min(dp[dp != 0])
  dp <- dp + min_dist
  diag(dp) <- 0

  cellPairwiseDistances(cds) <- dp

  gp <- igraph::graph_from_adjacency_matrix(dp, mode = "undirected", weighted = TRUE)
  dp_mst <- igraph::mst(gp)

  cds@auxOrderingData[["DDRTree"]]$pr_graph_cell_proj_tree <- dp_mst
  cds@auxOrderingData[["DDRTree"]]$pr_graph_cell_proj_dist <- P
  cds@auxOrderingData[["DDRTree"]]$pr_graph_cell_proj_closest_vertex <- closest_vertex_df

  cds
}

patch_namespace_function("extract_ddrtree_ordering", extract_ddrtree_ordering_fixed, "monocle")
patch_namespace_function("project2MST", project2MST_fixed, "monocle")


#5. ===========Build and save the all-HC Monocle2 data===========

match_genes <- function(genes, gene_universe) {
  gene_map <- setNames(gene_universe, toupper(gene_universe))
  matched <- unname(gene_map[toupper(genes)])
  unique(matched[!is.na(matched)])
}

build_stage_celltype_plot <- function(meta_df) {
  stage_short <- unname(stage_short_map[as.character(meta_df$stage)])
  celltype_for_plot <- ifelse(
    as.character(meta_df$stage) == "E16.5" & as.character(meta_df$original_celltype) == "iOHC",
    "iOHC",
    as.character(meta_df$scanvi_label)
  )

  factor(paste(stage_short, celltype_for_plot, sep = "_"), levels = stage_celltype_levels)
}

prepare_plot_metadata <- function(hc_cds) {
  plot_meta <- as.data.frame(pData(hc_cds))
  plot_meta$cell_id <- rownames(plot_meta)
  plot_meta$stage <- factor(as.character(plot_meta$stage), levels = target_stages)
  plot_meta$stage_short <- unname(stage_short_map[as.character(plot_meta$stage)])
  plot_meta$stage_celltype_plot <- build_stage_celltype_plot(plot_meta)
  pData(hc_cds)$stage_celltype_plot <- plot_meta$stage_celltype_plot
  plot_meta
}

extract_trajectory_data <- function(hc_cds, plot_meta) {
  cell_coord <- reducedDimS(hc_cds)

  cell_df <- data.frame(
    cell_id = colnames(cell_coord),
    Component_1 = as.numeric(cell_coord[1, ]),
    Component_2 = as.numeric(cell_coord[2, ]),
    stringsAsFactors = FALSE
  )

  cell_df <- merge(cell_df, plot_meta, by = "cell_id", all.x = TRUE, sort = FALSE)
  cell_df <- cell_df[match(colnames(cell_coord), cell_df$cell_id), , drop = FALSE]

  principal_coord <- reducedDimK(hc_cds)
  mst_graph <- minSpanningTree(hc_cds)

  if (is.null(colnames(principal_coord))) {
    colnames(principal_coord) <- igraph::V(mst_graph)$name[seq_len(ncol(principal_coord))]
  }

  principal_df <- data.frame(
    node_id = colnames(principal_coord),
    principal_1 = as.numeric(principal_coord[1, ]),
    principal_2 = as.numeric(principal_coord[2, ]),
    stringsAsFactors = FALSE
  )

  mst_edges <- igraph::as_data_frame(mst_graph, what = "edges")

  edge_df <- data.frame(
    x = principal_df$principal_1[match(mst_edges$from, principal_df$node_id)],
    y = principal_df$principal_2[match(mst_edges$from, principal_df$node_id)],
    xend = principal_df$principal_1[match(mst_edges$to, principal_df$node_id)],
    yend = principal_df$principal_2[match(mst_edges$to, principal_df$node_id)]
  )

  edge_df <- edge_df[complete.cases(edge_df), , drop = FALSE]

  list(cell_df = cell_df, edge_df = edge_df)
}

save_hc_monocle2_bundle <- function(hc_cds, plot_meta, trajectory_data, ordering_genes) {
  saveRDS(hc_cds, hc_cds_rds)
  saveRDS(trajectory_data, hc_trajectory_rds)
  utils::write.csv(plot_meta, hc_plot_meta_csv, row.names = FALSE)
  saveRDS(
    list(
      hc_cds = hc_cds,
      plot_meta = plot_meta,
      trajectory_data = trajectory_data,
      ordering_genes = ordering_genes,
      input_h5ad = input_h5ad
    ),
    hc_bundle_rds
  )
}

load_hc_monocle2_bundle <- function() {
  if (!file.exists(hc_bundle_rds)) {
    stop("The HC Monocle2 data bundle is missing. Run 01_build_hc_monocle2_data.R first.")
  }

  readRDS(hc_bundle_rds)
}

build_hc_monocle2_bundle <- function(force_rebuild = FALSE) {
  if (!force_rebuild && file.exists(hc_bundle_rds)) {
    return(readRDS(hc_bundle_rds))
  }

  message("Building all-HC Monocle2 object from the HVG5000 h5AD.")

  sce_meta <- read_h5ad_obs(
    input_h5ad,
    c(
      "stage", "period", "sample_id", "sample_raw",
      "integration_batch", "original_celltype",
      "scanvi_label", "stage_celltype"
    )
  )

  required_meta_cols <- c("stage", "scanvi_label", "original_celltype")
  missing_meta_cols <- setdiff(required_meta_cols, colnames(sce_meta))

  if (length(missing_meta_cols) > 0) {
    stop("Missing metadata columns: ", paste(missing_meta_cols, collapse = ", "))
  }

  expr_mat <- read_h5ad_sparse_x_gene_by_cell(input_h5ad)

  stage_keep <- sce_meta$stage %in% target_stages
  scanvi_hc_keep <- sce_meta$scanvi_label %in% target_labels
  e16_iohc_keep <- sce_meta$stage == "E16.5" & sce_meta$original_celltype %in% "iOHC"
  hc_keep <- stage_keep & (scanvi_hc_keep | e16_iohc_keep)

  expr_mat <- expr_mat[, hc_keep, drop = FALSE]
  hc_meta <- sce_meta[hc_keep, , drop = FALSE]
  rownames(hc_meta) <- colnames(expr_mat)

  hc_meta$stage <- factor(as.character(hc_meta$stage), levels = target_stages)
  hc_meta$stage_rank <- unname(stage_rank_map[as.character(hc_meta$stage)])
  hc_meta$hc_lineage <- ifelse(
    hc_meta$scanvi_label %in% c("IHC", "OHC"),
    as.character(hc_meta$scanvi_label),
    ifelse(hc_meta$original_celltype %in% "iOHC", "OHC", "HC")
  )
  hc_meta$hc_lineage <- factor(hc_meta$hc_lineage, levels = c("IHC", "OHC", "HC"))

  gene_keep <- Matrix::rowSums(expr_mat > 0) >= min_cells_gene
  expr_mat <- expr_mat[gene_keep, , drop = FALSE]

  gene_meta <- data.frame(
    gene_short_name = rownames(expr_mat),
    row.names = rownames(expr_mat)
  )

  hc_cds <- newCellDataSet(
    expr_mat,
    phenoData = new("AnnotatedDataFrame", data = hc_meta),
    featureData = new("AnnotatedDataFrame", data = gene_meta),
    lowerDetectionLimit = lower_detection_limit,
    expressionFamily = VGAM::uninormal()
  )

  hc_cds <- detectGenes(hc_cds, min_expr = lower_detection_limit)

  gene_mean <- Matrix::rowMeans(expr_mat)
  gene_var <- Matrix::rowMeans(expr_mat ^ 2) - gene_mean ^ 2
  gene_var[is.na(gene_var)] <- 0

  variable_genes <- names(sort(gene_var, decreasing = TRUE))[
    seq_len(min(ordering_gene_n, length(gene_var)))
  ]

  marker_genes <- c(
    "Sox2", "Gata3", "Atoh1", "Ccer2", "Dlk2", "Myo6", "Myo7a",
    "Otof", "Tmc1", "Pcp4", "Calb2", "Slc26a5", "Atp2b2",
    "Barhl1", "Lhx3", "Brf2", "Usf2", "Bdp1", "Arnt"
  )

  marker_genes_present <- intersect(marker_genes, rownames(expr_mat))
  ordering_genes <- unique(c(marker_genes_present, variable_genes))
  hc_cds <- setOrderingFilter(hc_cds, ordering_genes)

  hc_cds <- reduceDimension(
    hc_cds,
    max_components = 2,
    reduction_method = "DDRTree",
    norm_method = "none",
    verbose = FALSE
  )

  hc_cds <- orderCells(hc_cds)

  state_stage_table <- table(pData(hc_cds)$State, pData(hc_cds)$stage)
  root_state <- rownames(state_stage_table)[which.max(state_stage_table[, "E14.5"])]

  hc_cds <- orderCells(hc_cds, root_state = root_state)

  pt_cor <- suppressWarnings(cor(
    pData(hc_cds)$Pseudotime,
    pData(hc_cds)$stage_rank,
    method = "spearman",
    use = "complete.obs"
  ))

  if (!is.na(pt_cor) && pt_cor < 0) {
    pData(hc_cds)$Pseudotime <- max(pData(hc_cds)$Pseudotime, na.rm = TRUE) - pData(hc_cds)$Pseudotime
  }

  pData(hc_cds)$Pseudotime_scaled <- pData(hc_cds)$Pseudotime / max(pData(hc_cds)$Pseudotime, na.rm = TRUE)

  plot_meta <- prepare_plot_metadata(hc_cds)
  trajectory_data <- extract_trajectory_data(hc_cds, plot_meta)

  save_hc_monocle2_bundle(hc_cds, plot_meta, trajectory_data, ordering_genes)

  list(
    hc_cds = hc_cds,
    plot_meta = plot_meta,
    trajectory_data = trajectory_data,
    ordering_genes = ordering_genes,
    input_h5ad = input_h5ad
  )
}


#6. ===========Shared plotting helpers===========

save_pdf_plot <- function(plot_obj, filename, width, height) {
  dir.create(dirname(filename), recursive = TRUE, showWarnings = FALSE)
  pdf_device <- if (capabilities("cairo")) grDevices::cairo_pdf else grDevices::pdf

  ggplot2::ggsave(
    filename = filename,
    plot = plot_obj,
    width = width,
    height = height,
    units = "in",
    device = pdf_device,
    bg = "white"
  )
}

save_png_plot <- function(plot_obj, filename, width, height) {
  dir.create(dirname(filename), recursive = TRUE, showWarnings = FALSE)

  ggplot2::ggsave(
    filename = filename,
    plot = plot_obj,
    width = width,
    height = height,
    units = "in",
    dpi = img_dpi,
    bg = "white"
  )
}

theme_hc <- function(base_size = 12) {
  ggplot2::theme_classic(base_size = base_size) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(size = base_size + 1, hjust = 0.5, face = "plain"),
      axis.text = ggplot2::element_text(color = "black"),
      axis.title = ggplot2::element_text(color = "black"),
      legend.title = ggplot2::element_text(size = base_size - 1),
      legend.text = ggplot2::element_text(size = base_size - 2),
      aspect.ratio = 1
    )
}

scale_gene_expression <- function(x) {
  if (stats::sd(x) == 0 || all(is.na(x))) {
    return(rep(0, length(x)))
  }

  as.numeric(scale(x))
}

wrap_text <- function(x, width = 34) {
  vapply(x, function(label) paste(strwrap(label, width = width), collapse = "\n"), character(1))
}

clean_gene_key <- function(x) {
  x <- trimws(as.character(x))
  x <- sub("\\..*$", "", x)
  toupper(x)
}

match_marker_feature <- function(marker_aliases, feature_ids) {
  feature_lookup <- setNames(feature_ids, clean_gene_key(feature_ids))
  alias_keys <- clean_gene_key(marker_aliases)
  matched <- unname(feature_lookup[alias_keys])
  matched <- matched[!is.na(matched)]

  if (length(matched) == 0) {
    return(NA_character_)
  }

  matched[1]
}

get_hc_lineage <- function(stage, scanvi_label, original_celltype) {
  ifelse(stage == "E16.5" & original_celltype == "iOHC", "OHC", scanvi_label)
}

get_dotplot_group <- function(stage, hc_lineage) {
  period_group <- ifelse(
    stage %in% c("E14.5", "E16.5"),
    "Early",
    ifelse(stage %in% c("P1", "P7"), "Intermediate", "Mature")
  )

  paste(period_group, hc_lineage)
}
