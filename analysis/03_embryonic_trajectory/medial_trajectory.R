#1. ===========Load packages and parameters
set.seed(0)
options(stringsAsFactors = FALSE)

required_pkgs <- c(
  "rhdf5", "monocle", "Biobase", "Matrix", "VGAM", "igraph",
  "ggplot2", "gridExtra"
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
  library(VGAM)
  library(igraph)
  library(ggplot2)
  library(gridExtra)
})

base_dir <- normalizePath(
  if (dir.exists(file.path(getwd(), "analysis"))) getwd() else file.path(getwd(), "..", ".."),
  winslash = "/",
  mustWork = TRUE
)
data_root <- normalizePath(
  Sys.getenv("COCHLEA_DATA_DIR", file.path(base_dir, "Data")),
  winslash = "/",
  mustWork = FALSE
)
results_root <- normalizePath(
  Sys.getenv("COCHLEA_RESULTS_DIR", file.path(base_dir, "results")),
  winslash = "/",
  mustWork = FALSE
)
input_h5ad_medial_candidates <- c(
  file.path(results_root, "03_embryonic_trajectory/inputs/embryonic_medial_monocle2_input.h5ad"),
  file.path(data_root, "Monocle2_Embryonic_trajectory/embryonic_medial_monocle2_input.h5ad"),
  file.path(data_root, "embryonic_medial_monocle2_input.h5ad")
)
input_h5ad_medial <- input_h5ad_medial_candidates[which(file.exists(input_h5ad_medial_candidates))[1]]
if (is.na(input_h5ad_medial)) input_h5ad_medial <- input_h5ad_medial_candidates[1]

output_dir <- file.path(results_root, "03_embryonic_trajectory/figures")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

stage_order <- c("E9.5", "E11.5", "E13.5", "E14.5", "E16.5")
stage_rank_map <- c("E9.5" = 1, "E11.5" = 2, "E13.5" = 3, "E14.5" = 4, "E16.5" = 5)
stage_short_map <- c("E9.5" = "E9", "E11.5" = "E11", "E13.5" = "E13.5", "E14.5" = "E14", "E16.5" = "E16")

min_cells_gene <- 10
lower_detection_limit <- 0.2
fate_de_pval_cutoff <- 0.01
fate_de_max_genes <- 300
point_size <- 0.78
point_alpha <- 0.88

medial_celltype_levels <- c(
  "OV epithelial cells", "Prosensory domain", "Medial domain",
  "M.PsC", "IHC", "IPhC", "IBC", "IPC"
)

celltype_label_map <- c(
  "OV epithelial cells" = "OV",
  "Prosensory domain" = "PsD",
  "Medial domain" = "M.PsD",
  "M.PsC" = "M.PsC",
  "IHC" = "IHC",
  "IPhC" = "IPhC",
  "IBC" = "IBC",
  "IPC" = "IPC"
)

stage_colors <- c(
  "E9" = "#4C78A8",
  "E11" = "#72B7B2",
  "E13.5" = "#F2A541",
  "E14" = "#D96C75",
  "E16" = "#59A14F"
)

celltype_colors <- c(
  "OV" = "#B9A7C8",
  "PsD" = "#8A8A8A",
  "M.PsD" = "#6D5CA8",
  "M.PsC" = "#6D6AAE",
  "IHC" = "#B83232",
  "IPhC" = "#B8A1C8",
  "IBC" = "#D58AA8",
  "IPC" = "#4C9BC7"
)


#2. ===========Define medial marker genes
marker_genes_from_xlsx <- list(
  OV = c("Epcam", "Fbxo2", "Tbx2", "Pax2", "Foxg1", "Six1", "Eya1", "Sox2", "Lmx1a", "Tbx1", "Sp5", "Id3"),
  PsD = c("Sox2", "Jag1", "Lfng", "Eya1", "Six1", "Notch1", "Notch2", "Cdkn1b", "Fgf20"),
  M.PsD = c("Fgf10", "Fgf20", "Eya1", "Six1", "Jag1", "Lfng", "Sox2"),
  M.PsC = c("Fgf20", "Ebf1", "Anxa5", "Sox2", "Eya1", "Cdkn1b", "Jag1", "Crym", "Ntf3", "Hmga2", "Lockd", "Tectb"),
  IHC = c("Selenom", "S100a1", "Pcp4", "Acbd7", "Fgf8", "Pvalb", "Ccer2", "Ccl21a"),
  IPhC = c("Hes5", "Matn4", "Anxa5", "Fabp7", "Gjb2"),
  IPC = c("Gm45645", "Npy", "S100b", "Fam159b", "Tectb")
)

marker_genes_literature <- list(
  HC_fate = c("Atoh1", "Gfi1", "Myo6", "Myo7a", "Barhl1", "Lhx3", "Ikzf2"),
  IBC = c("Id1", "Id2", "Id3", "Hes1", "Krt18", "Anxa5")
)

medial_marker_genes <- unique(unlist(c(marker_genes_from_xlsx, marker_genes_literature), use.names = FALSE))


#3. ===========Patch Monocle2 for current igraph
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
      if (igraph::degree(dp_mst, v = parent_node_name) > 2) curr_state <- curr_state + 1
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


#4. ===========Load h5AD input
normalize_celltype <- function(x) {
  x <- as.character(x)
  x[grepl("^IBC[-_ ]?like$", x)] <- "IBC"
  x
}

load_h5ad_data <- function(input_h5ad, celltype_col = "trajectory_celltype") {
  if (!file.exists(input_h5ad)) {
    stop("Input h5AD not found: ", input_h5ad)
  }

  cell_id <- as.character(rhdf5::h5read(input_h5ad, "obs/_index"))
  gene_id <- as.character(rhdf5::h5read(input_h5ad, "var/_index"))

  x_data <- as.numeric(rhdf5::h5read(input_h5ad, "X/data"))
  x_indices <- as.integer(rhdf5::h5read(input_h5ad, "X/indices")) + 1L
  x_indptr <- as.integer(rhdf5::h5read(input_h5ad, "X/indptr"))

  n_cells <- length(cell_id)
  n_genes <- length(gene_id)
  cell_index <- rep.int(seq_len(n_cells), diff(x_indptr))

  expr_mat <- Matrix::sparseMatrix(
    i = x_indices,
    j = cell_index,
    x = x_data,
    dims = c(n_genes, n_cells),
    dimnames = list(gene_id, cell_id)
  )

  read_h5ad_categorical <- function(key) {
    categories <- as.character(rhdf5::h5read(input_h5ad, paste0("obs/", key, "/categories")))
    codes <- rhdf5::h5read(input_h5ad, paste0("obs/", key, "/codes")) + 1L
    as.character(categories[codes])
  }

  meta <- data.frame(
    stage = read_h5ad_categorical("stage"),
    trajectory_celltype = read_h5ad_categorical(celltype_col),
    trajectory_branch = read_h5ad_categorical("trajectory_branch"),
    row.names = cell_id,
    stringsAsFactors = FALSE
  )

  if (!celltype_col %in% colnames(meta)) {
    stop("Cell type column was not found: ", celltype_col)
  }

  expr_mat <- as(expr_mat, "dgCMatrix")
  meta <- meta[colnames(expr_mat), , drop = FALSE]

  meta$trajectory_celltype_raw <- as.character(meta[[celltype_col]])
  meta$trajectory_celltype <- normalize_celltype(meta[[celltype_col]])
  list(expr_mat = expr_mat, meta = meta)
}


#5. ===========Select fate-DE ordering genes
select_fate_de_genes <- function(expr_mat, meta) {
  cell_type <- as.character(meta$trajectory_celltype)
  fate_group <- ifelse(cell_type == "IHC", "IHC",
    ifelse(cell_type %in% c("M.PsC", "IPhC", "IBC", "IPC"), "supporting", "trunk")
  )
  fate_group <- factor(fate_group, levels = c("trunk", "IHC", "supporting"))

  keep_gene <- Matrix::rowSums(expr_mat > lower_detection_limit) >= min_cells_gene
  test_mat <- as.matrix(expr_mat[keep_gene, , drop = FALSE])
  gene_var <- apply(test_mat, 1, stats::var)
  test_mat <- test_mat[gene_var > 0, , drop = FALSE]

  p_values <- apply(test_mat, 1, function(x) {
    suppressWarnings(stats::kruskal.test(x, fate_group)$p.value)
  })

  q_values <- p.adjust(p_values, method = "BH")
  selected_genes <- names(q_values)[q_values <= fate_de_pval_cutoff]

  if (length(selected_genes) < 80) {
    selected_genes <- names(sort(q_values))[seq_len(min(fate_de_max_genes, length(q_values)))]
  } else {
    selected_genes <- selected_genes[order(q_values[selected_genes])]
    selected_genes <- selected_genes[seq_len(min(fate_de_max_genes, length(selected_genes)))]
  }

  unique(c(intersect(medial_marker_genes, rownames(expr_mat)), selected_genes))
}


#6. ===========Build and order medial Monocle2 object
select_root_state <- function(cds, root_label) {
  meta <- as.data.frame(pData(cds))
  root_table <- table(meta$State, meta$trajectory_celltype)

  if (!root_label %in% colnames(root_table)) {
    stop("Root label not found in Monocle2 states: ", root_label)
  }

  root_counts <- root_table[, root_label]
  if (all(root_counts == 0)) {
    stop("No cells found for root label: ", root_label)
  }

  candidate_states <- rownames(root_table)[root_counts == max(root_counts)]
  if (length(candidate_states) == 1) {
    return(candidate_states)
  }

  root_meta <- meta[as.character(meta$trajectory_celltype) == root_label, , drop = FALSE]
  root_meta <- root_meta[as.character(root_meta$State) %in% candidate_states, , drop = FALSE]
  state_stage <- tapply(root_meta$stage_rank, root_meta$State, median, na.rm = TRUE)
  names(state_stage)[which.min(state_stage)]
}

orient_pseudotime_from_root <- function(cds, root_label) {
  meta <- as.data.frame(pData(cds))
  root_cells <- as.character(meta$trajectory_celltype) == root_label
  root_pt <- stats::median(meta$Pseudotime[root_cells], na.rm = TRUE)
  other_pt <- stats::median(meta$Pseudotime[!root_cells], na.rm = TRUE)

  if (is.finite(root_pt) && is.finite(other_pt) && root_pt > other_pt) {
    pData(cds)$Pseudotime_oriented <- max(meta$Pseudotime, na.rm = TRUE) - meta$Pseudotime
  } else {
    pData(cds)$Pseudotime_oriented <- meta$Pseudotime
  }

  oriented <- pData(cds)$Pseudotime_oriented
  root_min <- min(oriented[root_cells], na.rm = TRUE)
  if (is.finite(root_min)) {
    oriented <- pmax(oriented - root_min, 0)
  }

  max_pt <- max(oriented, na.rm = TRUE)
  if (!is.finite(max_pt) || max_pt == 0) max_pt <- 1

  pData(cds)$Pseudotime_oriented <- oriented
  pData(cds)$Pseudotime_scaled <- oriented / max_pt
  cds
}

run_monocle_subset <- function(h5_data) {
  meta <- h5_data$meta
  expr_mat <- h5_data$expr_mat

  keep_cells <- rownames(meta)[
    meta$trajectory_celltype %in% medial_celltype_levels &
      as.character(meta$stage) %in% stage_order
  ]

  meta <- meta[keep_cells, , drop = FALSE]
  expr_mat <- expr_mat[, keep_cells, drop = FALSE]

  present_celltype_levels <- medial_celltype_levels[medial_celltype_levels %in% unique(meta$trajectory_celltype)]
  meta$stage <- factor(as.character(meta$stage), levels = stage_order)
  meta$stage_rank <- unname(stage_rank_map[as.character(meta$stage)])
  meta$trajectory_celltype <- factor(as.character(meta$trajectory_celltype), levels = present_celltype_levels)
  meta$stage_short <- factor(unname(stage_short_map[as.character(meta$stage)]), levels = unname(stage_short_map[stage_order]))
  meta$celltype_plot <- factor(
    unname(celltype_label_map[as.character(meta$trajectory_celltype)]),
    levels = unname(celltype_label_map[present_celltype_levels])
  )

  gene_keep <- Matrix::rowSums(expr_mat > lower_detection_limit) >= min_cells_gene
  expr_mat <- expr_mat[gene_keep, , drop = FALSE]

  ordering_genes <- select_fate_de_genes(expr_mat, meta)
  gene_meta <- data.frame(gene_short_name = rownames(expr_mat), row.names = rownames(expr_mat))

  cds <- newCellDataSet(
    expr_mat,
    phenoData = new("AnnotatedDataFrame", data = meta),
    featureData = new("AnnotatedDataFrame", data = gene_meta),
    lowerDetectionLimit = lower_detection_limit,
    expressionFamily = VGAM::uninormal()
  )

  cds <- detectGenes(cds, min_expr = lower_detection_limit)
  cds <- setOrderingFilter(cds, ordering_genes)
  cds <- reduceDimension(
    cds,
    max_components = 2,
    reduction_method = "DDRTree",
    norm_method = "none",
    verbose = FALSE
  )

  cds <- orderCells(cds)
  root_state <- select_root_state(cds, "OV epithelial cells")
  cds <- orderCells(cds, root_state = root_state)
  cds <- orient_pseudotime_from_root(cds, "OV epithelial cells")

  pt_cor <- suppressWarnings(cor(
    pData(cds)$Pseudotime_scaled,
    pData(cds)$stage_rank,
    method = "spearman",
    use = "complete.obs"
  ))

  qc <- data.frame(
    cells = ncol(expr_mat),
    genes = nrow(expr_mat),
    ordering_genes = length(ordering_genes),
    root_state = root_state,
    stage_pseudotime_cor = round(pt_cor, 3),
    stringsAsFactors = FALSE
  )

  list(cds = cds, qc = qc)
}


#7. ===========Plot real DDRTree trajectory
get_trajectory_data <- function(cds) {
  cell_coord <- reducedDimS(cds)

  cell_df <- data.frame(
    cell_id = colnames(cell_coord),
    Component_1 = -as.numeric(cell_coord[1, ]),
    Component_2 = as.numeric(cell_coord[2, ]),
    stringsAsFactors = FALSE
  )

  cell_meta <- as.data.frame(pData(cds))
  cell_df <- cbind(cell_df, cell_meta[cell_df$cell_id, , drop = FALSE])

  principal_coord <- reducedDimK(cds)
  mst_graph <- minSpanningTree(cds)

  if (is.null(colnames(principal_coord))) {
    colnames(principal_coord) <- igraph::V(mst_graph)$name[seq_len(ncol(principal_coord))]
  }

  principal_df <- data.frame(
    node_id = colnames(principal_coord),
    principal_1 = -as.numeric(principal_coord[1, ]),
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
  list(cells = cell_df, edges = edge_df)
}

make_branch_labels <- function(cell_df) {
  get_group <- function(celltypes) {
    keep <- as.character(cell_df$celltype_plot) %in% celltypes
    cell_df[keep, , drop = FALSE]
  }

  trunk_df <- get_group(c("OV", "PsD", "M.PsD"))
  ihc_df <- get_group("IHC")
  support_df <- get_group(c("M.PsC", "IPhC", "IBC", "IPC"))

  data.frame(
    label = c("OV/PsD/M.PsD", "IHC", "M.PsC/IPhC/IBC/IPC"),
    x = c(
      as.numeric(stats::quantile(trunk_df$Component_1, 0.08, na.rm = TRUE)) + 0.35,
      as.numeric(stats::quantile(ihc_df$Component_1, 0.72, na.rm = TRUE)),
      as.numeric(stats::quantile(support_df$Component_1, 0.94, na.rm = TRUE)) - 0.20
    ),
    y = c(
      stats::median(trunk_df$Component_2, na.rm = TRUE) + 0.48,
      as.numeric(stats::quantile(ihc_df$Component_2, 0.18, na.rm = TRUE)) - 0.28,
      as.numeric(stats::quantile(support_df$Component_2, 0.10, na.rm = TRUE)) - 0.05
    ),
    hjust = c(0, 0.5, 1),
    vjust = c(0.5, 0.5, 0.5),
    stringsAsFactors = FALSE
  )
}

plot_trajectory_panel <- function(cds, color_by, title_text, color_values = NULL, show_branch_labels = FALSE) {
  traj <- get_trajectory_data(cds)
  cell_df <- traj$cells
  cell_df$plot_value <- cell_df[[color_by]]
  cell_df <- cell_df[order(cell_df$Pseudotime_scaled), , drop = FALSE]

  p <- ggplot(cell_df, aes(x = Component_1, y = Component_2)) +
    geom_segment(
      data = traj$edges,
      aes(x = x, y = y, xend = xend, yend = yend),
      inherit.aes = FALSE,
      color = "grey20",
      linewidth = 0.35,
      alpha = 0.78
    ) +
    geom_point(aes(color = plot_value), size = point_size, alpha = point_alpha) +
    coord_equal() +
    theme_classic(base_size = 10) +
    labs(title = title_text, x = "DDRTree 1", y = "DDRTree 2") +
    theme(
      plot.title = element_text(size = 12.5, hjust = 0.5),
      axis.title = element_text(size = 9.5),
      axis.text = element_text(size = 8, color = "black"),
      legend.title = element_text(size = 9),
      legend.text = element_text(size = 8),
      legend.key.height = grid::unit(0.32, "cm"),
      aspect.ratio = 0.82
    )

  if (color_by == "Pseudotime_scaled") {
    p <- p +
      scale_color_gradientn(
        colors = c("#3D4FA3", "#A9D6C2", "#F2E85E", "#D64B22"),
        limits = c(0, 1),
        name = "Pseudotime"
      )
  } else {
    p <- p +
      scale_color_manual(
        values = color_values,
        drop = TRUE,
        name = ifelse(color_by == "stage_short", "Timepoint", "Cell type")
      )
  }

  if (show_branch_labels) {
    branch_label_df <- make_branch_labels(cell_df)
    p <- p +
      geom_text(
        data = branch_label_df,
        aes(x = x, y = y, label = label, hjust = hjust, vjust = vjust),
        inherit.aes = FALSE,
        size = 3.0,
        color = "black",
        check_overlap = TRUE
      )
  }

  p
}

save_plot_pair <- function(plot_obj, file_prefix, width, height) {
  pdf_path <- paste0(file_prefix, ".pdf")
  png_path <- paste0(file_prefix, ".png")

  ggsave(
    filename = pdf_path,
    plot = plot_obj,
    width = width,
    height = height,
    units = "in",
    device = grDevices::cairo_pdf,
    bg = "white"
  )

  ggsave(
    filename = png_path,
    plot = plot_obj,
    width = width,
    height = height,
    units = "in",
    dpi = 1200,
    bg = "white",
    limitsize = FALSE
  )

  c(pdf_path, png_path)
}


#8. ===========Run and save final medial trajectory
medial_data <- load_h5ad_data(input_h5ad_medial, celltype_col = "trajectory_celltype")
medial_result <- run_monocle_subset(medial_data)

p_celltype <- plot_trajectory_panel(
  medial_result$cds,
  color_by = "celltype_plot",
  title_text = "Medial trajectory by cell type",
  color_values = celltype_colors,
  show_branch_labels = TRUE
)

p_timepoint <- plot_trajectory_panel(
  medial_result$cds,
  color_by = "stage_short",
  title_text = "Medial trajectory by timepoint",
  color_values = stage_colors
)

p_pseudotime <- plot_trajectory_panel(
  medial_result$cds,
  color_by = "Pseudotime_scaled",
  title_text = "Medial pseudotime"
)

panel_medial <- gridExtra::arrangeGrob(
  p_celltype,
  p_timepoint,
  p_pseudotime,
  ncol = 3
)

saved_files <- save_plot_pair(
  panel_medial,
  file.path(output_dir, "Embryonic_E9_E16_panel_a_medial_real_fatede_trajectory"),
  width = 12.0,
  height = 4.3
)

cat(
  "Medial real DDRTree saved. cells=", medial_result$qc$cells,
  ", genes=", medial_result$qc$genes,
  ", ordering_genes=", medial_result$qc$ordering_genes,
  ", root_state=", medial_result$qc$root_state,
  ", stage_cor=", medial_result$qc$stage_pseudotime_cor,
  "\n",
  sep = ""
)
