#1. ===========Load packages and parameters
set.seed(0)
options(stringsAsFactors = FALSE)

required_pkgs <- c(
  "monocle", "Biobase", "Matrix", "VGAM", "igraph",
  "ggplot2", "grid", "gridExtra", "scales"
)

missing_pkgs <- required_pkgs[
  !vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_pkgs) > 0) {
  stop("Missing R packages: ", paste(missing_pkgs, collapse = ", "))
}

suppressPackageStartupMessages({
  library(monocle)
  library(Biobase)
  library(Matrix)
  library(VGAM)
  library(igraph)
  library(ggplot2)
  library(grid)
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
script_dir <- file.path(base_dir, "analysis", "03_embryonic_trajectory")
output_dir <- file.path(results_root, "03_embryonic_trajectory/figures")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

heatmap_min_genes <- 60
heatmap_max_genes <- 92
candidate_gene_n <- 500
side_bins <- 58
zscore_cap <- 3.0
min_cells_gene <- 10
lower_detection_limit <- 0.2

heatmap_colors <- c("#4F73A8", "#FAFAFA", "#8B0000")
heatmap_values <- scales::rescale(c(-zscore_cap, 0, zscore_cap))


#2. ===========Load refined trajectory functions
load_refined_environment <- function(script_path) {
  script_lines <- readLines(script_path, warn = FALSE)
  run_line <- grep("^#8\\. ===========Run", script_lines)[1]
  if (is.na(run_line)) stop("Run block was not found in script: ", script_path)

  refined_env <- new.env(parent = globalenv())
  eval(parse(text = paste(script_lines[seq_len(run_line - 1)], collapse = "\n")), envir = refined_env)
  refined_env
}

medial_env <- load_refined_environment(file.path(script_dir, "medial_trajectory.R"))
lateral_env <- load_refined_environment(file.path(script_dir, "lateral_trajectory.R"))


#3. ===========Rebuild the accepted Monocle2 trajectories
medial_data <- medial_env$load_h5ad_data(
  medial_env$input_h5ad_medial,
  celltype_col = "trajectory_celltype"
)

lateral_data <- lateral_env$load_h5ad_data(
  lateral_env$input_h5ad_lateral,
  celltype_col = "trajectory_celltype"
)

medial_result <- medial_env$run_monocle_subset(medial_data)
lateral_result <- lateral_env$run_monocle_subset(lateral_data)


#4. ===========Set branch direction and biological labels
branch_configs <- list(
  Medial = list(
    title = "Medial branch heatmap",
    branch_label = "M.PsD",
    left_title = "M.PsC/IPhC/IBC/IPC",
    right_title = "IHC",
    left_color = "#6D6AAE",
    right_color = "#B83232",
    branchpoint_labels = "M.PsD",
    left_labels = c("M.PsC", "IPhC", "IBC", "IPC"),
    right_labels = "IHC",
    marker_genes = c(
      "Fgf10", "Fgf20", "Sox2", "Jag1", "Lfng", "Eya1",
      "Anxa5", "Ntf3", "Crym", "Tectb", "Hes5", "Matn4",
      "Id1", "Id2", "Id3", "Npy", "S100b", "Fgf8", "Pvalb",
      "S100a1", "Pcp4", "Atoh1", "Gfi1", "Myo6", "Myo7a"
    ),
    key_gene_labels = c(
      "Sox2", "Jag1", "Fgf20", "Fgf10", "Eya1", "Lfng",
      "Anxa5", "Ntf3", "Crym", "Hes5", "Id1", "Id2", "Id3",
      "Npy", "S100b", "Atoh1", "Gfi1", "Fgf8", "Pvalb",
      "S100a1", "Pcp4", "Myo6", "Myo7a"
    )
  ),
  Lateral = list(
    title = "Lateral branch heatmap",
    branch_label = "L.PsD",
    left_title = "L.PsC/HeC",
    right_title = "OHC/iOHC",
    left_color = "#82B45F",
    right_color = "#9C2F2A",
    branchpoint_labels = "L.PsD",
    left_labels = c("L.PsC", "HeC"),
    right_labels = c("OHC", "iOHC"),
    marker_genes = c(
      "Bmp4", "Rorb", "Fgfr3", "Camta1", "Lockd", "Prox1",
      "Igfbp3", "Socs2", "Tectb", "Kazald1", "Fabp7", "Gata2",
      "Hes6", "Pcp4", "Rprm", "Lhfpl5", "Ccer2", "Pou4f3",
      "Atoh1", "Gfi1", "Myo6", "Myo7a"
    ),
    key_gene_labels = c(
      "Bmp4", "Rorb", "Fgfr3", "Camta1", "Lockd", "Prox1",
      "Igfbp3", "Socs2", "Kazald1", "Fabp7", "Gata2",
      "Hes6", "Pcp4", "Rprm", "Lhfpl5", "Ccer2", "Pou4f3",
      "Atoh1", "Gfi1", "Myo6", "Myo7a"
    )
  )
)


#5. ===========Select branch-associated genes
select_heatmap_genes <- function(cds, config) {
  meta <- as.data.frame(Biobase::pData(cds))
  expr_mat <- Biobase::exprs(cds)

  cell_type <- as.character(meta$celltype_plot)
  names(cell_type) <- rownames(meta)
  branch_cells <- rownames(meta)[cell_type %in% config$branchpoint_labels]
  left_cells <- rownames(meta)[cell_type %in% config$left_labels]
  right_cells <- rownames(meta)[cell_type %in% config$right_labels]

  if (length(branch_cells) == 0 || length(left_cells) == 0 || length(right_cells) == 0) {
    stop("Branch, left, or right cells are missing for: ", config$title)
  }

  path_cells <- unique(c(branch_cells, left_cells, right_cells))
  keep_gene <- Matrix::rowSums(expr_mat[, path_cells, drop = FALSE] > lower_detection_limit) >= min_cells_gene
  expr_use <- expr_mat[keep_gene, path_cells, drop = FALSE]

  branch_mean <- Matrix::rowMeans(expr_use[, branch_cells, drop = FALSE])
  left_mean <- Matrix::rowMeans(expr_use[, left_cells, drop = FALSE])
  right_mean <- Matrix::rowMeans(expr_use[, right_cells, drop = FALSE])
  gene_mean <- Matrix::rowMeans(expr_use)
  gene_var <- Matrix::rowMeans(expr_use * expr_use) - gene_mean ^ 2

  terminal_bias <- abs(left_mean - right_mean)
  terminal_gain <- pmax(left_mean, right_mean) - branch_mean
  branch_transition <- abs(branch_mean - (left_mean + right_mean) / 2)
  dynamic_score <- 1.20 * terminal_bias +
    0.45 * pmax(terminal_gain, 0) +
    0.25 * branch_transition +
    0.35 * sqrt(pmax(gene_var, 0))

  ranked_genes <- names(sort(dynamic_score, decreasing = TRUE))
  seed_genes <- intersect(config$marker_genes, rownames(expr_mat))
  selected_genes <- unique(c(seed_genes, ranked_genes))

  if (length(selected_genes) < heatmap_min_genes) {
    stop("Too few genes were selected for branch heatmap.")
  }

  selected_genes[seq_len(min(length(selected_genes), candidate_gene_n))]
}


#6. ===========Build direction-controlled heatmap matrix
clip_values <- function(x, lower, upper) {
  pmax(pmin(x, upper), lower)
}

make_rank_bins <- function(cell_ids, pseudotime, n_bins) {
  cell_ids <- cell_ids[order(pseudotime[cell_ids], na.last = NA)]
  if (length(cell_ids) == 0) return(vector("list", n_bins))

  bin_id <- cut(
    seq_along(cell_ids),
    breaks = n_bins,
    labels = FALSE,
    include.lowest = TRUE
  )

  split(cell_ids, factor(bin_id, levels = seq_len(n_bins)))
}

smooth_matrix <- function(mat, window = 5) {
  half_window <- floor(window / 2)
  smoothed <- mat

  for (j in seq_len(ncol(mat))) {
    left <- max(1, j - half_window)
    right <- min(ncol(mat), j + half_window)
    smoothed[, j] <- rowMeans(mat[, left:right, drop = FALSE], na.rm = TRUE)
  }

  smoothed
}

assign_peak_group <- function(peak_col, n_bins) {
  group_levels <- c(
    "Left terminal", "Left transition", "Branchpoint",
    "Right transition", "Right terminal"
  )

  breaks <- c(
    0,
    floor(n_bins * 0.20),
    floor(n_bins * 0.43),
    ceiling(n_bins * 0.57),
    ceiling(n_bins * 0.80),
    n_bins
  )

  breaks <- cummax(breaks)
  breaks[length(breaks)] <- n_bins

  cut(
    peak_col,
    breaks = breaks,
    labels = group_levels,
    include.lowest = TRUE
  )
}

select_balanced_genes <- function(heatmap_z, config) {
  peak_col <- max.col(heatmap_z, ties.method = "first")
  peak_group <- assign_peak_group(peak_col, ncol(heatmap_z))
  amplitude <- apply(heatmap_z, 1, function(x) max(x, na.rm = TRUE) - min(x, na.rm = TRUE))

  quota <- c(
    "Left terminal" = 16,
    "Left transition" = 20,
    "Branchpoint" = 18,
    "Right transition" = 20,
    "Right terminal" = 16
  )

  seed_genes <- intersect(config$marker_genes, rownames(heatmap_z))
  selected_genes <- character()

  for (group_name in names(quota)) {
    group_genes <- rownames(heatmap_z)[peak_group == group_name]
    group_genes <- group_genes[order(amplitude[group_genes], decreasing = TRUE)]

    group_seed <- intersect(seed_genes, group_genes)
    group_seed <- head(group_seed[order(amplitude[group_seed], decreasing = TRUE)], 4)

    group_pool <- setdiff(group_genes, group_seed)
    group_pick <- c(group_seed, head(group_pool, max(quota[group_name] - length(group_seed), 0)))
    selected_genes <- c(selected_genes, group_pick)
  }

  if (length(selected_genes) < heatmap_min_genes) {
    fallback_genes <- rownames(heatmap_z)[order(amplitude, decreasing = TRUE)]
    selected_genes <- unique(c(selected_genes, fallback_genes))
  }

  selected_genes <- unique(selected_genes)
  selected_genes[seq_len(min(length(selected_genes), heatmap_max_genes))]
}

spread_label_y <- function(label_df, min_gap = 3.2, lower = 2, upper) {
  if (nrow(label_df) <= 1) {
    label_df$label_y <- label_df$plot_y
    return(label_df)
  }

  label_df <- label_df[order(label_df$plot_y), , drop = FALSE]
  label_y <- pmin(pmax(label_df$plot_y, lower), upper)

  for (i in seq_len(nrow(label_df))[-1]) {
    if ((label_y[i] - label_y[i - 1]) < min_gap) {
      label_y[i] <- label_y[i - 1] + min_gap
    }
  }

  overflow <- max(label_y) - upper
  if (overflow > 0) label_y <- label_y - overflow

  for (i in rev(seq_len(nrow(label_df) - 1))) {
    if ((label_y[i + 1] - label_y[i]) < min_gap) {
      label_y[i] <- label_y[i + 1] - min_gap
    }
  }

  underflow <- lower - min(label_y)
  if (underflow > 0) label_y <- label_y + underflow

  if (min(label_y) < lower || max(label_y) > upper) {
    label_y <- seq(lower, upper, length.out = nrow(label_df))
  }

  label_df$label_y <- label_y
  label_df
}

build_branch_heatmap <- function(cds, config) {
  meta <- as.data.frame(Biobase::pData(cds))
  expr_mat <- Biobase::exprs(cds)
  candidate_genes <- select_heatmap_genes(cds, config)
  expr_use <- expr_mat[candidate_genes, , drop = FALSE]

  cell_type <- as.character(meta$celltype_plot)
  names(cell_type) <- rownames(meta)
  pseudotime <- meta$Pseudotime_scaled
  names(pseudotime) <- rownames(meta)

  branch_cells <- rownames(meta)[cell_type %in% config$branchpoint_labels]
  left_cells <- rownames(meta)[cell_type %in% config$left_labels]
  right_cells <- rownames(meta)[cell_type %in% config$right_labels]

  left_path_cells <- unique(c(branch_cells, left_cells))
  right_path_cells <- unique(c(branch_cells, right_cells))

  left_bins <- rev(make_rank_bins(left_path_cells, pseudotime, side_bins))
  right_bins <- make_rank_bins(right_path_cells, pseudotime, side_bins)
  bin_list <- c(left_bins, right_bins)

  bin_mat <- vapply(bin_list, function(bin_cells) {
    if (length(bin_cells) == 0) return(rep(NA_real_, length(candidate_genes)))
    Matrix::rowMeans(expr_use[, bin_cells, drop = FALSE])
  }, numeric(length(candidate_genes)))

  rownames(bin_mat) <- candidate_genes
  colnames(bin_mat) <- paste0("Bin_", seq_len(ncol(bin_mat)))
  bin_mat <- smooth_matrix(bin_mat, window = 5)

  keep_gene <- apply(bin_mat, 1, function(x) stats::sd(x, na.rm = TRUE) > 1e-6)
  bin_mat <- bin_mat[keep_gene, , drop = FALSE]

  heatmap_z <- t(scale(t(bin_mat)))
  heatmap_z[is.na(heatmap_z)] <- 0

  selected_genes <- select_balanced_genes(heatmap_z, config)
  heatmap_z <- heatmap_z[selected_genes, , drop = FALSE]
  heatmap_z <- clip_values(heatmap_z, -zscore_cap, zscore_cap)

  peak_col <- max.col(heatmap_z, ties.method = "first")
  peak_group <- assign_peak_group(peak_col, ncol(heatmap_z))

  gene_info <- data.frame(
    gene = rownames(heatmap_z),
    module = as.integer(peak_group),
    peak_col = peak_col,
    stringsAsFactors = FALSE
  )

  gene_info <- gene_info[order(gene_info$module, gene_info$peak_col), , drop = FALSE]
  heatmap_z <- heatmap_z[gene_info$gene, , drop = FALSE]
  gene_info$plot_y <- seq_len(nrow(gene_info))

  heatmap_df <- as.data.frame(as.table(heatmap_z), stringsAsFactors = FALSE)
  colnames(heatmap_df) <- c("gene", "bin", "z_score")
  heatmap_df$x <- as.integer(sub("Bin_", "", heatmap_df$bin))
  heatmap_df$y <- match(heatmap_df$gene, gene_info$gene)

  module_counts <- as.integer(table(factor(gene_info$module, levels = seq_len(5))))
  module_breaks <- cumsum(module_counts)
  module_breaks <- module_breaks[module_breaks < nrow(gene_info)] + 0.5

  label_genes <- config$key_gene_labels[config$key_gene_labels %in% gene_info$gene]
  label_genes <- head(label_genes, 22)
  gene_label_df <- gene_info[match(label_genes, gene_info$gene), , drop = FALSE]
  gene_label_df$display_gene <- gene_label_df$gene
  gene_label_df <- spread_label_y(
    gene_label_df,
    min_gap = 3.2,
    lower = 2,
    upper = max(2, nrow(gene_info) - 1)
  )

  list(
    heatmap_df = heatmap_df,
    gene_info = gene_info,
    gene_label_df = gene_label_df,
    module_breaks = module_breaks,
    center_x = side_bins + 0.5,
    n_bins = ncol(heatmap_z),
    n_genes = nrow(heatmap_z),
    left_cells = length(left_cells),
    right_cells = length(right_cells),
    branch_cells = length(branch_cells)
  )
}


#7. ===========Plot polished branch heatmaps
plot_branch_heatmap <- function(heatmap_obj, config, show_legend = TRUE) {
  label_y <- 0.55
  arrow_y <- 2.15
  heatmap_height <- heatmap_obj$n_genes
  key_gene_x <- heatmap_obj$n_bins + 4.8
  key_gene_segment_x <- heatmap_obj$n_bins + 0.5
  legend_position <- if (show_legend) "right" else "none"

  ggplot2::ggplot(heatmap_obj$heatmap_df, ggplot2::aes(x = x, y = y, fill = z_score)) +
    ggplot2::geom_tile(width = 1, height = 1) +
    ggplot2::geom_vline(
      xintercept = heatmap_obj$center_x,
      color = "white",
      linewidth = 0.65
    ) +
    ggplot2::geom_hline(
      yintercept = heatmap_obj$module_breaks,
      color = "white",
      linewidth = 0.35
    ) +
    ggplot2::annotate(
      "segment",
      x = heatmap_obj$center_x - 1.5,
      xend = 5,
      y = -arrow_y,
      yend = -arrow_y,
      arrow = grid::arrow(length = grid::unit(0.10, "inches"), type = "closed"),
      linewidth = 0.45,
      color = config$left_color
    ) +
    ggplot2::annotate(
      "segment",
      x = heatmap_obj$center_x + 1.5,
      xend = heatmap_obj$n_bins - 4,
      y = -arrow_y,
      yend = -arrow_y,
      arrow = grid::arrow(length = grid::unit(0.10, "inches"), type = "closed"),
      linewidth = 0.45,
      color = config$right_color
    ) +
    ggplot2::annotate(
      "text",
      x = heatmap_obj$center_x / 2,
      y = -label_y,
      label = config$left_title,
      size = 2.75,
      color = "black"
    ) +
    ggplot2::annotate(
      "text",
      x = heatmap_obj$center_x,
      y = -label_y,
      label = config$branch_label,
      size = 2.75,
      fontface = "bold",
      color = "black"
    ) +
    ggplot2::annotate(
      "text",
      x = heatmap_obj$center_x + side_bins / 2,
      y = -label_y,
      label = config$right_title,
      size = 2.75,
      color = "black"
    ) +
    ggplot2::geom_segment(
      data = heatmap_obj$gene_label_df,
      ggplot2::aes(
        x = key_gene_segment_x,
        xend = key_gene_x - 0.7,
        y = plot_y,
        yend = label_y
      ),
      inherit.aes = FALSE,
      color = "grey35",
      linewidth = 0.22
    ) +
    ggplot2::geom_text(
      data = heatmap_obj$gene_label_df,
      ggplot2::aes(x = key_gene_x, y = label_y, label = display_gene),
      inherit.aes = FALSE,
      hjust = 0,
      size = 2.25,
      color = "black",
      lineheight = 0.92
    ) +
    ggplot2::annotate(
      "text",
      x = key_gene_x,
      y = -1.7,
      label = "Key genes",
      hjust = 0,
      size = 2.85,
      color = "black"
    ) +
    ggplot2::scale_fill_gradientn(
      colors = heatmap_colors,
      values = heatmap_values,
      limits = c(-zscore_cap, zscore_cap),
      breaks = c(-2, -1, 0, 1, 2),
      oob = scales::squish,
      name = "z-score"
    ) +
    ggplot2::scale_x_continuous(expand = c(0, 0)) +
    ggplot2::scale_y_reverse(expand = c(0, 0)) +
    ggplot2::coord_cartesian(
      xlim = c(0.5, heatmap_obj$n_bins + 22.5),
      ylim = c(heatmap_height + 0.5, -3.2),
      clip = "off"
    ) +
    ggplot2::labs(title = config$title, x = NULL, y = NULL) +
    ggplot2::theme_classic(base_size = 10) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(size = 12, hjust = 0.5, face = "bold"),
      axis.text = ggplot2::element_blank(),
      axis.ticks = ggplot2::element_blank(),
      axis.line = ggplot2::element_blank(),
      legend.title = ggplot2::element_text(size = 8),
      legend.text = ggplot2::element_text(size = 7),
      legend.key.height = grid::unit(0.36, "cm"),
      legend.key.width = grid::unit(0.20, "cm"),
      legend.position = legend_position,
      plot.margin = grid::unit(c(0.12, 0.18, 0.10, 0.16), "cm")
    )
}

medial_heatmap <- build_branch_heatmap(medial_result$cds, branch_configs$Medial)
lateral_heatmap <- build_branch_heatmap(lateral_result$cds, branch_configs$Lateral)

p_medial <- plot_branch_heatmap(medial_heatmap, branch_configs$Medial, show_legend = TRUE)
p_lateral <- plot_branch_heatmap(lateral_heatmap, branch_configs$Lateral, show_legend = TRUE)


#8. ===========Save split labeled Panel C heatmaps
save_plot_pair <- function(plot_obj, file_prefix, width, height) {
  ggplot2::ggsave(
    filename = paste0(file_prefix, ".pdf"),
    plot = plot_obj,
    width = width,
    height = height,
    units = "in",
    device = grDevices::cairo_pdf,
    bg = "white"
  )

  ggplot2::ggsave(
    filename = paste0(file_prefix, ".png"),
    plot = plot_obj,
    width = width,
    height = height,
    units = "in",
    dpi = 1200,
    bg = "white",
    limitsize = FALSE
  )
}

medial_prefix <- file.path(
  output_dir,
  "Embryonic_E9_E16_panel_c_medial_branch_heatmap_labeled"
)
lateral_prefix <- file.path(
  output_dir,
  "Embryonic_E9_E16_panel_c_lateral_branch_heatmap_labeled"
)

save_plot_pair(p_medial, medial_prefix, width = 7.4, height = 5.25)
save_plot_pair(p_lateral, lateral_prefix, width = 7.4, height = 5.25)

cat(
  "Split labeled Panel C branch heatmaps saved. medial_genes=", medial_heatmap$n_genes,
  ", lateral_genes=", lateral_heatmap$n_genes,
  ", medial_labels=", nrow(medial_heatmap$gene_label_df),
  ", lateral_labels=", nrow(lateral_heatmap$gene_label_df),
  ", medial_direction=", branch_configs$Medial$left_title, "<-", branch_configs$Medial$branch_label, "->", branch_configs$Medial$right_title,
  ", lateral_direction=", branch_configs$Lateral$left_title, "<-", branch_configs$Lateral$branch_label, "->", branch_configs$Lateral$right_title,
  "\n",
  sep = ""
)
