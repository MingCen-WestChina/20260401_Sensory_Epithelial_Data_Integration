#1. ===========Load shared data===========

common_path <- file.path("analysis", "04_hair_cell_trajectory", "common.R")
if (!file.exists(common_path)) common_path <- "common.R"
source(common_path)

bundle <- build_hc_monocle2_bundle(force_rebuild = FALSE)
hc_cds <- bundle$hc_cds
plot_meta <- bundle$plot_meta
trajectory_data <- bundle$trajectory_data
ordering_genes <- bundle$ordering_genes


#2. ===========Trajectory plotting helpers===========

theme_trajectory <- function(base_size = 12, square_panel = TRUE) {
  plot_theme <- theme_classic(base_size = base_size) +
    theme(
      plot.title = element_text(size = base_size + 1, hjust = 0.5, face = "plain"),
      axis.title = element_text(size = base_size, color = "black"),
      axis.text = element_text(size = base_size - 1, color = "black"),
      legend.title = element_text(size = base_size - 1, color = "black"),
      legend.text = element_text(size = base_size - 2, color = "black"),
      legend.key.height = grid::unit(0.34, "cm"),
      legend.key.width = grid::unit(0.34, "cm"),
      axis.line = element_line(color = "black", linewidth = 0.45),
      axis.ticks = element_line(color = "black", linewidth = 0.4),
      plot.margin = margin(5.5, 5.5, 5.5, 5.5)
    )

  if (square_panel) {
    plot_theme <- plot_theme + theme(aspect.ratio = 1)
  }

  plot_theme
}

get_lineage_cells <- function(lineage_name) {
  cell_df <- trajectory_data$cell_df[
    trajectory_data$cell_df$hc_lineage == lineage_name,
    ,
    drop = FALSE
  ]
  cell_df$stage <- factor(as.character(cell_df$stage), levels = target_stages)
  cell_df$stage_celltype_plot <- factor(
    as.character(cell_df$stage_celltype_plot),
    levels = stage_celltype_levels
  )
  cell_df
}

add_trajectory_backbone <- function(plot_obj) {
  plot_obj +
    geom_segment(
      data = trajectory_data$edge_df,
      aes(x = x, y = y, xend = xend, yend = yend),
      inherit.aes = FALSE,
      color = "grey55",
      linewidth = 0.35
    )
}

plot_lineage_stage <- function(lineage_name, use_recolor = FALSE) {
  cell_df <- get_lineage_cells(lineage_name)
  palette_use <- if (use_recolor) stage_colors_recolor else stage_colors
  stage_breaks <- target_stages[target_stages %in% as.character(cell_df$stage)]

  p <- ggplot(cell_df, aes(x = Component_1, y = Component_2))
  p <- add_trajectory_backbone(p)

  p +
    geom_point(aes(color = stage), size = 1.05, alpha = 0.9) +
    scale_color_manual(
      values = palette_use,
      breaks = stage_breaks,
      drop = TRUE,
      na.value = "grey80"
    ) +
    guides(color = guide_legend(override.aes = list(size = 3, alpha = 1), ncol = 1)) +
    theme_trajectory(12) +
    labs(
      title = paste0(lineage_name, " on all-HC Monocle2 trajectory | stage"),
      x = "Component 1",
      y = "Component 2",
      color = "Stage"
    )
}

plot_lineage_stage_celltype <- function(lineage_name, use_recolor = FALSE) {
  cell_df <- get_lineage_cells(lineage_name)
  palette_use <- if (use_recolor) stage_celltype_colors_recolor else stage_celltype_colors
  celltype_breaks <- stage_celltype_levels[
    stage_celltype_levels %in% as.character(cell_df$stage_celltype_plot)
  ]

  p <- ggplot(cell_df, aes(x = Component_1, y = Component_2))
  p <- add_trajectory_backbone(p)

  p +
    geom_point(aes(color = stage_celltype_plot), size = 1.05, alpha = 0.9) +
    scale_color_manual(
      values = palette_use,
      breaks = celltype_breaks,
      labels = gsub("_", " ", celltype_breaks),
      drop = TRUE,
      na.value = "grey80"
    ) +
    guides(color = guide_legend(override.aes = list(size = 3, alpha = 1), ncol = 1)) +
    theme_trajectory(12) +
    labs(
      title = paste0(lineage_name, " on all-HC Monocle2 trajectory | stage and cell type"),
      x = "Component 1",
      y = "Component 2",
      color = "Stage_celltype"
    )
}

plot_lineage_pseudotime <- function(lineage_name, use_recolor = FALSE) {
  cell_df <- get_lineage_cells(lineage_name)
  color_values <- if (use_recolor) pseudotime_colors_recolor else c("#F2E85E", "#BDBDBD", "#3F4C9A")

  p <- ggplot(cell_df, aes(x = Component_1, y = Component_2))
  p <- add_trajectory_backbone(p)

  p +
    geom_point(aes(color = Pseudotime), size = 1.05, alpha = 0.9) +
    scale_color_gradientn(colors = color_values, name = "Pseudotime") +
    theme_trajectory(12) +
    labs(
      title = paste0(lineage_name, " on all-HC Monocle2 trajectory | pseudotime"),
      x = "Component 1",
      y = "Component 2"
    )
}

plot_lineage_distribution <- function(lineage_name) {
  meta_use <- plot_meta[plot_meta$hc_lineage == lineage_name, , drop = FALSE]
  meta_use$stage <- factor(as.character(meta_use$stage), levels = target_stages)

  ggplot(meta_use, aes(x = stage, y = Pseudotime, fill = stage)) +
    geom_violin(width = 0.85, color = "grey35", linewidth = 0.25, trim = TRUE) +
    geom_boxplot(width = 0.18, outlier.shape = NA, color = "black", linewidth = 0.25, fill = "white") +
    geom_jitter(aes(color = stage), width = 0.12, size = 0.35, alpha = 0.25, show.legend = FALSE) +
    scale_fill_manual(values = stage_colors, drop = FALSE) +
    scale_color_manual(values = stage_colors, drop = FALSE) +
    theme_classic(base_size = 12) +
    labs(
      title = paste0(lineage_name, " pseudotime distribution"),
      x = "Stage",
      y = "Monocle2 pseudotime",
      fill = "Stage"
    ) +
    theme(
      plot.title = element_text(size = 13, hjust = 0.5),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "none"
    )
}


#3. ===========Save IHC and OHC trajectory panels===========

trajectory_jobs <- list(
  list(lineage = "IHC", out_dir = IHC_OUTPUT_DIR, stem = "IHC_trajectory_stage", plot = plot_lineage_stage("IHC"), width = 5.39),
  list(lineage = "IHC", out_dir = IHC_OUTPUT_DIR, stem = "IHC_trajectory_stage_celltype", plot = plot_lineage_stage_celltype("IHC"), width = 5.69),
  list(lineage = "IHC", out_dir = IHC_OUTPUT_DIR, stem = "IHC_trajectory_pseudotime", plot = plot_lineage_pseudotime("IHC"), width = 5.39),
  list(lineage = "OHC", out_dir = OHC_OUTPUT_DIR, stem = "OHC_trajectory_stage", plot = plot_lineage_stage("OHC"), width = 5.39),
  list(lineage = "OHC", out_dir = OHC_OUTPUT_DIR, stem = "OHC_trajectory_stage_celltype", plot = plot_lineage_stage_celltype("OHC"), width = 5.69),
  list(lineage = "OHC", out_dir = OHC_OUTPUT_DIR, stem = "OHC_trajectory_pseudotime", plot = plot_lineage_pseudotime("OHC"), width = 5.39),
  list(lineage = "IHC", out_dir = IHC_OUTPUT_DIR, stem = "IHC_trajectory_stage_recolor_P1", plot = plot_lineage_stage("IHC", TRUE), width = 5.39),
  list(lineage = "IHC", out_dir = IHC_OUTPUT_DIR, stem = "IHC_trajectory_stage_celltype_recolor_P1", plot = plot_lineage_stage_celltype("IHC", TRUE), width = 5.69),
  list(lineage = "IHC", out_dir = IHC_OUTPUT_DIR, stem = "IHC_trajectory_pseudotime_recolor_P1", plot = plot_lineage_pseudotime("IHC", TRUE), width = 5.39),
  list(lineage = "OHC", out_dir = OHC_OUTPUT_DIR, stem = "OHC_trajectory_stage_recolor_P1", plot = plot_lineage_stage("OHC", TRUE), width = 5.39),
  list(lineage = "OHC", out_dir = OHC_OUTPUT_DIR, stem = "OHC_trajectory_stage_celltype_recolor_P1", plot = plot_lineage_stage_celltype("OHC", TRUE), width = 5.69),
  list(lineage = "OHC", out_dir = OHC_OUTPUT_DIR, stem = "OHC_trajectory_pseudotime_recolor_P1", plot = plot_lineage_pseudotime("OHC", TRUE), width = 5.39)
)

for (job in trajectory_jobs) {
  save_pdf_plot(
    job$plot,
    file.path(job$out_dir, paste0(job$stem, ".pdf")),
    width = job$width,
    height = 5.0
  )
}

save_pdf_plot(
  plot_lineage_stage_celltype("IHC"),
  file.path(HC_OUTPUT_DIR, "IHC_trajectory_stage_celltype.pdf"),
  width = 5.69,
  height = 5.0
)

save_pdf_plot(
  plot_lineage_stage_celltype("OHC"),
  file.path(HC_OUTPUT_DIR, "OHC_trajectory_stage_celltype.pdf"),
  width = 5.69,
  height = 5.0
)

save_pdf_plot(
  plot_lineage_distribution("IHC"),
  file.path(IHC_OUTPUT_DIR, "IHC_pseudotime_distribution_stage.pdf"),
  width = 5.2,
  height = 4.2
)

save_pdf_plot(
  plot_lineage_distribution("OHC"),
  file.path(OHC_OUTPUT_DIR, "OHC_pseudotime_distribution_stage.pdf"),
  width = 5.2,
  height = 4.2
)


#4. ===========Build lineage gene modules for heatmaps===========

smooth_vector <- function(x, window_size = 7) {
  if (length(x) < window_size) return(as.numeric(x))
  y <- as.numeric(stats::filter(x, rep(1 / window_size, window_size), sides = 2))
  y[is.na(y)] <- x[is.na(y)]
  y
}

fill_na_vector <- function(x) {
  if (all(is.na(x))) return(rep(0, length(x)))
  x[is.na(x)] <- mean(x, na.rm = TRUE)
  x
}

clip_values <- function(x, lower, upper) {
  pmax(pmin(x, upper), lower)
}

heatmap_seed_genes <- list(
  "IHC" = c(
    "Sox2", "Atoh1", "Ccer2", "Fgf8", "Dlk2", "Otof", "Tmc1",
    "Calb2", "Pvalb", "Cabp2", "Ctbp2", "Myo6", "Myo7a", "Gata3",
    "Barhl1", "Lhx3", "S100a1", "Pcdh15", "Cdh23", "Cib2", "Espn",
    "Grhl2", "Slc17a8", "Nefl", "Shtn1"
  ),
  "OHC" = c(
    "Sox2", "Atoh1", "Ccer2", "Hes6", "Lhfpl5", "Pcp4", "Insm1",
    "Bcl11b", "Cib2", "Calb1", "Ocm", "Ikzf2", "Kcnq4", "Strip2",
    "Slc26a5", "Atp2b2", "Tmc1", "Myo6", "Myo7a", "Pcdh15", "Cdh23",
    "Gfi1", "Pou4f3", "Barhl1", "S100a1", "Lhx3"
  )
)

build_lineage_modules <- function(lineage_name, n_bins = 100, n_genes = 350, n_modules = 4) {
  cell_df <- get_lineage_cells(lineage_name)
  cell_df <- cell_df[order(cell_df$Pseudotime, cell_df$cell_id), , drop = FALSE]
  cells_use <- cell_df$cell_id

  expr_mat <- exprs(hc_cds)
  gene_universe <- rownames(expr_mat)
  seed_genes <- match_genes(heatmap_seed_genes[[lineage_name]], gene_universe)
  candidate_genes <- unique(c(seed_genes, intersect(ordering_genes, gene_universe)))
  candidate_genes <- candidate_genes[Matrix::rowSums(expr_mat[candidate_genes, cells_use, drop = FALSE] > 0) >= min_cells_gene]

  bin_id <- cut(
    seq_along(cells_use),
    breaks = n_bins,
    include.lowest = TRUE,
    labels = FALSE
  )

  bin_matrix <- sapply(seq_len(n_bins), function(bin_i) {
    bin_cells <- cells_use[bin_id == bin_i]
    Matrix::rowMeans(expr_mat[candidate_genes, bin_cells, drop = FALSE])
  })

  rownames(bin_matrix) <- candidate_genes
  gene_var <- apply(bin_matrix, 1, stats::var)
  selected_genes <- unique(c(seed_genes, names(sort(gene_var, decreasing = TRUE))[seq_len(min(n_genes, length(gene_var)))]))
  selected_genes <- selected_genes[selected_genes %in% rownames(bin_matrix)]

  z_matrix <- t(apply(bin_matrix[selected_genes, , drop = FALSE], 1, function(x) {
    z <- scale_gene_expression(fill_na_vector(x))
    z <- smooth_vector(z, 7)
    clip_values(z, -3, 3)
  }))

  set.seed(0)
  module_id <- stats::kmeans(z_matrix, centers = n_modules, nstart = 50)$cluster
  module_peak <- tapply(seq_len(nrow(z_matrix)), module_id, function(idx) {
    mean(max.col(z_matrix[idx, , drop = FALSE], ties.method = "first"))
  })
  module_order <- names(sort(module_peak))
  module_id <- factor(module_id, levels = module_order, labels = seq_along(module_order))

  gene_order <- unlist(lapply(levels(module_id), function(module_name) {
    genes <- rownames(z_matrix)[module_id == module_name]
    genes[order(max.col(z_matrix[genes, , drop = FALSE], ties.method = "first"))]
  }))

  z_matrix <- z_matrix[gene_order, , drop = FALSE]
  colnames(z_matrix) <- seq_len(ncol(z_matrix))
  module_id <- module_id[gene_order]

  heatmap_df <- as.data.frame(as.table(z_matrix), stringsAsFactors = FALSE)
  colnames(heatmap_df) <- c("gene", "bin", "z")
  heatmap_df$bin <- as.integer(sub("^V", "", heatmap_df$bin))
  heatmap_df$gene <- factor(heatmap_df$gene, levels = rev(gene_order))
  heatmap_df$module <- module_id[as.character(heatmap_df$gene)]

  list(
    lineage = lineage_name,
    genes = gene_order,
    modules = module_id,
    heatmap_df = heatmap_df,
    bin_matrix = bin_matrix,
    z_matrix = z_matrix
  )
}

if (file.exists(hc_modules_rds)) {
  module_bundle <- readRDS(hc_modules_rds)
} else {
  module_bundle <- list(
    IHC = build_lineage_modules("IHC"),
    OHC = build_lineage_modules("OHC")
  )
  saveRDS(module_bundle, hc_modules_rds)
}

plot_heatmap_only <- function(module_obj, title_text) {
  ggplot(module_obj$heatmap_df, aes(x = bin, y = gene, fill = z)) +
    geom_raster() +
    facet_grid(module ~ ., scales = "free_y", space = "free_y") +
    scale_fill_gradientn(
      colors = c("#4055A8", "#8FC7BD", "#F1E85B", "#F08022"),
      limits = c(-3, 3),
      breaks = c(-3, -2, -1, 0, 1, 2, 3),
      name = "Relative\nexpression"
    ) +
    theme_classic(base_size = 11) +
    labs(title = title_text, x = "Pseudotime", y = NULL) +
    theme(
      plot.title = element_text(size = 13, hjust = 0),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      strip.background = element_blank(),
      strip.text.y = element_text(size = 10, color = "black"),
      legend.title = element_text(size = 9),
      legend.text = element_text(size = 8),
      panel.spacing.y = grid::unit(0.05, "lines"),
      plot.margin = margin(8, 12, 8, 8)
    )
}

save_pdf_plot(
  plot_heatmap_only(module_bundle$IHC, "IHC pseudotime gene dynamics"),
  file.path(IHC_OUTPUT_DIR, "IHC_pseudotime_heatmap_panel_c.pdf"),
  width = 6.6,
  height = 5.0
)

save_pdf_plot(
  plot_heatmap_only(module_bundle$OHC, "OHC pseudotime gene dynamics"),
  file.path(OHC_OUTPUT_DIR, "OHC_pseudotime_heatmap_panel_c.pdf"),
  width = 6.6,
  height = 5.0
)


#5. ===========Reverse-engineered gene projection panels===========

plot_gene_projection <- function(lineage_name, genes, output_file) {
  cell_df <- get_lineage_cells(lineage_name)
  expr_mat <- exprs(hc_cds)
  genes_use <- match_genes(genes, rownames(expr_mat))

  projection_df <- do.call(
    rbind,
    lapply(genes_use, function(gene_id) {
      values <- as.numeric(expr_mat[gene_id, cell_df$cell_id])
      value_scaled <- if (max(values) == min(values)) {
        rep(0, length(values))
      } else {
        (values - min(values)) / (max(values) - min(values))
      }

      data.frame(
        cell_df[, c("cell_id", "Component_1", "Component_2")],
        gene = gene_id,
        expression = value_scaled,
        stringsAsFactors = FALSE
      )
    })
  )

  ggplot(projection_df, aes(x = Component_1, y = Component_2)) +
    geom_segment(
      data = trajectory_data$edge_df,
      aes(x = x, y = y, xend = xend, yend = yend),
      inherit.aes = FALSE,
      color = "grey72",
      linewidth = 0.25
    ) +
    geom_point(aes(color = expression), size = 0.7, alpha = 0.9, stroke = 0) +
    facet_wrap(~gene, ncol = 4) +
    scale_color_gradientn(
      colors = c("#F4E64E", "#BDBDBD", "#364A9B"),
      limits = c(0, 1),
      name = "Relative\nexpression"
    ) +
    theme_trajectory(10, square_panel = FALSE) +
    labs(
      title = paste0(lineage_name, " marker projection on all-HC Monocle2 trajectory"),
      x = "Component 1",
      y = "Component 2"
    ) +
    theme(
      strip.background = element_blank(),
      strip.text = element_text(size = 10, face = "italic"),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      legend.position = "bottom"
    ) -> plot_obj

  save_pdf_plot(plot_obj, output_file, width = 7.8, height = 5.6)
}

plot_gene_projection(
  "IHC",
  c("Sox2", "Atoh1", "Ccer2", "Fgf8", "Pvalb", "Calb2", "Otof", "Barhl1", "Lhx3"),
  file.path(IHC_OUTPUT_DIR, "IHC_gene_projection_panel_d.pdf")
)

plot_gene_projection(
  "OHC",
  c("Hes6", "Pcp4", "Ccer2", "Lhfpl5", "Myo6", "Myo7a", "Cib2", "Calb1", "Ocm", "Kcnq4", "Slc26a5", "Atp2b2"),
  file.path(OHC_OUTPUT_DIR, "OHC_gene_projection_panel_d.pdf")
)


#6. ===========Reverse-engineered gene-set AUC panels===========

gene_sets <- list(
  IHC = list(
    Early_IHC_program = c("Sox2", "Atoh1", "Ccer2", "Selenom", "S100a1", "Fgf8"),
    Differentiating_IHC_program = c("Dlk2", "Nefl", "Shtn1", "Pvalb", "Cabp2", "Ctbp2"),
    Mature_IHC_program = c("Otof", "Calb2", "Tmc1", "Slc17a8", "Myo6", "Myo7a"),
    TF_reference_program = c("Sox2", "Atoh1", "Gata3", "Barhl1", "Lhx3", "Brf2", "Usf2", "Bdp1", "Arnt")
  ),
  OHC = list(
    Early_OHC_program = c("Hes6", "Insm1", "Bcl11b", "Lhfpl5", "Pcp4", "Cib2"),
    Differentiating_OHC_program = c("Calb1", "Ocm", "Kcnq4", "Ikzf2", "Strip2", "Calca"),
    Mature_OHC_program = c("Slc26a5", "Atp2b2", "Tmc1", "Myo7a", "Six2", "Lbh", "Strc"),
    TF_reference_program = c("Sox2", "Atoh1", "Gata3", "Barhl1", "Lhx3", "Brf2", "Usf2", "Bdp1", "Arnt")
  )
)

score_gene_sets <- function(lineage_name) {
  cell_df <- get_lineage_cells(lineage_name)
  expr_mat <- exprs(hc_cds)
  set_names <- names(gene_sets[[lineage_name]])

  score_df <- do.call(
    rbind,
    lapply(set_names, function(set_name) {
      genes_use <- match_genes(gene_sets[[lineage_name]][[set_name]], rownames(expr_mat))
      if (length(genes_use) == 0) {
        score <- rep(0, nrow(cell_df))
      } else {
        score <- Matrix::colMeans(expr_mat[genes_use, cell_df$cell_id, drop = FALSE])
        score <- as.numeric(score)
        score <- if (max(score) == min(score)) rep(0, length(score)) else (score - min(score)) / (max(score) - min(score))
      }

      data.frame(
        cell_df[, c("cell_id", "Component_1", "Component_2")],
        gene_set = set_name,
        auc_score = score,
        stringsAsFactors = FALSE
      )
    })
  )

  score_df
}

plot_auc_score <- function(lineage_name, output_file) {
  score_df <- score_gene_sets(lineage_name)

  plot_obj <- ggplot(score_df, aes(x = Component_1, y = Component_2)) +
    geom_segment(
      data = trajectory_data$edge_df,
      aes(x = x, y = y, xend = xend, yend = yend),
      inherit.aes = FALSE,
      color = "grey76",
      linewidth = 0.25
    ) +
    geom_point(aes(color = auc_score), size = 0.65, alpha = 0.9, stroke = 0) +
    facet_wrap(~gene_set, ncol = 2) +
    scale_color_gradientn(
      colors = c("#F4E64E", "#BDBDBD", "#364A9B"),
      limits = c(0, 1),
      name = "AUC"
    ) +
    theme_trajectory(10, square_panel = FALSE) +
    labs(
      title = paste0(lineage_name, " gene-set AUCell AUC on all-HC trajectory"),
      x = "Component 1",
      y = "Component 2"
    ) +
    theme(
      strip.background = element_blank(),
      strip.text = element_text(size = 9, face = "bold"),
      legend.position = "right"
    )

  save_pdf_plot(plot_obj, output_file, width = 7.0, height = 5.6)

  score_df
}

plot_auc_high_cells <- function(score_df, lineage_name, output_file) {
  score_df$high_auc <- ave(score_df$auc_score, score_df$gene_set, FUN = function(x) x >= stats::quantile(x, 0.75))
  score_df$high_auc <- factor(score_df$high_auc > 0, levels = c(FALSE, TRUE))

  plot_obj <- ggplot(score_df, aes(x = Component_1, y = Component_2)) +
    geom_segment(
      data = trajectory_data$edge_df,
      aes(x = x, y = y, xend = xend, yend = yend),
      inherit.aes = FALSE,
      color = "grey78",
      linewidth = 0.25
    ) +
    geom_point(aes(color = high_auc), size = 0.65, alpha = 0.9, stroke = 0) +
    facet_wrap(~gene_set, ncol = 2) +
    scale_color_manual(values = c("FALSE" = "#D9D9D9", "TRUE" = "#8F1D82"), name = "High AUC") +
    theme_trajectory(10, square_panel = FALSE) +
    labs(
      title = paste0(lineage_name, " gene-set AUCell high-AUC cells"),
      x = "Component 1",
      y = "Component 2"
    ) +
    theme(
      strip.background = element_blank(),
      strip.text = element_text(size = 9, face = "bold"),
      legend.position = "right"
    )

  save_pdf_plot(plot_obj, output_file, width = 7.0, height = 5.6)
}

ihc_auc <- plot_auc_score("IHC", file.path(IHC_OUTPUT_DIR, "IHC_gene_set_AUCell_AUC_score.pdf"))
plot_auc_high_cells(ihc_auc, "IHC", file.path(IHC_OUTPUT_DIR, "IHC_gene_set_AUCell_AUC_high_cells.pdf"))

ohc_auc <- plot_auc_score("OHC", file.path(OHC_OUTPUT_DIR, "OHC_gene_set_AUCell_AUC_score.pdf"))
plot_auc_high_cells(ohc_auc, "OHC", file.path(OHC_OUTPUT_DIR, "OHC_gene_set_AUCell_AUC_high_cells.pdf"))


#7. ===========Reverse-engineered lineage marker dotplots===========

lineage_marker_groups <- list(
  IHC = list(
    E14_IHC = c("Ccer2", "Selenom", "S100a1", "Atoh1", "Fgf8", "Pvalb"),
    E16_IHC = c("Ccer2", "Fgf8", "Tbx2", "Kcnn3", "Otof", "Dlk2", "Nefl", "Calb2"),
    P1_IHC = c("Tmc1", "Myo6", "Myo7a", "Cabp2", "Azin1", "Otof"),
    P7_IHC = c("Sox2", "Atoh1", "Gata3", "Barhl1", "Lhx3"),
    P14_IHC = c("Brf2", "Usf2", "Bdp1", "Arnt"),
    P28_IHC = c("Otof", "Calb2", "Myo7a"),
    Shared_HC = c("Sox2", "Atoh1", "Gata3"),
    TF_reference = c("Barhl1", "Lhx3", "Brf2", "Usf2", "Bdp1", "Arnt")
  ),
  OHC = list(
    E14_OHC = c("Hes6", "Pcp4", "Rprm", "Ccer2", "Selenom", "Atoh1"),
    E16_iOHC = c("Hes6", "Lhfpl5", "Myo6", "Myo7a", "Cib2"),
    E16_OHC = c("Calca", "Veph1", "Strip2", "Calb1", "Six2"),
    P1_OHC = c("Pcp4", "Tmc1", "Atp2b2", "Lhfpl5", "Pcp4"),
    P7_OHC = c("Cib1", "Sox2", "Gata3"),
    P14_OHC = c("Barhl1", "Lhx3"),
    P28_OHC = c("Brf2", "Usf2", "Bdp1", "Arnt"),
    Shared_HC = c("Sox2", "Atoh1", "Gata3"),
    TF_reference = c("Barhl1", "Lhx3", "Brf2", "Usf2", "Bdp1", "Arnt")
  )
)

plot_lineage_marker_dotplot <- function(lineage_name, output_file, part_file) {
  marker_groups <- lineage_marker_groups[[lineage_name]]
  marker_table <- data.frame(
    marker_group = rep(names(marker_groups), lengths(marker_groups)),
    gene_label = unlist(marker_groups, use.names = FALSE),
    stringsAsFactors = FALSE
  )
  marker_table <- marker_table[!duplicated(paste(marker_table$marker_group, marker_table$gene_label)), , drop = FALSE]
  marker_table$gene <- match_genes(marker_table$gene_label, rownames(exprs(hc_cds)))[
    match(toupper(marker_table$gene_label), toupper(match_genes(marker_table$gene_label, rownames(exprs(hc_cds)))))
  ]
  marker_table$gene <- vapply(marker_table$gene_label, function(g) {
    matched <- match_genes(g, rownames(exprs(hc_cds)))
    if (length(matched) == 0) NA_character_ else matched[1]
  }, character(1))
  marker_table <- marker_table[!is.na(marker_table$gene), , drop = FALSE]
  marker_table$marker_id <- paste(marker_table$marker_group, marker_table$gene_label, seq_len(nrow(marker_table)), sep = "__")

  cell_df <- get_lineage_cells(lineage_name)
  group_levels <- stage_celltype_levels[stage_celltype_levels %in% as.character(cell_df$stage_celltype_plot)]

  dot_df <- do.call(
    rbind,
    lapply(seq_len(nrow(marker_table)), function(i) {
      do.call(
        rbind,
        lapply(group_levels, function(group_id) {
          cells_use <- cell_df$cell_id[cell_df$stage_celltype_plot == group_id]
          values <- as.numeric(exprs(hc_cds)[marker_table$gene[i], cells_use])

          data.frame(
            marker_id = marker_table$marker_id[i],
            gene_label = marker_table$gene_label[i],
            marker_group = marker_table$marker_group[i],
            stage_celltype_plot = group_id,
            avg_exp = mean(values),
            pct_exp = mean(values > lower_detection_limit) * 100,
            stringsAsFactors = FALSE
          )
        })
      )
    })
  )

  dot_df$avg_exp_scaled <- ave(dot_df$avg_exp, dot_df$marker_id, FUN = scale_gene_expression)
  dot_df$avg_exp_scaled <- pmax(pmin(dot_df$avg_exp_scaled, 1), -1)
  dot_df$marker_id <- factor(dot_df$marker_id, levels = marker_table$marker_id)
  dot_df$stage_celltype_plot <- factor(dot_df$stage_celltype_plot, levels = group_levels)
  dot_df$marker_group <- factor(dot_df$marker_group, levels = names(marker_groups))

  make_plot <- function(plot_df, title_text) {
    ggplot(plot_df, aes(x = marker_id, y = stage_celltype_plot)) +
      geom_point(aes(size = pct_exp, color = avg_exp_scaled), alpha = 0.95, stroke = 0) +
      facet_grid(. ~ marker_group, scales = "free_x", space = "free_x") +
      scale_x_discrete(labels = setNames(marker_table$gene_label, marker_table$marker_id)) +
      scale_size_area(max_size = 4.2, limits = c(0, 100), breaks = c(0, 25, 50, 75, 100), name = "% Exp.") +
      scale_color_gradientn(colors = c("#D9D9D9", "#D894C4", "#8F1D82"), limits = c(-1, 1), name = "Avg. Exp.") +
      theme_classic(base_size = 9) +
      labs(title = title_text, x = NULL, y = NULL) +
      theme(
        plot.title = element_text(size = 12, hjust = 0.5),
        strip.background = element_blank(),
        strip.text.x = element_text(angle = 45, hjust = 0, size = 6.5, color = "black"),
        axis.text.x = element_text(angle = 60, hjust = 1, vjust = 1, size = 6.5, color = "black"),
        axis.text.y = element_text(size = 8.5, color = "black"),
        panel.spacing.x = grid::unit(0.12, "lines")
      )
  }

  save_pdf_plot(
    make_plot(dot_df, paste0(lineage_name, " marker dotplot")),
    output_file,
    width = 8.4,
    height = 4.9
  )

  first_groups <- names(marker_groups)[seq_len(min(4, length(marker_groups)))]
  part_df <- dot_df[as.character(dot_df$marker_group) %in% first_groups, , drop = FALSE]
  part_df$marker_group <- factor(as.character(part_df$marker_group), levels = first_groups)

  save_pdf_plot(
    make_plot(part_df, paste0(lineage_name, " marker dotplot | part 1")),
    part_file,
    width = 7.2,
    height = 4.8
  )
}

plot_lineage_marker_dotplot(
  "IHC",
  file.path(IHC_OUTPUT_DIR, "IHC_marker_dotplot_curated_full.pdf"),
  file.path(IHC_OUTPUT_DIR, "IHC_marker_dotplot_curated_parts.pdf")
)

plot_lineage_marker_dotplot(
  "OHC",
  file.path(OHC_OUTPUT_DIR, "OHC_marker_dotplot_curated_full.pdf"),
  file.path(OHC_OUTPUT_DIR, "OHC_marker_dotplot_curated_parts.pdf")
)

message("Saved IHC and OHC lineage panels.")
