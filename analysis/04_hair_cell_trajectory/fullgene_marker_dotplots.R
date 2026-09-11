#1. ===========Load shared workflow===========

common_path <- file.path("analysis", "04_hair_cell_trajectory", "common.R")
if (!file.exists(common_path)) common_path <- "common.R"
source(common_path)

panel_b_dir <- file.path(HC_OUTPUT_DIR, "Panel_b_dotplot")
dir.create(panel_b_dir, recursive = TRUE, showWarnings = FALSE)


#2. ===========Define full-gene marker list===========

dotplot_group_levels <- c(
  "Early IHC",
  "Intermediate IHC",
  "Mature IHC",
  "Early OHC",
  "Intermediate OHC",
  "Mature OHC"
)

dotplot_strip_colors <- c(
  "Early IHC" = "#C9972B",
  "Intermediate IHC" = "#777777",
  "Mature IHC" = "#252525",
  "Early OHC" = "#E3B23C",
  "Intermediate OHC" = "#9A9A9A",
  "Mature OHC" = "#000000"
)

make_marker_item <- function(group, label, aliases = NULL) {
  if (is.null(aliases)) {
    aliases <- label
  }

  list(group = group, label = label, aliases = aliases)
}

marker_items <- list(
  make_marker_item("Early IHC", "Ccer2"),
  make_marker_item("Early IHC", "Atoh1"),
  make_marker_item("Early IHC", "Sox2"),
  make_marker_item("Early IHC", "Selenom"),
  make_marker_item("Early IHC", "S100a1"),
  make_marker_item("Early IHC", "Ccl21a"),
  make_marker_item("Early IHC", "Fgf8"),
  make_marker_item("Early IHC", "Brip1"),
  make_marker_item("Early IHC", "Msx1"),
  make_marker_item("Intermediate IHC", "Dlk2"),
  make_marker_item("Intermediate IHC", "Nefl"),
  make_marker_item("Intermediate IHC", "Shtn1"),
  make_marker_item("Intermediate IHC", "Pvalb"),
  make_marker_item("Intermediate IHC", "Cabp2"),
  make_marker_item("Intermediate IHC", "Ctbp2"),
  make_marker_item("Intermediate IHC", "Cacna1d"),
  make_marker_item("Intermediate IHC", "Atp2a3"),
  make_marker_item("Intermediate IHC", "Kcnma1/BK", c("Kcnma1", "BK")),
  make_marker_item("Mature IHC", "Tmc1"),
  make_marker_item("Mature IHC", "Slc17a8/Vglut3", c("Slc17a8", "Vglut3")),
  make_marker_item("Mature IHC", "Otof"),
  make_marker_item("Mature IHC", "Calb2"),
  make_marker_item("Mature IHC", "Tbx2"),
  make_marker_item("Mature IHC", "Rprm"),
  make_marker_item("Mature IHC", "Acbd7"),
  make_marker_item("Mature IHC", "Igfbp5"),
  make_marker_item("Mature IHC", "Pgm2l1"),
  make_marker_item("Early OHC", "Hes6"),
  make_marker_item("Early OHC", "Insm1"),
  make_marker_item("Early OHC", "Bcl11b"),
  make_marker_item("Early OHC", "Lhfpl5"),
  make_marker_item("Early OHC", "Pcp4"),
  make_marker_item("Early OHC", "Pou4f3"),
  make_marker_item("Early OHC", "Cdkn1c"),
  make_marker_item("Early OHC", "Cib2"),
  make_marker_item("Intermediate OHC", "Calb1"),
  make_marker_item("Intermediate OHC", "Ocm"),
  make_marker_item("Intermediate OHC", "Kcnq4"),
  make_marker_item("Intermediate OHC", "Ikzf2", c("Ikzf2", "Helios")),
  make_marker_item("Intermediate OHC", "Strip2"),
  make_marker_item("Intermediate OHC", "Calca"),
  make_marker_item("Intermediate OHC", "Veph1"),
  make_marker_item("Intermediate OHC", "Atp2b2"),
  make_marker_item("Intermediate OHC", "Lmo7"),
  make_marker_item("Intermediate OHC", "Sorbs2"),
  make_marker_item("Intermediate OHC", "Aqp11"),
  make_marker_item("Mature OHC", "Slc26a5/Prestin", c("Slc26a5", "Prestin")),
  make_marker_item("Mature OHC", "Strc"),
  make_marker_item("Mature OHC", "Myo7a"),
  make_marker_item("Mature OHC", "Six2"),
  make_marker_item("Mature OHC", "Lbh"),
  make_marker_item("Mature OHC", "Lpin2"),
  make_marker_item("Mature OHC", "Chst2"),
  make_marker_item("Mature OHC", "Pde6d"),
  make_marker_item("Mature OHC", "Ldhb"),
  make_marker_item("Mature OHC", "Mmd")
)

marker_labels <- vapply(marker_items, `[[`, character(1), "label")
marker_keep <- !duplicated(toupper(marker_labels))
marker_items <- marker_items[marker_keep]

marker_table <- data.frame(
  marker_group = vapply(marker_items, `[[`, character(1), "group"),
  gene_label = vapply(marker_items, `[[`, character(1), "label"),
  stringsAsFactors = FALSE
)

marker_table$marker_id <- paste(marker_table$marker_group, marker_table$gene_label, sep = "__")
marker_table$marker_group <- factor(marker_table$marker_group, levels = dotplot_group_levels)


#3. ===========Read integrated metadata===========

meta_df <- read_h5ad_obs(input_h5ad, c("stage", "scanvi_label", "original_celltype"))
meta_df$cell_id <- rownames(meta_df)
meta_df$stage <- as.character(meta_df$stage)
meta_df$scanvi_label <- as.character(meta_df$scanvi_label)
meta_df$original_celltype <- as.character(meta_df$original_celltype)


#4. ===========Extract marker expression from full-gene h5AD files===========

stage_expr_list <- list()
stage_meta_list <- list()
feature_hit_by_stage <- matrix(
  FALSE,
  nrow = nrow(marker_table),
  ncol = length(fullgene_h5ad_by_stage),
  dimnames = list(marker_table$marker_id, names(fullgene_h5ad_by_stage))
)

for (stage_name in names(fullgene_h5ad_by_stage)) {
  message("Processing full-gene marker expression: ", stage_name)

  stage_file <- fullgene_h5ad_by_stage[[stage_name]]
  stage_gene_ids <- read_h5ad_var_index(stage_file)
  stage_cell_ids <- paste0(stage_name, "__", as.character(rhdf5::h5read(stage_file, "obs/_index")))

  marker_feature <- vapply(
    seq_len(nrow(marker_table)),
    function(marker_i) match_marker_feature(marker_items[[marker_i]]$aliases, stage_gene_ids),
    character(1)
  )

  present_marker <- !is.na(marker_feature)
  feature_hit_by_stage[marker_table$marker_id[present_marker], stage_name] <- TRUE

  unique_features <- unique(marker_feature[present_marker])
  feature_index <- match(unique_features, stage_gene_ids)
  expr_stage <- read_h5ad_dense_x_rows(stage_file, feature_index)
  rownames(expr_stage) <- unique_features
  colnames(expr_stage) <- stage_cell_ids

  matched_cells <- intersect(stage_cell_ids, meta_df$cell_id)
  if (length(matched_cells) == 0) {
    stop("No matched cells between full-gene data and integrated metadata for stage: ", stage_name)
  }

  stage_meta <- meta_df[match(matched_cells, meta_df$cell_id), , drop = FALSE]
  rownames(stage_meta) <- stage_meta$cell_id
  stage_meta$hc_lineage <- get_hc_lineage(
    stage_meta$stage,
    stage_meta$scanvi_label,
    stage_meta$original_celltype
  )

  stage_keep <- stage_meta$stage %in% target_stages & stage_meta$hc_lineage %in% target_labels
  stage_meta <- stage_meta[stage_keep, , drop = FALSE]

  if (nrow(stage_meta) == 0) {
    next
  }

  stage_meta$dotplot_group <- get_dotplot_group(stage_meta$stage, stage_meta$hc_lineage)
  stage_meta <- stage_meta[stage_meta$dotplot_group %in% dotplot_group_levels, , drop = FALSE]
  stage_meta$dotplot_group <- factor(stage_meta$dotplot_group, levels = dotplot_group_levels)

  stage_marker_expr <- matrix(
    0,
    nrow = nrow(marker_table),
    ncol = nrow(stage_meta),
    dimnames = list(marker_table$marker_id, stage_meta$cell_id)
  )

  for (marker_i in seq_len(nrow(marker_table))) {
    feature_id <- marker_feature[marker_i]
    if (!is.na(feature_id)) {
      stage_marker_expr[marker_i, ] <- as.numeric(expr_stage[feature_id, stage_meta$cell_id])
    }
  }

  stage_expr_list[[stage_name]] <- stage_marker_expr
  stage_meta_list[[stage_name]] <- stage_meta
}

expr_dot <- do.call(cbind, unname(stage_expr_list))
dot_meta <- do.call(rbind, unname(stage_meta_list))
rownames(dot_meta) <- dot_meta$cell_id
dot_meta <- dot_meta[colnames(expr_dot), , drop = FALSE]

present_marker_ids <- rownames(feature_hit_by_stage)[rowSums(feature_hit_by_stage) > 0]
marker_table <- marker_table[marker_table$marker_id %in% present_marker_ids, , drop = FALSE]
expr_dot <- expr_dot[marker_table$marker_id, , drop = FALSE]
marker_table$gene_index <- seq_len(nrow(marker_table))


#5. ===========Calculate full-gene dotplot statistics===========

dot_df <- do.call(
  rbind,
  lapply(seq_len(nrow(marker_table)), function(marker_i) {
    marker_id <- marker_table$marker_id[marker_i]

    do.call(
      rbind,
      lapply(dotplot_group_levels, function(group_name) {
        cells_use <- rownames(dot_meta)[dot_meta$dotplot_group == group_name]
        values <- as.numeric(expr_dot[marker_id, cells_use])

        data.frame(
          marker_id = marker_id,
          gene_label = marker_table$gene_label[marker_i],
          marker_group = as.character(marker_table$marker_group[marker_i]),
          dotplot_group = group_name,
          avg_exp = mean(values),
          pct_exp = mean(values > lower_detection_limit) * 100,
          gene_index = marker_table$gene_index[marker_i],
          stringsAsFactors = FALSE
        )
      })
    )
  })
)

dot_df$avg_exp_scaled <- ave(dot_df$avg_exp, dot_df$marker_id, FUN = scale_gene_expression)
dot_df$avg_exp_scaled <- pmax(pmin(dot_df$avg_exp_scaled, 1), -1)
dot_df$dotplot_group <- factor(dot_df$dotplot_group, levels = dotplot_group_levels)
dot_df$marker_id <- factor(dot_df$marker_id, levels = marker_table$marker_id)


#6. ===========Plot horizontal full-gene dotplots===========

plot_fullgene_horizontal <- function(plot_df, table_df, output_file, width, height) {
  annotation_df <- table_df
  annotation_df$x <- seq_len(nrow(annotation_df))

  plot_obj <- ggplot(plot_df, aes(x = marker_id, y = dotplot_group)) +
    geom_point(aes(size = pct_exp, color = avg_exp_scaled), alpha = 0.95, stroke = 0) +
    facet_grid(. ~ marker_group, scales = "free_x", space = "free_x") +
    scale_x_discrete(labels = setNames(table_df$gene_label, table_df$marker_id)) +
    scale_y_discrete(limits = rev(dotplot_group_levels)) +
    scale_size_area(
      max_size = 4.4,
      limits = c(0, 100),
      breaks = c(0, 25, 50, 75, 100),
      name = "% Exp."
    ) +
    scale_color_gradientn(
      colors = c("#D9D9D9", "#D894C4", "#8F1D82"),
      limits = c(-1, 1),
      breaks = c(-1, -0.5, 0, 0.5, 1),
      name = "Avg. Exp."
    ) +
    theme_classic(base_size = 8.5) +
    labs(x = NULL, y = NULL) +
    theme(
      strip.background = element_blank(),
      strip.text.x = element_text(angle = 35, hjust = 0, size = 7, color = "black"),
      axis.text.x = element_text(angle = 60, hjust = 1, vjust = 1, color = "black", size = 7),
      axis.text.y = element_text(color = "black", size = 8.5),
      legend.title = element_text(size = 7.2),
      legend.text = element_text(size = 6.4),
      panel.spacing.x = grid::unit(0.14, "lines"),
      plot.margin = margin(8, 8, 8, 8)
    )

  save_pdf_plot(plot_obj, output_file, width = width, height = height)
}

plot_fullgene_horizontal(
  dot_df,
  marker_table,
  file.path(panel_b_dir, "HC_panel_b_fullgene_marker_dotplot.pdf"),
  width = max(8.2, 0.15 * nrow(marker_table) + 3.0),
  height = 2.6
)

plot_fullgene_horizontal(
  dot_df,
  marker_table,
  file.path(panel_b_dir, "HC_panel_b_fullgene_marker_dotplot_reordered_P1.pdf"),
  width = max(10.2, 0.18 * nrow(marker_table) + 3.2),
  height = 2.8
)


#7. ===========Plot vertical full-gene dotplot===========

group_x_map <- setNames(seq_along(dotplot_group_levels), dotplot_group_levels)
marker_table$gene_y <- nrow(marker_table) - marker_table$gene_index + 1
dot_df$group_x <- unname(group_x_map[as.character(dot_df$dotplot_group)])
dot_df$gene_y <- marker_table$gene_y[match(as.character(dot_df$marker_id), marker_table$marker_id)]

p_vertical <- ggplot(dot_df, aes(x = group_x, y = gene_y)) +
  geom_point(aes(size = pct_exp, color = avg_exp_scaled), alpha = 0.95, stroke = 0) +
  scale_x_continuous(
    breaks = unname(group_x_map),
    labels = dotplot_group_levels,
    limits = c(0.35, length(dotplot_group_levels) + 0.55),
    expand = c(0, 0)
  ) +
  scale_y_continuous(
    breaks = marker_table$gene_y,
    labels = marker_table$gene_label,
    limits = c(0.45, nrow(marker_table) + 0.55),
    expand = c(0, 0)
  ) +
  scale_size_area(
    max_size = 4.4 * 0.92,
    limits = c(0, 100),
    breaks = c(0, 25, 50, 75, 100),
    name = "% Exp."
  ) +
  scale_color_gradientn(
    colors = c("#D9D9D9", "#D894C4", "#8F1D82"),
    limits = c(-1, 1),
    breaks = c(-1, -0.5, 0, 0.5, 1),
    name = "Avg. Exp."
  ) +
  theme_classic(base_size = 8.5) +
  labs(x = NULL, y = NULL) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, color = "black", size = 10),
    axis.text.y = element_text(color = "black", size = 8.8),
    axis.ticks.x = element_blank(),
    axis.ticks.y = element_line(linewidth = 0.25),
    axis.line = element_line(linewidth = 0.35),
    legend.title = element_text(size = 7.2),
    legend.text = element_text(size = 6.4),
    legend.key.height = grid::unit(0.28, "cm"),
    legend.key.width = grid::unit(0.24, "cm"),
    legend.spacing.y = grid::unit(0.04, "cm"),
    legend.box.spacing = grid::unit(0.12, "cm"),
    plot.margin = margin(8, 8, 16, 6)
  )

save_pdf_plot(
  p_vertical,
  file.path(panel_b_dir, "HC_panel_b_fullgene_marker_dotplot_vertical_P1.pdf"),
  width = 3.1,
  height = max(7.2, 0.17 * nrow(marker_table) + 1.2)
)

message("Saved HC full-gene marker dotplots.")
