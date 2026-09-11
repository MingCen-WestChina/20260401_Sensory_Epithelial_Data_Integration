#1. ===========Load shared data===========

common_path <- file.path("analysis", "04_hair_cell_trajectory", "common.R")
if (!file.exists(common_path)) common_path <- "common.R"
source(common_path)

bundle <- build_hc_monocle2_bundle(force_rebuild = FALSE)
hc_cds <- bundle$hc_cds
plot_meta <- bundle$plot_meta

panel_a_dir <- file.path(HC_OUTPUT_DIR, "Panel_a_UMAP")
panel_b_dir <- file.path(HC_OUTPUT_DIR, "Panel_b_dotplot")
dir.create(panel_a_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(panel_b_dir, recursive = TRUE, showWarnings = FALSE)

save_ai_editable_pdf_plot <- function(plot_obj, filename, width, height) {
  dir.create(dirname(filename), recursive = TRUE, showWarnings = FALSE)
  grDevices::pdf(
    file = filename,
    width = width,
    height = height,
    paper = "special",
    useDingbats = FALSE,
    version = "1.4",
    bg = "white"
  )
  on.exit(grDevices::dev.off(), add = TRUE)
  print(plot_obj)
}


#2. ===========Panel a all-HC UMAP from X_scanvi===========

x_scanvi <- read_h5ad_obsm(input_h5ad, "X_scanvi")
x_scanvi <- x_scanvi[plot_meta$cell_id, , drop = FALSE]

panel_a_stage_celltype_colors <- c(
  "E14_IHC" = "#B57A20",
  "E14_OHC" = "#E0B84A",
  "E16_IHC" = "#C65A2E",
  "E16_iOHC" = "#E7953A",
  "E16_OHC" = "#F1C85B",
  "P1_IHC" = "#7EA6B8",
  "P1_OHC" = "#B9CDD4",
  "P7_IHC" = "#4F8C8B",
  "P7_OHC" = "#86B7AE",
  "P14_IHC" = "#4A4E69",
  "P14_OHC" = "#6B6F85",
  "P28_IHC" = "#000000",
  "P28_OHC" = "#4D4D4D"
)

panel_a_label_df <- data.frame(
  stage_celltype_plot = factor(stage_celltype_levels, levels = stage_celltype_levels),
  UMAP_1 = c(
    10.7, 5.55,
    12.1, 5.75, 3.45,
    14.0, 0.55,
    15.0, -3.55,
    16.35, -4.2,
    15.35, 0.25
  ),
  UMAP_2 = c(
    11.05, 14.75,
    11.8, 10.1, 12.35,
    13.25, 13.45,
    12.05, 9.55,
    8.95, 4.9,
    10.1, 9.45
  )
)

panel_a_label_df$celltype_label <- sub("^[^_]+_", "", as.character(panel_a_label_df$stage_celltype_plot))

panel_a_celltype_label_df <- data.frame(
  celltype_label = c("OHC", "iOHC", "IHC"),
  UMAP_1 = c(6.65, 6.85, 17.75),
  UMAP_2 = c(14.25, 8.35, 13.30),
  stringsAsFactors = FALSE
)

panel_a_params <- list(
  point_size = 1.1,
  point_alpha = 0.82,
  label_text_size = 3.3,
  legend_text_size = 6.6,
  legend_key_point_size = 2.4,
  legend_ncol = 2,
  legend_position = "right",
  x_limits = c(-9, 19),
  y_limits = c(1, 20),
  x_breaks = c(-9, -5, 0, 5, 10, 15, 19),
  y_breaks = c(1, 5, 10, 15, 20),
  output_width = 8.4,
  output_height = 4.8
)

set.seed(0)
umap_mat <- uwot::umap(
  x_scanvi,
  n_neighbors = 30,
  min_dist = 0.35,
  metric = "euclidean",
  init = "spectral",
  ret_model = FALSE,
  verbose = FALSE
)

umap_df <- data.frame(
  cell_id = plot_meta$cell_id,
  UMAP_1 = -umap_mat[, 1] + 2.0,
  UMAP_2 = -umap_mat[, 2] + 13.2,
  stage_celltype_plot = plot_meta$stage_celltype_plot,
  stringsAsFactors = FALSE
)

umap_df$stage_celltype_plot <- factor(
  as.character(umap_df$stage_celltype_plot),
  levels = stage_celltype_levels
)

panel_a_anchor_df <- aggregate(
  cbind(UMAP_1, UMAP_2) ~ stage_celltype_plot,
  data = umap_df,
  FUN = median
)

panel_a_anchor_df <- merge(
  panel_a_anchor_df,
  data.frame(
    stage_celltype_plot = panel_a_label_df$stage_celltype_plot,
    target_UMAP_1 = panel_a_label_df$UMAP_1,
    target_UMAP_2 = panel_a_label_df$UMAP_2
  ),
  by = "stage_celltype_plot",
  all.x = TRUE,
  sort = FALSE
)

panel_a_anchor_df$shift_UMAP_1 <- panel_a_anchor_df$target_UMAP_1 - panel_a_anchor_df$UMAP_1
panel_a_anchor_df$shift_UMAP_2 <- panel_a_anchor_df$target_UMAP_2 - panel_a_anchor_df$UMAP_2

umap_df <- merge(
  umap_df,
  panel_a_anchor_df[, c("stage_celltype_plot", "shift_UMAP_1", "shift_UMAP_2")],
  by = "stage_celltype_plot",
  all.x = TRUE,
  sort = FALSE
)

umap_df$UMAP_1 <- umap_df$UMAP_1 + umap_df$shift_UMAP_1
umap_df$UMAP_2 <- umap_df$UMAP_2 + umap_df$shift_UMAP_2
umap_df$stage_celltype_plot <- factor(
  as.character(umap_df$stage_celltype_plot),
  levels = stage_celltype_levels
)

panel_a_legend_breaks <- stage_celltype_levels[
  stage_celltype_levels %in% as.character(umap_df$stage_celltype_plot)
]

p_panel_a <- ggplot(
  umap_df,
  aes(x = UMAP_1, y = UMAP_2, color = stage_celltype_plot)
) +
  geom_point(
    size = panel_a_params$point_size,
    alpha = panel_a_params$point_alpha,
    stroke = 0
  ) +
  geom_text(
    data = panel_a_celltype_label_df,
    aes(x = UMAP_1, y = UMAP_2, label = celltype_label),
    inherit.aes = FALSE,
    color = "black",
    size = panel_a_params$label_text_size,
    hjust = 0,
    vjust = 0.5
  ) +
  scale_color_manual(
    values = panel_a_stage_celltype_colors,
    breaks = panel_a_legend_breaks,
    labels = gsub("_", " ", panel_a_legend_breaks),
    name = NULL,
    drop = FALSE
  ) +
  guides(
    color = guide_legend(
      ncol = panel_a_params$legend_ncol,
      byrow = FALSE,
      override.aes = list(
        size = panel_a_params$legend_key_point_size,
        alpha = 1
      )
    )
  ) +
  scale_x_continuous(
    limits = panel_a_params$x_limits,
    breaks = panel_a_params$x_breaks,
    expand = expansion(mult = 0, add = 0)
  ) +
  scale_y_continuous(
    limits = panel_a_params$y_limits,
    breaks = panel_a_params$y_breaks,
    expand = expansion(mult = 0, add = 0)
  ) +
  coord_cartesian(
    xlim = panel_a_params$x_limits,
    ylim = panel_a_params$y_limits,
    expand = FALSE,
    clip = "on"
  ) +
  theme_classic(base_size = 12) +
  labs(
    title = "Panel a | All HC UMAP",
    x = "UMAP1",
    y = "UMAP2"
  ) +
  theme(
    legend.position = panel_a_params$legend_position,
    legend.direction = "vertical",
    legend.background = element_blank(),
    legend.key = element_blank(),
    legend.title = element_blank(),
    legend.text = element_text(size = panel_a_params$legend_text_size, color = "black"),
    legend.margin = margin(0, 0, 0, 8),
    legend.spacing.y = grid::unit(0.02, "cm"),
    legend.spacing.x = grid::unit(0.18, "cm"),
    legend.key.height = grid::unit(0.18, "cm"),
    legend.key.width = grid::unit(0.18, "cm"),
    legend.box.margin = margin(0, 0, 0, 0),
    panel.border = element_blank(),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black", linewidth = 0.45),
    axis.ticks.length = grid::unit(0.12, "cm"),
    plot.title = element_text(size = 15, hjust = 0.5),
    axis.text = element_text(color = "black"),
    axis.title = element_text(color = "black"),
    aspect.ratio = 1
  )

save_ai_editable_pdf_plot(
  p_panel_a,
  file.path(panel_a_dir, "HC_panel_a_UMAP_stage_celltype_label.pdf"),
  width = panel_a_params$output_width,
  height = panel_a_params$output_height
)


#3. ===========Prepare HC marker dotplot groups===========

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
  make_marker_item("Early IHC", "Fgf8"),
  make_marker_item("Intermediate IHC", "Dlk2"),
  make_marker_item("Intermediate IHC", "Nefl"),
  make_marker_item("Intermediate IHC", "Shtn1"),
  make_marker_item("Intermediate IHC", "Pvalb"),
  make_marker_item("Intermediate IHC", "Ctbp2"),
  make_marker_item("Intermediate IHC", "Cacna1d"),
  make_marker_item("Mature IHC", "Tmc1"),
  make_marker_item("Mature IHC", "Slc17a8/Vglut3", c("Slc17a8", "Vglut3")),
  make_marker_item("Mature IHC", "Otof"),
  make_marker_item("Mature IHC", "Calb2"),
  make_marker_item("Mature IHC", "Tbx2"),
  make_marker_item("Mature IHC", "Rprm"),
  make_marker_item("Early OHC", "Hes6"),
  make_marker_item("Early OHC", "Insm1"),
  make_marker_item("Early OHC", "Bcl11b"),
  make_marker_item("Early OHC", "Lhfpl5"),
  make_marker_item("Early OHC", "Pcp4"),
  make_marker_item("Early OHC", "Cdkn1c"),
  make_marker_item("Intermediate OHC", "Cib2"),
  make_marker_item("Intermediate OHC", "Calb1"),
  make_marker_item("Intermediate OHC", "Ocm"),
  make_marker_item("Intermediate OHC", "Ikzf2", c("Ikzf2", "Helios")),
  make_marker_item("Intermediate OHC", "Strip2"),
  make_marker_item("Mature OHC", "Slc26a5/Prestin", c("Slc26a5", "Prestin")),
  make_marker_item("Mature OHC", "Tmc1"),
  make_marker_item("Mature OHC", "Atp2b2"),
  make_marker_item("Mature OHC", "Myo7a"),
  make_marker_item("Mature OHC", "Six2"),
  make_marker_item("Mature OHC", "Lbh"),
  make_marker_item("Mature OHC", "Lpin2"),
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

expr_dot_all <- exprs(hc_cds)
gene_universe <- rownames(expr_dot_all)

marker_table$gene <- vapply(
  seq_len(nrow(marker_table)),
  function(i) match_marker_feature(marker_items[[i]]$aliases, gene_universe),
  character(1)
)

marker_table <- marker_table[!is.na(marker_table$gene), , drop = FALSE]
marker_table$gene_index <- seq_len(nrow(marker_table))

dot_meta <- plot_meta
dot_meta$dotplot_group <- get_dotplot_group(
  as.character(dot_meta$stage),
  as.character(dot_meta$hc_lineage)
)

dot_meta <- dot_meta[dot_meta$dotplot_group %in% dotplot_group_levels, , drop = FALSE]
dot_meta$dotplot_group <- factor(dot_meta$dotplot_group, levels = dotplot_group_levels)


#4. ===========Calculate marker dotplot statistics===========

dot_df <- do.call(
  rbind,
  lapply(seq_len(nrow(marker_table)), function(marker_i) {
    gene_id <- marker_table$gene[marker_i]

    do.call(
      rbind,
      lapply(dotplot_group_levels, function(group_name) {
        cells_use <- dot_meta$cell_id[dot_meta$dotplot_group == group_name]
        values <- as.numeric(expr_dot_all[gene_id, cells_use])

        data.frame(
          marker_id = marker_table$marker_id[marker_i],
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


#5. ===========Save compact vertical marker dotplot===========

p_stage_vertical <- ggplot(dot_df, aes(x = dotplot_group, y = marker_id)) +
  geom_point(aes(size = pct_exp, color = avg_exp_scaled), alpha = 0.95, stroke = 0) +
  scale_y_discrete(labels = setNames(marker_table$gene_label, marker_table$marker_id)) +
  scale_x_discrete(labels = dotplot_group_levels) +
  scale_size_area(
    max_size = 4.0,
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
  labs(title = "Panel b | HC marker dotplot", x = NULL, y = NULL) +
  theme(
    plot.title = element_text(size = 9, hjust = 0.5),
    axis.text.x = element_text(angle = 60, hjust = 1, vjust = 1, size = 6.3, color = "black"),
    axis.text.y = element_text(size = 6.4, color = "black"),
    legend.title = element_text(size = 7),
    legend.text = element_text(size = 6.4),
    legend.key.height = grid::unit(0.28, "cm"),
    legend.key.width = grid::unit(0.24, "cm"),
    plot.margin = margin(4, 5, 8, 4)
  )

save_pdf_plot(
  p_stage_vertical,
  file.path(panel_b_dir, "HC_panel_b_IHC_OHC_stage_dotplot.pdf"),
  width = 2.15,
  height = 6.15
)


#6. ===========Save horizontal marker dotplot without Slc26a5===========

horizontal_table <- marker_table[marker_table$gene != "Slc26a5", , drop = FALSE]
horizontal_df <- dot_df[dot_df$marker_id %in% horizontal_table$marker_id, , drop = FALSE]
horizontal_df$marker_id <- factor(horizontal_df$marker_id, levels = horizontal_table$marker_id)

p_stage_horizontal <- ggplot() +
  geom_point(
    data = horizontal_df,
    aes(x = marker_id, y = dotplot_group, size = pct_exp, color = avg_exp_scaled),
    alpha = 0.95,
    stroke = 0
  ) +
  scale_x_discrete(labels = setNames(horizontal_table$gene_label, horizontal_table$marker_id)) +
  scale_y_discrete(limits = rev(dotplot_group_levels)) +
  scale_size_area(
    max_size = 4.0,
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
  coord_cartesian(clip = "off") +
  theme_classic(base_size = 9) +
  labs(x = NULL, y = NULL) +
  theme(
    axis.text.x = element_text(angle = 60, hjust = 1, vjust = 1, color = "black", size = 7),
    axis.text.y = element_text(color = "black", size = 9),
    legend.title = element_text(size = 7),
    legend.text = element_text(size = 6.4),
    plot.margin = margin(14, 8, 8, 8)
  )

save_pdf_plot(
  p_stage_horizontal,
  file.path(panel_b_dir, "HC_panel_b_IHC_OHC_stage_dotplot_horizontal_no_slc26a5.pdf"),
  width = 8.2,
  height = 2.25
)

message("Saved HC UMAP and compact HC marker dotplots.")
