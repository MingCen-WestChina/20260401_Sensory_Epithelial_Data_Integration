#1. ===========Load USH2A plot data===========

common_path <- file.path("analysis", "05_mouse_ush2a", "common.R")
if (!file.exists(common_path)) common_path <- "common.R"
source(common_path)

plot_data <- load_ush2a_plot_data()
umap_df <- merge_expression_by_cell(plot_data$umap_df, plot_data$expression_meta)


#2. ===========Plot Ush2a expression on all-HC UMAP===========

label_df <- aggregate(
  cbind(UMAP_1, UMAP_2) ~ stage_celltype_plot,
  data = umap_df,
  FUN = median
)

label_df$stage_celltype_plot <- factor(
  as.character(label_df$stage_celltype_plot),
  levels = stage_celltype_levels
)
label_df <- label_df[!is.na(label_df$stage_celltype_plot), , drop = FALSE]

p_umap <- ggplot() +
  geom_point(
    data = umap_df[!umap_df$ush2a_detected, , drop = FALSE],
    aes(x = UMAP_1, y = UMAP_2),
    color = ush2a_background_color,
    size = 0.60,
    alpha = 0.90,
    stroke = 0
  ) +
  geom_point(
    data = umap_df[umap_df$ush2a_detected, , drop = FALSE],
    aes(x = UMAP_1, y = UMAP_2, color = ush2a_relative_expr),
    size = 0.78,
    alpha = 0.95,
    stroke = 0
  ) +
  ggrepel::geom_text_repel(
    data = label_df,
    aes(x = UMAP_1, y = UMAP_2, label = stage_celltype_plot),
    inherit.aes = FALSE,
    family = "serif",
    color = "black",
    size = 3.25,
    min.segment.length = 0.05,
    segment.color = "grey55",
    segment.size = 0.25,
    box.padding = 0.12,
    point.padding = 0.08,
    max.overlaps = Inf
  ) +
  ush2a_color_scale() +
  guides(color = guide_colorbar(barheight = grid::unit(1.60, "cm"), barwidth = grid::unit(0.32, "cm"))) +
  theme_ush2a(base_size = 12, square_panel = FALSE) +
  labs(
    title = "Ush2a expression overlaid on all-HC UMAP",
    x = "UMAP 1",
    y = "UMAP 2"
  ) +
  theme(
    legend.position = "right",
    plot.margin = margin(5.5, 5.5, 5.5, 5.5)
  )

save_pdf_plot(
  p_umap,
  file.path(USH2A_FIGURE_DIR, "USH2A_all_HC_UMAP.pdf"),
  width = 475 / 72,
  height = 417 / 72
)
