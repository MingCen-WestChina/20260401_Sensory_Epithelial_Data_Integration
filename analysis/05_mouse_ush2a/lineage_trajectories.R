#1. ===========Load USH2A plot data===========

common_path <- file.path("analysis", "05_mouse_ush2a", "common.R")
if (!file.exists(common_path)) common_path <- "common.R"
source(common_path)

plot_data <- load_ush2a_plot_data()
trajectory_cells <- merge_expression_by_cell(
  plot_data$trajectory_data$cell_df,
  plot_data$expression_meta
)
trajectory_edges <- plot_data$trajectory_data$edge_df


#2. ===========Plot Ush2a expression on lineage trajectories===========

plot_ush2a_lineage_trajectory <- function(lineage_name) {
  lineage_df <- trajectory_cells[
    as.character(trajectory_cells$hc_lineage) == lineage_name,
    ,
    drop = FALSE
  ]

  ggplot() +
    geom_segment(
      data = trajectory_edges,
      aes(x = x, y = y, xend = xend, yend = yend),
      inherit.aes = FALSE,
      color = "grey55",
      linewidth = 0.35
    ) +
    geom_point(
      data = lineage_df[!lineage_df$ush2a_detected, , drop = FALSE],
      aes(x = Component_1, y = Component_2),
      color = ush2a_background_color,
      size = 1.20,
      alpha = 0.90,
      stroke = 0
    ) +
    geom_point(
      data = lineage_df[lineage_df$ush2a_detected, , drop = FALSE],
      aes(x = Component_1, y = Component_2, color = ush2a_relative_expr),
      size = 1.45,
      alpha = 0.95,
      stroke = 0
    ) +
    ush2a_color_scale() +
    guides(color = guide_colorbar(barheight = grid::unit(1.60, "cm"), barwidth = grid::unit(0.32, "cm"))) +
    theme_ush2a(base_size = 12, square_panel = TRUE) +
    labs(
      title = paste0("Ush2a expression overlaid on ", lineage_name, " trajectory"),
      x = "Component 1",
      y = "Component 2"
    ) +
    theme(
      legend.position = "right",
      plot.margin = margin(5.5, 5.5, 5.5, 5.5)
    )
}

save_pdf_plot(
  plot_ush2a_lineage_trajectory("IHC"),
  file.path(USH2A_FIGURE_DIR, "USH2A_IHC_Monocle2_trajectory.pdf"),
  width = 399 / 72,
  height = 360 / 72
)

save_pdf_plot(
  plot_ush2a_lineage_trajectory("OHC"),
  file.path(USH2A_FIGURE_DIR, "USH2A_OHC_Monocle2_trajectory.pdf"),
  width = 399 / 72,
  height = 360 / 72
)
