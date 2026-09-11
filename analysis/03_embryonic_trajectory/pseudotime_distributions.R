#1. ===========Load packages and parameters
set.seed(0)
options(stringsAsFactors = FALSE)

required_pkgs <- c("Biobase", "ggplot2", "grid", "gridExtra")
missing_pkgs <- required_pkgs[
  !vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_pkgs) > 0) {
  stop("Missing R packages: ", paste(missing_pkgs, collapse = ", "))
}

suppressPackageStartupMessages({
  library(Biobase)
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

# stage_gap_scale controls the distance between adjacent stage columns.
# 1.00 is the default discrete-like spacing; smaller values put stages closer.
stage_gap_scale <- 0.78

# x_side_blank controls how much blank x-axis space is kept on both sides.
# Larger values make columns look closer while keeping each panel square.
x_side_blank <- 0.58

# These two only control the width of each stage column, not the spacing between stages.
jitter_width <- 0.05
violin_width <- 0.48
box_width <- 0.486
panel_aspect_ratio <- 1

output_width <- 7.25
output_height <- 4.05


#2. ===========Load refined Monocle2 functions
load_refined_environment <- function(script_path) {
  script_lines <- readLines(script_path, warn = FALSE)
  run_line <- grep("^#8\\. ===========Run", script_lines)[1]
  if (is.na(run_line)) {
    stop("Run block was not found in script: ", script_path)
  }

  refined_env <- new.env(parent = globalenv())
  eval(parse(text = paste(script_lines[seq_len(run_line - 1)], collapse = "\n")), envir = refined_env)
  refined_env
}

medial_env <- load_refined_environment(file.path(script_dir, "medial_trajectory.R"))
lateral_env <- load_refined_environment(file.path(script_dir, "lateral_trajectory.R"))


#3. ===========Run matched medial and lateral trajectories
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

extract_pseudotime_df <- function(result_obj, branch_name) {
  meta <- as.data.frame(Biobase::pData(result_obj$cds))
  data.frame(
    branch = branch_name,
    stage = meta$stage_short,
    stage_rank = meta$stage_rank,
    celltype = meta$celltype_plot,
    pseudotime = pmin(pmax(meta$Pseudotime_scaled, 0), 1),
    stringsAsFactors = FALSE
  )
}

plot_df <- rbind(
  extract_pseudotime_df(medial_result, "Medial"),
  extract_pseudotime_df(lateral_result, "Lateral")
)

stage_levels <- c("E9", "E11", "E13.5", "E14", "E16")
stage_colors <- c(
  "E9" = "#4C78A8",
  "E11" = "#72B7B2",
  "E13.5" = "#F2A541",
  "E14" = "#D96C75",
  "E16" = "#59A14F"
)

plot_df$stage <- factor(plot_df$stage, levels = stage_levels)


#4. ===========Build compact numeric x positions
x_center <- (length(stage_levels) + 1) / 2
stage_x_map <- setNames(
  x_center + (seq_along(stage_levels) - x_center) * stage_gap_scale,
  stage_levels
)

plot_df$stage_x <- unname(stage_x_map[as.character(plot_df$stage)])
x_limits <- range(stage_x_map) + c(-x_side_blank, x_side_blank)


#5. ===========Plot pseudotime distribution
make_distribution_panel <- function(branch_name) {
  branch_df <- plot_df[plot_df$branch == branch_name, , drop = FALSE]

  ggplot2::ggplot(branch_df, ggplot2::aes(x = stage_x, y = pseudotime, fill = stage)) +
    ggplot2::geom_violin(
      ggplot2::aes(group = stage),
      width = violin_width,
      trim = TRUE,
      color = "grey65",
      linewidth = 0.18,
      alpha = 0.22
    ) +
    ggplot2::geom_boxplot(
      ggplot2::aes(group = stage),
      width = box_width,
      outlier.shape = NA,
      color = "grey35",
      linewidth = 0.25,
      fill = "white"
    ) +
    ggplot2::geom_jitter(
      ggplot2::aes(color = stage),
      width = jitter_width,
      height = 0,
      size = 0.35,
      alpha = 0.25,
      show.legend = FALSE
    ) +
    ggplot2::scale_fill_manual(values = stage_colors, drop = FALSE) +
    ggplot2::scale_color_manual(values = stage_colors, drop = FALSE) +
    ggplot2::scale_x_continuous(
      breaks = unname(stage_x_map),
      labels = names(stage_x_map),
      limits = x_limits,
      expand = c(0, 0)
    ) +
    ggplot2::scale_y_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, 0.25),
      expand = ggplot2::expansion(mult = c(0.02, 0.04))
    ) +
    ggplot2::theme_classic(base_size = 9.2) +
    ggplot2::labs(
      title = paste0(branch_name, " pseudotime distribution"),
      x = "Stage",
      y = "Scaled Monocle2 pseudotime"
    ) +
    ggplot2::theme(
      aspect.ratio = panel_aspect_ratio,
      plot.title = ggplot2::element_text(size = 10.6, hjust = 0.5),
      axis.title = ggplot2::element_text(size = 9.4, color = "black"),
      axis.text.x = ggplot2::element_text(size = 8.5, color = "black"),
      axis.text.y = ggplot2::element_text(size = 8.5, color = "black"),
      legend.position = "none",
      plot.margin = grid::unit(c(0.06, 0.08, 0.06, 0.06), "in")
    )
}

p_medial <- make_distribution_panel("Medial")
p_lateral <- make_distribution_panel("Lateral")

panel_distribution <- gridExtra::arrangeGrob(
  p_medial,
  p_lateral,
  ncol = 2
)


#6. ===========Save final panel
file_prefix <- file.path(
  output_dir,
  "Embryonic_E9_E16_panel_b_pseudotime_distribution_square_tight"
)

ggplot2::ggsave(
  filename = paste0(file_prefix, ".pdf"),
  plot = panel_distribution,
  width = output_width,
  height = output_height,
  units = "in",
  device = grDevices::cairo_pdf,
  bg = "white"
)

ggplot2::ggsave(
  filename = paste0(file_prefix, ".png"),
  plot = panel_distribution,
  width = output_width,
  height = output_height,
  units = "in",
  dpi = 1200,
  bg = "white",
  limitsize = FALSE
)

cat(
  "Pseudotime distribution saved. medial_cells=", medial_result$qc$cells,
  ", lateral_cells=", lateral_result$qc$cells,
  ", medial_stage_cor=", medial_result$qc$stage_pseudotime_cor,
  ", lateral_stage_cor=", lateral_result$qc$stage_pseudotime_cor,
  "\n",
  sep = ""
)
