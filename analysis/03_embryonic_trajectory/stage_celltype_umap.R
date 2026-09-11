#1. ===========Load packages and parameters
set.seed(0)
options(stringsAsFactors = FALSE)

required_pkgs <- c("rhdf5", "uwot", "ggplot2", "grid")

missing_pkgs <- required_pkgs[
  !vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_pkgs) > 0) {
  stop("Missing R packages: ", paste(missing_pkgs, collapse = ", "))
}

suppressPackageStartupMessages({
  library(rhdf5)
  library(uwot)
  library(ggplot2)
  library(grid)
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
input_h5ad_lateral_candidates <- c(
  file.path(results_root, "03_embryonic_trajectory/inputs/embryonic_lateral_monocle2_input.h5ad"),
  file.path(data_root, "Monocle2_Embryonic_trajectory/embryonic_lateral_monocle2_input.h5ad"),
  file.path(data_root, "embryonic_lateral_monocle2_input.h5ad")
)
input_h5ad_medial <- input_h5ad_medial_candidates[which(file.exists(input_h5ad_medial_candidates))[1]]
input_h5ad_lateral <- input_h5ad_lateral_candidates[which(file.exists(input_h5ad_lateral_candidates))[1]]
if (is.na(input_h5ad_medial)) input_h5ad_medial <- input_h5ad_medial_candidates[1]
if (is.na(input_h5ad_lateral)) input_h5ad_lateral <- input_h5ad_lateral_candidates[1]
output_pdf <- file.path(
  results_root,
  "03_embryonic_trajectory",
  "figures",
  "Embryonic_E9_E16_medial_lateral_UMAP_celltype_label_stage_celltype_legend.pdf"
)
dir.create(dirname(output_pdf), recursive = TRUE, showWarnings = FALSE)

stage_order <- c("E9.5", "E11.5", "E13.5", "E14.5", "E16.5")
stage_short_map <- c(
  "E9.5" = "E9",
  "E11.5" = "E11",
  "E13.5" = "E13.5",
  "E14.5" = "E14",
  "E16.5" = "E16"
)

celltype_label_map <- c(
  "OV epithelial cells" = "OV",
  "Prosensory domain" = "PsD",
  "Medial domain" = "M.PsD",
  "Lateral domain" = "L.PsD",
  "IBC-like" = "IBC"
)

celltype_order <- c(
  "OV", "PsD", "M.PsD", "L.PsD", "M.PsC", "L.PsC",
  "IHC", "OHC", "iOHC", "IPhC", "IBC", "IPC", "HeC"
)

point_size <- 0.54
point_alpha <- 0.82
label_size <- 3.25
min_label_cells <- 5
umap_neighbors <- 30
umap_min_dist <- 0.32
legend_ncol <- 2
legend_text_size <- 7.2
legend_point_size <- 2.2
plot_width <- 10.6
plot_height <- 6.2
axis_break_step <- 5
label_outline_scale <- 0.0018


#2. ===========Read h5AD metadata and latent embedding
read_h5ad_categorical <- function(input_h5ad, key) {
  categories <- rhdf5::h5read(input_h5ad, paste0("obs/", key, "/categories"))
  codes <- rhdf5::h5read(input_h5ad, paste0("obs/", key, "/codes"))
  
  values <- rep(NA_character_, length(codes))
  valid_codes <- !is.na(codes) & codes >= 0
  values[valid_codes] <- categories[codes[valid_codes] + 1L]
  values
}

read_trajectory_h5ad <- function(input_h5ad, branch_name) {
  if (!file.exists(input_h5ad)) {
    stop("Input h5AD not found: ", input_h5ad)
  }
  
  cell_id <- rhdf5::h5read(input_h5ad, "obs/_index")
  latent <- rhdf5::h5read(input_h5ad, "obsm/X_scanvi")
  
  if (ncol(latent) == length(cell_id)) {
    latent <- t(latent)
  }
  
  if (nrow(latent) != length(cell_id)) {
    stop("X_scanvi dimensions do not match obs rows for: ", input_h5ad)
  }
  
  meta <- data.frame(
    cell_id = cell_id,
    stage = read_h5ad_categorical(input_h5ad, "stage"),
    trajectory_celltype = read_h5ad_categorical(input_h5ad, "trajectory_celltype"),
    trajectory_branch = read_h5ad_categorical(input_h5ad, "trajectory_branch"),
    source_branch = branch_name,
    stringsAsFactors = FALSE
  )
  
  list(meta = meta, latent = latent)
}

get_axis_breaks <- function(values, step) {
  value_range <- range(values, na.rm = TRUE)
  seq(
    floor(value_range[1] / step) * step,
    ceiling(value_range[2] / step) * step,
    by = step
  )
}


#3. ===========Combine medial and lateral branch cells
medial_data <- read_trajectory_h5ad(input_h5ad_medial, "Medial")
lateral_data <- read_trajectory_h5ad(input_h5ad_lateral, "Lateral")

combined_meta <- rbind(medial_data$meta, lateral_data$meta)
combined_latent <- rbind(medial_data$latent, lateral_data$latent)

keep_cell <- !duplicated(combined_meta$cell_id)
combined_meta <- combined_meta[keep_cell, , drop = FALSE]
combined_latent <- combined_latent[keep_cell, , drop = FALSE]

celltype_plot <- ifelse(
  combined_meta$trajectory_celltype %in% names(celltype_label_map),
  unname(celltype_label_map[combined_meta$trajectory_celltype]),
  combined_meta$trajectory_celltype
)

combined_meta$stage <- factor(combined_meta$stage, levels = stage_order)
combined_meta$stage_short <- unname(stage_short_map[as.character(combined_meta$stage)])
combined_meta$celltype_plot <- factor(celltype_plot, levels = celltype_order)

stage_celltype_levels <- unlist(lapply(unname(stage_short_map[stage_order]), function(stage_name) {
  paste(stage_name, celltype_order, sep = "_")
}))

combined_meta$stage_celltype_plot <- paste(
  combined_meta$stage_short,
  as.character(combined_meta$celltype_plot),
  sep = "_"
)

present_levels <- stage_celltype_levels[
  stage_celltype_levels %in% unique(combined_meta$stage_celltype_plot)
]

combined_meta$stage_celltype_plot <- factor(
  combined_meta$stage_celltype_plot,
  levels = present_levels
)

present_legend_levels <- gsub("_", " ", present_levels)
combined_meta$stage_celltype_legend <- factor(
  gsub("_", " ", as.character(combined_meta$stage_celltype_plot)),
  levels = present_legend_levels
)


#4. ===========Run UMAP on X_scanvi
umap_coord <- uwot::umap(
  combined_latent,
  n_neighbors = umap_neighbors,
  min_dist = umap_min_dist,
  metric = "cosine",
  init = "spectral",
  n_threads = 1,
  verbose = FALSE
)

plot_df <- cbind(
  combined_meta,
  data.frame(
    UMAP_1 = umap_coord[, 1],
    UMAP_2 = umap_coord[, 2]
  )
)

plot_df <- plot_df[order(plot_df$stage, plot_df$stage_celltype_plot), , drop = FALSE]

x_breaks <- get_axis_breaks(plot_df$UMAP_1, axis_break_step)
y_breaks <- get_axis_breaks(plot_df$UMAP_2, axis_break_step)


#5. ===========Build QC summary and label anchors
qc_summary <- data.frame(
  metric = c("cells", "stage_celltype_labels", "duplicated_cells_removed"),
  value = c(nrow(plot_df), length(present_levels), sum(!keep_cell))
)

print(qc_summary, row.names = FALSE)

label_df <- aggregate(
  cbind(UMAP_1, UMAP_2) ~ celltype_plot,
  data = plot_df,
  FUN = median
)

label_count <- as.data.frame(table(plot_df$celltype_plot), stringsAsFactors = FALSE)
colnames(label_count) <- c("celltype_plot", "cell_count")

label_df <- merge(label_df, label_count, by = "celltype_plot", all.x = TRUE)
label_df <- label_df[label_df$cell_count >= min_label_cells, , drop = FALSE]
label_df$celltype_plot <- factor(label_df$celltype_plot, levels = celltype_order)
label_df <- label_df[order(label_df$celltype_plot), , drop = FALSE]
label_df$label_text <- as.character(label_df$celltype_plot)

label_nudge <- data.frame(
  celltype_plot = c(
    "OV", "PsD", "M.PsD", "L.PsD", "M.PsC", "L.PsC",
    "IHC", "OHC", "iOHC", "IPhC", "IBC", "IPC", "HeC"
  ),
  nudge_x = c(-0.85, 1.45, 0.20, -2.45, -1.35, 2.25, -0.60, 1.55, -1.15, -1.25, 0.95, -0.30, -0.95),
  nudge_y = c(-0.70, 1.85, 1.85, 1.15, 1.55, 2.05, 1.55, 1.50, 1.35, -1.00, 0.50, -0.75, -0.65),
  stringsAsFactors = FALSE
)

label_df <- merge(label_df, label_nudge, by = "celltype_plot", all.x = TRUE, sort = FALSE)
label_df$nudge_x[is.na(label_df$nudge_x)] <- 0
label_df$nudge_y[is.na(label_df$nudge_y)] <- 0
label_df$label_x <- label_df$UMAP_1 + label_df$nudge_x
label_df$label_y <- label_df$UMAP_2 + label_df$nudge_y

label_outline_x <- diff(range(plot_df$UMAP_1, na.rm = TRUE)) * label_outline_scale
label_outline_y <- diff(range(plot_df$UMAP_2, na.rm = TRUE)) * label_outline_scale
outline_offsets <- expand.grid(
  dx = c(-label_outline_x, 0, label_outline_x),
  dy = c(-label_outline_y, 0, label_outline_y)
)
outline_offsets <- outline_offsets[outline_offsets$dx != 0 | outline_offsets$dy != 0, , drop = FALSE]

label_outline_df <- label_df[rep(seq_len(nrow(label_df)), each = nrow(outline_offsets)), ]
label_outline_df$dx <- rep(outline_offsets$dx, times = nrow(label_df))
label_outline_df$dy <- rep(outline_offsets$dy, times = nrow(label_df))

palette_values <- c(
  "#7FA6C4", "#E7A64A", "#79B7B2", "#C95F69", "#69A85D",
  "#A889C2", "#8C8C8C", "#4E79A7", "#F28E2B", "#E15759",
  "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
  "#9C755F", "#BAB0AC", "#86BCB6", "#D37295", "#499894",
  "#B6992D", "#AF7AA1", "#6B6ECF", "#BD9E39", "#8CD17D",
  "#A0CBE8", "#FFBE7D", "#FABFD2", "#C7C7C7", "#8CD0C3"
)

plot_colors <- setNames(
  rep(palette_values, length.out = length(levels(plot_df$stage_celltype_plot))),
  levels(plot_df$stage_celltype_plot)
)


#6. ===========Plot UMAP
p_umap <- ggplot2::ggplot(
  plot_df,
  ggplot2::aes(x = UMAP_1, y = UMAP_2, color = stage_celltype_plot)
) +
  ggplot2::geom_point(size = point_size, alpha = point_alpha, stroke = 0) +
  ggplot2::geom_text(
    data = label_outline_df,
    ggplot2::aes(x = label_x + dx, y = label_y + dy, label = label_text),
    inherit.aes = FALSE,
    color = "white",
    size = label_size
  ) +
  ggplot2::geom_text(
    data = label_df,
    ggplot2::aes(x = label_x, y = label_y, label = label_text),
    inherit.aes = FALSE,
    color = "black",
    size = label_size
  ) +
  ggplot2::scale_color_manual(
    values = plot_colors,
    breaks = present_levels,
    labels = present_legend_levels,
    drop = FALSE
  ) +
  ggplot2::scale_x_continuous(breaks = x_breaks) +
  ggplot2::scale_y_continuous(breaks = y_breaks) +
  ggplot2::guides(
    color = ggplot2::guide_legend(
      ncol = legend_ncol,
      byrow = FALSE,
      override.aes = list(size = legend_point_size, alpha = 1)
    )
  ) +
  ggplot2::coord_equal() +
  ggplot2::labs(
    title = "Embryonic medial/lateral UMAP",
    x = "UMAP 1",
    y = "UMAP 2",
    color = NULL
  ) +
  ggplot2::theme_classic(base_size = 10) +
  ggplot2::theme(
    aspect.ratio = 1,
    plot.title = ggplot2::element_text(size = 13.5, hjust = 0.5, face = "plain"),
    axis.title = ggplot2::element_text(size = 11.5, color = "black"),
    axis.text = ggplot2::element_text(size = 9.2, color = "black"),
    axis.line = ggplot2::element_line(linewidth = 0.45, color = "black"),
    axis.ticks = ggplot2::element_line(linewidth = 0.35, color = "black"),
    axis.ticks.length = grid::unit(0.08, "cm"),
    legend.position = "right",
    legend.justification = "center",
    legend.direction = "vertical",
    legend.background = ggplot2::element_blank(),
    legend.box.background = ggplot2::element_blank(),
    legend.margin = ggplot2::margin(0, 0, 0, 4),
    legend.key = ggplot2::element_blank(),
    legend.key.size = grid::unit(0.20, "cm"),
    legend.spacing.x = grid::unit(0.08, "cm"),
    legend.spacing.y = grid::unit(0.02, "cm"),
    legend.text = ggplot2::element_text(size = legend_text_size, color = "black"),
    plot.margin = grid::unit(c(0.12, 0.18, 0.12, 0.12), "in")
  )


#7. ===========Save PDF and display UMAP
ggplot2::ggsave(
  filename = output_pdf,
  plot = p_umap,
  width = plot_width,
  height = plot_height,
  units = "in",
  device = grDevices::pdf,
  useDingbats = FALSE,
  bg = "white"
)

cat("Saved PDF: ", output_pdf, "\n", sep = "")
print(p_umap)
