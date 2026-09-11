#1. ===========Load packages and set paths
set.seed(0)
options(stringsAsFactors = FALSE)

required_pkgs <- c("rhdf5", "Matrix", "ggplot2", "grid", "ragg")
missing_pkgs <- required_pkgs[
  !vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_pkgs) > 0) {
  stop("Missing R packages: ", paste(missing_pkgs, collapse = ", "))
}

suppressPackageStartupMessages({
  library(rhdf5)
  library(Matrix)
  library(ggplot2)
  library(grid)
})

args_full <- commandArgs(trailingOnly = FALSE)
script_arg <- sub("^--file=", "", grep("^--file=", args_full, value = TRUE)[1])
script_dir <- dirname(normalizePath(script_arg, winslash = "/", mustWork = TRUE))
project_root <- normalizePath(file.path(script_dir, "../.."), winslash = "/", mustWork = TRUE)

env_path <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  normalizePath(if (nzchar(value)) value else default, winslash = "/", mustWork = FALSE)
}

data_dir <- env_path("COCHLEA_DATA_DIR", file.path(project_root, "Data"))
results_dir <- env_path("COCHLEA_RESULTS_DIR", file.path(project_root, "results"))

resolve_input_path <- function(candidates) {
  existing <- candidates[file.exists(candidates)]
  if (length(existing) > 0) existing[[1]] else candidates[[1]]
}
input_h5ad_medial <- resolve_input_path(c(
  file.path(results_dir, "03_embryonic_trajectory/inputs/embryonic_medial_monocle2_input.h5ad"),
  file.path(data_dir, "Monocle2_Embryonic_trajectory/embryonic_medial_monocle2_input.h5ad")
))
input_h5ad_lateral <- resolve_input_path(c(
  file.path(results_dir, "03_embryonic_trajectory/inputs/embryonic_lateral_monocle2_input.h5ad"),
  file.path(data_dir, "Monocle2_Embryonic_trajectory/embryonic_lateral_monocle2_input.h5ad")
))
output_dir <- file.path(results_dir, "figures", "fig2")
font_family <- Sys.getenv("COCHLEA_FONT_FAMILY", unset = "Arial")

if ("--check-inputs" %in% commandArgs(trailingOnly = TRUE)) {
  missing_inputs <- c(input_h5ad_medial, input_h5ad_lateral)[
    !file.exists(c(input_h5ad_medial, input_h5ad_lateral))
  ]
  if (length(missing_inputs) > 0) stop("Missing input h5AD: ", paste(missing_inputs, collapse = ", "))
  message("Validated Figure 2B inputs: ", input_h5ad_medial, " and ", input_h5ad_lateral)
  quit(save = "no", status = 0)
}

stage_order <- c("E9.5", "E11.5", "E13.5", "E14.5", "E16.5")
stage_short_map <- c(
  "E9.5" = "E9",
  "E11.5" = "E11",
  "E13.5" = "E13",
  "E14.5" = "E14",
  "E16.5" = "E16"
)

celltype_levels <- c(
  "OV", "PsD", "M.PsD", "L.PsD", "M.PsC", "L.PsC",
  "IPhC", "IBC", "IPC", "HeC", "IHC", "iOHC", "OHC"
)

celltype_label_map <- c(
  "OV epithelial cells" = "OV",
  "Prosensory domain" = "PsD",
  "Medial domain" = "M.PsD",
  "Lateral domain" = "L.PsD",
  "M.PsC" = "M.PsC",
  "L.PsC" = "L.PsC",
  "IHC" = "IHC",
  "OHC" = "OHC",
  "iOHC" = "iOHC",
  "IPhC" = "IPhC",
  "IBC-like" = "IBC",
  "IBC" = "IBC",
  "IPC" = "IPC",
  "HeC" = "HeC"
)

strip_colors <- c(
  "OV" = "#B9A7C8",
  "PsD" = "#8A8A8A",
  "M.PsD" = "#6D5CA8",
  "L.PsD" = "#45A5A0",
  "M.PsC" = "#6D6AAE",
  "L.PsC" = "#82B45F",
  "IPhC" = "#B8A1C8",
  "IBC" = "#D58AA8",
  "IPC" = "#4C9BC7",
  "HeC" = "#D39B43",
  "IHC" = "#B83232",
  "iOHC" = "#D66A4E",
  "OHC" = "#9C2F2A"
)

stage_celltype_colors <- c(
  "E9 OV" = "#7FA6C4",
  "E11 OV" = "#E7A64A",
  "E13 PsD" = "#79B7B2",
  "E13 M.PsD" = "#C95F69",
  "E13 L.PsD" = "#69A85D",
  "E14 M.PsC" = "#A889C2",
  "E14 L.PsC" = "#8C8C8C",
  "E14 IHC" = "#4E79A7",
  "E14 OHC" = "#F28E2B",
  "E16 M.PsC" = "#E15759",
  "E16 L.PsC" = "#76B7B2",
  "E16 IHC" = "#59A14F",
  "E16 OHC" = "#EDC948",
  "E16 iOHC" = "#B07AA1",
  "E16 IPhC" = "#FF9DA7",
  "E16 IBC" = "#9C755F",
  "E16 IPC" = "#BAB0AC",
  "E16 HeC" = "#86BCB6"
)

lower_detection_limit <- 0.1
min_cells_marker <- 3
color_limit <- 1
max_dot_size <- 4.98 / ggplot2::.pt

gene_axis_text_size <- 6.430
celltype_axis_text_size <- 6.000
full_row_bullet_text_size <- 11.000
top_label_text_size <- 5.000
panel_label_text_size <- 14.000
legend_title_text_size <- 4.091
legend_text_size <- 3.636

full_width <- 395.276 / 72
full_height <- 2.990
part1_width <- 2.450
part2_width <- 2.950
part3_width <- 3.400
split_height <- 2.990

plot_margin_top <- 20.149
plot_margin_right <- 11.935
plot_margin_bottom <- 4.645
plot_margin_left <- 47.450

row_bullet_offset_pt <- c(
  "E9 OV" = -23.460,
  "E11 OV" = -27.460,
  "E13 PsD" = -29.460,
  "E13 M.PsD" = -36.460,
  "E13 L.PsD" = -34.460,
  "E14 M.PsC" = -36.460,
  "E14 L.PsC" = -34.460,
  "E14 IHC" = -29.460,
  "E14 OHC" = -31.460,
  "E16 M.PsC" = -36.458,
  "E16 L.PsC" = -35.458,
  "E16 IPhC" = -32.458,
  "E16 IBC" = -29.458,
  "E16 IPC" = -29.458,
  "E16 HeC" = -31.458,
  "E16 IHC" = -29.458,
  "E16 iOHC" = -33.458,
  "E16 OHC" = -31.458
)


#2. ===========Define the marker genes shown in Fig2B
make_marker_item <- function(group, label, aliases = NULL) {
  if (is.null(aliases)) aliases <- label
  list(group = group, label = label, aliases = aliases)
}

marker_items <- list(
  make_marker_item("OV", "Meis2"),
  make_marker_item("OV", "Foxg1"),
  make_marker_item("OV", "Gata3"),
  make_marker_item("PsD", "Sox2"),
  make_marker_item("PsD", "Fgf20"),
  make_marker_item("PsD", "Jag1"),
  make_marker_item("PsD", "Lgr5"),
  make_marker_item("M.PsD", "Tbx2"),
  make_marker_item("M.PsD", "Fgf10"),
  make_marker_item("M.PsD", "Sulf1"),
  make_marker_item("M.PsD", "Ebf1"),
  make_marker_item("L.PsD", "Bmp4"),
  make_marker_item("L.PsD", "Rorb"),
  make_marker_item("L.PsD", "Fst"),
  make_marker_item("M.PsC", "Ntf3"),
  make_marker_item("M.PsC", "Crym"),
  make_marker_item("M.PsC", "Anxa5"),
  make_marker_item("L.PsC", "Prox1"),
  make_marker_item("L.PsC", "Igfbp3"),
  make_marker_item("IPhC", "Hes5"),
  make_marker_item("IPhC", "Gjb2"),
  make_marker_item("IBC", "Sparcl1"),
  make_marker_item("IBC", "Id3"),
  make_marker_item("IPC", "Ngfr"),
  make_marker_item("IPC", "Npy"),
  make_marker_item("IPC", "S100b"),
  make_marker_item("HeC", "Gata2"),
  make_marker_item("HeC", "Kazald1"),
  make_marker_item("IHC", "Atoh1"),
  make_marker_item("IHC", "Gfi1"),
  make_marker_item("IHC", "Fgf8"),
  make_marker_item("IHC", "Pvalb"),
  make_marker_item("IHC", "Myo7a"),
  make_marker_item("IHC", "Otof"),
  make_marker_item("iOHC", "Hes6"),
  make_marker_item("iOHC", "Ccer2"),
  make_marker_item("iOHC", "Pcp4"),
  make_marker_item("iOHC", "Lhfpl5"),
  make_marker_item("OHC", "Bcl11b"),
  make_marker_item("OHC", "Rasd2"),
  make_marker_item("OHC", "Barhl1"),
  make_marker_item("OHC", "Espn"),
  make_marker_item("OHC", "Myo6")
)


#3. ===========Read the preserved h5AD expression matrices
read_h5ad_categorical <- function(input_h5ad, key) {
  categories <- rhdf5::h5read(input_h5ad, paste0("obs/", key, "/categories"))
  codes <- rhdf5::h5read(input_h5ad, paste0("obs/", key, "/codes")) + 1L
  categories[codes]
}

read_trajectory_h5ad <- function(input_h5ad, branch_name) {
  if (!file.exists(input_h5ad)) stop("Input h5AD not found: ", input_h5ad)

  cell_id <- rhdf5::h5read(input_h5ad, "obs/_index")
  gene_id <- rhdf5::h5read(input_h5ad, "var/_index")
  x_data <- as.numeric(rhdf5::h5read(input_h5ad, "X/data"))
  x_indices <- as.integer(rhdf5::h5read(input_h5ad, "X/indices")) + 1L
  x_indptr <- as.integer(rhdf5::h5read(input_h5ad, "X/indptr"))

  n_cells <- length(cell_id)
  n_genes <- length(gene_id)
  cell_index <- rep.int(seq_len(n_cells), diff(x_indptr))
  expr_cell_gene <- Matrix::sparseMatrix(
    i = cell_index,
    j = x_indices,
    x = x_data,
    dims = c(n_cells, n_genes)
  )
  expr_mat <- Matrix::t(expr_cell_gene)
  rownames(expr_mat) <- gene_id
  colnames(expr_mat) <- cell_id

  meta <- data.frame(
    cell_id = cell_id,
    stage = read_h5ad_categorical(input_h5ad, "stage"),
    trajectory_celltype = read_h5ad_categorical(input_h5ad, "trajectory_celltype"),
    source_branch = branch_name,
    stringsAsFactors = FALSE
  )
  rownames(meta) <- cell_id
  list(expr_mat = expr_mat, meta = meta)
}

medial_data <- read_trajectory_h5ad(input_h5ad_medial, "Medial")
lateral_data <- read_trajectory_h5ad(input_h5ad_lateral, "Lateral")

if (!identical(rownames(medial_data$expr_mat), rownames(lateral_data$expr_mat))) {
  stop("Medial and lateral h5AD files do not have the same gene order.")
}

expr_mat <- Matrix::cbind2(medial_data$expr_mat, lateral_data$expr_mat)
meta <- rbind(medial_data$meta, lateral_data$meta)
keep_cell <- !duplicated(meta$cell_id)
expr_mat <- expr_mat[, keep_cell, drop = FALSE]
meta <- meta[keep_cell, , drop = FALSE]

meta$celltype_plot <- ifelse(
  meta$trajectory_celltype %in% names(celltype_label_map),
  unname(celltype_label_map[meta$trajectory_celltype]),
  meta$trajectory_celltype
)
meta <- meta[meta$celltype_plot %in% celltype_levels, , drop = FALSE]
expr_mat <- expr_mat[, rownames(meta), drop = FALSE]
meta$stage <- factor(as.character(meta$stage), levels = stage_order)
meta$stage_short <- unname(stage_short_map[as.character(meta$stage)])
meta$celltype_plot <- factor(meta$celltype_plot, levels = celltype_levels)
meta$stage_celltype_plot <- paste(meta$stage_short, meta$celltype_plot)


#4. ===========Resolve markers and calculate dotplot statistics
resolve_marker_gene <- function(aliases, gene_names) {
  hit <- aliases[aliases %in% gene_names]
  if (length(hit) == 0) return(NA_character_)
  hit[1]
}

marker_table <- do.call(
  rbind,
  lapply(marker_items, function(item) {
    data.frame(
      marker_group = item$group,
      gene_label = item$label,
      gene_id = resolve_marker_gene(item$aliases, rownames(expr_mat)),
      stringsAsFactors = FALSE
    )
  })
)
marker_table <- marker_table[!is.na(marker_table$gene_id), , drop = FALSE]
marker_table <- marker_table[!duplicated(marker_table$gene_id), , drop = FALSE]
detected_cells <- Matrix::rowSums(
  expr_mat[marker_table$gene_id, , drop = FALSE] > lower_detection_limit
)
marker_table <- marker_table[detected_cells >= min_cells_marker, , drop = FALSE]
marker_table$marker_group <- factor(marker_table$marker_group, levels = celltype_levels)
marker_table <- marker_table[order(marker_table$marker_group), , drop = FALSE]
marker_table$marker_id <- paste(marker_table$marker_group, marker_table$gene_label, sep = "__")

expr_dot <- expr_mat[marker_table$gene_id, , drop = FALSE]
rownames(expr_dot) <- marker_table$marker_id

stage_celltype_levels <- unlist(
  lapply(unname(stage_short_map[stage_order]), function(stage_name) {
    paste(stage_name, celltype_levels)
  })
)
dotplot_group_levels <- stage_celltype_levels[
  stage_celltype_levels %in% unique(meta$stage_celltype_plot)
]
y_groups <- rev(dotplot_group_levels)
y_map <- setNames(seq_along(y_groups), y_groups)

scale_gene_expression <- function(x) {
  if (length(unique(x)) <= 1 || stats::sd(x) == 0) return(rep(0, length(x)))
  as.numeric(scale(x))
}

dot_df <- do.call(
  rbind,
  lapply(seq_len(nrow(marker_table)), function(marker_i) {
    marker_id <- marker_table$marker_id[marker_i]
    do.call(
      rbind,
      lapply(dotplot_group_levels, function(group_name) {
        cells_use <- rownames(meta)[meta$stage_celltype_plot == group_name]
        values <- as.numeric(expr_dot[marker_id, cells_use])
        data.frame(
          marker_id = marker_id,
          dotplot_group = group_name,
          avg_exp = mean(values),
          pct_exp = mean(values > lower_detection_limit) * 100,
          group_y = unname(y_map[group_name]),
          stringsAsFactors = FALSE
        )
      })
    )
  })
)
dot_df$avg_exp_scaled <- ave(dot_df$avg_exp, dot_df$marker_id, FUN = scale_gene_expression)
dot_df$avg_exp_scaled <- pmax(pmin(dot_df$avg_exp_scaled, color_limit), -color_limit)


#5. ===========Build the full and split Fig2B plots
make_dotplot <- function(
  marker_groups,
  plot_width,
  show_legend = TRUE,
  show_panel_label = FALSE,
  legend_x = 0.970,
  row_bullet_text_size = celltype_axis_text_size
) {
  marker_subset <- marker_table[marker_table$marker_group %in% marker_groups, , drop = FALSE]
  marker_subset$gene_index <- seq_len(nrow(marker_subset))
  dot_subset <- dot_df[dot_df$marker_id %in% marker_subset$marker_id, , drop = FALSE]
  dot_subset$gene_index <- marker_subset$gene_index[
    match(dot_subset$marker_id, marker_subset$marker_id)
  ]

  annotation_y <- length(y_groups) + 0.50
  label_y <- length(y_groups) + 0.842
  marker_count <- nrow(marker_subset)
  x_lower <- 0.5408 - 0.01 * marker_count
  x_upper <- 1.03 * marker_count + 0.4576
  panel_width_pt <- plot_width * 72 - plot_margin_left - plot_margin_right
  point_per_x_unit <- panel_width_pt / (x_upper - x_lower)
  annotation_df <- aggregate(gene_index ~ marker_group, data = marker_subset, FUN = mean)
  annotation_df$label_x <- annotation_df$gene_index - 0.210 / point_per_x_unit
  y_label_df <- data.frame(
    label = y_groups,
    group_y = seq_along(y_groups) + 0.060,
    label_color = unname(stage_celltype_colors[y_groups]),
    label_x = x_lower - 0.900 / point_per_x_unit,
    bullet_x = x_lower + unname(row_bullet_offset_pt[y_groups]) / point_per_x_unit,
    stringsAsFactors = FALSE
  )

  ggplot() +
    geom_tile(
      data = marker_subset,
      aes(x = gene_index, y = annotation_y, fill = marker_group),
      height = 0.13,
      width = 0.92
    ) +
    geom_text(
      data = annotation_df,
      aes(x = label_x, y = label_y, label = marker_group),
      angle = 35,
      hjust = 0,
      vjust = 0.5,
      size = top_label_text_size,
      size.unit = "pt",
      family = font_family,
      fontface = "plain",
      color = "black"
    ) +
    geom_point(
      data = dot_subset,
      aes(x = gene_index, y = group_y, size = pct_exp, color = avg_exp_scaled),
      alpha = 0.95,
      stroke = 0
    ) +
    geom_text(
      data = y_label_df,
      aes(x = label_x, y = group_y, label = label),
      inherit.aes = FALSE,
      hjust = 1,
      family = font_family,
      fontface = "plain",
      size = celltype_axis_text_size,
      size.unit = "pt",
      color = "black"
    ) +
    # Draw the row colour key as a geometric point rather than a Unicode
    # bullet.  This avoids corrupted glyphs on R installations whose startup
    # locale is not UTF-8 while preserving the intended filled-circle key.
    geom_point(
      data = y_label_df,
      aes(x = bullet_x, y = group_y),
      inherit.aes = FALSE,
      shape = 16,
      size = 0.56 * row_bullet_text_size / ggplot2::.pt,
      color = y_label_df$label_color,
      show.legend = FALSE
    ) +
    scale_x_continuous(
      breaks = marker_subset$gene_index,
      labels = marker_subset$gene_label,
      expand = expansion(mult = c(0.010, 0.030))
    ) +
    scale_y_continuous(
      breaks = seq_along(y_groups),
      labels = y_groups,
      limits = c(0.55, length(y_groups) + 1.18),
      expand = c(0, 0)
    ) +
    scale_fill_manual(values = strip_colors, guide = "none") +
    scale_size_area(
      max_size = max_dot_size,
      limits = c(0, 100),
      breaks = c(0, 25, 50, 75, 100),
      name = "% Exp."
    ) +
    scale_color_gradientn(
      colors = c("#D9D9D9", "#D894C4", "#8F1D82"),
      limits = c(-color_limit, color_limit),
      breaks = c(-1, -0.5, 0, 0.5, 1),
      name = "Avg. Exp."
    ) +
    guides(
      color = guide_colorbar(
        order = 1,
        title.position = "top",
        barwidth = grid::unit(0.193, "cm"),
        barheight = grid::unit(0.824, "cm")
      ),
      size = guide_legend(
        order = 2,
        title.position = "top",
        override.aes = list(color = "black")
      )
    ) +
    coord_cartesian(
      xlim = c(0.54, marker_count + 0.46),
      clip = "off"
    ) +
    theme_classic(base_size = 7, base_family = font_family) +
    labs(
      x = NULL,
      y = NULL,
      tag = if (show_panel_label) "B" else NULL
    ) +
    theme(
      axis.text.x = element_text(
        angle = 60,
        hjust = 1,
        vjust = 1,
        color = "black",
        size = gene_axis_text_size,
        family = font_family,
        face = "italic",
        margin = margin(t = 0, unit = "pt")
      ),
      axis.text.y = element_blank(),
      axis.ticks.x = element_line(linewidth = 0.187),
      axis.ticks.y = element_blank(),
      axis.ticks.length = grid::unit(1.203, "pt"),
      axis.line = element_line(linewidth = 0.263),
      legend.position = if (show_legend) "inside" else "none",
      legend.position.inside = c(legend_x, 0.247),
      legend.justification = c(0.40, 0.5),
      legend.title = element_text(
        size = legend_title_text_size,
        family = font_family,
        face = "plain"
      ),
      legend.text = element_text(
        size = legend_text_size,
        family = font_family,
        face = "plain"
      ),
      legend.key.height = grid::unit(0.21, "cm"),
      legend.key.width = grid::unit(0.233, "cm"),
      legend.spacing.y = grid::unit(-0.15, "cm"),
      legend.box.spacing = grid::unit(0.06, "cm"),
      legend.background = element_blank(),
      legend.box.background = element_blank(),
      legend.key = element_blank(),
      plot.tag.position = c(-0.11548, 1.02861),
      plot.tag = element_text(
        family = "Times New Roman",
        face = "plain",
        size = panel_label_text_size,
        hjust = 0,
        vjust = 1
      ),
      plot.margin = margin(
        plot_margin_top,
        plot_margin_right,
        plot_margin_bottom,
        plot_margin_left,
        unit = "pt"
      )
    )
}

p_full <- make_dotplot(
  celltype_levels,
  plot_width = full_width,
  show_legend = TRUE,
  show_panel_label = TRUE,
  legend_x = 0.9965,
  row_bullet_text_size = full_row_bullet_text_size
)
p_part1 <- make_dotplot(
  c("OV", "PsD", "M.PsD", "L.PsD"),
  plot_width = part1_width,
  show_legend = FALSE,
  show_panel_label = FALSE
)
p_part2 <- make_dotplot(
  c("M.PsC", "L.PsC", "IPhC", "IBC", "IPC", "HeC"),
  plot_width = part2_width,
  show_legend = FALSE,
  show_panel_label = FALSE
)
p_part3 <- make_dotplot(
  c("IHC", "iOHC", "OHC"),
  plot_width = part3_width,
  show_legend = TRUE,
  show_panel_label = FALSE,
  legend_x = 0.98
)


#6. ===========Save editable PDFs and matched 1201 dpi PNGs
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

save_plot_pair <- function(plot_obj, file_stem, width, height) {
  ggplot2::ggsave(
    filename = file.path(output_dir, paste0(file_stem, ".pdf")),
    plot = plot_obj,
    width = width,
    height = height,
    units = "in",
    device = grDevices::cairo_pdf,
    bg = "white",
    limitsize = FALSE
  )

  ggplot2::ggsave(
    filename = file.path(output_dir, paste0(file_stem, ".png")),
    plot = plot_obj,
    width = width,
    height = height,
    units = "in",
    dpi = 1201,
    device = ragg::agg_png,
    background = "white",
    limitsize = FALSE
  )
}

save_plot_pair(p_full, "fig2b_marker_dotplot_full", full_width, full_height)
save_plot_pair(p_part1, "fig2b_marker_dotplot_part1", part1_width, split_height)
save_plot_pair(p_part2, "fig2b_marker_dotplot_part2", part2_width, split_height)
save_plot_pair(p_part3, "fig2b_marker_dotplot_part3", part3_width, split_height)

cat("Fig2B saved: four editable PDFs and four matched 1201 dpi PNGs.\n")
