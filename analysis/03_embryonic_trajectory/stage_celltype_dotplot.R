#1. ===========Load packages and parameters
set.seed(0)
options(stringsAsFactors = FALSE)

required_pkgs <- c("rhdf5", "Matrix", "ggplot2", "grid")
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
output_dir <- file.path(results_root, "03_embryonic_trajectory/figures")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

file_prefix <- file.path(output_dir, "Embryonic_E9_E16_medial_lateral_stage_marker_dotplot_compact_V2")

stage_order <- c("E9.5", "E11.5", "E13.5", "E14.5", "E16.5")
stage_short_map <- c(
  "E9.5" = "E9",
  "E11.5" = "E11",
  "E13.5" = "E13.5",
  "E14.5" = "E14",
  "E16.5" = "E16"
)

celltype_levels <- c(
  "OV", "PsD", "M.PsD", "L.PsD",
  "M.PsC", "L.PsC", "IPhC", "IBC", "IPC", "HeC",
  "IHC", "iOHC", "OHC"
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

lower_detection_limit <- 0.1
min_cells_marker <- 3
color_limit <- 1
max_dot_size <- 4.1

gene_axis_text_size <- 7.8
celltype_axis_text_size <- 9.0
top_label_text_size <- 2.75
legend_title_text_size <- 7.2
legend_text_size <- 6.4


#2. ===========Define marker genes
make_marker_item <- function(group, label, aliases = NULL) {
  if (is.null(aliases)) aliases <- label
  list(group = group, label = label, aliases = aliases)
}

marker_items <- list(
  make_marker_item("OV", "Epcam"),
  make_marker_item("OV", "Pax2"),
  make_marker_item("OV", "Foxg1"),
  make_marker_item("OV", "Gata3"),
  make_marker_item("OV", "Isl1"),
  make_marker_item("PsD", "Sox2"),
  make_marker_item("PsD", "Sox9"),
  make_marker_item("PsD", "Lgr5"),
  make_marker_item("PsD", "Jag1"),
  make_marker_item("PsD", "Fgf20"),
  make_marker_item("M.PsD", "Fgf10"),
  make_marker_item("M.PsD", "Tbx2"),
  make_marker_item("M.PsD", "Sulf1"),
  make_marker_item("L.PsD", "Bmp4"),
  make_marker_item("L.PsD", "Rorb"),
  make_marker_item("L.PsD", "Fgfr3"),
  make_marker_item("M.PsC", "Anxa5"),
  make_marker_item("M.PsC", "Crym"),
  make_marker_item("M.PsC", "Ntf3"),
  make_marker_item("L.PsC", "Prox1"),
  make_marker_item("L.PsC", "Igfbp3"),
  make_marker_item("L.PsC", "Socs2"),
  make_marker_item("IPhC", "Hes5"),
  make_marker_item("IPhC", "Gjb2"),
  make_marker_item("IBC", "Id2"),
  make_marker_item("IBC", "Id3"),
  make_marker_item("IPC", "Npy"),
  make_marker_item("IPC", "S100b"),
  make_marker_item("HeC", "Gata2"),
  make_marker_item("HeC", "Kazald1"),
  make_marker_item("IHC", "Atoh1"),
  make_marker_item("IHC", "Gfi1"),
  make_marker_item("IHC", "Fgf8"),
  make_marker_item("IHC", "Pvalb"),
  make_marker_item("IHC", "Myo7a"),
  make_marker_item("iOHC", "Hes6"),
  make_marker_item("iOHC", "Ccer2"),
  make_marker_item("iOHC", "Pcp4"),
  make_marker_item("iOHC", "Lhfpl5"),
  make_marker_item("OHC", "Ikzf2"),
  make_marker_item("OHC", "Calb1"),
  make_marker_item("OHC", "Kcnq4"),
  make_marker_item("OHC", "Myo6"),
  make_marker_item("OHC", "Bcl11b")
)


#3. ===========Read h5AD sparse expression and metadata
read_h5ad_categorical <- function(input_h5ad, key) {
  categories <- rhdf5::h5read(input_h5ad, paste0("obs/", key, "/categories"))
  codes <- rhdf5::h5read(input_h5ad, paste0("obs/", key, "/codes")) + 1L
  categories[codes]
}

read_trajectory_h5ad <- function(input_h5ad, branch_name) {
  if (!file.exists(input_h5ad)) {
    stop("Input h5AD not found: ", input_h5ad)
  }

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
meta$stage_celltype_plot <- paste(
  meta$stage_short,
  as.character(meta$celltype_plot),
  sep = "_"
)


#4. ===========Resolve and filter marker genes
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

detected_cells <- Matrix::rowSums(expr_mat[marker_table$gene_id, , drop = FALSE] > lower_detection_limit)
marker_table <- marker_table[detected_cells >= min_cells_marker, , drop = FALSE]
marker_table$marker_group <- factor(marker_table$marker_group, levels = celltype_levels)
marker_table <- marker_table[order(marker_table$marker_group), , drop = FALSE]
marker_table$gene_index <- seq_len(nrow(marker_table))
marker_table$marker_id <- paste(marker_table$marker_group, marker_table$gene_label, sep = "__")

expr_dot <- expr_mat[marker_table$gene_id, , drop = FALSE]
rownames(expr_dot) <- marker_table$marker_id


#5. ===========Calculate dotplot statistics
scale_gene_expression <- function(x) {
  if (length(unique(x)) <= 1 || stats::sd(x) == 0) {
    return(rep(0, length(x)))
  }
  as.numeric(scale(x))
}

stage_celltype_levels <- unlist(lapply(unname(stage_short_map[stage_order]), function(stage_name) {
  paste(stage_name, celltype_levels, sep = "_")
}))

dotplot_group_levels <- stage_celltype_levels[
  stage_celltype_levels %in% unique(meta$stage_celltype_plot)
]

y_groups <- rev(dotplot_group_levels)
y_map <- setNames(seq_along(y_groups), y_groups)

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
          gene_label = marker_table$gene_label[marker_i],
          marker_group = as.character(marker_table$marker_group[marker_i]),
          dotplot_group = group_name,
          avg_exp = mean(values),
          pct_exp = mean(values > lower_detection_limit) * 100,
          gene_index = marker_table$gene_index[marker_i],
          group_y = unname(y_map[group_name]),
          stringsAsFactors = FALSE
        )
      })
    )
  })
)

dot_df$avg_exp_scaled <- ave(dot_df$avg_exp, dot_df$marker_id, FUN = scale_gene_expression)
dot_df$avg_exp_scaled <- pmax(pmin(dot_df$avg_exp_scaled, color_limit), -color_limit)


#6. ===========Plot horizontal dotplot
annotation_y <- length(y_groups) + 0.50
label_y <- length(y_groups) + 0.80

annotation_df <- aggregate(gene_index ~ marker_group, data = marker_table, FUN = mean)
annotation_df$marker_group <- factor(annotation_df$marker_group, levels = celltype_levels)

p_horizontal <- ggplot() +
  geom_tile(
    data = marker_table,
    aes(x = gene_index, y = annotation_y, fill = marker_group),
    height = 0.13,
    width = 0.92
  ) +
  geom_text(
    data = annotation_df,
    aes(x = gene_index, y = label_y, label = marker_group),
    angle = 35,
    hjust = 0,
    vjust = 0.5,
    size = top_label_text_size,
    color = "black"
  ) +
  geom_point(
    data = dot_df,
    aes(x = gene_index, y = group_y, size = pct_exp, color = avg_exp_scaled),
    alpha = 0.95,
    stroke = 0
  ) +
  scale_x_continuous(
    breaks = marker_table$gene_index,
    labels = marker_table$gene_label,
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
    size = guide_legend(title.position = "top", override.aes = list(color = "black")),
    color = guide_colorbar(
      title.position = "top",
      barwidth = grid::unit(0.34, "cm"),
      barheight = grid::unit(1.45, "cm")
    )
  ) +
  coord_cartesian(clip = "off") +
  theme_classic(base_size = 8.5) +
  labs(title = NULL, x = NULL, y = NULL) +
  theme(
    axis.text.x = element_text(
      angle = 60,
      hjust = 1,
      vjust = 1,
      color = "black",
      size = gene_axis_text_size
    ),
    axis.text.y = element_text(color = "black", size = celltype_axis_text_size),
    axis.ticks.x = element_line(linewidth = 0.25),
    axis.ticks.y = element_blank(),
    axis.line = element_line(linewidth = 0.35),
    legend.position = "right",
    legend.title = element_text(size = legend_title_text_size),
    legend.text = element_text(size = legend_text_size),
    legend.key.height = grid::unit(0.28, "cm"),
    legend.key.width = grid::unit(0.24, "cm"),
    legend.spacing.y = grid::unit(0.04, "cm"),
    legend.box.spacing = grid::unit(0.12, "cm"),
    plot.margin = margin(46, 8, 12, 6)
  )


#7. ===========Save compact horizontal dotplot
save_plot_pair <- function(plot_obj, output_prefix, width, height) {
  ggplot2::ggsave(
    filename = paste0(output_prefix, ".pdf"),
    plot = plot_obj,
    width = width,
    height = height,
    units = "in",
    device = grDevices::cairo_pdf,
    bg = "white"
  )

  ggplot2::ggsave(
    filename = paste0(output_prefix, ".png"),
    plot = plot_obj,
    width = width,
    height = height,
    units = "in",
    dpi = 1200,
    bg = "white",
    limitsize = FALSE
  )
}

horizontal_width <- max(9.8, 0.155 * nrow(marker_table) + 3.1)
horizontal_height <- max(5.3, 0.18 * length(dotplot_group_levels) + 2.0)

save_plot_pair(
  p_horizontal,
  paste0(file_prefix, "_horizontal"),
  width = horizontal_width,
  height = horizontal_height
)

cat(
  "Compact embryonic marker dotplot saved. cells=", ncol(expr_mat),
  ", markers=", nrow(marker_table),
  ", groups=", length(dotplot_group_levels),
  ", output=", file_prefix,
  "_horizontal.pdf/.png\n",
  sep = ""
)
