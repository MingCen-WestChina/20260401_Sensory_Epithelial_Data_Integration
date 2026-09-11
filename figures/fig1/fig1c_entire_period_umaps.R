#!/usr/bin/env Rscript

# Reproduce the cell-state and developmental-stage UMAPs used in Figure 1C.

set.seed(0)
options(stringsAsFactors = FALSE)

required_pkgs <- c("rhdf5", "uwot", "ggplot2", "ragg")
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
})

#1. ===========Paths and parameters===========
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/")))
  }
  normalizePath(getwd(), winslash = "/")
}

env_path <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  normalizePath(if (nzchar(value)) value else default, winslash = "/", mustWork = FALSE)
}

script_dir <- get_script_dir()
project_root <- normalizePath(file.path(script_dir, "../.."), winslash = "/", mustWork = TRUE)
data_dir <- env_path("COCHLEA_DATA_DIR", file.path(project_root, "Data"))
results_dir <- env_path("COCHLEA_RESULTS_DIR", file.path(project_root, "results"))
output_dir <- file.path(results_dir, "figures", "fig1")
resolve_input_path <- function(candidates) {
  existing <- candidates[file.exists(candidates)]
  if (length(existing) > 0) existing[[1]] else candidates[[1]]
}
h5ad_file <- resolve_input_path(c(
  file.path(results_dir, "02_scanvi_integration/all_stages/all_stages_scanvi.h5ad"),
  file.path(data_dir, "Integration_3types_after_training/Entire period files/all_retained_scanvi_HVG5000_integrated.h5ad"),
  file.path(data_dir, "Entire period/EntirePeriod_scanvi_HVG5000/all_retained_scanvi_HVG5000_integrated.h5ad")
))
check_inputs <- "--check-inputs" %in% commandArgs(trailingOnly = TRUE)

RANDOM_SEED <- 0
UMAP_NEIGHBORS <- 30
UMAP_MIN_DIST <- 0.35
UMAP_METRIC <- "euclidean"
UMAP_INIT <- "spectral"
POINT_SIZE <- 0.48
POINT_ALPHA <- 0.82
FONT_FAMILY <- Sys.getenv("COCHLEA_FONT_FAMILY", unset = "Arial")
FIGURE_WIDTH_IN <- 7.43653
FIGURE_HEIGHT_IN <- 9.56
stage_order <- c("E9.5", "E11.5", "E13.5", "E14.5", "E16.5", "P1", "P7", "P14", "P28")
stage_colors <- c(
  "E9.5" = "#1F77B4", "E11.5" = "#FF7F0E", "E13.5" = "#2CA02C",
  "E14.5" = "#D62728", "E16.5" = "#9467BD", "P1" = "#8C564B",
  "P7" = "#E377C2", "P14" = "#BCBD22", "P28" = "#17BECF"
)

if (!file.exists(h5ad_file)) stop("Input h5AD not found: ", h5ad_file)

#2. ===========Minimal h5AD readers===========
h5_table <- rhdf5::h5ls(h5ad_file, recursive = TRUE)
h5_paths <- sub("^//", "/", paste0(h5_table$group, "/", h5_table$name))
h5_has <- function(path) paste0("/", sub("^/+", "", path)) %in% h5_paths

read_obs_col <- function(column_name) {
  base_path <- paste0("obs/", column_name)
  categories_path <- paste0(base_path, "/categories")
  codes_path <- paste0(base_path, "/codes")
  if (h5_has(categories_path) && h5_has(codes_path)) {
    categories <- as.character(rhdf5::h5read(h5ad_file, categories_path))
    codes <- as.integer(rhdf5::h5read(h5ad_file, codes_path))
    values <- rep(NA_character_, length(codes))
    valid <- !is.na(codes) & codes >= 0
    values[valid] <- categories[codes[valid] + 1L]
    return(values)
  }
  as.character(rhdf5::h5read(h5ad_file, base_path))
}

read_obsm <- function(obsm_key) {
  matrix_value <- rhdf5::h5read(h5ad_file, paste0("obsm/", obsm_key))
  cell_id <- as.character(rhdf5::h5read(h5ad_file, "obs/_index"))
  if (nrow(matrix_value) == length(cell_id)) {
    rownames(matrix_value) <- cell_id
  } else if (ncol(matrix_value) == length(cell_id)) {
    matrix_value <- t(matrix_value)
    rownames(matrix_value) <- cell_id
  } else {
    stop("Unexpected obsm matrix shape for: ", obsm_key)
  }
  matrix_value
}

format_stage_celltype <- function(value) gsub("_", " ", as.character(value))

extract_celltype <- function(stage_celltype, stage) {
  result <- as.character(stage_celltype)
  for (i in seq_along(result)) {
    underscore_prefix <- paste0(stage[i], "_")
    space_prefix <- paste0(stage[i], " ")
    if (startsWith(result[i], underscore_prefix)) {
      result[i] <- substring(result[i], nchar(underscore_prefix) + 1L)
    } else if (startsWith(result[i], space_prefix)) {
      result[i] <- substring(result[i], nchar(space_prefix) + 1L)
    } else {
      result[i] <- sub("^[^_]+_", "", result[i])
    }
  }
  format_stage_celltype(result)
}

celltype_family <- function(label) {
  if (label %in% "OHC") return("outer_hair")
  if (label %in% c("IHC", "iOHC")) return("inner_hair")
  if (label %in% c("KO-1", "KO-2", "KO-3", "KO-4", "L KO", "M KO", "ML KO")) return("ko")
  if (label %in% c("GER", "L.GER", "GER/Hmgn2", "GER/KO", "LER")) return("ger_ler")
  if (label %in% c("L.PsC", "M.PsC")) return("psc")
  if (label %in% c("DC", "DC-1/2", "DC-3")) return("dc")
  if (label %in% c("IPC", "IPhC", "ISC", "IdC")) return("pillar_support")
  if (label %in% c("HeC", "IB", "PC", "OSC", "CC/OSC", "CC/OS", "OPC")) return("support_epithelium")
  if (label %in% c("OV epithelial cells", "Prosensory domain", "Lateral domain", "Medial domain")) return("early_domain")
  "other"
}

family_color_values <- function(family, n) {
  palettes <- list(
    outer_hair = c("#FFBB78", "#FF7F0E", "#E6550D", "#D55E00", "#A6761D"),
    inner_hair = c("#9ECAE1", "#6BAED6", "#3182BD", "#1F77B4", "#0072B2"),
    ko = c("#DE9ED6", "#CE6DBD", "#A55194", "#7B4173", "#882255"),
    ger_ler = c("#C7E9C0", "#A1D99B", "#74C476", "#31A354", "#117733"),
    psc = c("#CEDB9C", "#B5CF6B", "#8CA252", "#637939"),
    dc = c("#FF9896", "#EE6677", "#D6616B", "#AD494A", "#843C39"),
    pillar_support = c("#DADAEB", "#BCBDDC", "#9E9AC8", "#756BB1", "#393B79"),
    support_epithelium = c("#8DD3C7", "#44AA99", "#1B9E77", "#009E73", "#66A61E"),
    early_domain = c("#FDD0A2", "#E7CB94", "#C49C94", "#8C564B"),
    other = c("#BEBADA", "#BC80BD", "#AA3377")
  )
  colors <- palettes[[family]]
  if (is.null(colors)) colors <- palettes$other
  if (n == 1L) return(colors[ceiling(length(colors) / 2)])
  grDevices::colorRampPalette(colors)(n)
}

#3. ===========Read metadata and integrated latent coordinates===========
cell_id <- as.character(rhdf5::h5read(h5ad_file, "obs/_index"))
meta <- data.frame(
  cell_id = cell_id,
  stage = read_obs_col("stage"),
  stage_celltype = read_obs_col("stage_celltype"),
  stringsAsFactors = FALSE
)
x_scanvi <- read_obsm("X_scanvi")
x_scanvi <- x_scanvi[meta$cell_id, , drop = FALSE]
if (!identical(rownames(x_scanvi), meta$cell_id)) stop("X_scanvi rows and cell IDs are not aligned.")
if (!all(unique(meta$stage) %in% stage_order)) stop("Unexpected developmental stage labels in h5AD.")
if (check_inputs) {
  message("Validated Figure 1C input: ", nrow(meta), " cells and ", ncol(x_scanvi), " latent dimensions.")
  quit(save = "no", status = 0)
}

#4. ===========Compute one deterministic UMAP for both panels===========
set.seed(RANDOM_SEED)
umap_mat <- uwot::umap(
  x_scanvi,
  n_neighbors = UMAP_NEIGHBORS,
  min_dist = UMAP_MIN_DIST,
  metric = UMAP_METRIC,
  init = UMAP_INIT,
  ret_model = FALSE,
  n_threads = 1,
  verbose = FALSE
)
plot_df <- data.frame(
  cell_id = meta$cell_id,
  UMAP1 = umap_mat[, 1],
  UMAP2 = umap_mat[, 2],
  stage = factor(meta$stage, levels = stage_order),
  stage_celltype = meta$stage_celltype,
  stringsAsFactors = FALSE
)
plot_df$celltype <- extract_celltype(plot_df$stage_celltype, plot_df$stage)
stage_celltype_order <- unique(unlist(lapply(stage_order, function(stage) {
  sort(unique(plot_df$stage_celltype[plot_df$stage == stage]))
}), use.names = FALSE))
plot_df$stage_celltype <- factor(plot_df$stage_celltype, levels = stage_celltype_order)

#5. ===========Cell-state labels and colors===========
label_order <- c(
  "OV epithelial cells", "Prosensory domain", "Lateral domain", "Medial domain",
  "GER", "GER/Hmgn2", "GER/KO", "L.GER", "LER", "L.PsC", "M.PsC", "IHC",
  "iOHC", "OHC", "IPC", "IPhC", "ISC", "IdC", "HeC", "IB", "PC", "OSC",
  "CC/OSC", "CC/OS", "OPC", "DC", "DC-1/2", "DC-3", "KO-1", "KO-2",
  "KO-3", "KO-4", "L KO", "M KO", "ML KO"
)
label_df <- aggregate(cbind(UMAP1, UMAP2) ~ celltype, data = plot_df, FUN = median)
label_df$celltype <- factor(label_df$celltype, levels = c(label_order, sort(setdiff(label_df$celltype, label_order))))
label_df <- label_df[order(label_df$celltype), , drop = FALSE]
label_df$celltype <- as.character(label_df$celltype)
manual_nudge <- data.frame(
  celltype = c("OV epithelial cells", "Prosensory domain", "Lateral domain", "Medial domain", "IHC", "OHC", "HeC", "ISC", "IdC", "IPhC", "IPC", "OPC", "LER", "L.PsC", "M.PsC", "DC", "OSC", "CC/OSC", "DC-1/2", "DC-3", "KO-1", "KO-2", "KO-3", "KO-4", "L KO", "M KO", "ML KO"),
  dx = c(-0.85, -0.20, 0.10, 0.45, -0.20, -0.20, 0.15, -0.10, 0.12, 0.18, -0.14, -0.45, 0.10, -0.05, 0.05, 0.12, -0.05, 0.10, 0.45, -0.25, 0.00, 0.10, 0.10, 0.00, 0.00, 0.00, 0.00),
  dy = c(0.28, -0.18, 0.15, -0.50, 0.12, -0.10, 0.14, -0.10, 0.12, 0.12, -0.10, 0.25, 0.12, -0.10, -0.08, 0.10, 0.12, 0.12, -0.05, -0.20, -0.08, 0.08, -0.08, 0.00, 0.08, 0.08, -0.08)
)
nudge_index <- match(label_df$celltype, manual_nudge$celltype)
has_nudge <- !is.na(nudge_index)
label_df$UMAP1[has_nudge] <- label_df$UMAP1[has_nudge] + manual_nudge$dx[nudge_index[has_nudge]]
label_df$UMAP2[has_nudge] <- label_df$UMAP2[has_nudge] + manual_nudge$dy[nudge_index[has_nudge]]

group_df <- unique(plot_df[, c("stage_celltype", "celltype")])
group_df <- group_df[match(stage_celltype_order, as.character(group_df$stage_celltype)), , drop = FALSE]
group_df$family <- vapply(group_df$celltype, celltype_family, character(1))
plot_colors <- character(nrow(group_df))
for (family in unique(group_df$family)) {
  index <- which(group_df$family == family)
  plot_colors[index] <- family_color_values(family, length(index))
}
plot_colors <- setNames(plot_colors, as.character(group_df$stage_celltype))

#6. ===========Draw and save the two Figure 1C components===========
base_theme <- theme_classic(base_size = 11, base_family = FONT_FAMILY) +
  theme(
    axis.title = element_text(size = 17, color = "black"),
    axis.text = element_text(size = 10, color = "black"),
    axis.ticks = element_line(linewidth = 0.45, color = "black"),
    axis.line = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.75)
  )

celltype_plot <- ggplot(plot_df, aes(UMAP1, UMAP2, color = stage_celltype)) +
  geom_point(size = POINT_SIZE, alpha = POINT_ALPHA, stroke = 0) +
  geom_text(data = label_df, aes(UMAP1, UMAP2, label = celltype), inherit.aes = FALSE,
            color = "white", size = 8.0 / 2.845276, family = FONT_FAMILY) +
  geom_text(data = label_df, aes(UMAP1, UMAP2, label = celltype), inherit.aes = FALSE,
            color = "black", size = 8.0 / 2.845276, family = FONT_FAMILY) +
  scale_color_manual(values = plot_colors, breaks = stage_celltype_order,
                     labels = format_stage_celltype, name = NULL) +
  coord_equal() + labs(x = "UMAP1", y = "UMAP2") + base_theme +
  guides(color = guide_legend(ncol = 6, byrow = FALSE,
                              override.aes = list(size = 2.2, alpha = 1))) +
  theme(legend.position = "bottom", legend.justification = "left",
        legend.direction = "horizontal", legend.text = element_text(size = 7),
        legend.key = element_blank(), legend.title = element_blank())

stage_counts <- table(factor(plot_df$stage, levels = stage_order))
stage_labels <- paste0(stage_order, " (n = ", format(as.integer(stage_counts), big.mark = ","), ")")
stage_plot <- ggplot(plot_df, aes(UMAP1, UMAP2, color = stage)) +
  geom_point(size = POINT_SIZE, alpha = POINT_ALPHA, stroke = 0) +
  scale_color_manual(values = stage_colors, breaks = stage_order, labels = stage_labels, name = NULL) +
  coord_equal() + labs(x = "UMAP1", y = "UMAP2") + base_theme +
  guides(color = guide_legend(ncol = 5, byrow = TRUE,
                              override.aes = list(size = 2.2, alpha = 1))) +
  theme(legend.position = "bottom", legend.text = element_text(size = 8.5),
        legend.key = element_blank(), legend.title = element_blank())

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
save_pair <- function(plot_object, stem) {
  ggsave(file.path(output_dir, paste0(stem, ".pdf")), plot_object,
         width = FIGURE_WIDTH_IN, height = FIGURE_HEIGHT_IN, units = "in",
         device = grDevices::cairo_pdf, bg = "white", family = FONT_FAMILY)
  ggsave(file.path(output_dir, paste0(stem, ".png")), plot_object,
         width = FIGURE_WIDTH_IN, height = FIGURE_HEIGHT_IN, units = "in",
         dpi = 1201, device = ragg::agg_png, bg = "white")
}
save_pair(celltype_plot, "fig1c_entire_period_celltype_umap")
save_pair(stage_plot, "fig1c_entire_period_stage_umap")
message("Saved Figure 1C cell-state and stage UMAPs from one shared embedding.")
