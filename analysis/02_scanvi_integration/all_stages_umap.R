suppressPackageStartupMessages({
  library(rhdf5)
  library(uwot)
  library(ggplot2)
})

# This helper is called after the Python scANVI workflow saves the h5ad.
# It does not change the model, labels, genes, or h5ad content. It only redraws
# the final full-period stage_celltype UMAP from the saved X_scanvi embedding.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop(
    "Usage: Rscript all_stages_umap.R <input_h5ad> <output_dir>"
  )
}

h5ad_file <- args[[1]]
output_dir <- args[[2]]

if (!file.exists(h5ad_file)) {
  stop("Input h5ad does not exist: ", h5ad_file)
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)


# 1. ===========UMAP and plot parameters===========
RANDOM_SEED <- 0
UMAP_NEIGHBORS <- 30
UMAP_MIN_DIST <- 0.35
UMAP_METRIC <- "euclidean"
UMAP_INIT <- "spectral"
UMAP_THREADS <- 1
UMAP_VERBOSE <- FALSE

POINT_SIZE <- 0.48
POINT_ALPHA <- 0.82

CELLTYPE_LABEL_SIZE <- 8.0 / 2.845276
LABEL_OUTLINE_X <- 0.045
LABEL_OUTLINE_Y <- 0.045

TITLE_SIZE <- 17
AXIS_TITLE_SIZE <- 17
AXIS_TEXT_SIZE <- 9.5
PANEL_BORDER_SIZE <- 0.75
TICK_SIZE <- 0.45
TICK_LENGTH_CM <- 0.12

LABEL_PDF_WIDTH <- 8.2
LABEL_PDF_HEIGHT <- 8.1
LEGEND_PDF_WIDTH <- 10.4
LEGEND_PDF_HEIGHT <- 8.1
PNG_DPI <- 1201

LEGEND_TEXT_SIZE <- 6.6
LEGEND_POINT_SIZE <- 2.2
LEGEND_NCOL <- 2
LEGEND_BYROW <- FALSE
LEGEND_KEY_BOX_SIZE_CM <- 0.20
LEGEND_SPACING_X_CM <- 0.08
LEGEND_SPACING_Y_CM <- 0.02
LEGEND_MARGIN_LEFT_PT <- 4

stage_order <- c(
  "E9.5", "E11.5", "E13.5", "E14.5", "E16.5",
  "P1", "P7", "P14", "P28"
)


# 2. ===========Minimal h5ad readers===========
h5_table <- rhdf5::h5ls(h5ad_file, recursive = TRUE)
h5_paths <- sub("^//", "/", paste0(h5_table$group, "/", h5_table$name))

h5_has <- function(path) {
  paste0("/", sub("^/+", "", path)) %in% h5_paths
}

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
  mat <- rhdf5::h5read(h5ad_file, paste0("obsm/", obsm_key))
  cell_id <- as.character(rhdf5::h5read(h5ad_file, "obs/_index"))

  if (nrow(mat) == length(cell_id)) {
    rownames(mat) <- cell_id
  } else if (ncol(mat) == length(cell_id)) {
    mat <- t(mat)
    rownames(mat) <- cell_id
  } else {
    stop("Unexpected obsm matrix shape for: ", obsm_key)
  }

  mat
}

format_stage_celltype <- function(stage_celltype) {
  gsub("_", " ", as.character(stage_celltype))
}

extract_celltype_label <- function(stage_celltype, stage) {
  stage_celltype <- as.character(stage_celltype)
  stage <- as.character(stage)
  label <- stage_celltype

  for (i in seq_along(label)) {
    if (is.na(label[i]) || is.na(stage[i])) {
      next
    }

    underscore_prefix <- paste0(stage[i], "_")
    space_prefix <- paste0(stage[i], " ")

    if (startsWith(label[i], underscore_prefix)) {
      label[i] <- substring(label[i], nchar(underscore_prefix) + 1L)
    } else if (startsWith(label[i], space_prefix)) {
      label[i] <- substring(label[i], nchar(space_prefix) + 1L)
    } else {
      label[i] <- sub("^[^_]+_", "", label[i])
    }
  }

  format_stage_celltype(label)
}

make_label_outline_df <- function(label_df, outline_x, outline_y) {
  outline_offsets <- expand.grid(
    dx = c(-outline_x, 0, outline_x),
    dy = c(-outline_y, 0, outline_y)
  )
  outline_offsets <- outline_offsets[
    outline_offsets$dx != 0 | outline_offsets$dy != 0,
    ,
    drop = FALSE
  ]

  outline_df <- label_df[rep(seq_len(nrow(label_df)), each = nrow(outline_offsets)), ]
  outline_df$dx <- rep(outline_offsets$dx, times = nrow(label_df))
  outline_df$dy <- rep(outline_offsets$dy, times = nrow(label_df))
  outline_df
}

celltype_family <- function(celltype_label) {
  if (celltype_label %in% c("OHC")) return("outer_hair")
  if (celltype_label %in% c("IHC", "iOHC")) return("inner_hair")
  if (celltype_label %in% c("KO-1", "KO-2", "KO-3", "KO-4", "L KO", "M KO", "ML KO")) return("ko")
  if (celltype_label %in% c("GER", "L.GER", "GER/Hmgn2", "GER/KO", "LER")) return("ger_ler")
  if (celltype_label %in% c("L.PsC", "M.PsC")) return("psc")
  if (celltype_label %in% c("DC", "DC-1/2", "DC-3")) return("dc")
  if (celltype_label %in% c("IPC", "IPhC", "ISC", "IdC")) return("pillar_support")
  if (celltype_label %in% c("HeC", "IB", "PC", "OSC", "CC/OSC", "CC/OS", "OPC")) return("support_epithelium")
  if (celltype_label %in% c("OV epithelial cells", "Prosensory domain", "Lateral domain", "Medial domain")) return("early_domain")
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
  if (is.null(colors)) {
    colors <- palettes$other
  }
  if (n == 1L) {
    return(colors[ceiling(length(colors) / 2)])
  }
  grDevices::colorRampPalette(colors)(n)
}

make_family_palette <- function(stage_celltype_order, stage_celltype_df) {
  stage_celltype_df <- stage_celltype_df[
    match(stage_celltype_order, stage_celltype_df$stage_celltype),
    ,
    drop = FALSE
  ]
  stage_celltype_df$family <- vapply(
    stage_celltype_df$celltype_label,
    celltype_family,
    character(1)
  )

  plot_colors <- character(nrow(stage_celltype_df))
  for (family in unique(stage_celltype_df$family)) {
    family_index <- which(stage_celltype_df$family == family)
    family_colors <- family_color_values(family, length(family_index))
    plot_colors[family_index] <- family_colors
  }

  setNames(plot_colors, stage_celltype_df$stage_celltype)
}


# 3. ===========Read metadata and X_scanvi===========
cell_id <- as.character(rhdf5::h5read(h5ad_file, "obs/_index"))

meta <- data.frame(
  cell_id = cell_id,
  stage = read_obs_col("stage"),
  stage_celltype = read_obs_col("stage_celltype"),
  stringsAsFactors = FALSE
)

x_scanvi <- read_obsm("X_scanvi")
x_scanvi <- x_scanvi[meta$cell_id, , drop = FALSE]


# 4. ===========Recompute full-period UMAP from X_scanvi===========
set.seed(RANDOM_SEED)

umap_mat <- uwot::umap(
  x_scanvi,
  n_neighbors = UMAP_NEIGHBORS,
  min_dist = UMAP_MIN_DIST,
  metric = UMAP_METRIC,
  init = UMAP_INIT,
  ret_model = FALSE,
  n_threads = UMAP_THREADS,
  verbose = UMAP_VERBOSE
)

plot_df <- data.frame(
  cell_id = meta$cell_id,
  UMAP_1 = umap_mat[, 1],
  UMAP_2 = umap_mat[, 2],
  stage = meta$stage,
  stage_celltype = meta$stage_celltype,
  stringsAsFactors = FALSE
)

stage_celltype_order <- unlist(lapply(stage_order, function(stage) {
  sort(unique(plot_df$stage_celltype[plot_df$stage == stage]))
}), use.names = FALSE)

stage_celltype_order <- unique(stage_celltype_order)
stage_celltype_order <- c(
  stage_celltype_order,
  sort(setdiff(unique(plot_df$stage_celltype), stage_celltype_order))
)

plot_df$stage_celltype <- factor(
  plot_df$stage_celltype,
  levels = stage_celltype_order
)

plot_df$celltype_label <- extract_celltype_label(
  plot_df$stage_celltype,
  plot_df$stage
)


# 5. ===========Labels and colors===========
label_df <- aggregate(
  cbind(UMAP_1, UMAP_2) ~ celltype_label,
  data = plot_df,
  FUN = median
)

names(label_df)[names(label_df) == "celltype_label"] <- "label_text"

label_order <- c(
  "OV epithelial cells", "Prosensory domain", "Lateral domain", "Medial domain",
  "GER", "GER/Hmgn2", "GER/KO", "L.GER", "LER", "L.PsC", "M.PsC",
  "IHC", "iOHC", "OHC", "IPC", "IPhC", "ISC", "IdC",
  "HeC", "IB", "PC", "OSC", "CC/OSC", "CC/OS", "OPC",
  "DC", "DC-1/2", "DC-3", "KO-1", "KO-2", "KO-3", "KO-4",
  "L KO", "M KO", "ML KO"
)

label_df$label_text <- factor(
  label_df$label_text,
  levels = c(label_order, sort(setdiff(as.character(label_df$label_text), label_order)))
)
label_df <- label_df[order(label_df$label_text), , drop = FALSE]
label_df$label_text <- as.character(label_df$label_text)

manual_label_nudge <- data.frame(
  label_text = c(
    "OV epithelial cells", "Prosensory domain", "Lateral domain", "Medial domain",
    "IHC", "OHC", "HeC", "ISC", "IdC", "IPhC", "IPC", "OPC",
    "LER", "L.PsC", "M.PsC", "DC", "OSC", "CC/OSC",
    "DC-1/2", "DC-3", "KO-1", "KO-2", "KO-3", "KO-4", "L KO", "M KO", "ML KO"
  ),
  shift_UMAP_1 = c(
    -0.85, -0.20, 0.10, 0.45,
    -0.20, -0.20, 0.15, -0.10, 0.12, 0.18, -0.14, -0.45,
    0.10, -0.05, 0.05, 0.12, -0.05, 0.10,
    0.45, -0.25, 0.00, 0.10, 0.10, 0.00, 0.00, 0.00, 0.00
  ),
  shift_UMAP_2 = c(
    0.28, -0.18, 0.15, -0.50,
    0.12, -0.10, 0.14, -0.10, 0.12, 0.12, -0.10, 0.25,
    0.12, -0.10, -0.08, 0.10, 0.12, 0.12,
    -0.05, -0.20, -0.08, 0.08, -0.08, 0.00, 0.08, 0.08, -0.08
  ),
  stringsAsFactors = FALSE
)

nudge_index <- match(label_df$label_text, manual_label_nudge$label_text)
has_nudge <- !is.na(nudge_index)
label_df$UMAP_1[has_nudge] <- label_df$UMAP_1[has_nudge] +
  manual_label_nudge$shift_UMAP_1[nudge_index[has_nudge]]
label_df$UMAP_2[has_nudge] <- label_df$UMAP_2[has_nudge] +
  manual_label_nudge$shift_UMAP_2[nudge_index[has_nudge]]

label_outline_df <- make_label_outline_df(
  label_df,
  outline_x = LABEL_OUTLINE_X,
  outline_y = LABEL_OUTLINE_Y
)

stage_celltype_df <- unique(
  plot_df[, c("stage_celltype", "stage", "celltype_label")]
)
stage_celltype_df$stage_celltype <- as.character(stage_celltype_df$stage_celltype)
stage_celltype_df <- stage_celltype_df[
  match(stage_celltype_order, stage_celltype_df$stage_celltype),
  ,
  drop = FALSE
]

plot_colors <- make_family_palette(
  stage_celltype_order = stage_celltype_order,
  stage_celltype_df = stage_celltype_df
)


# 6. ===========Shared plot theme===========
base_umap <- ggplot(
  plot_df,
  aes(x = UMAP_1, y = UMAP_2, color = stage_celltype)
) +
  geom_point(size = POINT_SIZE, alpha = POINT_ALPHA, stroke = 0) +
  scale_color_manual(
    values = plot_colors,
    breaks = stage_celltype_order,
    labels = format_stage_celltype,
    name = NULL
  ) +
  coord_equal() +
  labs(
    title = "scANVI: refined cell type",
    x = "UMAP1",
    y = "UMAP2"
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(size = TITLE_SIZE, hjust = 0.5, color = "black"),
    axis.title = element_text(size = AXIS_TITLE_SIZE, color = "black"),
    axis.text = element_text(size = AXIS_TEXT_SIZE, color = "black"),
    axis.ticks = element_line(linewidth = TICK_SIZE, color = "black"),
    axis.ticks.length = grid::unit(TICK_LENGTH_CM, "cm"),
    axis.line = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = PANEL_BORDER_SIZE)
  )

celltype_label_outline_layer <- geom_text(
  data = label_outline_df,
  aes(x = UMAP_1 + dx, y = UMAP_2 + dy, label = label_text),
  inherit.aes = FALSE,
  color = "white",
  size = CELLTYPE_LABEL_SIZE,
  fontface = "plain"
)

celltype_label_layer <- geom_text(
  data = label_df,
  aes(x = UMAP_1, y = UMAP_2, label = label_text),
  inherit.aes = FALSE,
  color = "black",
  size = CELLTYPE_LABEL_SIZE,
  fontface = "plain"
)

p_label <- base_umap +
  celltype_label_outline_layer +
  celltype_label_layer +
  theme(legend.position = "none")

p_legend <- base_umap +
  celltype_label_outline_layer +
  celltype_label_layer +
  guides(
    color = guide_legend(
      ncol = LEGEND_NCOL,
      byrow = LEGEND_BYROW,
      override.aes = list(size = LEGEND_POINT_SIZE, alpha = 1)
    )
  ) +
  theme(
    legend.position = "right",
    legend.justification = "top",
    legend.direction = "vertical",
    legend.text = element_text(size = LEGEND_TEXT_SIZE, color = "black"),
    legend.key = element_blank(),
    legend.key.size = grid::unit(LEGEND_KEY_BOX_SIZE_CM, "cm"),
    legend.spacing.x = grid::unit(LEGEND_SPACING_X_CM, "cm"),
    legend.spacing.y = grid::unit(LEGEND_SPACING_Y_CM, "cm"),
    legend.margin = margin(0, 0, 0, LEGEND_MARGIN_LEFT_PT),
    legend.background = element_blank(),
    legend.box.background = element_blank(),
    legend.title = element_blank()
  )


# 7. ===========Save figures===========
label_pdf <- file.path(output_dir, "all_stages_umap_labels.pdf")
legend_pdf <- file.path(output_dir, "all_stages_umap_legend.pdf")
label_png <- "all_stages_umap_labels.png"

ggsave(
  filename = file.path(output_dir, label_png),
  plot = p_label,
  width = LABEL_PDF_WIDTH,
  height = LABEL_PDF_HEIGHT,
  units = "in",
  dpi = PNG_DPI,
  bg = "white"
)

ggsave(
  filename = label_pdf,
  plot = p_label,
  width = LABEL_PDF_WIDTH,
  height = LABEL_PDF_HEIGHT,
  units = "in",
  device = grDevices::pdf
)

ggsave(
  filename = legend_pdf,
  plot = p_legend,
  width = LEGEND_PDF_WIDTH,
  height = LEGEND_PDF_HEIGHT,
  units = "in",
  device = grDevices::pdf
)

cat("Saved label PDF: ", label_pdf, "\n", sep = "")
cat("Saved legend PDF: ", legend_pdf, "\n", sep = "")
