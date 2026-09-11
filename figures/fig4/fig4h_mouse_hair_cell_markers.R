#1. ===========Load packages and set paths===========

set.seed(0)
options(stringsAsFactors = FALSE)

required_pkgs <- c("rhdf5", "uwot", "grid", "ragg", "scales")
missing_pkgs <- required_pkgs[
  !vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_pkgs) > 0) {
  stop("Missing R packages: ", paste(missing_pkgs, collapse = ", "))
}

suppressPackageStartupMessages({
  library(rhdf5)
  library(grid)
})

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/")))
  }
  normalizePath(getwd(), winslash = "/")
}

SCRIPT_DIR <- get_script_dir()
PROJECT_ROOT <- normalizePath(
  file.path(SCRIPT_DIR, "..", ".."),
  winslash = "/",
  mustWork = TRUE
)

env_path <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  normalizePath(if (nzchar(value)) value else default, winslash = "/", mustWork = FALSE)
}

DATA_DIR <- env_path("COCHLEA_DATA_DIR", file.path(PROJECT_ROOT, "Data"))
RESULTS_DIR <- env_path("COCHLEA_RESULTS_DIR", file.path(PROJECT_ROOT, "results"))
OUTPUT_DIR <- file.path(RESULTS_DIR, "figures", "fig4")
resolve_input_path <- function(candidates) {
  existing <- candidates[file.exists(candidates)]
  if (length(existing) > 0) existing[[1]] else candidates[[1]]
}
PLOT_DATA_RDS <- resolve_input_path(c(
  file.path(RESULTS_DIR, "05_mouse_ush2a/data/ush2a_hc_plot_data.rds"),
  file.path(DATA_DIR, "USH2A_HCs/ush2a_hc_plot_data.rds")
))
H5AD_FILE <- resolve_input_path(c(
  file.path(RESULTS_DIR, "02_scanvi_integration/all_stages/all_stages_scanvi.h5ad"),
  file.path(DATA_DIR, "Entire period/EntirePeriod_scanvi_HVG5000/all_retained_scanvi_HVG5000_integrated.h5ad"),
  file.path(DATA_DIR, "Integration_3types_after_training/Entire period files/all_retained_scanvi_HVG5000_integrated.h5ad")
))

FULLGENE_H5AD_BY_STAGE <- setNames(
  vapply(
    c("E14.5", "E16.5", "P1", "P7", "P14", "P28"),
    function(stage) resolve_input_path(c(
      file.path(RESULTS_DIR, "01_stage_clustering/h5ad_for_integration", paste0(stage, "_for_scnvi_normalized.h5ad")),
      file.path(DATA_DIR, "h5ad_for_Integration", paste0(stage, "_for_scnvi_normalized.h5ad"))
    )),
    character(1)
  ),
  c("E14.5", "E16.5", "P1", "P7", "P14", "P28")
)

all_inputs <- c(PLOT_DATA_RDS, H5AD_FILE, FULLGENE_H5AD_BY_STAGE)
if (any(!file.exists(all_inputs))) {
  stop("Missing Figure 4H input files: ", paste(all_inputs[!file.exists(all_inputs)], collapse = ", "))
}

if ("--check-inputs" %in% commandArgs(trailingOnly = TRUE)) {
  message("Validated Figure 4H inputs: integrated h5AD, plot-data RDS, and six stage h5AD files.")
  quit(save = "no", status = 0)
}

plot_data <- readRDS(PLOT_DATA_RDS)
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)


#2. ===========Set genes and current Figure 4H vector geometry===========

GENES <- c("Slc26a5", "Slc17a8", "Atoh1", "Myo7a")
DISPLAY_GENES <- setNames(toupper(GENES), GENES)
COLOR_LIMITS <- c(Slc26a5 = 4, Slc17a8 = 4, Atoh1 = 3, Myo7a = 4)
COLOR_TICKS <- list(
  Slc26a5 = 0:4,
  Slc17a8 = 0:4,
  Atoh1 = 0:3,
  Myo7a = 0:4
)

PNG_DPI <- 1300
ARIAL <- Sys.getenv("COCHLEA_FONT_FAMILY", unset = "Arial")
PAGE_WIDTH <- 143
PAGE_HEIGHT <- 131
AXIS <- c(left = 8.761, top = 13.845, right = 122.319, bottom = 118.393)
COLORBAR <- c(left = 123.566, top = 13.845, right = 127.051, bottom = 118.393)
BACKGROUND_COLOR <- "#B9D9EA"
EXPRESSION_COLORS <- c("#FFD26A", "#F98C20", "#DE2D2D")
POINT_DIAMETER_PT <- 1.55
POINT_SYMBOL_FACTOR <- 1.68 / 2.25

lwd_pt <- function(width_pt) width_pt / 0.75
point_size <- function(diameter_pt) unit(diameter_pt / POINT_SYMBOL_FACTOR, "pt")
x_pt <- function(value) unit(value, "pt")
y_pt <- function(top) unit(PAGE_HEIGHT - top, "pt")


#3. ===========Rebuild the current Figure 4H HC UMAP coordinates===========

h5_index <- rhdf5::h5ls(H5AD_FILE, recursive = TRUE)
h5_full_paths <- sub(
  "^//", "/",
  paste0(h5_index$group, "/", h5_index$name)
)

h5_path_exists <- function(path) {
  paste0("/", sub("^/+", "", path)) %in% h5_full_paths
}

read_obs_column <- function(column_name) {
  base_path <- paste0("obs/", column_name)
  categories_path <- paste0(base_path, "/categories")
  codes_path <- paste0(base_path, "/codes")

  if (h5_path_exists(categories_path) && h5_path_exists(codes_path)) {
    categories <- as.character(rhdf5::h5read(H5AD_FILE, categories_path))
    codes <- as.integer(rhdf5::h5read(H5AD_FILE, codes_path))
    values <- rep(NA_character_, length(codes))
    valid <- !is.na(codes) & codes >= 0
    values[valid] <- categories[codes[valid] + 1L]
    return(values)
  }

  as.character(rhdf5::h5read(H5AD_FILE, base_path))
}

cell_id <- as.character(rhdf5::h5read(H5AD_FILE, "obs/_index"))
sce_meta <- data.frame(
  stage = read_obs_column("stage"),
  original_celltype = read_obs_column("original_celltype"),
  scanvi_label = read_obs_column("scanvi_label"),
  row.names = cell_id
)

x_scanvi <- rhdf5::h5read(H5AD_FILE, "obsm/X_scanvi")
if (ncol(x_scanvi) == length(cell_id)) x_scanvi <- t(x_scanvi)
rownames(x_scanvi) <- cell_id

target_stages <- c("E14.5", "E16.5", "P1", "P7", "P14", "P28")
stage_short_map <- c(
  "E14.5" = "E14", "E16.5" = "E16", "P1" = "P1",
  "P7" = "P7", "P14" = "P14", "P28" = "P28"
)
stage_celltype_levels <- c(
  "E14_IHC", "E14_OHC", "E16_IHC", "E16_iOHC", "E16_OHC",
  "P1_IHC", "P1_OHC", "P7_IHC", "P7_OHC", "P14_IHC",
  "P14_OHC", "P28_IHC", "P28_OHC"
)

celltype_for_plot <- ifelse(
  sce_meta$stage == "E16.5" & sce_meta$original_celltype == "iOHC",
  "iOHC",
  sce_meta$scanvi_label
)
sce_meta$stage_celltype_plot <- factor(
  paste(unname(stage_short_map[sce_meta$stage]), celltype_for_plot, sep = "_"),
  levels = stage_celltype_levels
)

hc_keep <- sce_meta$stage %in% target_stages &
  (sce_meta$scanvi_label %in% c("IHC", "OHC") |
     (sce_meta$stage == "E16.5" & sce_meta$original_celltype == "iOHC")) &
  !is.na(sce_meta$stage_celltype_plot)

set.seed(0)
umap_mat <- uwot::umap(
  x_scanvi[hc_keep, , drop = FALSE],
  n_neighbors = 30,
  min_dist = 0.35,
  metric = "euclidean",
  init = "spectral",
  ret_model = FALSE,
  n_threads = 1,
  verbose = FALSE
)

umap_df <- data.frame(
  cell_id = rownames(sce_meta)[hc_keep],
  UMAP_1 = -umap_mat[, 1] + 2.0,
  UMAP_2 = -umap_mat[, 2] + 13.2,
  stage_celltype_plot = sce_meta$stage_celltype_plot[hc_keep]
)

target_anchor_df <- data.frame(
  stage_celltype_plot = factor(stage_celltype_levels, levels = stage_celltype_levels),
  UMAP_1 = c(10.7, 5.55, 12.1, 5.75, 3.45, 14.0, 0.55, 15.0, -3.55, 16.35, -4.2, 15.35, 0.25),
  UMAP_2 = c(11.05, 14.75, 11.8, 10.1, 12.35, 13.25, 13.45, 12.05, 9.55, 8.95, 4.9, 10.1, 9.45)
)
current_anchor_df <- aggregate(
  cbind(UMAP_1, UMAP_2) ~ stage_celltype_plot,
  data = umap_df,
  FUN = median
)
shift_df <- merge(
  current_anchor_df,
  target_anchor_df,
  by = "stage_celltype_plot",
  suffixes = c("_current", "_target"),
  sort = FALSE
)
shift_df$shift_1 <- shift_df$UMAP_1_target - shift_df$UMAP_1_current
shift_df$shift_2 <- shift_df$UMAP_2_target - shift_df$UMAP_2_current
umap_df <- merge(
  umap_df,
  shift_df[, c("stage_celltype_plot", "shift_1", "shift_2")],
  by = "stage_celltype_plot",
  sort = FALSE
)
umap_df$UMAP_1 <- umap_df$UMAP_1 + umap_df$shift_1
umap_df$UMAP_2 <- umap_df$UMAP_2 + umap_df$shift_2


#4. ===========Extract four genes from saved full-gene h5AD files===========

extract_gene_expression <- function() {
  expression <- matrix(
    0,
    nrow = nrow(plot_data$umap_df),
    ncol = length(GENES),
    dimnames = list(plot_data$umap_df$cell_id, GENES)
  )

  for (stage in names(FULLGENE_H5AD_BY_STAGE)) {
    h5ad_file <- FULLGENE_H5AD_BY_STAGE[[stage]]
    target_cells <- grep(paste0("^", stage, "__"), rownames(expression), value = TRUE)
    raw_target_cells <- sub(paste0("^", stage, "__"), "", target_cells)
    h5_cell_ids <- as.character(h5read(h5ad_file, "obs/_index"))
    h5_gene_ids <- as.character(h5read(h5ad_file, "var/_index"))
    row_index <- match(raw_target_cells, h5_cell_ids)
    gene_index <- match(GENES, h5_gene_ids) - 1L

    if (anyNA(row_index)) {
      stop("Some HC cells were not found in the full-gene h5AD for stage: ", stage)
    }

    encoding_type <- as.character(h5readAttributes(h5ad_file, "X")[["encoding-type"]])
    stage_expression <- matrix(0, nrow = length(row_index), ncol = length(GENES))
    present_genes <- which(!is.na(gene_index))

    if (identical(encoding_type, "array")) {
      dense_values <- h5read(
        h5ad_file,
        "X",
        index = list(gene_index[present_genes] + 1L, row_index)
      )
      dense_values <- matrix(
        dense_values,
        nrow = length(present_genes),
        ncol = length(row_index)
      )
      stage_expression[, present_genes] <- t(dense_values)
    } else if (identical(encoding_type, "csr_matrix")) {
      order_rows <- order(row_index)
      ordered_index <- row_index[order_rows]
      x_indptr <- as.integer(h5read(h5ad_file, "X/indptr"))
      row_lengths <- x_indptr[ordered_index + 1L] - x_indptr[ordered_index]
      positions <- unlist(Map(
        function(start, length) {
          if (length > 0) seq.int(start + 1L, length.out = length) else integer()
        },
        x_indptr[ordered_index],
        row_lengths
      ), use.names = FALSE)
      cell_group <- rep.int(seq_along(ordered_index), row_lengths)
      x_indices <- as.integer(h5read(h5ad_file, "X/indices", index = list(positions)))
      x_data <- as.numeric(h5read(h5ad_file, "X/data", index = list(positions)))
      ordered_expression <- matrix(0, nrow = length(ordered_index), ncol = length(GENES))

      for (gene_col in present_genes) {
        present <- x_indices == gene_index[gene_col]
        ordered_expression[cell_group[present], gene_col] <- x_data[present]
      }
      stage_expression <- ordered_expression[order(order_rows), , drop = FALSE]
    } else {
      stop("Unsupported X encoding for stage: ", stage)
    }

    expression[target_cells, ] <- stage_expression
  }

  expression
}

expression <- extract_gene_expression()
expression_order <- match(umap_df$cell_id, rownames(expression))
if (anyNA(expression_order)) {
  stop("Some Figure 4H HC cells were not found in the full-gene expression matrix.")
}
expression <- expression[expression_order, , drop = FALSE]
if (!identical(umap_df$cell_id, rownames(expression))) {
  stop("The Figure 4H coordinates and gene-expression rows are not in identical order.")
}

#5. ===========Define editable vector drawing helpers===========

draw_text <- function(label, x, top, fontsize, hjust = 0.5, rot = 0, fontface = "plain") {
  grid.text(
    label,
    x = x_pt(x),
    y = y_pt(top),
    just = c(hjust, 0.5),
    rot = rot,
    gp = gpar(
      col = "black",
      fontsize = fontsize,
      fontfamily = ARIAL,
      fontface = fontface
    )
  )
}

draw_line <- function(x0, top0, x1, top1, width_pt = 0.5) {
  grid.lines(
    x = unit(c(x0, x1), "pt"),
    y = unit(PAGE_HEIGHT - c(top0, top1), "pt"),
    gp = gpar(col = "black", lwd = lwd_pt(width_pt), lineend = "butt")
  )
}

draw_vector_colorbar <- function(gene, limit) {
  gradient <- linearGradient(
    colours = EXPRESSION_COLORS,
    stops = seq(0, 1, length.out = length(EXPRESSION_COLORS)),
    x1 = unit(0.5, "npc"),
    y1 = unit(0, "npc"),
    x2 = unit(0.5, "npc"),
    y2 = unit(1, "npc")
  )
  grid.rect(
    x = x_pt(mean(COLORBAR[c("left", "right")])),
    y = y_pt(mean(COLORBAR[c("top", "bottom")])),
    width = unit(COLORBAR[["right"]] - COLORBAR[["left"]], "pt"),
    height = unit(COLORBAR[["bottom"]] - COLORBAR[["top"]], "pt"),
    gp = gpar(fill = gradient, col = "black", lwd = lwd_pt(0.5))
  )

  ticks <- COLOR_TICKS[[gene]]
  tick_top <- COLORBAR[["bottom"]] - ticks / limit * diff(COLORBAR[c("top", "bottom")])
  labels <- formatC(ticks, format = "f", digits = 0)

  for (i in seq_along(ticks)) {
    draw_line(COLORBAR[["right"]], tick_top[i], 128.794, tick_top[i], 0.5)
    draw_text(labels[i], 130.539, tick_top[i] + 1.273, 6.9725, hjust = 0)
  }
}


#6. ===========Draw one Figure 4H gene-expression panel===========

draw_gene_panel <- function(gene) {
  values <- expression[, gene]
  limit <- COLOR_LIMITS[[gene]]
  point_viewport <- viewport(
    x = x_pt(AXIS[["left"]]),
    y = y_pt(AXIS[["bottom"]]),
    width = unit(AXIS[["right"]] - AXIS[["left"]], "pt"),
    height = unit(AXIS[["bottom"]] - AXIS[["top"]], "pt"),
    just = c("left", "bottom"),
    xscale = c(-9, 19),
    yscale = c(1, 20),
    clip = "on"
  )

  pushViewport(point_viewport)
  undetected <- values <= 0
  detected_order <- order(values, na.last = TRUE)
  detected_order <- detected_order[values[detected_order] > 0]
  expression_pal <- scales::gradient_n_pal(EXPRESSION_COLORS, space = "Lab")

  grid.points(
    unit(umap_df$UMAP_1[undetected], "native"),
    unit(umap_df$UMAP_2[undetected], "native"),
    pch = 16,
    size = point_size(POINT_DIAMETER_PT),
    gp = gpar(col = BACKGROUND_COLOR)
  )
  grid.points(
    unit(umap_df$UMAP_1[detected_order], "native"),
    unit(umap_df$UMAP_2[detected_order], "native"),
    pch = 16,
    size = point_size(POINT_DIAMETER_PT),
    gp = gpar(
      col = expression_pal(
        scales::rescale(pmin(values[detected_order], limit), from = c(0, limit))
      )
    )
  )
  popViewport()

  draw_line(AXIS[["left"]], AXIS[["top"]], AXIS[["right"]], AXIS[["top"]])
  draw_line(AXIS[["right"]], AXIS[["top"]], AXIS[["right"]], AXIS[["bottom"]])
  draw_line(AXIS[["right"]], AXIS[["bottom"]], AXIS[["left"]], AXIS[["bottom"]])
  draw_line(AXIS[["left"]], AXIS[["bottom"]], AXIS[["left"]], AXIS[["top"]])

  draw_text(DISPLAY_GENES[[gene]], 65.540, 9.637, 6.9725, fontface = "italic")
  draw_text("UMAP1", 68.567, 123.890, 6.0706)
  draw_text("UMAP2", 4.568, 67.254, 6.0706, rot = 90)
  draw_vector_colorbar(gene, limit)
}


#7. ===========Save four editable PDFs and 1300 dpi PNGs===========

render_gene <- function(gene) {
  stem <- paste0("fig4h_", tolower(DISPLAY_GENES[[gene]]), "_hc_umap")
  pdf_file <- file.path(OUTPUT_DIR, paste0(stem, ".pdf"))
  png_file <- file.path(OUTPUT_DIR, paste0(stem, ".png"))

  grDevices::cairo_pdf(
    pdf_file,
    width = PAGE_WIDTH / 72,
    height = PAGE_HEIGHT / 72,
    family = ARIAL,
    bg = "white"
  )
  grid.newpage()
  draw_gene_panel(gene)
  grDevices::dev.off()

  ragg::agg_png(
    png_file,
    width = PAGE_WIDTH / 72,
    height = PAGE_HEIGHT / 72,
    units = "in",
    res = PNG_DPI,
    scaling = 1,
    background = "white"
  )
  grid.newpage()
  draw_gene_panel(gene)
  grDevices::dev.off()
}

invisible(lapply(GENES, render_gene))
message("Saved four current Figure 4H hair-cell marker UMAPs.")
