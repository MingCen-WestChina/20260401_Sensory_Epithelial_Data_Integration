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
  library(uwot)
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

if (!file.exists(PLOT_DATA_RDS) || !file.exists(H5AD_FILE)) {
  stop("Required saved RDS or h5AD data were not found.")
}

if ("--check-inputs" %in% commandArgs(trailingOnly = TRUE)) {
  message("Validated Figure 4D-G inputs: ", PLOT_DATA_RDS, " and ", H5AD_FILE)
  quit(save = "no", status = 0)
}

plot_data <- readRDS(PLOT_DATA_RDS)
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)


#2. ===========Define exact vector drawing helpers===========

PNG_DPI <- 1300
ARIAL <- Sys.getenv("COCHLEA_FONT_FAMILY", unset = "Arial")
POINT_SYMBOL_FACTOR <- 1.68 / 2.25

lwd_pt <- function(width_pt) width_pt / 0.75
point_size <- function(diameter_pt) unit(diameter_pt / POINT_SYMBOL_FACTOR, "pt")

make_page <- function(crop) {
  list(
    crop = crop,
    width_pt = crop[["right"]] - crop[["left"]],
    height_pt = crop[["bottom"]] - crop[["top"]]
  )
}

x_unit <- function(page, x) unit(x - page$crop[["left"]], "pt")
y_unit <- function(page, top) unit(page$crop[["bottom"]] - top, "pt")

draw_line <- function(page, x0, top0, x1, top1, color, width_pt, lineend = "butt") {
  grid.lines(
    x = unit(c(x0, x1) - page$crop[["left"]], "pt"),
    y = unit(page$crop[["bottom"]] - c(top0, top1), "pt"),
    gp = gpar(col = color, lwd = lwd_pt(width_pt), lineend = lineend)
  )
}

draw_text <- function(
  page, label, x, top, fontsize, family = ARIAL,
  hjust = 0.5, vjust = 0.5, rot = 0, fontface = "plain"
) {
  grid.text(
    label,
    x = x_unit(page, x),
    y = y_unit(page, top),
    just = c(hjust, vjust),
    rot = rot,
    gp = gpar(
      col = "black",
      fontsize = fontsize,
      fontfamily = family,
      fontface = fontface
    )
  )
}

draw_vector_bar <- function(page, x0, x1, top0, top1, colors) {
  gradient <- linearGradient(
    colours = colors,
    stops = seq(0, 1, length.out = length(colors)),
    x1 = unit(0.5, "npc"),
    y1 = unit(0, "npc"),
    x2 = unit(0.5, "npc"),
    y2 = unit(1, "npc")
  )

  grid.rect(
    x = x_unit(page, (x0 + x1) / 2),
    y = y_unit(page, (top0 + top1) / 2),
    width = unit(x1 - x0, "pt"),
    height = unit(top1 - top0, "pt"),
    gp = gpar(fill = gradient, col = NA)
  )
}

render_figure <- function(stem, page, draw_function) {
  pdf_file <- file.path(OUTPUT_DIR, paste0(stem, ".pdf"))
  png_file <- file.path(OUTPUT_DIR, paste0(stem, ".png"))

  grDevices::cairo_pdf(
    pdf_file,
    width = page$width_pt / 72,
    height = page$height_pt / 72,
    family = ARIAL,
    bg = "white"
  )
  grid.newpage()
  draw_function()
  grDevices::dev.off()

  ragg::agg_png(
    png_file,
    width = page$width_pt / 72,
    height = page$height_pt / 72,
    units = "in",
    res = PNG_DPI,
    scaling = 1,
    background = "white"
  )
  grid.newpage()
  draw_function()
  grDevices::dev.off()
}


#3. ===========Rebuild the saved-latent-space HC UMAP for current panel D===========

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

expression_columns <- c(
  "cell_id", "ush2a_detected", "ush2a_relative_expr", "ush2a_expr"
)
umap_df <- merge(
  umap_df,
  plot_data$expression_meta[, expression_columns],
  by = "cell_id",
  sort = FALSE
)

#4. ===========Prepare trajectory and stage-dotplot data===========

trajectory_expression <- plot_data$expression_meta[, expression_columns]
trajectory_cells <- merge(
  plot_data$trajectory_data$cell_df,
  trajectory_expression,
  by = "cell_id",
  all.x = TRUE,
  sort = FALSE
)
trajectory_cells <- trajectory_cells[
  match(plot_data$trajectory_data$cell_df$cell_id, trajectory_cells$cell_id),
  ,
  drop = FALSE
]
trajectory_edges <- plot_data$trajectory_data$edge_df

summary_df <- plot_data$detection_summary
expression_summary <- aggregate(
  ush2a_expr ~ stage + hc_lineage,
  data = plot_data$expression_meta,
  FUN = mean
)
names(expression_summary)[3] <- "mean_expression"
summary_df <- merge(
  summary_df,
  expression_summary,
  by = c("stage", "hc_lineage"),
  sort = FALSE
)
summary_df$stage <- factor(as.character(summary_df$stage), levels = target_stages)
summary_df$hc_lineage <- factor(as.character(summary_df$hc_lineage), levels = c("IHC", "OHC"))
summary_df$avg_exp_scaled <- as.numeric(scale(summary_df$mean_expression))
summary_df$avg_exp_scaled <- pmax(pmin(summary_df$avg_exp_scaled, 1), -1)


#5. ===========Draw current panel D===========

G_PAGE <- make_page(c(left = 18.5, top = 456.0, right = 229.0, bottom = 743.0))
G_AXIS <- c(left = 40.6535, top = 538.2497, right = 222.7875, bottom = 720.3837)
G_X_RANGE <- c(-9, 19)
G_Y_RANGE <- c(1, 20)
G_COLORS <- c("#FFD26A", "#F98C20", "#DE2D2D")

draw_panel_G <- function() {
  point_vp <- viewport(
    x = x_unit(G_PAGE, G_AXIS[["left"]]),
    y = y_unit(G_PAGE, G_AXIS[["bottom"]]),
    width = unit(G_AXIS[["right"]] - G_AXIS[["left"]], "pt"),
    height = unit(G_AXIS[["bottom"]] - G_AXIS[["top"]], "pt"),
    just = c("left", "bottom"),
    xscale = G_X_RANGE,
    yscale = G_Y_RANGE,
    clip = "on"
  )

  pushViewport(point_vp)
  background_df <- umap_df[!umap_df$ush2a_detected, , drop = FALSE]
  detected_df <- umap_df[umap_df$ush2a_detected, , drop = FALSE]
  detected_df <- detected_df[order(detected_df$ush2a_relative_expr), , drop = FALSE]
  expression_pal <- scales::gradient_n_pal(G_COLORS, space = "Lab")

  grid.points(
    unit(background_df$UMAP_1, "native"),
    unit(background_df$UMAP_2, "native"),
    pch = 16,
    size = point_size(1.506),
    gp = gpar(col = "#B9D9EA")
  )
  grid.points(
    unit(detected_df$UMAP_1, "native"),
    unit(detected_df$UMAP_2, "native"),
    pch = 16,
    size = point_size(1.615),
    gp = gpar(col = expression_pal(detected_df$ush2a_relative_expr))
  )

  label_df <- data.frame(
    label = c("OHC", "iOHC", "IHC"),
    x = c(-4.239, 6.105, 12.895),
    y = c(13.907, 8.120, 10.419)
  )
  grid.text(
    label_df$label,
    x = unit(label_df$x, "native"),
    y = unit(label_df$y, "native"),
    just = c("left", "centre"),
    gp = gpar(col = "black", fontsize = 5.7682, fontfamily = ARIAL)
  )
  popViewport()

  draw_line(G_PAGE, 40.6535, 538.2497, 40.6535, 720.3837, "black", 0.638)
  draw_line(G_PAGE, 40.6535, 720.3837, 222.7875, 720.3837, "black", 0.638)

  g_x_ticks <- c(40.6535, 66.6745, 99.2007, 131.7205, 164.2467, 196.7729, 222.7875)
  for (tick_x in g_x_ticks) {
    draw_line(G_PAGE, tick_x, 720.3838, tick_x, 722.5628, "black", 0.528)
  }
  g_y_ticks <- c(720.3837, 682.0380, 634.1043, 586.1770, 538.2496)
  for (tick_y in g_y_ticks) {
    draw_line(G_PAGE, 38.4744, tick_y, 40.6534, tick_y, "black", 0.528)
  }

  g_x_labels <- c("", "-5", "0", "5", "10", "15", "19")
  for (i in seq_along(g_x_ticks)) {
    if (nzchar(g_x_labels[i])) {
      draw_text(G_PAGE, g_x_labels[i], g_x_ticks[i], 727.65, 6)
    }
  }
  g_y_labels <- c("", "5", "10", "15", "20")
  for (i in seq_along(g_y_ticks)) {
    if (nzchar(g_y_labels[i])) {
      draw_text(G_PAGE, g_y_labels[i], 36.49, g_y_ticks[i], 6, hjust = 1)
    }
  }
  draw_text(G_PAGE, "UMAP1", 134.53, 736.315, 7)
  draw_text(G_PAGE, "UMAP2", 24.98, 625.51, 7, rot = 90)

  draw_text(G_PAGE, "Ush2a", 191.7703, 462.8359, 7, fontface = "italic")
  draw_text(G_PAGE, "expression", 191.7668, 471.2359, 7)
  draw_vector_bar(G_PAGE, 184.3862, 193.4562, 481.9660, 527.3160, G_COLORS)

  legend_ticks <- c(482.0359, 493.3360, 504.6359, 515.9360, 527.2359)
  for (tick_y in legend_ticks) {
    draw_line(G_PAGE, 189.9962, tick_y, 193.4562, tick_y, "white", 0.37)
  }
  legend_labels <- c("1.00", "0.75", "0.50", "0.25", "0.00")
  legend_centres <- c(483.8633, 495.1613, 506.4593, 517.7573, 529.0553)
  for (i in seq_along(legend_labels)) {
    draw_text(G_PAGE, legend_labels[i], 195.4263, legend_centres[i], 6, hjust = 0)
  }
}


#6. ===========Draw current panels E and F===========

H_PAGE <- make_page(c(left = 239, top = 473, right = 402, bottom = 638))
I_PAGE <- make_page(c(left = 239, top = 652, right = 402, bottom = 817))

trajectory_spec <- list(
  IHC = list(
    page = H_PAGE,
    axis = c(left = 257.1449, top = 476.7618, right = 398.6419, bottom = 618.2598),
    x_range = c(-11.7058420784385, 17.9653766498941),
    y_range = c(-5.7194288734272, 8.8766120323462),
    x_ticks = c(265.2784, 312.9676, 360.6562),
    y_ticks = c(611.2856, 562.8142, 514.3422),
    y_text = c(612.47, 564.00, 515.525),
    x_text_top = 624.178,
    x_title = c(329.739, 631.616),
    y_title = c(245.592, 545.662),
    label = "IHC",
    label_center = c(353.197, 557.1325),
    arrow_start = c(276.0590, 499.9607),
    arrow_commands = matrix(c(
      276.4250, 510.1227, 277.0640, 518.6337, 277.6370, 524.9467,
      280.4390, 555.8197, 284.2470, 562.8367, 288.2420, 567.6107,
      290.0220, 569.7387, 294.8190, 575.3307, 302.9810, 578.1667,
      311.0670, 580.9767, 318.2770, 579.6287, 323.7910, 578.5977,
      329.4950, 577.5317, 334.0560, 575.5787, 342.9460, 571.7167,
      349.6730, 568.7937, 355.2280, 566.0867, 359.2650, 564.0317
    ), ncol = 6, byrow = TRUE),
    dash = c(2.013, 1.006),
    gradient = c(origin = 275.559, scale = 85.333),
    arrow_head = matrix(c(
      356.1184, 563.5668, 360.5954, 563.5668,
      357.7149, 567.3178, 360.2639, 563.5668
    ), ncol = 4, byrow = TRUE)
  ),
  OHC = list(
    page = I_PAGE,
    axis = c(left = 257.1684, top = 655.4677, right = 398.6654, bottom = 796.9657),
    x_range = c(-11.8317312854035, 18.7226740136948),
    y_range = c(-5.82742775615134, 9.05862118945305),
    x_ticks = c(265.6514, 311.9616, 358.2712),
    y_ticks = c(789.0998, 741.5722, 694.0467),
    y_text = c(790.284, 742.760, 695.235),
    x_text_top = 802.602,
    x_title = c(329.764, 810.322),
    y_title = c(245.615, 724.369),
    label = "OHC",
    label_center = c(352.2944, 739.1452),
    arrow_start = c(277.0876, 681.4189),
    arrow_commands = matrix(c(
      277.2156, 687.9629, 277.5076, 696.5989, 278.1876, 706.7129,
      280.4956, 741.0119, 284.1826, 746.3339, 287.2466, 749.3919,
      295.9126, 758.0369, 309.2846, 758.2349, 314.7936, 758.3169,
      324.0866, 758.4539, 331.5436, 755.8119, 338.3286, 753.4069,
      345.1306, 750.9969, 350.6326, 748.1789, 354.6836, 745.8119
    ), ncol = 6, byrow = TRUE),
    dash = c(2.015, 1.007),
    gradient = c(origin = 276.588, scale = 79.6841),
    arrow_head = matrix(c(
      351.5005, 745.2578, 355.9775, 745.2578,
      353.0970, 749.0088, 355.6460, 745.2578
    ), ncol = 4, byrow = TRUE)
  )
)

sample_bezier_path <- function(start, commands, step_pt = 0.04) {
  path_x <- start[1]
  path_y <- start[2]
  current <- start

  for (i in seq_len(nrow(commands))) {
    t <- seq(0, 1, length.out = 301)[-1]
    row <- commands[i, ]
    x <- (1 - t)^3 * current[1] +
      3 * (1 - t)^2 * t * row[1] +
      3 * (1 - t) * t^2 * row[3] +
      t^3 * row[5]
    y <- (1 - t)^3 * current[2] +
      3 * (1 - t)^2 * t * row[2] +
      3 * (1 - t) * t^2 * row[4] +
      t^3 * row[6]
    path_x <- c(path_x, x)
    path_y <- c(path_y, y)
    current <- row[5:6]
  }

  distance <- c(0, cumsum(sqrt(diff(path_x)^2 + diff(path_y)^2)))
  uniform_distance <- seq(0, max(distance), by = step_pt)
  data.frame(
    x = approx(distance, path_x, xout = uniform_distance)$y,
    top = approx(distance, path_y, xout = uniform_distance)$y,
    distance = uniform_distance
  )
}

draw_gradient_arrow <- function(spec) {
  path <- sample_bezier_path(spec$arrow_start, spec$arrow_commands)
  mid_distance <- head(path$distance, -1) + diff(path$distance) / 2
  draw_segment <- (mid_distance %% sum(spec$dash)) < spec$dash[1]
  segment_x <- (head(path$x, -1) + tail(path$x, -1)) / 2
  gradient_value <- pmin(pmax(
    (segment_x - spec$gradient[["origin"]]) / spec$gradient[["scale"]],
    0
  ), 1)
  color_matrix <- grDevices::colorRamp(c("#E83828", "#B9D8EA"), space = "rgb")
  segment_colors <- rgb(color_matrix(gradient_value), maxColorValue = 255)

  keep <- which(draw_segment)
  grid.segments(
    x0 = x_unit(spec$page, head(path$x, -1)[keep]),
    y0 = y_unit(spec$page, head(path$top, -1)[keep]),
    x1 = x_unit(spec$page, tail(path$x, -1)[keep]),
    y1 = y_unit(spec$page, tail(path$top, -1)[keep]),
    gp = gpar(col = segment_colors[keep], lwd = lwd_pt(1), lineend = "butt")
  )

  for (i in seq_len(nrow(spec$arrow_head))) {
    head_line <- spec$arrow_head[i, ]
    draw_line(
      spec$page,
      head_line[1], head_line[2], head_line[3], head_line[4],
      "#B5BAFF", 1
    )
  }
}

draw_trajectory_panel <- function(lineage) {
  spec <- trajectory_spec[[lineage]]
  page <- spec$page
  axis <- spec$axis
  lineage_df <- trajectory_cells[
    as.character(trajectory_cells$hc_lineage) == lineage,
    ,
    drop = FALSE
  ]

  point_vp <- viewport(
    x = x_unit(page, axis[["left"]]),
    y = y_unit(page, axis[["bottom"]]),
    width = unit(axis[["right"]] - axis[["left"]], "pt"),
    height = unit(axis[["bottom"]] - axis[["top"]], "pt"),
    just = c("left", "bottom"),
    xscale = spec$x_range,
    yscale = spec$y_range,
    clip = "on"
  )

  pushViewport(point_vp)
  grid.segments(
    x0 = unit(trajectory_edges$x, "native"),
    y0 = unit(trajectory_edges$y, "native"),
    x1 = unit(trajectory_edges$xend, "native"),
    y1 = unit(trajectory_edges$yend, "native"),
    gp = gpar(col = "#8C8C8C", lwd = lwd_pt(0.747), lineend = "round")
  )

  background_df <- lineage_df[!lineage_df$ush2a_detected, , drop = FALSE]
  detected_df <- lineage_df[lineage_df$ush2a_detected, , drop = FALSE]
  detected_df <- detected_df[order(detected_df$ush2a_relative_expr), , drop = FALSE]
  expression_pal <- scales::gradient_n_pal(
    c("#FFD166", "#F77F00", "#D62828"),
    space = "Lab"
  )

  grid.points(
    unit(background_df$Component_1, "native"),
    unit(background_df$Component_2, "native"),
    pch = 16,
    size = point_size(1.67),
    gp = gpar(col = "#B9D8EA")
  )
  grid.points(
    unit(detected_df$Component_1, "native"),
    unit(detected_df$Component_2, "native"),
    pch = 16,
    size = point_size(1.67),
    gp = gpar(col = expression_pal(detected_df$ush2a_relative_expr))
  )
  popViewport()

  draw_gradient_arrow(spec)
  draw_line(page, axis[["left"]], axis[["top"]], axis[["left"]], axis[["bottom"]], "black", 0.96)
  draw_line(page, axis[["left"]], axis[["bottom"]], axis[["right"]], axis[["bottom"]], "black", 0.96)

  for (tick_y in spec$y_ticks) {
    draw_line(page, axis[["left"]] - 1.486, tick_y, axis[["left"]], tick_y, "black", 0.854)
  }
  for (tick_x in spec$x_ticks) {
    draw_line(page, tick_x, axis[["bottom"]], tick_x, axis[["bottom"]] + 1.486, "black", 0.854)
  }

  for (i in seq_along(spec$x_ticks)) {
    draw_text(page, c("-10", "0", "10")[i], spec$x_ticks[i], spec$x_text_top, 5)
  }
  for (i in seq_along(spec$y_ticks)) {
    draw_text(page, c("-5", "0", "5")[i], axis[["left"]] - 2.63, spec$y_text[i], 5, hjust = 1)
  }

  draw_text(page, "Component 1", spec$x_title[1], spec$x_title[2], 6)
  draw_text(page, "Component 2", spec$y_title[1], spec$y_title[2], 6, rot = 90)
  draw_text(
    page,
    spec$label,
    spec$label_center[1],
    spec$label_center[2],
    6.465,
    family = ARIAL
  )
}


#7. ===========Draw current panel G===========

J_PAGE <- make_page(c(left = 23, top = 752, right = 232, bottom = 811))
J_STAGE_X <- c(48.4792, 69.9683, 91.4550, 112.9442, 134.4309, 155.9200)
J_ROW_TOP <- c(IHC = 764.6579, OHC = 782.2779)
J_COLORS <- c("#D9D9D9", "#D894C4", "#8F1D82")

draw_panel_J <- function() {
  dot_pal <- scales::gradient_n_pal(J_COLORS, values = scales::rescale(c(-1, 0, 1)), space = "Lab")
  summary_df$x <- J_STAGE_X[match(as.character(summary_df$stage), target_stages)]
  summary_df$top <- unname(J_ROW_TOP[as.character(summary_df$hc_lineage)])
  summary_df$diameter_pt <- 8.833 * sqrt(summary_df$detected_pct / 100)

  visible <- summary_df$diameter_pt > 0
  grid.points(
    x_unit(J_PAGE, summary_df$x[visible]),
    y_unit(J_PAGE, summary_df$top[visible]),
    pch = 16,
    size = point_size(summary_df$diameter_pt[visible]),
    gp = gpar(col = dot_pal(scales::rescale(summary_df$avg_exp_scaled[visible], from = c(-1, 1))))
  )

  draw_line(J_PAGE, 41.1756, 758.3131, 41.1756, 788.6211, "black", 0.747)
  draw_line(J_PAGE, 41.1756, 788.6211, 166.6626, 788.6211, "black", 0.747)
  for (tick_x in J_STAGE_X) {
    draw_line(J_PAGE, tick_x, 788.6210, tick_x, 789.8730, "black", 0.533)
  }

  draw_text(J_PAGE, "IHC", 39.3213, 766.018, 6, hjust = 1)
  draw_text(J_PAGE, "OHC", 39.3633, 783.640, 6, hjust = 1)
  for (i in seq_along(target_stages)) {
    draw_text(
      J_PAGE,
      target_stages[i],
      J_STAGE_X[i],
      791.15,
      6,
      hjust = 1,
      rot = 60
    )
  }

  draw_text(J_PAGE, "% Exp.", 188.748, 758.62, 5)
  size_values <- c(25, 50, 75, 100)
  size_centres <- c(770.3715, 777.4533, 786.6817, 797.6302)
  size_diameters <- c(4.416, 6.246, 7.651, 8.833)
  grid.points(
    x_unit(J_PAGE, rep(186.6925, 4)),
    y_unit(J_PAGE, size_centres),
    pch = 16,
    size = point_size(size_diameters),
    gp = gpar(col = "black")
  )

  draw_text(J_PAGE, "0", 195.06, 766.04, 4, hjust = 0)
  size_label_centres <- c(771.32, 778.41, 787.64, 798.59)
  for (i in seq_along(size_values)) {
    draw_text(J_PAGE, as.character(size_values[i]), 195.06, size_label_centres[i], 4, hjust = 0)
  }

  draw_text(J_PAGE, "Avg. Exp.", 217.61, 758.392, 5)
  draw_vector_bar(J_PAGE, 207.9138, 213.6144, 764.4340, 789.3874, J_COLORS)
  j_color_ticks <- c(764.5569, 770.6110, 776.6675, 782.7239, 788.7804)
  for (tick_y in j_color_ticks) {
    draw_line(J_PAGE, 207.9138, tick_y, 213.6144, tick_y, "white", 0.375)
  }
  j_color_labels <- c("1.0", "0.5", "0.0", "-0.5", "-1.0")
  j_color_centres <- c(765.51, 771.56, 777.62, 783.67, 789.73)
  for (i in seq_along(j_color_labels)) {
    draw_text(J_PAGE, j_color_labels[i], 216.12, j_color_centres[i], 4, hjust = 0)
  }
}


#8. ===========Save the four final panels===========

render_figure("fig4d_mouse_ush2a_hc_umap", G_PAGE, draw_panel_G)
render_figure("fig4e_mouse_ush2a_ihc_trajectory", H_PAGE, function() draw_trajectory_panel("IHC"))
render_figure("fig4f_mouse_ush2a_ohc_trajectory", I_PAGE, function() draw_trajectory_panel("OHC"))
render_figure("fig4g_mouse_ush2a_stage_dotplot", J_PAGE, draw_panel_J)

message("Saved current Figure 4D-G as vector PDFs and 1300 dpi PNGs.")
