#1. ===========Load shared workflow and set paths===========

set.seed(0)
options(stringsAsFactors = FALSE)

get_sourced_file_05 <- function() {
  frame_files <- vapply(
    sys.frames(),
    function(frame_env) {
      if (!is.null(frame_env$ofile)) {
        return(frame_env$ofile)
      }
      NA_character_
    },
    character(1)
  )

  frame_files <- frame_files[!is.na(frame_files)]
  if (length(frame_files) > 0) {
    return(normalizePath(frame_files[length(frame_files)], winslash = "/", mustWork = TRUE))
  }

  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd_args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE))
  }

  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

USH2A_SCRIPT_DIR <- dirname(get_sourced_file_05())
USH2A_ORG_ROOT <- normalizePath(file.path(USH2A_SCRIPT_DIR, ".."), winslash = "/", mustWork = TRUE)
PROJECT_ROOT_05 <- normalizePath(file.path(USH2A_ORG_ROOT, "..", ".."), winslash = "/", mustWork = TRUE)
DATA_ROOT_05 <- normalizePath(
  Sys.getenv("COCHLEA_DATA_DIR", file.path(PROJECT_ROOT_05, "Data")),
  winslash = "/",
  mustWork = FALSE
)
RESULTS_ROOT_05 <- normalizePath(
  Sys.getenv("COCHLEA_RESULTS_DIR", file.path(PROJECT_ROOT_05, "results")),
  winslash = "/",
  mustWork = FALSE
)

hc_common_script <- file.path(PROJECT_ROOT_05, "analysis", "04_hair_cell_trajectory", "common.R")
if (!file.exists(hc_common_script)) {
  stop("Missing shared HC workflow script: ", hc_common_script)
}
source(hc_common_script)

USH2A_OUTPUT_ROOT <- file.path(RESULTS_ROOT_05, "05_mouse_ush2a")
USH2A_DATA_DIR <- file.path(USH2A_OUTPUT_ROOT, "data")
USH2A_FIGURE_DIR <- file.path(USH2A_OUTPUT_ROOT, "USH2A in HCs")

dir.create(USH2A_DATA_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(USH2A_FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)

ush2a_data_rds <- file.path(USH2A_DATA_DIR, "ush2a_hc_plot_data.rds")
ush2a_expression_csv <- file.path(USH2A_DATA_DIR, "ush2a_hc_expression_metadata.csv")
ush2a_detection_csv <- file.path(USH2A_DATA_DIR, "ush2a_stage_detection_summary.csv")
ush2a_umap_csv <- file.path(USH2A_DATA_DIR, "ush2a_all_hc_umap_coordinates.csv")

ush2a_gene_aliases <- c("Ush2a", "USH2A", "ENSMUSG00000026609")
ush2a_background_color <- "#B9D9EA"
ush2a_expression_colors <- c("#FFD26A", "#F98C20", "#DE2D2D")


#2. ===========Prepare shared data helpers===========

load_hc_dependency_bundle <- function() {
  if (!file.exists(hc_bundle_rds)) {
    stop("Missing HC Monocle2 data bundle. Run analysis/04_hair_cell_trajectory/run_all.R first.")
  }

  bundle <- readRDS(hc_bundle_rds)
  required_items <- c("plot_meta", "trajectory_data", "input_h5ad")
  missing_items <- setdiff(required_items, names(bundle))

  if (length(missing_items) > 0) {
    stop("The HC Monocle2 data bundle is missing fields: ", paste(missing_items, collapse = ", "))
  }

  bundle
}

normalize_detected_expression <- function(expression_value) {
  relative_value <- rep(NA_real_, length(expression_value))
  detected <- !is.na(expression_value) & expression_value > lower_detection_limit

  if (!any(detected)) {
    return(relative_value)
  }

  detected_value <- expression_value[detected]
  value_range <- range(detected_value, na.rm = TRUE)

  if (diff(value_range) == 0) {
    relative_value[detected] <- 1
    return(relative_value)
  }

  relative_value[detected] <- (detected_value - value_range[1]) / diff(value_range)
  relative_value
}

extract_ush2a_fullgene_expression <- function(plot_meta) {
  metadata_columns <- c(
    "cell_id", "stage", "stage_short", "hc_lineage", "scanvi_label",
    "original_celltype", "stage_celltype_plot", "Pseudotime",
    "Pseudotime_scaled", "State"
  )
  metadata_columns <- intersect(metadata_columns, colnames(plot_meta))
  meta_df <- plot_meta[, metadata_columns, drop = FALSE]
  meta_df$cell_id <- as.character(meta_df$cell_id)

  stage_expression <- lapply(names(fullgene_h5ad_by_stage), function(stage_name) {
    message("Extracting Ush2a expression from full-gene h5AD: ", stage_name)

    stage_file <- fullgene_h5ad_by_stage[[stage_name]]
    stage_gene_ids <- read_h5ad_var_index(stage_file)
    feature_id <- match_marker_feature(ush2a_gene_aliases, stage_gene_ids)

    if (is.na(feature_id)) {
      stop("Ush2a was not found in full-gene h5AD for stage: ", stage_name)
    }

    feature_index <- match(feature_id, stage_gene_ids)
    expression_matrix <- read_h5ad_dense_x_rows(stage_file, feature_index)
    stage_cell_ids <- paste0(stage_name, "__", as.character(rhdf5::h5read(stage_file, "obs/_index")))
    expression_value <- as.numeric(expression_matrix[1, ])
    names(expression_value) <- stage_cell_ids

    matched_cells <- intersect(stage_cell_ids, meta_df$cell_id)
    if (length(matched_cells) == 0) {
      stop("No HC cells matched the full-gene h5AD for stage: ", stage_name)
    }

    stage_df <- meta_df[match(matched_cells, meta_df$cell_id), , drop = FALSE]
    stage_df$ush2a_gene <- feature_id
    stage_df$ush2a_expr <- expression_value[matched_cells]
    stage_df
  })

  expression_df <- do.call(rbind, stage_expression)
  rownames(expression_df) <- NULL

  expression_df$stage <- factor(as.character(expression_df$stage), levels = target_stages)
  expression_df$hc_lineage <- factor(as.character(expression_df$hc_lineage), levels = target_labels)
  expression_df$stage_celltype_plot <- factor(
    as.character(expression_df$stage_celltype_plot),
    levels = stage_celltype_levels
  )
  expression_df$ush2a_detected <- expression_df$ush2a_expr > lower_detection_limit
  expression_df$ush2a_relative_expr <- normalize_detected_expression(expression_df$ush2a_expr)

  expression_df <- expression_df[
    order(
      match(as.character(expression_df$hc_lineage), target_labels),
      match(as.character(expression_df$stage), target_stages),
      expression_df$cell_id
    ),
    ,
    drop = FALSE
  ]

  rownames(expression_df) <- NULL
  expression_df
}

summarize_ush2a_detection <- function(expression_df) {
  summary_template <- expand.grid(
    stage = target_stages,
    hc_lineage = target_labels,
    stringsAsFactors = FALSE
  )

  summary_list <- lapply(seq_len(nrow(summary_template)), function(i) {
    stage_name <- summary_template$stage[i]
    lineage_name <- summary_template$hc_lineage[i]
    cells_use <- expression_df[
      as.character(expression_df$stage) == stage_name &
        as.character(expression_df$hc_lineage) == lineage_name,
      ,
      drop = FALSE
    ]

    total_cells <- nrow(cells_use)
    detected_cells <- sum(cells_use$ush2a_detected, na.rm = TRUE)
    detected_pct <- if (total_cells == 0) NA_real_ else detected_cells / total_cells * 100

    data.frame(
      stage = stage_name,
      hc_lineage = lineage_name,
      detected_cells = detected_cells,
      total_cells = total_cells,
      detected_pct = detected_pct,
      label = paste0(detected_cells, "/", total_cells),
      stringsAsFactors = FALSE
    )
  })

  summary_df <- do.call(rbind, summary_list)
  summary_df$stage <- factor(summary_df$stage, levels = target_stages)
  summary_df$hc_lineage <- factor(summary_df$hc_lineage, levels = target_labels)
  summary_df
}

build_ush2a_umap_data <- function(plot_meta) {
  x_scanvi <- read_h5ad_obsm(input_h5ad, "X_scanvi")
  x_scanvi <- x_scanvi[plot_meta$cell_id, , drop = FALSE]

  set.seed(0)
  umap_matrix <- uwot::umap(
    x_scanvi,
    n_neighbors = 30,
    min_dist = 0.35,
    metric = "euclidean",
    init = "spectral",
    ret_model = FALSE,
    verbose = FALSE
  )

  umap_df <- data.frame(
    cell_id = plot_meta$cell_id,
    UMAP_1 = umap_matrix[, 1],
    UMAP_2 = umap_matrix[, 2],
    stage = as.character(plot_meta$stage),
    hc_lineage = as.character(plot_meta$hc_lineage),
    stage_celltype_plot = as.character(plot_meta$stage_celltype_plot),
    stringsAsFactors = FALSE
  )

  umap_df$stage <- factor(umap_df$stage, levels = target_stages)
  umap_df$hc_lineage <- factor(umap_df$hc_lineage, levels = target_labels)
  umap_df$stage_celltype_plot <- factor(
    umap_df$stage_celltype_plot,
    levels = stage_celltype_levels
  )

  umap_df
}


#3. ===========Build and save reusable USH2A plot data===========

build_ush2a_plot_data <- function(force_rebuild = FALSE) {
  if (!force_rebuild && file.exists(ush2a_data_rds)) {
    return(readRDS(ush2a_data_rds))
  }

  bundle <- load_hc_dependency_bundle()
  plot_meta <- bundle$plot_meta

  expression_df <- extract_ush2a_fullgene_expression(plot_meta)
  detection_summary <- summarize_ush2a_detection(expression_df)
  umap_df <- build_ush2a_umap_data(plot_meta)

  plot_data <- list(
    expression_meta = expression_df,
    detection_summary = detection_summary,
    umap_df = umap_df,
    trajectory_data = bundle$trajectory_data,
    input_h5ad = bundle$input_h5ad,
    fullgene_h5ad_by_stage = fullgene_h5ad_by_stage,
    detection_threshold = lower_detection_limit,
    relative_expression_method = "Detected cells are min-max scaled to 0-1 across all HC cells."
  )

  saveRDS(plot_data, ush2a_data_rds)
  utils::write.csv(expression_df, ush2a_expression_csv, row.names = FALSE)
  utils::write.csv(detection_summary, ush2a_detection_csv, row.names = FALSE)
  utils::write.csv(umap_df, ush2a_umap_csv, row.names = FALSE)

  plot_data
}

load_ush2a_plot_data <- function() {
  build_ush2a_plot_data(force_rebuild = FALSE)
}


#4. ===========Shared USH2A plotting helpers===========

ush2a_color_scale <- function() {
  scale_color_gradientn(
    colors = ush2a_expression_colors,
    limits = c(0, 1),
    breaks = c(0, 0.25, 0.50, 0.75, 1.00),
    labels = sprintf("%.2f", c(0, 0.25, 0.50, 0.75, 1.00)),
    name = "Ush2a\nrelative\nexpression"
  )
}

theme_ush2a <- function(base_size = 12, square_panel = FALSE) {
  plot_theme <- theme_classic(base_size = base_size, base_family = "serif") +
    theme(
      plot.title = element_text(size = base_size + 4, hjust = 0.5, face = "plain"),
      axis.title = element_text(size = base_size + 1, color = "black"),
      axis.text = element_text(size = base_size - 1, color = "black"),
      legend.title = element_text(size = base_size, color = "black"),
      legend.text = element_text(size = base_size - 2, color = "black"),
      legend.key.height = grid::unit(0.36, "cm"),
      legend.key.width = grid::unit(0.36, "cm"),
      axis.line = element_line(color = "black", linewidth = 0.45),
      axis.ticks = element_line(color = "black", linewidth = 0.4)
    )

  if (square_panel) {
    plot_theme <- plot_theme + theme(aspect.ratio = 1)
  }

  plot_theme
}

merge_expression_by_cell <- function(plot_df, expression_df) {
  expression_columns <- c(
    "cell_id", "ush2a_gene", "ush2a_expr",
    "ush2a_detected", "ush2a_relative_expr"
  )
  expression_use <- expression_df[, expression_columns, drop = FALSE]
  merged_df <- merge(plot_df, expression_use, by = "cell_id", all.x = TRUE, sort = FALSE)
  merged_df <- merged_df[match(plot_df$cell_id, merged_df$cell_id), , drop = FALSE]
  rownames(merged_df) <- NULL
  merged_df
}
