#1. ===========Load packages and parameters
set.seed(0)
options(stringsAsFactors = FALSE)

required_pkgs <- c(
  "monocle", "Biobase", "Matrix", "VGAM", "igraph",
  "ggplot2", "clusterProfiler", "org.Mm.eg.db", "AnnotationDbi"
)

missing_pkgs <- required_pkgs[
  !vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_pkgs) > 0) {
  stop("Missing R packages: ", paste(missing_pkgs, collapse = ", "))
}

suppressPackageStartupMessages({
  library(monocle)
  library(Biobase)
  library(Matrix)
  library(VGAM)
  library(igraph)
  library(ggplot2)
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

de_qval_cutoff <- 0.05
de_formula <- "~branch_group + sm.ns(Pseudotime_scaled, df=3)"
min_cells_gene <- 10
lower_detection_limit <- 0.2
side_bins <- 58
smooth_window <- 5
terms_per_module <- 10
min_entrez_per_module <- 5
min_de_genes <- 50
go_width <- 12.2
go_height <- 8.8


#2. ===========Load accepted trajectory functions
load_refined_environment <- function(script_path) {
  script_lines <- readLines(script_path, warn = FALSE)
  run_line <- grep("^#8\\. ===========Run", script_lines)[1]
  if (is.na(run_line)) stop("Run block was not found in script: ", script_path)

  refined_env <- new.env(parent = globalenv())
  eval(parse(text = paste(script_lines[seq_len(run_line - 1)], collapse = "\n")), envir = refined_env)
  refined_env
}

medial_env <- load_refined_environment(file.path(script_dir, "medial_trajectory.R"))
lateral_env <- load_refined_environment(file.path(script_dir, "lateral_trajectory.R"))

message("Rebuilding accepted medial and lateral trajectories.")
medial_data <- suppressMessages(suppressWarnings(medial_env$load_h5ad_data(
  medial_env$input_h5ad_medial,
  celltype_col = "trajectory_celltype"
)))

lateral_data <- suppressMessages(suppressWarnings(lateral_env$load_h5ad_data(
  lateral_env$input_h5ad_lateral,
  celltype_col = "trajectory_celltype"
)))

medial_result <- suppressMessages(suppressWarnings(medial_env$run_monocle_subset(medial_data)))
lateral_result <- suppressMessages(suppressWarnings(lateral_env$run_monocle_subset(lateral_data)))


#3. ===========Set branch labels and GO module labels
branch_configs <- list(
  Medial = list(
    lineage = "Medial",
    go_title = "Panel C | Medial GO Biological Enrichment",
    file_prefix = file.path(output_dir, "Embryonic_E9_E16_panel_c_medial_GO_terms_with_ID"),
    branchpoint_labels = "M.PsD",
    left_labels = c("M.PsC", "IPhC", "IBC", "IPC"),
    right_labels = "IHC",
    module_labels = c(
      "M1 | M.PsC/IPhC/IBC/IPC",
      "M2 | Left transition",
      "M3 | M.PsD",
      "M4 | IHC transition",
      "M5 | IHC"
    )
  ),
  Lateral = list(
    lineage = "Lateral",
    go_title = "Panel C | Lateral GO Biological Enrichment",
    file_prefix = file.path(output_dir, "Embryonic_E9_E16_panel_c_lateral_GO_terms_with_ID"),
    branchpoint_labels = "L.PsD",
    left_labels = c("L.PsC", "HeC"),
    right_labels = c("OHC", "iOHC"),
    module_labels = c(
      "M1 | L.PsC/HeC",
      "M2 | Left transition",
      "M3 | L.PsD",
      "M4 | OHC/iOHC transition",
      "M5 | OHC/iOHC"
    )
  )
)

generic_go_patterns <- c(
  "translation", "ribosom", "peptide biosynthetic", "amide biosynthetic",
  "oxidative phosphorylation", "electron transport chain", "ATP synthesis",
  "mRNA splicing", "spliceosom", "mRNA processing", "RNA processing",
  "catabolic process", "mitochondrial", "ubiquitin", "proteasome",
  "immune", "leukocyte", "lymphocyte", "B cell", "T cell",
  "erythrocyte", "blood coagulation", "cardiac", "heart",
  "kidney", "renal", "liver", "spermat", "olfactory"
)

relevant_go_patterns <- c(
  "inner ear", "cochlea", "\\bear\\b", "auditory", "sound",
  "hair cell", "receptor cell", "sensory organ", "sensory perception",
  "mechanosensory", "mechanical stimulus", "stereo", "cilium",
  "actin", "cytoskeleton", "cell projection", "epithelial", "epithelium",
  "cell fate", "differentiation", "development", "morphogenesis",
  "Notch", "Wnt", "BMP", "FGF", "planar cell polarity",
  "adhesion", "extracellular matrix", "synap", "vesicle",
  "ion transport", "ion transmembrane", "calcium", "potassium",
  "sodium", "membrane potential"
)

secondary_go_patterns <- c(
  "pattern specification", "tube morphogenesis", "regionalization",
  "neurogenesis", "cell migration", "cell junction", "signaling pathway"
)


#4. ===========Run complete branch differential analysis
filter_gene_symbols <- function(genes) {
  genes <- unique(as.character(genes))
  genes[!grepl("^(Rpl|Rps|Mrpl|Mrps|Mt-)", genes)]
}

assign_branch_group <- function(cds, config) {
  meta <- as.data.frame(Biobase::pData(cds))
  cell_type <- as.character(meta$celltype_plot)
  names(cell_type) <- rownames(meta)

  branch_cells <- rownames(meta)[cell_type %in% config$branchpoint_labels]
  left_cells <- rownames(meta)[cell_type %in% config$left_labels]
  right_cells <- rownames(meta)[cell_type %in% config$right_labels]

  if (length(branch_cells) == 0 || length(left_cells) == 0 || length(right_cells) == 0) {
    stop("Branch, left, or right cells are missing for: ", config$lineage)
  }

  branch_group <- rep(NA_character_, nrow(meta))
  names(branch_group) <- rownames(meta)
  branch_group[left_cells] <- "Left"
  branch_group[branch_cells] <- "Branchpoint"
  branch_group[right_cells] <- "Right"

  Biobase::pData(cds)$branch_group <- factor(
    branch_group[rownames(Biobase::pData(cds))],
    levels = c("Left", "Branchpoint", "Right")
  )

  list(
    cds = cds,
    path_cells = unique(c(branch_cells, left_cells, right_cells)),
    branch_cells = branch_cells,
    left_cells = left_cells,
    right_cells = right_cells
  )
}

get_expressed_genes <- function(cds, cells) {
  expr_mat <- Biobase::exprs(cds)
  keep_gene <- Matrix::rowSums(expr_mat[, cells, drop = FALSE] > lower_detection_limit) >= min_cells_gene
  filter_gene_symbols(rownames(expr_mat)[keep_gene])
}

run_branch_de <- function(cds, config) {
  branch_obj <- assign_branch_group(cds, config)
  branch_cds <- branch_obj$cds[, branch_obj$path_cells]
  expressed_genes <- get_expressed_genes(branch_obj$cds, branch_obj$path_cells)
  expressed_genes <- intersect(expressed_genes, rownames(branch_cds))

  message("Running branch differential test for ", config$lineage, ".")
  dgt <- suppressMessages(suppressWarnings(monocle::differentialGeneTest(
    branch_cds[expressed_genes, ],
    fullModelFormulaStr = de_formula,
    cores = 1
  )))

  dgt <- dgt[order(dgt$qval, dgt$pval, na.last = TRUE), , drop = FALSE]
  dgt <- dgt[is.finite(dgt$qval), , drop = FALSE]
  de_table <- dgt[dgt$qval < de_qval_cutoff, , drop = FALSE]

  if (nrow(de_table) < min_de_genes) {
    stop(
      config$lineage,
      " has too few significant DE genes at qval < ",
      de_qval_cutoff,
      ". Please loosen de_qval_cutoff after checking the trajectory."
    )
  }

  list(
    cds = branch_obj$cds,
    branch_cds = branch_cds,
    de_table = de_table,
    universe_genes = rownames(dgt),
    branch_cells = branch_obj$branch_cells,
    left_cells = branch_obj$left_cells,
    right_cells = branch_obj$right_cells
  )
}


#5. ===========Assign all DE genes to branch modules
make_rank_bins <- function(cell_ids, pseudotime, n_bins) {
  cell_ids <- cell_ids[order(pseudotime[cell_ids], na.last = NA)]
  if (length(cell_ids) == 0) return(vector("list", n_bins))

  bin_id <- cut(
    seq_along(cell_ids),
    breaks = n_bins,
    labels = FALSE,
    include.lowest = TRUE
  )

  split(cell_ids, factor(bin_id, levels = seq_len(n_bins)))
}

smooth_matrix <- function(mat, window = 5) {
  half_window <- floor(window / 2)
  smoothed <- mat

  for (j in seq_len(ncol(mat))) {
    left <- max(1, j - half_window)
    right <- min(ncol(mat), j + half_window)
    smoothed[, j] <- rowMeans(mat[, left:right, drop = FALSE], na.rm = TRUE)
  }

  smoothed
}

fill_missing_rows <- function(mat) {
  for (i in seq_len(nrow(mat))) {
    row_value <- mean(mat[i, ], na.rm = TRUE)
    if (!is.finite(row_value)) row_value <- 0
    mat[i, !is.finite(mat[i, ])] <- row_value
  }
  mat
}

assign_peak_group <- function(peak_col, n_bins) {
  group_levels <- c(
    "Left terminal", "Left transition", "Branchpoint",
    "Right transition", "Right terminal"
  )

  breaks <- c(
    0,
    floor(n_bins * 0.20),
    floor(n_bins * 0.43),
    ceiling(n_bins * 0.57),
    ceiling(n_bins * 0.80),
    n_bins
  )

  breaks <- cummax(breaks)
  breaks[length(breaks)] <- n_bins

  cut(
    peak_col,
    breaks = breaks,
    labels = group_levels,
    include.lowest = TRUE
  )
}

build_de_modules <- function(de_result, config) {
  cds <- de_result$cds
  meta <- as.data.frame(Biobase::pData(cds))
  expr_mat <- Biobase::exprs(cds)
  de_genes <- rownames(de_result$de_table)

  pseudotime <- meta$Pseudotime_scaled
  names(pseudotime) <- rownames(meta)

  left_path_cells <- unique(c(de_result$branch_cells, de_result$left_cells))
  right_path_cells <- unique(c(de_result$branch_cells, de_result$right_cells))
  path_cells <- unique(c(left_path_cells, right_path_cells))
  expr_use <- expr_mat[de_genes, path_cells, drop = FALSE]

  left_bins <- rev(make_rank_bins(left_path_cells, pseudotime, side_bins))
  right_bins <- make_rank_bins(right_path_cells, pseudotime, side_bins)
  bin_list <- c(left_bins, right_bins)

  bin_mat <- vapply(bin_list, function(bin_cells) {
    if (length(bin_cells) == 0) return(rep(NA_real_, length(de_genes)))
    Matrix::rowMeans(expr_use[, bin_cells, drop = FALSE])
  }, numeric(length(de_genes)))

  rownames(bin_mat) <- de_genes
  colnames(bin_mat) <- paste0("Bin_", seq_len(ncol(bin_mat)))
  bin_mat <- smooth_matrix(bin_mat, window = smooth_window)
  bin_mat <- fill_missing_rows(bin_mat)

  keep_gene <- apply(bin_mat, 1, function(x) stats::sd(x, na.rm = TRUE) > 1e-6)
  bin_mat <- bin_mat[keep_gene, , drop = FALSE]

  z_mat <- t(scale(t(bin_mat)))
  z_mat[!is.finite(z_mat)] <- 0

  peak_col <- max.col(z_mat, ties.method = "first")
  peak_group <- assign_peak_group(peak_col, ncol(z_mat))
  module_id <- as.integer(peak_group)

  gene_info <- data.frame(
    gene = rownames(z_mat),
    module = module_id,
    module_label = config$module_labels[module_id],
    peak_col = peak_col,
    pval = de_result$de_table[rownames(z_mat), "pval"],
    qval = de_result$de_table[rownames(z_mat), "qval"],
    stringsAsFactors = FALSE
  )

  gene_info <- gene_info[order(gene_info$module, gene_info$peak_col, gene_info$qval), , drop = FALSE]

  list(
    gene_info = gene_info,
    universe_genes = de_result$universe_genes,
    module_counts = table(factor(gene_info$module, levels = seq_along(config$module_labels)))
  )
}


#6. ===========Run GO enrichment
parse_gene_ratio <- function(gene_ratio) {
  vapply(strsplit(as.character(gene_ratio), "/", fixed = TRUE), function(parts) {
    if (length(parts) != 2) return(NA_real_)
    numerator <- suppressWarnings(as.numeric(parts[1]))
    denominator <- suppressWarnings(as.numeric(parts[2]))
    if (!is.finite(numerator) || !is.finite(denominator) || denominator == 0) return(NA_real_)
    numerator / denominator
  }, numeric(1))
}

wrap_text <- function(x, width = 44) {
  vapply(x, function(value) paste(strwrap(value, width = width), collapse = "\n"), character(1))
}

get_entrez_map <- function(genes) {
  genes <- filter_gene_symbols(genes)
  if (length(genes) == 0) {
    return(data.frame(SYMBOL = character(), ENTREZID = character()))
  }

  gene_map <- suppressMessages(suppressWarnings(AnnotationDbi::select(
    org.Mm.eg.db::org.Mm.eg.db,
    keys = genes,
    keytype = "SYMBOL",
    columns = c("SYMBOL", "ENTREZID")
  )))

  gene_map <- gene_map[!is.na(gene_map$ENTREZID), c("SYMBOL", "ENTREZID"), drop = FALSE]
  gene_map <- gene_map[!duplicated(gene_map), , drop = FALSE]
  gene_map
}

select_go_terms <- function(enrich_df) {
  if (is.null(enrich_df) || nrow(enrich_df) == 0) {
    return(data.frame())
  }

  enrich_df <- enrich_df[order(enrich_df$p.adjust, -enrich_df$Count), , drop = FALSE]
  enrich_df$GeneRatioValue <- parse_gene_ratio(enrich_df$GeneRatio)

  generic_hit <- grepl(
    paste(generic_go_patterns, collapse = "|"),
    enrich_df$Description,
    ignore.case = TRUE
  )

  term_pool <- enrich_df[!generic_hit, , drop = FALSE]
  if (nrow(term_pool) == 0) term_pool <- enrich_df

  relevant_hit <- grepl(
    paste(relevant_go_patterns, collapse = "|"),
    term_pool$Description,
    ignore.case = TRUE
  )

  secondary_hit <- grepl(
    paste(secondary_go_patterns, collapse = "|"),
    term_pool$Description,
    ignore.case = TRUE
  )

  term_pool$term_priority <- ifelse(relevant_hit, 1, ifelse(secondary_hit, 2, 3))
  term_pool <- term_pool[
    order(term_pool$term_priority, term_pool$p.adjust, -term_pool$Count, -term_pool$GeneRatioValue),
    ,
    drop = FALSE
  ]
  term_pool <- term_pool[!duplicated(term_pool$Description), , drop = FALSE]
  term_pool <- head(term_pool, terms_per_module)

  data.frame(
    ID = term_pool$ID,
    Description = term_pool$Description,
    GeneRatio = term_pool$GeneRatio,
    GeneRatioValue = term_pool$GeneRatioValue,
    Count = term_pool$Count,
    p.adjust = term_pool$p.adjust,
    stringsAsFactors = FALSE
  )
}

run_go_enrichment <- function(module_result, config) {
  universe_map <- get_entrez_map(module_result$universe_genes)
  universe_entrez <- unique(universe_map$ENTREZID)
  module_list <- split(module_result$gene_info$gene, module_result$gene_info$module)

  go_list <- lapply(names(module_list), function(module_id) {
    module_genes <- filter_gene_symbols(module_list[[module_id]])
    module_entrez <- unique(universe_map$ENTREZID[universe_map$SYMBOL %in% module_genes])

    if (length(module_entrez) < min_entrez_per_module) {
      return(data.frame())
    }

    ego <- tryCatch(
      suppressMessages(suppressWarnings(clusterProfiler::enrichGO(
        gene = module_entrez,
        universe = universe_entrez,
        OrgDb = org.Mm.eg.db::org.Mm.eg.db,
        keyType = "ENTREZID",
        ont = "BP",
        pAdjustMethod = "BH",
        pvalueCutoff = 1,
        qvalueCutoff = 1,
        minGSSize = 5,
        maxGSSize = 500,
        readable = TRUE
      ))),
      error = function(e) NULL
    )

    go_use <- select_go_terms(as.data.frame(ego))
    if (nrow(go_use) == 0) return(data.frame())

    go_use$module <- as.integer(module_id)
    go_use$module_label <- config$module_labels[as.integer(module_id)]
    go_use$n_module_genes <- length(module_genes)
    go_use
  })

  go_df <- do.call(rbind, go_list)
  if (is.null(go_df)) go_df <- data.frame()
  go_df
}


#7. ===========Plot GO barplots
plot_go_barplot <- function(go_df, config) {
  if (is.null(go_df) || nrow(go_df) == 0) {
    stop("No GO terms were available for: ", config$lineage)
  }

  go_df$module <- as.integer(go_df$module)
  go_df$module_label <- factor(go_df$module_label, levels = config$module_labels)
  go_df <- go_df[is.finite(go_df$Count), , drop = FALSE]

  go_df <- do.call(
    rbind,
    lapply(split(go_df, go_df$module), function(module_df) {
      module_df <- module_df[
        order(module_df$p.adjust, -module_df$Count, -module_df$GeneRatioValue),
        ,
        drop = FALSE
      ]
      module_df <- head(module_df, terms_per_module)
      module_df$rank_id <- seq_len(nrow(module_df))
      module_df
    })
  )

  go_df$term_label <- wrap_text(paste0(go_df$ID, ": ", go_df$Description), width = 44)
  go_df$term_plot <- paste(go_df$module_label, go_df$term_label, sep = "___")

  term_levels <- unlist(lapply(split(go_df, go_df$module), function(module_df) {
    rev(module_df$term_plot)
  }))

  go_df$term_plot <- factor(go_df$term_plot, levels = unique(term_levels))

  finite_padjust <- go_df$p.adjust[is.finite(go_df$p.adjust)]
  go_df$p.adjust[!is.finite(go_df$p.adjust)] <- max(finite_padjust, 0.3)

  ggplot2::ggplot(
    go_df,
    ggplot2::aes(
      x = Count,
      y = term_plot,
      fill = p.adjust
    )
  ) +
    ggplot2::geom_col(width = 0.72, color = "grey35", linewidth = 0.12) +
    ggplot2::facet_wrap(
      ~module_label,
      ncol = 2,
      scales = "free_y"
    ) +
    ggplot2::scale_y_discrete(
      labels = function(x) sub("^.*___", "", x)
    ) +
    ggplot2::scale_x_continuous(
      breaks = function(x) pretty(x, n = 5),
      expand = ggplot2::expansion(mult = c(0, 0.05))
    ) +
    ggplot2::scale_fill_gradientn(
      colours = c("#B2182B", "#E08214", "#FDB863"),
      name = "Adjusted\nP value"
    ) +
    ggplot2::labs(
      title = config$go_title,
      x = "Gene number",
      y = NULL
    ) +
    ggplot2::theme_bw(base_size = 11) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(size = 16, hjust = 0.5, face = "plain"),
      strip.background = ggplot2::element_rect(fill = "grey93", color = "grey45", linewidth = 0.35),
      strip.text = ggplot2::element_text(size = 11, face = "plain"),
      axis.title.x = ggplot2::element_text(size = 11.5),
      axis.text.x = ggplot2::element_text(size = 9.2, color = "black"),
      axis.text.y = ggplot2::element_text(size = 7.4, color = "black", lineheight = 0.82),
      panel.grid.major.y = ggplot2::element_line(color = "grey92", linewidth = 0.20),
      panel.grid.major.x = ggplot2::element_line(color = "grey88", linewidth = 0.22),
      panel.grid.minor = ggplot2::element_blank(),
      legend.position = "right",
      legend.title = ggplot2::element_text(size = 9),
      legend.text = ggplot2::element_text(size = 8),
      plot.margin = ggplot2::margin(8, 18, 10, 8)
    )
}

save_go_plot <- function(plot_obj, file_prefix) {
  ggplot2::ggsave(
    filename = paste0(file_prefix, ".pdf"),
    plot = plot_obj,
    width = go_width,
    height = go_height,
    units = "in",
    device = grDevices::cairo_pdf,
    bg = "white"
  )

  ggplot2::ggsave(
    filename = paste0(file_prefix, ".png"),
    plot = plot_obj,
    width = go_width,
    height = go_height,
    units = "in",
    dpi = 1200,
    bg = "white",
    limitsize = FALSE
  )
}


#8. ===========Run GO analysis and save final figures
run_lineage_go <- function(cds, config) {
  de_result <- run_branch_de(cds, config)
  module_result <- build_de_modules(de_result, config)
  go_df <- run_go_enrichment(module_result, config)
  go_plot <- plot_go_barplot(go_df, config)
  save_go_plot(go_plot, config$file_prefix)

  cat(
    config$lineage,
    " GO saved. tested_genes=", length(de_result$universe_genes),
    ", de_genes=", nrow(de_result$de_table),
    ", module_genes=", paste(as.integer(module_result$module_counts), collapse = "/"),
    ", go_terms=", nrow(go_df),
    "\n",
    sep = ""
  )

  invisible(list(
    de_result = de_result,
    module_result = module_result,
    go_df = go_df,
    go_plot = go_plot
  ))
}

medial_go <- run_lineage_go(medial_result$cds, branch_configs$Medial)
lateral_go <- run_lineage_go(lateral_result$cds, branch_configs$Lateral)

cat(
  "Embryonic branch GO analysis finished. Output files: ",
  branch_configs$Medial$file_prefix,
  ".pdf/.png; ",
  branch_configs$Lateral$file_prefix,
  ".pdf/.png\n",
  sep = ""
)
