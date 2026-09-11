#1. ===========Load shared data and modules===========

common_path <- file.path("analysis", "04_hair_cell_trajectory", "common.R")
if (!file.exists(common_path)) common_path <- "common.R"
source(common_path)

required_enrichment_pkgs <- c("clusterProfiler", "org.Mm.eg.db", "AnnotationDbi")
missing_enrichment_pkgs <- required_enrichment_pkgs[
  !vapply(required_enrichment_pkgs, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_enrichment_pkgs) > 0) {
  stop("Missing enrichment packages: ", paste(missing_enrichment_pkgs, collapse = ", "))
}

if (!file.exists(hc_modules_rds)) {
  lineage_script <- file.path("analysis", "04_hair_cell_trajectory", "lineage_panels.R")
  if (!file.exists(lineage_script)) lineage_script <- "lineage_panels.R"
  source(lineage_script)
}

module_bundle <- readRDS(hc_modules_rds)

panel_e_dir <- file.path(HC_OUTPUT_DIR, "panel_e_IHC_pseudotime_heatmap")
panel_f_dir <- file.path(HC_OUTPUT_DIR, "panel_f_OHC_pseudotime_heatmap")
dir.create(panel_e_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(panel_f_dir, recursive = TRUE, showWarnings = FALSE)

enrichment_rds <- file.path(DATA_OUTPUT_DIR, "hc_monocle2_lineage_enrichment.rds")


#2. ===========Plot HC heatmap-only panels===========

plot_heatmap_only <- function(module_obj, title_text) {
  ggplot(module_obj$heatmap_df, aes(x = bin, y = gene, fill = z)) +
    geom_raster() +
    facet_grid(module ~ ., scales = "free_y", space = "free_y") +
    scale_fill_gradientn(
      colors = c("#4055A8", "#8FC7BD", "#F1E85B", "#F08022"),
      limits = c(-3, 3),
      breaks = c(-3, -2, -1, 0, 1, 2, 3),
      name = "Relative\nexpression"
    ) +
    theme_classic(base_size = 11) +
    labs(title = title_text, x = "Pseudotime", y = NULL) +
    theme(
      plot.title = element_text(size = 13, hjust = 0),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      strip.background = element_blank(),
      strip.text.y = element_text(size = 10, color = "black"),
      legend.title = element_text(size = 9),
      legend.text = element_text(size = 8),
      panel.spacing.y = grid::unit(0.05, "lines"),
      plot.margin = margin(8, 12, 8, 8)
    )
}

p_panel_e_heatmap <- plot_heatmap_only(module_bundle$IHC, "Panel e | IHC pseudotime heatmap")
p_panel_f_heatmap <- plot_heatmap_only(module_bundle$OHC, "Panel f | OHC pseudotime heatmap")

save_pdf_plot(
  p_panel_e_heatmap,
  file.path(panel_e_dir, "HC_panel_e_IHC_heatmap_only.pdf"),
  width = 10.4,
  height = 7.8
)

save_pdf_plot(
  p_panel_e_heatmap,
  file.path(panel_e_dir, "HC_panel_e_IHC_heatmap_only_new.pdf"),
  width = 10.4,
  height = 7.8
)

save_pdf_plot(
  p_panel_f_heatmap,
  file.path(panel_f_dir, "HC_panel_f_OHC_heatmap_only.pdf"),
  width = 10.4,
  height = 7.8
)


#3. ===========Run GO and KEGG enrichment by module===========

map_symbols_to_entrez <- function(symbols) {
  mapped <- AnnotationDbi::mapIds(
    org.Mm.eg.db::org.Mm.eg.db,
    keys = unique(symbols),
    keytype = "SYMBOL",
    column = "ENTREZID",
    multiVals = "first"
  )

  unique(unname(mapped[!is.na(mapped)]))
}

select_enrichment_terms <- function(enrich_df, database_name, n_terms = 20) {
  if (is.null(enrich_df) || nrow(enrich_df) == 0) {
    return(data.frame())
  }

  enrich_df <- enrich_df[order(enrich_df$p.adjust, -enrich_df$Count), , drop = FALSE]
  enrich_df <- head(enrich_df, n_terms)
  enrich_df$database <- database_name
  enrich_df
}

run_module_enrichment <- function(module_obj) {
  module_levels <- levels(module_obj$modules)

  go_df <- do.call(
    rbind,
    lapply(module_levels, function(module_name) {
      genes <- names(module_obj$modules)[module_obj$modules == module_name]
      entrez <- map_symbols_to_entrez(genes)

      if (length(entrez) < 5) {
        return(data.frame())
      }

      ego <- suppressMessages(suppressWarnings(clusterProfiler::enrichGO(
        gene = entrez,
        OrgDb = org.Mm.eg.db::org.Mm.eg.db,
        keyType = "ENTREZID",
        ont = "BP",
        pAdjustMethod = "BH",
        pvalueCutoff = 1,
        qvalueCutoff = 1,
        readable = TRUE
      )))

      out <- select_enrichment_terms(as.data.frame(ego), "GO", 20)
      if (nrow(out) == 0) return(out)
      out$module <- module_name
      out
    })
  )

  kegg_df <- do.call(
    rbind,
    lapply(module_levels, function(module_name) {
      genes <- names(module_obj$modules)[module_obj$modules == module_name]
      entrez <- map_symbols_to_entrez(genes)

      if (length(entrez) < 5) {
        return(data.frame())
      }

      ekegg <- tryCatch(
        suppressMessages(suppressWarnings(clusterProfiler::enrichKEGG(
          gene = entrez,
          organism = "mmu",
          pAdjustMethod = "BH",
          pvalueCutoff = 1,
          qvalueCutoff = 1
        ))),
        error = function(e) NULL
      )

      out <- select_enrichment_terms(as.data.frame(ekegg), "KEGG", 20)
      if (nrow(out) == 0) return(out)
      out$module <- module_name
      out
    })
  )

  list(go_df = go_df, kegg_df = kegg_df)
}

if (file.exists(enrichment_rds)) {
  enrichment_bundle <- readRDS(enrichment_rds)
} else {
  enrichment_bundle <- list(
    IHC = run_module_enrichment(module_bundle$IHC),
    OHC = run_module_enrichment(module_bundle$OHC)
  )
  saveRDS(enrichment_bundle, enrichment_rds)
}


#4. ===========Plot enrichment barplots===========

format_enrichment_df <- function(enrich_df, include_id = FALSE) {
  if (is.null(enrich_df) || nrow(enrich_df) == 0) {
    return(data.frame(
      module = factor("1", levels = c("1")),
      label = "No enriched term",
      p_adjust = 1,
      Count = 0,
      stringsAsFactors = FALSE
    ))
  }

  out <- enrich_df
  out$label <- if (include_id) {
    paste0(out$Description, " (", out$ID, ")")
  } else {
    out$Description
  }

  out$label <- wrap_text(out$label, width = 42)
  out$p_adjust <- pmax(out$p.adjust, .Machine$double.xmin)
  out$module <- factor(out$module, levels = sort(unique(as.character(out$module))))
  out
}

plot_enrichment_barplot <- function(enrich_df, title_text, include_id = FALSE) {
  plot_df <- format_enrichment_df(enrich_df, include_id = include_id)
  plot_df <- plot_df[order(plot_df$module, plot_df$p_adjust), , drop = FALSE]
  plot_df$label <- factor(plot_df$label, levels = rev(unique(plot_df$label)))

  ggplot(plot_df, aes(x = -log10(p_adjust), y = label, fill = module)) +
    geom_col(width = 0.72, color = "grey20", linewidth = 0.15) +
    facet_grid(module ~ ., scales = "free_y", space = "free_y") +
    scale_fill_manual(
      values = c("1" = "#D94B3D", "2" = "#F17C22", "3" = "#4059B7", "4" = "#A13DB8"),
      guide = "none"
    ) +
    theme_classic(base_size = 10) +
    labs(title = title_text, x = "-log10(adjusted P value)", y = NULL) +
    theme(
      plot.title = element_text(size = 13, hjust = 0.5),
      axis.text.y = element_text(size = 7.2, color = "black", lineheight = 0.9),
      axis.text.x = element_text(size = 9, color = "black"),
      axis.title.x = element_text(size = 12, color = "black"),
      strip.background = element_blank(),
      strip.text.y = element_text(size = 9, color = "black"),
      panel.spacing.y = grid::unit(0.1, "lines"),
      plot.margin = margin(8, 16, 8, 8)
    )
}

save_pdf_plot(
  plot_enrichment_barplot(enrichment_bundle$IHC$go_df, "Panel e | IHC GO Biological Enrichment", include_id = TRUE),
  file.path(panel_e_dir, "HC_panel_e_IHC_GO_terms_GOID.pdf"),
  width = 13.2,
  height = 10.6
)

save_pdf_plot(
  plot_enrichment_barplot(enrichment_bundle$IHC$go_df, "Panel e | IHC GO Biological Enrichment", include_id = FALSE),
  file.path(panel_e_dir, "HC_panel_e_IHC_GO_terms_only.pdf"),
  width = 13.2,
  height = 10.6
)

save_pdf_plot(
  plot_enrichment_barplot(enrichment_bundle$IHC$kegg_df, "Panel e | IHC KEGG Pathway Enrichment", include_id = FALSE),
  file.path(panel_e_dir, "HC_panel_e_IHC_KEGG_terms_only.pdf"),
  width = 13.2,
  height = 10.6
)

save_pdf_plot(
  plot_enrichment_barplot(enrichment_bundle$OHC$go_df, "Panel f | OHC GO Biological Enrichment", include_id = TRUE),
  file.path(panel_f_dir, "HC_panel_f_OHC_GO_terms_GOID.pdf"),
  width = 13.2,
  height = 10.6
)

save_pdf_plot(
  plot_enrichment_barplot(enrichment_bundle$OHC$go_df, "Panel f | OHC GO Biological Enrichment", include_id = FALSE),
  file.path(panel_f_dir, "HC_panel_f_OHC_GO_terms_only.pdf"),
  width = 13.2,
  height = 10.6
)

save_pdf_plot(
  plot_enrichment_barplot(enrichment_bundle$OHC$kegg_df, "Panel f | OHC KEGG Pathway Enrichment", include_id = FALSE),
  file.path(panel_f_dir, "HC_panel_f_OHC_KEGG_terms_only.pdf"),
  width = 13.2,
  height = 10.6
)

message("Saved HC pseudotime heatmap and enrichment panels.")
