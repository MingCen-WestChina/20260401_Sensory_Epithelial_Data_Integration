#1. ===========Load USH2A detection data===========

set.seed(0)

get_current_script_dir <- function() {
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
    return(dirname(normalizePath(frame_files[length(frame_files)], winslash = "/", mustWork = TRUE)))
  }

  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

find_project_root <- function(start_dirs) {
  for (start_dir in unique(start_dirs)) {
    current_dir <- normalizePath(start_dir, winslash = "/", mustWork = TRUE)

    repeat {
      common_script <- file.path(current_dir, "analysis", "05_mouse_ush2a", "common.R")
      if (file.exists(common_script)) {
        return(current_dir)
      }

      parent_dir <- dirname(current_dir)
      if (identical(parent_dir, current_dir)) {
        break
      }

      current_dir <- parent_dir
    }
  }

  stop("Cannot find the repository root that contains analysis/05_mouse_ush2a/common.R.")
}

project_root <- find_project_root(c(getwd(), get_current_script_dir()))
common_script <- file.path(project_root, "analysis", "05_mouse_ush2a", "common.R")

suppressPackageStartupMessages(
  suppressWarnings(source(common_script))
)

plot_data <- load_ush2a_plot_data()
summary_df <- plot_data$detection_summary


#2. ===========Prepare dotplot coordinates===========

expression_df <- plot_data$expression_meta
expression_summary_df <- aggregate(
  ush2a_expr ~ stage + hc_lineage,
  data = expression_df,
  FUN = function(values) mean(values, na.rm = TRUE)
)
colnames(expression_summary_df)[colnames(expression_summary_df) == "ush2a_expr"] <- "mean_expression"

summary_df <- merge(
  summary_df,
  expression_summary_df,
  by = c("stage", "hc_lineage"),
  all.x = TRUE,
  sort = FALSE
)

summary_df$stage <- factor(as.character(summary_df$stage), levels = target_stages)
summary_df$hc_lineage <- factor(as.character(summary_df$hc_lineage), levels = target_labels)

scale_gene_expression <- function(x) {
  if (stats::sd(x) == 0 || all(is.na(x))) {
    return(rep(0, length(x)))
  }

  as.numeric(scale(x))
}

color_limit <- 1
max_dot_size <- 7.0

summary_df$avg_exp_scaled <- scale_gene_expression(summary_df$mean_expression)
summary_df$avg_exp_scaled <- pmax(pmin(summary_df$avg_exp_scaled, color_limit), -color_limit)


#3. ===========Plot detected-cell dotplot across development===========

p_detection_dotplot <- ggplot(
  summary_df,
  aes(x = stage, y = hc_lineage)
) +
  geom_point(
    aes(size = detected_pct, color = avg_exp_scaled),
    alpha = 0.95,
    stroke = 0
  ) +
  scale_x_discrete(limits = target_stages, expand = expansion(add = c(0.34, 0.50))) +
  scale_y_discrete(limits = rev(target_labels), expand = expansion(add = c(0.36, 0.36))) +
  scale_size_area(
    name = "% Exp.",
    max_size = max_dot_size,
    limits = c(0, 100),
    breaks = c(0, 25, 50, 75, 100)
  ) +
  scale_color_gradientn(
    name = "Avg. Exp.",
    colors = c("#D9D9D9", "#D894C4", "#8F1D82"),
    limits = c(-color_limit, color_limit),
    breaks = c(-1, -0.5, 0, 0.5, 1)
  ) +
  coord_fixed(ratio = 0.82, clip = "off") +
  theme_classic(base_size = 8.5) +
  labs(title = NULL, x = NULL, y = NULL) +
  guides(
    size = guide_legend(
      title.position = "top",
      override.aes = list(color = "black")
    ),
    color = guide_colorbar(
      title.position = "top",
      barwidth = grid::unit(0.34, "cm"),
      barheight = grid::unit(1.45, "cm")
    )
  ) +
  theme(
    axis.text.x = element_text(
      angle = 60,
      hjust = 1,
      vjust = 1,
      color = "black",
      size = 10
    ),
    axis.text.y = element_text(color = "black", size = 10),
    axis.ticks.x = element_line(linewidth = 0.25),
    axis.ticks.y = element_blank(),
    axis.line = element_line(linewidth = 0.35),
    legend.position = "right",
    legend.title = element_text(size = 7.2),
    legend.text = element_text(size = 6.4),
    legend.key.height = grid::unit(0.28, "cm"),
    legend.key.width = grid::unit(0.24, "cm"),
    legend.spacing.y = grid::unit(0.04, "cm"),
    legend.box.spacing = grid::unit(0.06, "cm"),
    plot.margin = margin(12, 8, 12, 6)
  )


#4. ===========Save dotplot PDF to the figure folder===========

figure_dir <- USH2A_FIGURE_DIR
output_pdf <- file.path(figure_dir, "USH2A_stage_detection_dotplot_horizontal.pdf")

get_versioned_pdf <- function(filename, version_id) {
  filename_base <- tools::file_path_sans_ext(filename)
  filename_ext <- paste0(".", tools::file_ext(filename))
  paste0(filename_base, "_", version_id, filename_ext)
}

save_pdf_plot_fallback <- function(plot_obj, filename, width, height) {
  candidate_file <- filename

  for (attempt_id in seq_len(5)) {
    save_result <- suppressWarnings(
      try(
        save_pdf_plot(plot_obj, candidate_file, width = width, height = height),
        silent = TRUE
      )
    )

    if (!inherits(save_result, "try-error")) {
      return(candidate_file)
    }

    candidate_file <- get_versioned_pdf(filename, attempt_id)
  }

  stop("Cannot save the dotplot PDF. Please close any open PDF preview and rerun this script.")
}

saved_pdf <- save_pdf_plot_fallback(
  p_detection_dotplot,
  output_pdf,
  width = 4.10,
  height = 2.55
)

message("Saved dotplot PDF: ", saved_pdf)
