#1. ===========Run the complete USH2A-in-HCs workflow===========

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = FALSE))
} else {
  file.path(getwd(), "analysis", "05_mouse_ush2a")
}

script_files <- file.path(
  script_dir,
  c(
    "extract_expression.R",
    "hair_cell_umap.R",
    "lineage_trajectories.R",
    "stage_detection_dotplot.R"
  )
)

for (script_file in script_files) {
  message("Running ", script_file)
  source(script_file)
}
