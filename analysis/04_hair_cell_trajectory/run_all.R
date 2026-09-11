#1. ===========Run all organized HC Monocle2 scripts===========

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = FALSE))
} else {
  file.path(getwd(), "analysis", "04_hair_cell_trajectory")
}

scripts_to_run <- file.path(
  script_dir,
  c(
    "build_trajectory.R",
    "umap_and_marker_dotplots.R",
    "fullgene_marker_dotplots.R",
    "lineage_panels.R",
    "heatmaps_and_enrichment.R"
  )
)

for (script_file in scripts_to_run) {
  message("Running: ", script_file)
  source(script_file)
}

message("Finished organized 04_hc_monocle2 workflow.")
