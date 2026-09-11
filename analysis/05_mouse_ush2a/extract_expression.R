#1. ===========Build reusable USH2A plot data===========

common_path <- file.path("analysis", "05_mouse_ush2a", "common.R")
if (!file.exists(common_path)) common_path <- "common.R"
source(common_path)

build_ush2a_plot_data(force_rebuild = TRUE)
