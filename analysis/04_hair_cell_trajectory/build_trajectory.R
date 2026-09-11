#1. ===========Load shared workflow===========

common_path <- file.path("analysis", "04_hair_cell_trajectory", "common.R")
if (!file.exists(common_path)) common_path <- "common.R"
source(common_path)


#2. ===========Build and save all-HC Monocle2 data===========

bundle <- build_hc_monocle2_bundle(force_rebuild = TRUE)

message("Saved all-HC Monocle2 CellDataSet: ", hc_cds_rds)
message("Saved all-HC plotting metadata: ", hc_plot_meta_csv)
message("Saved all-HC trajectory data: ", hc_trajectory_rds)
message("Saved all-HC data bundle: ", hc_bundle_rds)
