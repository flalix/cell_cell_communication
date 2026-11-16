#!/usr/bin/env Rscript
# run_rctd_pdac.R - RCTD (spacexr, full mode) on the 21 PDAC Visium sections, as Chen et al. 2025 (Cancer Cell;
# STAR Methods: "RCTD algorithm from spacexr in full mode ... default parameters ... normalized to sum 1").
#
# Inputs: pdac/results/rctd/inputs/ (SingleCellPDAC.export_rctd_inputs, mp_pdac01 or src/programs/run_rctd_pdac.py):
#   reference/{counts.mtx, genes.tsv, cells.tsv}   tissue cells (<= 3,000 per type), cell_type, nUMI
#   <section>/{counts.mtx, genes.tsv, spots.tsv}   one folder per section (shared genes of the three panels)
# Outputs: pdac/results/rctd/weights/<section>.tsv (one per section; existing files are kept: resumable)
#          pdac/results/rctd/pdac_rctd_weights.tsv.gz (all sections, rows sum to 1)
#
# Usage: R of the renv conda env + spacexr 2.2.1 in the project library .R_lib (renv itself is not modified):
#   R_LIBS_USER=.R_lib /home/flavio/miniforge3/envs/renv/bin/Rscript src/programs/run_rctd_pdac.R [cores] [section ...]
# (install once: R_LIBS_USER=.R_lib .../renv/bin/Rscript -e 'remotes::install_github("dmcable/spacexr",
#  build_vignettes = FALSE, upgrade = "never", lib = Sys.getenv("R_LIBS_USER"))')

suppressPackageStartupMessages({
  library(spacexr)
  library(Matrix)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
cores <- if (length(args) >= 1) as.integer(args[1]) else 8
only <- if (length(args) >= 2) args[-1] else NULL

root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "..", ".."))
d_in <- file.path(root, "pdac", "results", "rctd", "inputs")
d_out <- file.path(root, "pdac", "results", "rctd", "weights")
dir.create(d_out, showWarnings = FALSE, recursive = TRUE)

read_counts <- function(d) {
  m <- as(readMM(file.path(d, "counts.mtx")), "CsparseMatrix")
  rownames(m) <- fread(file.path(d, "genes.tsv"), header = FALSE)$V1
  m
}

t0 <- Sys.time()
msg <- function(...) cat(sprintf("[%6.0f s] ", as.numeric(difftime(Sys.time(), t0, units = "secs"))), ..., "\n", sep = "")

# ---- reference
ref_cells <- fread(file.path(d_in, "reference", "cells.tsv"))
ref_counts <- read_counts(file.path(d_in, "reference"))
colnames(ref_counts) <- ref_cells$barcode
ct <- factor(make.names(ref_cells$cell_type))          # RCTD wants syntactic names; mapped back below
names(ct) <- ref_cells$barcode
name_map <- setNames(unique(ref_cells$cell_type), make.names(unique(ref_cells$cell_type)))
nUMI_ref <- setNames(ref_cells$nUMI, ref_cells$barcode)
reference <- Reference(ref_counts, ct, nUMI_ref, n_max_cells = 10000)
msg("reference: ", ncol(ref_counts), " cells, ", nlevels(ct), " cell types, ", nrow(ref_counts), " genes")

sections <- setdiff(list.dirs(d_in, full.names = FALSE, recursive = FALSE), "reference")
if (!is.null(only)) sections <- intersect(sections, only)
for (s in sections) {
  f_out <- file.path(d_out, paste0(s, ".tsv"))
  if (file.exists(f_out)) { msg(s, ": exists"); next }
  spots <- fread(file.path(d_in, s, "spots.tsv"))
  counts <- read_counts(file.path(d_in, s))
  colnames(counts) <- spots$barcode
  coords <- data.frame(x = spots$x, y = spots$y, row.names = spots$barcode)
  puck <- SpatialRNA(coords, counts, setNames(spots$nUMI, spots$barcode))
  rctd <- create.RCTD(puck, reference, max_cores = cores)
  rctd <- run.RCTD(rctd, doublet_mode = "full")
  w <- as.matrix(normalize_weights(rctd@results$weights))
  colnames(w) <- name_map[colnames(w)]
  dt <- data.table(barcode = rownames(w), w)
  fwrite(dt, f_out, sep = "\t")
  msg(s, ": ", nrow(w), " of ", nrow(spots), " spots")
}

# ---- merge (all sections present)
fs <- file.path(d_out, paste0(setdiff(list.dirs(d_in, full.names = FALSE, recursive = FALSE), "reference"), ".tsv"))
if (all(file.exists(fs))) {
  all_w <- rbindlist(lapply(fs, fread), fill = TRUE)
  for (j in names(all_w)[-1]) set(all_w, which(is.na(all_w[[j]])), j, 0)
  fwrite(all_w, file.path(root, "pdac", "results", "rctd", "pdac_rctd_weights.tsv.gz"), sep = "\t")
  msg("merged: ", nrow(all_w), " spots x ", ncol(all_w) - 1, " cell types")
}
