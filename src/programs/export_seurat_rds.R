#!/usr/bin/env Rscript
# export_seurat_rds.R - export a Seurat .rds (v3/v4/v5) to a plain bundle (Matrix Market + CSV) without the Seurat
# package: slots are read directly with SeuratObject, so SCTAssay / semla extras never have to be resolved.
# Assembled into h5ad by heart_dvlp_build_h5ad.py (or libs/single_cell_heart_dvlp_lib.assemble_bundle).
#
#   Rscript export_seurat_rds.R <input.rds> <outdir> [assay=RNA] [layer=counts]
#
# <outdir>/: counts.mtx (features x cells), barcodes.csv, features.csv, metadata.csv, reduction_<name>.csv,
#            manifest.json
# (adapted from HDCA_heart_dev/src/libs/convert_seurat_to_h5ad.R, 2026/11/15)
suppressPackageStartupMessages({ library(SeuratObject); library(Matrix) })

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript export_seurat_rds.R <input.rds> <outdir> [assay] [layer]")
in_rds <- args[[1]]; outdir <- args[[2]]
assay <- if (length(args) >= 3) args[[3]] else "RNA"
layer <- if (length(args) >= 4) args[[4]] else "counts"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

t0 <- Sys.time()
obj <- readRDS(in_rds)
message(sprintf("read %s in %.1f s; assays: %s", basename(in_rds), as.numeric(Sys.time() - t0, units = "secs"),
                paste(names(obj@assays), collapse = ",")))
a <- obj@assays[[assay]]
if (is.null(a)) stop(sprintf("assay %s not found", assay))

# Assay (v3/v4): slots counts/data; Assay5: list of layers
mat <- NULL
if ("layers" %in% slotNames(a)) {
  ln <- names(a@layers); hit <- ln[ln == layer | startsWith(ln, paste0(layer, "."))]
  if (length(hit) == 0) stop(sprintf("layer %s not in %s", layer, paste(ln, collapse = ",")))
  mats <- lapply(hit, function(h) a@layers[[h]])
  mat <- if (length(mats) == 1) mats[[1]] else do.call(cbind, mats)
  feats <- rownames(a@features)[seq_len(nrow(mat))]
  cells <- rownames(a@cells)[seq_len(ncol(mat))]
  rownames(mat) <- feats; colnames(mat) <- cells
} else {
  mat <- slot(a, layer)
  if (length(mat) == 0 || nrow(mat) == 0) stop(sprintf("slot %s is empty", layer))
}
mat <- as(mat, "CsparseMatrix")
writeMM(mat, file.path(outdir, "counts.mtx"))
write.csv(data.frame(barcode = colnames(mat)), file.path(outdir, "barcodes.csv"), row.names = FALSE)
write.csv(data.frame(feature = rownames(mat)), file.path(outdir, "features.csv"), row.names = FALSE)

meta <- obj@meta.data[colnames(mat), , drop = FALSE]
write.csv(meta, file.path(outdir, "metadata.csv"), row.names = TRUE)

reds <- names(obj@reductions)
for (r in reds) {
  emb <- obj@reductions[[r]]@cell.embeddings
  emb <- emb[intersect(colnames(mat), rownames(emb)), , drop = FALSE]
  write.csv(emb, file.path(outdir, sprintf("reduction_%s.csv", r)), row.names = TRUE)
}
writeLines(sprintf('{"source":"%s","assay":"%s","layer":"%s","n_features":%d,"n_cells":%d,"reductions":[%s],"meta_columns":%d}',
                   basename(in_rds), assay, layer, nrow(mat), ncol(mat),
                   paste(sprintf('"%s"', reds), collapse = ","), ncol(meta)),
           file.path(outdir, "manifest.json"))
message(sprintf("bundle %s: %d features x %d cells, reductions: %s (%.1f min)", outdir, nrow(mat), ncol(mat),
                paste(reds, collapse = ","), as.numeric(Sys.time() - t0, units = "mins")))
