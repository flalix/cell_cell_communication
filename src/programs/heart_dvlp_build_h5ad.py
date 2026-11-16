#!/usr/bin/env python
"""heart_dvlp_build_h5ad.py - build the two complete h5ad files of the heart_dvlp project (Lazar et al. 2025).

Steps (run once, from the project root):
  1. python src/programs/heart_dvlp_download.py                       # 38 Visium sections + metadata (Mendeley part 1)
  2. Rscript src/programs/export_seurat_rds.R heart_dvlp/data/rds/HDCA_heart_sc_full_extended.rds heart_dvlp/data/bundles/sc_reference
     Rscript src/programs/export_seurat_rds.R heart_dvlp/data/rds/HDCA_heart_clustering.rds       heart_dvlp/data/bundles/st_clustering
     (RDS objects: Mendeley part 2, doi:10.17632/w65jtfsvpr.1, R_objects/; heart_dvlp_download.py --with-rds <name>
      downloads them; the copies in data/rds/ come from /home/flavio/uv/HDCA_heart_dev/download_dataset/data)
     Rscript: an R with SeuratObject + Matrix (e.g. /home/flavio/miniforge3/envs/renv/bin/Rscript); Seurat itself is
     not needed.
  3. python src/programs/heart_dvlp_build_h5ad.py [--force] [--only visium|reference]

Writes heart_dvlp/data/hdca_heart_visium_38sections.h5ad and heart_dvlp/data/hdca_heart_sc_reference.h5ad.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT0 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT0 / "src"))
from libs.single_cell_heart_dvlp_lib import SingleCellHeart  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", choices=["visium", "reference"], default=None)
    a = ap.parse_args()
    sch = SingleCellHeart(ROOT0)
    t0 = time.time()
    if a.only in (None, "visium"):
        ad_vis = sch.build_visium_h5ad(force=a.force)
        print(ad_vis)
    if a.only in (None, "reference"):
        ad_ref = sch.build_reference_h5ad(force=a.force)
        print(ad_ref)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
