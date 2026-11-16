#!/usr/bin/env python
"""
pdac_build_h5ad.py - the complete h5ad files of the PDAC project (Chen et al. 2025, Cancer Cell; GEO GSE278694).

GEO publishes no h5ad: Space Ranger outputs of 21 Visium sections (GSE278687_RAW.tar) and combined MTX + metadata of
the scRNA (GSE278688) and FFPE snRNA (GSE278689). Downloads (pdac/data/geo/, 2.3 GB):

    B=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn
    curl -O $B/GSE278687/suppl/GSE278687_RAW.tar                 # -> pdac/data/geo/GSE278687/ ; tar xf -> raw/
    curl -O $B/GSE278688/suppl/GSE278688_sc_{barcodes.tsv,features.tsv,matrix.mtx,metadata.csv}.gz
    curl -O $B/GSE278689/suppl/GSE278689_sn_{barcodes.tsv,features.tsv,matrix.mtx,metadata.csv}.gz

Steps (SingleCellPDAC): unpack_visium -> pdac/data/visium/<section>/ ; build_sections_table -> pdac_sections.tsv ;
build_visium_h5ad -> pdac_visium_21sections.h5ad ; build_reference_h5ad -> pdac_sc_sn_reference.h5ad.

Usage:
    python src/programs/pdac_build_h5ad.py --step all            # or: visium | reference ; --force to rebuild
"""
import argparse
import sys
import time
from pathlib import Path

ROOT0 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT0 / "src"))

import warnings                                           # noqa: E402
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from libs.single_cell_pdac_lib import SingleCellPDAC      # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--step", choices=["visium", "reference", "all"], default="all")
p.add_argument("--force", action="store_true")
args = p.parse_args()

t0 = time.time()
scp = SingleCellPDAC(ROOT0)
if args.step in ("visium", "all"):
    scp.unpack_visium(force=args.force)
    scp.build_sections_table(force=args.force)
    a = scp.build_visium_h5ad(force=args.force)
    print(f"[{time.time() - t0:6.0f} s] visium {a.shape}")
    del a
if args.step in ("reference", "all"):
    a = scp.build_reference_h5ad(force=args.force)
    print(f"[{time.time() - t0:6.0f} s] reference {a.shape}")
