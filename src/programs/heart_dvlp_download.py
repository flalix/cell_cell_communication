#!/usr/bin/env python
"""
heart_dvlp_download.py - download the processed data of Lazar et al. 2025 (Nat Genet 57:2756) from Mendeley Data.

Part 1 (doi:10.17632/fhtb99mdzd.1): Space Ranger outputs of the 38 Visium sections, metadata, extended tables/text,
       Cell Ranger h5 of the 21 scRNA libraries, ISS (large, skipped by default).
Part 2 (doi:10.17632/w65jtfsvpr.1): Seurat RDS objects (scRNA reference, ST clustering, ST harmony/semla).

    python heart_dvlp_download.py [--root ROOT] [--with-cellranger] [--with-rds NAME ...] [--with-iss]

Files land in ROOT (default /home/flavio/uv/cell_cell_communication/heart_dvlp/data):
    visium/<section>/filtered_feature_bc_matrix.h5, visium/<section>/spatial/{tissue_hires_image.png,...}
    metadata/, extended_tables/, extended_text/, cellranger/, rds/
Existing files with the right size are skipped. The file inventory is written to ROOT/mendeley_files.tsv.
(curl is used: the Mendeley API refuses the default urllib client.)
"""
import argparse
import json
import os
import subprocess
import sys
import time

import pandas as pd

API = "https://data.mendeley.com/public-api/datasets"
FILE_URL = "https://data.mendeley.com/public-files/datasets/{ds}/files/{fid}/file_downloaded"
DATASETS = {"fhtb99mdzd": "part1", "w65jtfsvpr": "part2"}
TOP_MAP = {"2_Visium_spaceranger_data_ST": "visium", 
           "4_Extended_data_tables": "extended_tables",
           "6_Extended_text": "extended_text", 
           "7_Metadata": "metadata",
           "1_Chromium_cellranger_data_SC": "cellranger", 
           "3_ISS": "iss", 
           "5_Extended_data_figures": "extended_figures",
           "R_objects": "rds"}


def _get_json(url):
    out = subprocess.run(["curl", "-s", "-m", "60", url], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def inventory():
    rows = []
    for ds in DATASETS:
        folders = _get_json(f"{API}/{ds}/folders/1")
        byid = {f["id"]: f for f in folders}

        def fpath(fid):
            parts = []
            while fid in byid:
                parts.append(byid[fid]["name"])
                fid = byid[fid].get("parent_id")
            return "/".join(parts[::-1])

        for fid in ["root"] + list(byid):
            for f in _get_json(f"{API}/{ds}/files?folder_id={fid}&version=1"):
                prefix = fpath(fid) + "/" if fid != "root" else ""
                rows.append(dict(dataset=ds, path=prefix + f["filename"], size=f.get("size"), id=f["id"]))
    
    return pd.DataFrame(rows)


def local_path(root, path):
    parts = path.split("/")
    if len(parts) == 1:
        return os.path.join(root, "extended_text", parts[0])
    parts[0] = TOP_MAP.get(parts[0], parts[0])
    return os.path.join(root, *parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/flavio/uv/cell_cell_communication/heart_dvlp/data")
    ap.add_argument("--with-cellranger", action="store_true")
    ap.add_argument("--with-iss", action="store_true")
    ap.add_argument("--with-figures", action="store_true")
    ap.add_argument("--with-rds", nargs="*", default=[], help="RDS file names of part 2, e.g. 220114_STdata_harmony_semla.rds")
    
    a = ap.parse_args()
    
    os.makedirs(a.root, exist_ok=True)
    
    df = inventory()
    df.to_csv(os.path.join(a.root, "mendeley_files.tsv"), sep="\t", index=False)
    
    keep = df.path.str.match(r"^(2_Visium|4_Extended_data_tables|6_Extended_text|7_Metadata)") | ~df.path.str.contains("/")
    keep &= df.dataset.eq("fhtb99mdzd")
    
    if a.with_cellranger:
        keep |= df.path.str.startswith("1_Chromium")
    if a.with_iss:
        keep |= df.path.str.startswith("3_ISS")
    if a.with_figures:
        keep |= df.path.str.startswith("5_Extended_data_figures")
    for name in a.with_rds:
        keep |= df.path.eq(f"R_objects/{name}")
    
    df_sel = df[keep]
    print(f"{len(df_sel)} files, {df_sel['size'].sum() / 1e9:.2f} GB")

    t0, n, fails = time.time(), 0, []
    for r in df_sel.itertuples():
        dst = local_path(a.root, r.path)
        if os.path.exists(dst) and os.path.getsize(dst) == r.size:
            continue
        
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        
        subprocess.run(["curl", "-sL", "--retry", "3", "-m", "3600", FILE_URL.format(ds=r.dataset, fid=r.id), "-o", dst])
        
        if not (os.path.exists(dst) and os.path.getsize(dst) == r.size):
            fails.append(r.path)
        n += 1
    
    print(f"downloaded {n} files in {time.time() - t0:.0f} s; failures: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
