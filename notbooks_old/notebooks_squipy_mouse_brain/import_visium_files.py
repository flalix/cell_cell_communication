"""
Download the 5 mouse brain Visium sections from Kleshchevnikov et al. 2022 (cell2location; E-MTAB-11114)

and merge them into one AnnData: mouse_brain/data/visium_mouse_brain_5sections.h5ad
"""

import urllib.request, zipfile
from pathlib import Path
import pandas as pd, scanpy as sc, anndata as ad

ROOT0 = Path("/home/flavio/uv/cell_cell_communication")
root_data = ROOT0 / "mouse_brain" / "data"
root_data.mkdir(parents=True, exist_ok=True)

url = "https://cell2location.cog.sanger.ac.uk/tutorial/mouse_brain_visium_wo_cloupe_data.zip"
filename = root_data / "mouse_brain_visium_wo_cloupe_data.zip"          # ~771 MB

if not filename.exists():
    urllib.request.urlretrieve(url, filename)

fname_dir = root_data / "mouse_brain_visium_wo_cloupe_data"
if not fname_dir.exists():
    zipfile.ZipFile(filename).extractall(root_data)

meta = pd.read_csv(fname_dir / "Visium_mouse.csv")                      # sample sheet (slide / capture area)
ads = []
for s in meta["sample_name"]:
    a = sc.read_visium(fname_dir / "rawdata" / s, count_file="filtered_feature_bc_matrix.h5", library_id=s)
    a.var_names_make_unique()
    a.obs["sample"] = s
    a.var["SYMBOL"] = a.var_names
    a.var_names = a.var.pop("gene_ids")                           # Ensembl IDs as var_names
    a.obs_names = s + "_" + a.obs_names
    ads.append(a)

adata = ad.concat(ads, join="outer", uns_merge="unique")          # keeps all 5 images in uns['spatial']
adata.var["SYMBOL"] = ads[0].var["SYMBOL"].reindex(adata.var_names)
adata.var["MT_gene"] = adata.var["SYMBOL"].str.startswith("mt-")
adata.obs = adata.obs.merge(meta.set_index("sample_name"), left_on="sample", right_index=True, how="left")

filename = root_data / "visium_mouse_brain_5sections.h5ad"
adata.write_h5ad(filename, compression="gzip")

print(f"Merged AnnData object: {filename}")
print(adata)