"""Generates mp_mouse_brain02_region_astrocytes.ipynb (Fig 3, regional astrocyte subtypes). Run once; then edit the notebook freely."""
import nbformat as nbf
from pathlib import Path

out = Path(__file__).resolve().parent.parent / "mp_mouse_brain02_region_astrocytes.ipynb"   # notebooks_multiproject/
if out.exists():
    raise SystemExit(f"{out} exists: not overwritten")
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""### Mouse brain - cell2location paper, Figure 3: regional astrocyte subtypes (mp_mouse_brain02)

**Paper**: Kleshchevnikov V. *et al.* *Comprehensive mapping of tissue cell architecture via integrated single cell and spatial transcriptomics.* bioRxiv 2020.11.15.378125 (Nat Biotechnol 40, 661-671, 2022).

The authors: 3,013 astrocyte nuclei were re-integrated across the 6 snRNA-seq samples (BBKNN) and Leiden-clustered into 17 subclusters; clusters that were not distinct in location (first cell2location mapping) or markers were merged into **10 regional astrocyte subtypes**, with 4 grey-matter subtypes in the diencephalon (HYPO, THAL_hab, THAL_med, THAL_lat).

| section | content | paper |
|---|---|---|
| 1 | astrocyte reference (10 subtypes, 13 pre-merge clusters) | |
| 2 | UMAP of the subtypes | Fig 3A |
| 3 | subtype marker dot plot | Fig 3B |
| 4 | Visium map and brain regions | |
| 5 | subtype densities and brain regions | Fig 3C |
| 6 | relative density of each subtype across regions | Fig 3D |
| 7 | the 4 diencephalic subtypes | Fig 3E |
| 8 | marker expression in the diencephalon (smFISH substitute) | Fig 3F |

**Classes**: **SingleCellNeuro** (**single_cell_neuro_lib.py**, Fig 3 methods and **NEURO_PARAMS**) on top of **SingleCell** (**single_cell_lib.py**: dot plot, labelled UMAP, region outlines).

<div class="alert alert-warning">
<b>Differences from the paper</b>

- **Fig 3A-B** use the paper's astrocyte object (UMAP, 10 subtypes, 13 pre-merge clusters in **annotation_1_old**; the 17 original subclusters are not published) and raw counts of the same nuclei from **all_cells_20200625.h5ad**. The marker list was read from the published figure.
- **Fig 3C-E** use the paper's cell2location map (2 of the 5 sections: ST8059048 mouse 1, ST8059051 mouse 2) and our rule-based region labels (**mouse_brain_regions.tsv**), not the paper's 29 annotated regions.
- **Fig 3F**: the smFISH images are not public. It is replaced by Visium expression of the same markers (Ttll3, Angpt1, Lgr6, Aldh1a1) in the diencephalon plus a Spearman test against the subtype densities. Spot expression includes all cell types of a spot, so it is not astrocyte-specific.
</div>""")

md("## Loading packages")
code("""import os, sys
from pathlib import Path
from importlib.metadata import version
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42

ROOT0 = Path('/home/flavio/uv/cell_cell_communication')
ROOT_SRC = ROOT0 / 'src'
if str(ROOT_SRC) not in sys.path:
    sys.path.append(str(ROOT_SRC))

from libs.single_cell_neuro_lib import SingleCellNeuro, NEURO_PARAMS
from libs.matplotlib_figure_style import apply_figure_style
from libs.Basic import pdwritecsv, pdreadcsv

pd.set_option('display.max_colwidth', 120)
pd.set_option('display.width', 250)
print({p: version(p) for p in ['scanpy', 'squidpy', 'anndata', 'cell2location']})""")

md("""## Parameters

File names, cutoffs and figure settings. **params** overrides **NEURO_PARAMS**; the Fig 3 lists (subtype order, marker groups, panel subtypes, region areas, colours) are **NEURO_PARAMS** entries starting with **astro_**, **fig3** and **area_**.""")
code("""project = 'mouse_brain'

# ---- input files (relative to root_data = ROOT0 / project / 'data')
fname_astro      = 'snrna/mouse_brain_snRNA_with_UMAP_astrocytes_cellxgene.h5ad'   # paper astrocyte object
fname_snrna      = 'snrna/all_cells_20200625.h5ad'                                 # raw counts, all genes
fname_map        = 'c2l_paper_map/sp_with_clusters.h5ad'                           # paper cell2location map
fname_regions    = 'mouse_brain_regions.tsv'
fname_cell_types = 'mouse_brain_cell_types.tsv'

# ---- output files (figures in root_figure, tables in root_result)
fig_3A = 'mb02_fig3A_astro_umap.png'
fig_3B = 'mb02_fig3B_astro_markers.png'
fig_3C = 'mb02_fig3C_astro_regions_{lib}.png'
fig_3D = 'mb02_fig3D_astro_region_dotplot.png'
fig_3E = 'mb02_fig3E_diencephalic_astro.png'
fig_3F = 'mb02_fig3F_diencephalic_markers.png'
tsv_markers     = 'mb02_fig3B_marker_fraction_relative.tsv'
tsv_region_dens = 'mb02_fig3D_astro_region_density.tsv'
tsv_rho         = 'mb02_fig3F_marker_density_rho.tsv'

# ---- settings (override NEURO_PARAMS)
params = dict(
    region_label_override={'11': 'CTX_L1', '21': 'edge_LowQ'},   # as in mp_mouse_brain01 (edge fraction evidence)
    density_quantile=0.992,         # colour cap of density panels
    fig3F_smooth=True,              # Fig 3F display: mean over spot + 6 neighbours
    outline_min_spots=15,           # region outlines ignore groups of fewer spots
)
library_main = 'ST8059048'          # mouse 1 (Fig 3C); Fig 3E/3F show both sections

apply_figure_style()
scn = SingleCellNeuro(ROOT0, project, params=params)
root_data, root_figure, root_result = scn.root_data, scn.root_figure, scn.root_result
print(scn)""")
code("""df_params = scn.calc_df_params()
df_params[df_params.parameter.str.startswith(('astro_', 'fig3', 'area_', 'diencephalon', 'outline'))]""")

md("""### 1. Astrocyte reference

**load_astro_reference** takes the 3,013 astrocyte nuclei of the paper's astrocyte object (UMAP, colours, **annotation_1** = 10 subtypes, **annotation_1_old** = 13 clusters before the location-based merge) and their raw counts for all 31,053 genes from the full snRNA-seq file.""")
code("""adata_astro = scn.load_astro_reference(fname_astro, fname_snrna)
print(adata_astro.shape, 'raw counts:', scn.is_raw_counts(adata_astro))
adata_astro""")
code("""df_astro_sample = pd.crosstab(adata_astro.obs[scn.params['astro_col']], adata_astro.obs[scn.sample_col], margins=True)
df_astro_sample.loc[list(scn.params['astro_order']) + ['All']]""")
md("""**Merge**: which pre-merge clusters (**annotation_1_old**) form each of the 10 subtypes (e.g. Astro_AMY_1 + Astro_AMY_2 -> Astro_AMY).""")
code("""df_merge = pd.crosstab(adata_astro.obs[scn.params['astro_col_premerge']], adata_astro.obs[scn.params['astro_col']])
df_merge = df_merge.loc[:, [c for c in scn.params['astro_order'] if c in df_merge.columns]]
df_merge[df_merge.sum(1) > 0]""")
code("""df_cell_types = pdreadcsv(fname_cell_types, root_data, verbose=True)
df_cell_types[df_cell_types.broad_class == 'Astrocytes'][['cell_type_id', 'cell_type_name', 'n_nuclei', 'main_region', 'description']]""")

md("### 2. UMAP of the regional astrocyte subtypes (Fig 3A)")
code("""fig = scn.plot_astro_umap(adata_astro, fname=fig_3A)
plt.show()""")

md("""### 3. Astrocyte subtype markers (Fig 3B)

Dot size = fraction of nuclei with count > 0; colour = mean log1p(CP10k) relative to the subtype with the highest value (0-1 per gene). Marker groups as in the paper: pan-astrocyte, white matter, diencephalic, telencephalic and subtype-specific markers.""")
code("""fig, df_frac, df_rel = scn.plot_astro_markers(adata_astro, fname=fig_3B)
plt.show()""")
code("""df_markers = pd.concat({'frac': df_frac.T, 'rel': df_rel.T}, axis=1)
df_markers.columns = [f'{a}_{scn.astro_short(t)}' for a, t in df_markers.columns]
pdwritecsv(df_markers.round(4).reset_index(names='gene'), tsv_markers, root_result, verbose=True)""")
md("""**The 4 diencephalic markers validated by smFISH in the paper** - each should peak in its own subtype.""")
code("""rows = []
for t, g in scn.params['fig3F_markers'].items():
    col = df_rel.loc[:, g].iloc[:, 0] if df_rel.loc[:, g].ndim > 1 else df_rel.loc[:, g]
    frac = df_frac.loc[:, g].iloc[:, 0] if df_frac.loc[:, g].ndim > 1 else df_frac.loc[:, g]
    rows.append(dict(marker=g, paper_subtype=scn.astro_short(t), top_subtype=scn.astro_short(col.idxmax()),
                     frac_in_paper_subtype=round(float(frac[t]), 3), max_frac_other=round(float(frac.drop(t).max()), 3)))
df_marker_check = pd.DataFrame(rows)
df_marker_check""")

md("""### 4. Visium map and brain regions

The paper's cell2location map (cells per spot for the 59 types, including the 10 astrocyte subtypes) and the region labels of mp_mouse_brain01.""")
code("""adata_map = scn.load_paper_map(fname_map)
df_abund = scn.calc_df_abundance(adata_map)
df_regions = scn.calc_df_regions(adata_map)
scn.add_region_labels(df_regions, adata_map)
print(adata_map.shape, scn.library_ids(adata_map))
df_brain_regions = pdreadcsv(fname_regions, root_data, verbose=True)
adata_map.obs['region_label'].value_counts().to_frame('n_spots').T""")
code("""astro_types = list(scn.params['astro_order'])
df_abund[astro_types].describe().T[['mean', '50%', 'max']].round(3)""")

md("""### 5. Astrocyte subtype densities and brain regions (Fig 3C)

Left: densities of 5 subtypes (HYPO, CTX, HPC, THAL_lat, THAL_hab; colour scale 0 to the 99.2% quantile, cells per spot). Right: brain regions from location clustering. The paper shows both halves on one section; here they are two panels of the same section.""")
code("""for lib in scn.library_ids(adata_map):
    fig = scn.plot_astro_density_regions(adata_map, library_id=lib, df_abund=df_abund,
                                         colors=scn.astro_colors(adata_astro), fname=fig_3C.format(lib=lib))
    plt.show()""")

md("""### 6. Relative density of each astrocyte subtype across brain regions (Fig 3D)

Mean density per region over both sections, divided by the subtype's maximum over regions (0-1). Top bar: broad brain areas; bottom bar: regions (colours of **mouse_brain_regions.tsv** maps). The tissue-rim artefact **edge_LowQ** is not shown.""")
code("""df_astro_mean, df_astro_rel = scn.calc_df_astro_region_density(adata_map, df_abund=df_abund)
fig = scn.plot_astro_region_dotplot(df_astro_rel, fname=fig_3D)
plt.show()""")
code("""df_astro_peak = pd.DataFrame({'peak_region': df_astro_rel.idxmax(1),
                              'peak_density': df_astro_mean.max(1).round(3),
                              'second_region': df_astro_rel.apply(lambda r: r.drop(r.idxmax()).idxmax(), axis=1)})
df_astro_peak.index = [scn.astro_short(t) for t in df_astro_peak.index]
pdwritecsv(df_astro_mean.round(4).reset_index(names='astro_subtype'), tsv_region_dens, root_result, verbose=True)
df_astro_peak""")
md("""**Astro_STR** peaks in the hippocampus (CA1, CA3) rather than the striatum; the paper's Fig 3D also shows the STR row over the hippocampal columns.""")

md("""### 7. The 4 diencephalic astrocyte subtypes (Fig 3E)

Densities (white -> purple, cells per spot) over the H&E; yellow outlines = thalamus (THAL + HAB) and hypothalamus. The paper showed THAL_med and THAL_lat from a better section of the same brain; here both sections are shown (mouse 1 top, mouse 2 bottom).""")
code("""fig = scn.plot_diencephalic_astro(adata_map, df_abund=df_abund, fname=fig_3E)
plt.show()""")

md("""### 8. Marker expression in the diencephalon (Fig 3F substitute)

Visium log1p(CP10k) of the smFISH markers, diencephalon spots only (THAL, HAB, HYPO). Display: mean over each spot and its 6 neighbours (**fig3F_smooth**), because single spots detect these genes rarely.""")
code("""fig = scn.plot_diencephalic_markers(adata_map, fname=fig_3F)
plt.show()""")
md("""**Spearman rho** between each marker's spot expression (raw, not smoothed) and each diencephalic subtype's density, over the diencephalon spots of both sections. A marker that tracks its subtype has its highest rho on the diagonal.""")
code("""df_rho = scn.calc_df_marker_density_rho(adata_map, df_abund=df_abund)
print(f"diencephalon spots: {df_rho.attrs['n_spots']:,}")
pdwritecsv(df_rho.reset_index(names='marker'), tsv_rho, root_result, verbose=True)
df_rho""")

md("""### Notes for interpretation

- The subtype densities come from the paper's own cell2location run on 2 sections; the region names are our rule-based labels.
- Fig 3B marker fractions are computed from raw counts of the paper's 3,013 astrocyte nuclei; the published dendrogram row order is reproduced with **astro_order**.
- Fig 3F is not an smFISH result: Visium spots mix cell types, and Ttll3, Angpt1 and Lgr6 are detected in few spots. Aldh1a1 is also expressed outside astrocytes.""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"display_name": "cell-cell-communication (3.12.12)", "language": "python", "name": "python3"}
nbf.write(nb, out)
print("written", out, len(C), "cells")
