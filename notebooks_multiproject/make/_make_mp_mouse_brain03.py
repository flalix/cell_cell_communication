"""Generates mp_mouse_brain03_subregions.ipynb. Run once; then edit the notebook freely."""
import nbformat as nbf
from pathlib import Path

out = Path(__file__).resolve().parent.parent / "mp_mouse_brain03_subregions.ipynb"   # notebooks_multiproject/
if out.exists():
    raise SystemExit(f"{out} exists: not overwritten")
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""### Mouse brain - subregions of each brain region (mp_mouse_brain03)

For each of the 17 brain regions of mp_mouse_brain01 (**mouse_brain_regions.tsv**), the Visium spots of the region are clustered on **gene expression** (own UMAP per region) to look for **subregions**, which are then **mapped on the tissue** and described.

**Outputs** (all figures are Plotly, interactive here and saved as HTML in **root_figure**)

- **table region**: region_id, signature genes, functional description (+ size, enriched cell types)
- **table subregion**: region_id, sub_region_id, subregion signature genes, other genes, not present genes, functional description (+ diagnostics)
- long table of the 4 gene sets of every subregion (**.tsv.gz**)

**Data**: the paper's cell2location map (2 sections: ST8059048 mouse 1, ST8059051 mouse 2; 5,396 spots, 31,040 genes, raw counts) with the region labels of mp_mouse_brain01.""")

md("""### Critique of the request and how it is handled

1. **Gene sets must partition the genes** (total = region signature + subregion signature + not present + other). Implemented as 4 disjoint sets per subregion; a check asserts that the counts add up.
2. **"Not present" needs a definition.** Most genes are zero in most Visium spots, so "absent" alone would list thousands of genes that are absent everywhere. Here: detected in **>= 30%** of the rest of the region, in **<= 5%** of the subregion, and significantly lower (Wilcoxon, BH FDR <= 0.05).
3. **"Other genes" is very large** (~14,000 detected genes per region). The table shows its count and the most expressed members (genes shared by the whole region). The full list is in the long gene-set file.
4. **Circularity.** The regions were made by clustering cell-type abundances; re-clustering abundances would rediscover the merged clusters. Subregions are therefore clustered on **gene expression**; abundances are used only to describe them.
5. **Two sections from two mice.** Clusters can split by section instead of anatomy: section correction (ComBat on the HVGs) and a **single-section** flag.
6. **A UMAP cluster is not a subregion until it is mapped.** Every subregion is drawn on the H&E and gets a **spatial coherence ratio** (fraction of its spots' neighbours in the same subregion / chance).
7. **Confounders found while doing it**: **sequencing depth** (low- or high-UMI spots form their own clusters and inflate gene-set sizes) and **borders** (55-µm spots at a region edge mix two regions). Both are reported (**depth_ratio**, **border_region**, flags in **status**).
8. **Resolution limits.** Small or thin regions (HPC_CA1 68 spots; cortical layers 1-2 spots thick) cannot be subdivided reliably; a minimum subregion size is enforced.
9. **edge_LowQ** is a tissue-rim artefact, not anatomy: it is in the region table but not subclustered.
10. **Functional descriptions are interpretations.** They are anchored on signature genes, enriched cell types (cell2location), the bordering region and the flags, and kept in an editable table (**mouse_brain_subregion_descriptions.tsv**); entries whose signature changed after a parameter change are marked **CHECK**.""")

md("## Loading packages")
code("""import os, sys
from pathlib import Path
from importlib.metadata import version
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import plotly.io as pio
pio.renderers.default = "notebook_connected"      # interactive figures in the notebook (plotly.js from CDN)

ROOT0 = Path('/home/flavio/uv/cell_cell_communication')
ROOT_SRC = ROOT0 / 'src'
if str(ROOT_SRC) not in sys.path:
    sys.path.append(str(ROOT_SRC))

from libs.single_cell_neuro_lib import SingleCellNeuro, NEURO_PARAMS
from libs.Basic import pdwritecsv, pdreadcsv

pd.set_option('display.max_colwidth', 120)
pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 200)
print({p: version(p) for p in ['scanpy', 'anndata', 'plotly']})""")

md("""## Parameters

**params** overrides **NEURO_PARAMS** / **DEFAULT_PARAMS** (subclustering: **sub_**; gene sets and flags: **region_sig_**, **sub_sig_**, **notpres_**, **subregion_**, ...).""")
code("""project = 'mouse_brain'

# ---- input files (relative to root_data)
fname_map          = 'c2l_paper_map/sp_with_clusters.h5ad'
fname_regions      = 'mouse_brain_regions.tsv'
fname_descriptions = 'mouse_brain_subregion_descriptions.tsv'     # curated subregion descriptions (editable)

# ---- output files
fig_region_heatmap = 'mb03_region_signature_heatmap.html'
fig_region_sub     = 'mb03_subregions_{region}.html'               # one per region
fig_diagnostics    = 'mb03_subregion_diagnostics.html'
fig_map            = 'mb03_subregion_map_{lib}.html'
tsv_region_table   = 'mb03_region_table.tsv'
tsv_sub_table      = 'mb03_subregion_table.tsv'
tsv_gene_sets      = 'mb03_subregion_gene_sets.tsv.gz'
tsv_grid           = 'mb03_subcluster_resolution_grid.tsv'

params = dict(
    region_label_override={'11': 'CTX_L1', '21': 'edge_LowQ'},   # as in mp_mouse_brain01
    subregion_skip=('edge_LowQ',),
    # subclustering
    sub_n_hvg=2000, sub_n_pcs=20, sub_n_neighbors=15, sub_batch='combat',
    sub_resolutions=(0.2, 0.3, 0.4, 0.5, 0.6, 0.8), sub_min_size=20, sub_max_k=6,
    # gene sets
    region_sig_lfc=1.0, region_sig_padj=0.05, region_sig_min_pct=0.25,
    sub_sig_lfc=0.75, sub_sig_padj=0.05, sub_sig_min_pct=0.20,
    notpres_max_pct=0.05, notpres_min_pct_rest=0.30, notpres_padj=0.05,
    gene_universe='detected', universe_min_pct=0.01,
    # flags
    coherence_min_ratio=1.5, sub_min_sig_genes=3, section_min_frac=0.10,
    low_depth_ratio=0.6, high_depth_ratio=2.0, border_min_frac=0.25,
    plotly_include_js='cdn',
)
scn = SingleCellNeuro(ROOT0, project, params=params)
root_data, root_figure, root_result = scn.root_data, scn.root_figure, scn.root_result
print(scn)""")

md("""### 1. Spots, regions and expression

**prepare_expression**: gene symbols as **var_names**, raw counts in **layers['counts']**, **X** = log1p(CP10k). The paper map also carries **.raw** (Ensembl names), which is dropped so every test runs on **X**.""")
code("""adata_map = scn.load_paper_map(fname_map)
df_regions = scn.calc_df_regions(adata_map)
scn.add_region_labels(df_regions, adata_map)
df_abund = scn.calc_df_abundance(adata_map)

adata_expr = scn.prepare_expression(adata_map)
df_brain_regions = pdreadcsv(fname_regions, root_data)
print(adata_expr)
adata_expr.obs['region_label'].value_counts().to_frame('n_spots').T""")

md("""### 2. Region signatures (table region)

Each region vs all other spots (Wilcoxon on log1p CP10k): signature = logFC >= 1, FDR <= 0.05, detected in >= 25% of the region's spots. Enriched cell types: mean cell2location density in the region / in all other spots.""")
code("""df_rank_region, region_sig = scn.calc_df_region_signature(adata_expr)
df_region_table = scn.calc_df_region_table(adata_expr, df_rank_region, region_sig, df_abund, df_brain_regions)
df_region_table[['region_id', 'n_spots', 'n_signature_genes', 'signature_genes', 'enriched_cell_types']]""")
code("""fig = scn.plotly_region_signature_heatmap(adata_expr, df_rank_region, n_per_region=3)
scn.save_plotly(fig, fig_region_heatmap)
fig.show()""")

md("""### 3. Subclustering each region (own UMAP)

**subcluster_adaptive** per region: genes detected in >= 2% of the region's spots, 2,000 HVGs (per section), ComBat by section, PCA (20), kNN (15), UMAP, Leiden over **sub_resolutions**. The resolution kept is the highest at which every subregion has >= **sub_min_size** spots and there are <= **sub_max_k** subregions.""")
code("""subs, df_grid = scn.run_subregions(adata_expr)
pdwritecsv(df_grid, tsv_grid, root_result)
df_grid[df_grid.chosen][['region_id', 'resolution', 'k', 'min_size']]""")

md("""### 4. Gene sets of each subregion (partition)

Per subregion (Wilcoxon, subregion vs rest of its region), universe = genes detected in >= 1% of the region's spots:

| set | definition |
|---|---|
| region_signature | in the region signature (section 2) |
| subregion_signature | logFC >= 0.75, FDR <= 0.05, detected in >= 20% of the subregion; not in the region signature |
| not_present | detected in >= 30% of the rest of the region, <= 5% of the subregion, FDR <= 0.05 |
| other | the rest of the universe |""")
code("""df_gene_sets = scn.calc_df_subregion_gene_sets(adata_expr, region_sig)
df_set_counts = df_gene_sets.groupby(['sub_region_id', 'gene_set'], sort=False).size().unstack(fill_value=0)
df_set_counts['total'] = df_set_counts.sum(1)
n_universe = df_gene_sets.groupby('sub_region_id', sort=False).size()
assert (df_set_counts['total'] == n_universe.loc[df_set_counts.index]).all()
print(f"{len(df_set_counts)} subregions; the 4 sets partition every universe")
pdwritecsv(df_gene_sets, tsv_gene_sets, root_result, verbose=True)
df_set_counts.head(10)""")

md("""### 5. Subregion table

Diagnostics: **coherence_ratio** (spatial contiguity; 1 = random), **depth_ratio** (median UMIs of the subregion / region), **border_region** / **border_frac** (most frequent other region among its spots' neighbours), section fractions, **status** (supported / weak + flags). Functional descriptions come from **mouse_brain_subregion_descriptions.tsv** (curated); missing entries get an automatic text.""")
code("""df_descriptions = pdreadcsv(fname_descriptions, root_data)
df_sub_table = scn.calc_df_subregion_table(adata_expr, df_gene_sets, df_abund, df_descriptions=df_descriptions)

# requested columns first
first = ['region_id', 'sub_region_id', 'subregion_signature_genes', 'n_other', 'other_genes_top_expressed',
         'not_present_genes', 'functional_description']
df_sub_table = df_sub_table[first + [c for c in df_sub_table.columns if c not in first]]
print(df_sub_table.status.str.split(';').str[0].value_counts().to_dict())
print(df_sub_table.description_source.value_counts().to_dict())
df_sub_table[['sub_region_id', 'n_spots', 'coherence_ratio', 'depth_ratio', 'border_region', 'n_subregion_signature',
              'n_not_present', 'status']]""")
code("""fig = scn.plotly_subregion_diagnostics(df_sub_table)
scn.save_plotly(fig, fig_diagnostics)
fig.show()""")

md("""### 6. Each region: UMAP and map of its subregions

Per region: UMAP coloured by subregion, UMAP coloured by section (a good subregion contains both sections), and the subregions on the H&E of each section (cropped to the region). Click a legend entry to hide / isolate a subregion in all panels; hover shows status and signature genes.""")
code("""for region, adata_sub in subs.items():
    fig = scn.plotly_region_subregions(region, adata_sub, adata_map, df_sub_table=df_sub_table)
    scn.save_plotly(fig, fig_region_sub.format(region=region.replace('/', '-')))
    fig.show()""")

md("""### 7. Map of all subregions

Both sections, every subregion in a shade of its region's colour (legend grouped by region; double-click a subregion to isolate it). Hover: status, signature genes, absent genes, enriched cell types. **edge_LowQ** is grey (not subclustered).""")
code("""for lib in scn.library_ids(adata_map):
    fig = scn.plotly_subregion_map(adata_expr, adata_map, lib, df_sub_table=df_sub_table)
    scn.save_plotly(fig, fig_map.format(lib=lib))
    fig.show()""")

md("### 8. Tables")
md("""**Table region**""")
code("""# rebuilt now that the subregions exist (n_subregions)
df_region_table = scn.calc_df_region_table(adata_expr, df_rank_region, region_sig, df_abund, df_brain_regions)
first = ['region_id', 'signature_genes', 'functional_description']
df_region_table = df_region_table[first + [c for c in df_region_table.columns if c not in first]]
pdwritecsv(df_region_table, tsv_region_table, root_result, verbose=True)
df_region_table""")
md("""**Table subregion**""")
code("""pdwritecsv(df_sub_table, tsv_sub_table, root_result, verbose=True)
df_sub_table[['region_id', 'sub_region_id', 'status', 'subregion_signature_genes', 'not_present_genes',
              'enriched_cell_types', 'functional_description']]""")

md("""### Notes for interpretation

- **supported** = spatially coherent (ratio >= 1.5) with >= 3 own signature genes; **weak** subregions are not distinct from the rest of their region.
- **low-depth / high-depth** subregions are partly technical: gene-set sizes and "not present" lists are inflated. **border with X** subregions mix two regions in the same 55-µm spots.
- **single-section** subregions may be real levels present in one section only (e.g. HYPO_4 mammillary region, STR_4 lateral hypothalamus) or section artefacts.
- Several subregions reveal **region-label errors** inherited from the region clustering (e.g. STR_4 is lateral hypothalamus, HAB_1 / HAB_2 / HAB_5 are not habenula, HPC_DG_3 / HPC_DG_4 are edge artefacts). A next step is to relabel these spots and rerun.
- Subregion IDs and gene sets depend on the parameters; after changing them, check **description_source** for **CHECK** entries and update **mouse_brain_subregion_descriptions.tsv**.""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"display_name": "cell-cell-communication (3.12.12)", "language": "python", "name": "python3"}
nbf.write(nb, out)
print("written", out, len(C), "cells")
