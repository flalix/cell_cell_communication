"""Generates mp_lymphoid03_subregions.ipynb (structure of mp_mouse_brain03_subregions.ipynb). Run once; then edit the
notebook freely."""
import nbformat as nbf
from pathlib import Path

out = Path(__file__).resolve().parent.parent / "mp_lymphoid03_subregions.ipynb"   # notebooks_multiproject/
if out.exists():
    raise SystemExit(f"{out} exists: not overwritten")
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""### Human lymph node - subregions of each region (mp_lymphoid03)

For each of the 20 regions of mp_lymphoid01 (**lymph01_region_labels.tsv**: Leiden clusters of the cell2location abundances), the Visium spots of the region are clustered on **gene expression** (own UMAP per region) to look for **subregions**, which are then **mapped on the tissue** and described. It follows **mp_mouse_brain03_subregions.ipynb** (same methods and gene-set definitions) and continues **mp_lymphoid02_regions.ipynb**: the subregion signatures are scored on the scRNA reference, the reference whose UMAP is Figure 1 of mp_lymphoid02 (**scl.umap_cell_type_plot**).

#### Libs: **src/libs**

- **single_cell_lib**
  - Class **SingleCell**: generic methods; section **8c. Subregions** (prepare_expression, run_subregions, gene sets, tables, Plotly figures), shared with mp_mouse_brain03
  - **DEFAULT_PARAMS**
- **single_cell_lymphoid_lib**
  - Class **SingleCellLymphoid(SingleCell)**: add_region_ids, calc_df_subregion_ref_score, plotly_subregion_ref_heatmap, plot_ref_umap_subregion_scores
  - **LYMPHOID_PARAMS**, **LYMPHOID_LINEAGE**

| section | content |
|---|---|
| 1 | spots, regions (curated **lymph_node_regions.tsv**) and expression |
| 2 | region signatures (table region) |
| 3 | subclustering each region (own UMAP) |
| 4 | gene sets of each subregion (partition) |
| 5 | subregion table and diagnostics |
| 6 | each region: UMAP and map of its subregions |
| 7 | map of all subregions |
| 8 | subregion signatures on the scRNA reference (cell types; reference UMAP) |
| 9 | tables |

**Outputs**: Plotly figures (interactive here, saved as HTML in **root_figure**), one PNG (reference UMAP grid), tables in **root_result**; prefix **lymph03_**.

- **table region**: region_id, signature genes, functional description (+ size, enriched cell types)
- **table subregion**: region_id, sub_region_id, subregion signature genes, other genes, not present genes, functional description (+ diagnostics, reference cell types)
- long table of the 4 gene sets of every subregion (**.tsv.gz**)

**Data**: the cell2location map of mp_lymphoid01 (**sp.h5ad**; V1_Human_Lymph_Node, one section, 4,035 spots, 10,217 genes, raw counts) with its 20 region labels; the scRNA reference **lymphoid_organ_sc.h5ad** (73,260 cells, 34 cell types in **Subset**).""")

md("""### Critique of the request and how it is handled

1. **Gene sets must partition the genes** (total = region signature + subregion signature + not present + other). Implemented as 4 disjoint sets per subregion; a check asserts that the counts add up.
2. **\"Not present\" needs a definition.** Here: detected in **>= 30%** of the rest of the region, in **<= 5%** of the subregion, and significantly lower (Wilcoxon, BH FDR <= 0.05). In the lymph node this set is almost always **empty**: a 55-µm spot holds about 30 cells of several types, and lymphocytes share most of their genes, so a gene detected in 30% of a region's spots is rarely absent from a whole subregion. This is a property of the tissue, not a failure of the test.
3. **\"Other genes\" is very large** (several thousand detected genes per region). The table shows its count and the most expressed members; the full list is in the long gene-set file.
4. **Circularity.** The regions were made by clustering cell-type abundances; re-clustering abundances would rediscover the merged clusters. Subregions are therefore clustered on **gene expression**; abundances describe them. The reference scores of section 8 are gene-level, but cell2location used the same reference, so they are a consistency check, not independent evidence.
5. **One section from one donor.** There is no section batch to correct (no ComBat) and no \"single-section\" flag; nothing here can be replicated across sections or donors. The second UMAP panel of section 6 shows **log10 UMIs** instead of sections (depth is the main technical confounder).
6. **A UMAP cluster is not a subregion until it is mapped.** Every subregion is drawn on the H&E and gets a **spatial coherence ratio** (fraction of its spots' neighbours in the same subregion / chance).
7. **Borders mean something else in the lymph node.** The abundance regions are interleaved patches: on average 24-83% of a region's spot neighbours belong to another region. A fixed border cutoff (as in the brain) flags almost every subregion (66 of 69 at 0.25). Here a subregion is flagged **border with X** only when it is more on its region's edge than the region itself (**border_rel** = border_frac / region_border_frac >= 1.25, **params['border_min_rel']**).
8. **Depth.** **depth_ratio** (median UMIs of the subregion / region) is reported; no subregion falls below 0.6 or above 2.0 with the default parameters.
9. **Resolution limits.** Small regions (R16 88 spots, R18 43) stay one cluster; a minimum subregion size is enforced (**sub_min_size**).
10. **R19 is a low-quality strip** (median 945 UMIs vs 15,000-24,000 elsewhere; 3 cells per spot): it is in the region table but not subclustered (**subregion_skip**), like edge_LowQ in the brain.
11. **Region names.** The names of mp_lymphoid01 are the top-3 cell types and are not unique (clusters 5, 12 and 16 are all \"T_CD4+_naive / T_CD8+_naive / T_Treg\"), so regions get ids **R00-R19** (= region_cluster) and a curated name in **lymph_node_regions.tsv**.
12. **Functional descriptions are interpretations.** They rest on signature genes, enriched cell types (cell2location), the reference cell types expressing the signature, coherence and depth; anatomical words (sinus, medulla, trabecula, HEV) are inferred from genes, not read from the H&E. They are kept in an editable table (**lymph_node_subregion_descriptions.tsv**); entries whose signature changed after a parameter change are marked **CHECK**.""")

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
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
%config InlineBackend.print_figure_kwargs = {'bbox_inches': 'tight', 'dpi': 90}   # inline display only; files are saved at 300 dpi
import plotly.io as pio
pio.renderers.default = "notebook_connected"      # interactive figures in the notebook (plotly.js from CDN)

ROOT0 = Path('/home/flavio/uv/cell_cell_communication')
ROOT_SRC = ROOT0 / 'src'
if str(ROOT_SRC) not in sys.path:
    sys.path.append(str(ROOT_SRC))

from libs.single_cell_lymphoid_lib import SingleCellLymphoid, LYMPHOID_PARAMS, LYMPHOID_LINEAGE
from libs.matplotlib_figure_style import apply_figure_style
from libs.Basic import pdwritecsv, pdreadcsv

pd.set_option('display.max_colwidth', 120)
pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 200)
print({p: version(p) for p in ['scanpy', 'anndata', 'plotly']})""")

md("""## Parameters

**params** overrides **LYMPHOID_PARAMS** / **DEFAULT_PARAMS** (subclustering: **sub_**; gene sets and flags: **region_sig_**, **sub_sig_**, **notpres_**, **border_**, ...; reference scores: **ref_**).""")
code("""project = 'lymphoid_tissue'

# ---- input files
c2l_dir            = 'lymph_nodes_analysis'                      # root_result / c2l_dir
map_subdir         = 'cell2location_map'
fname_map          = 'sp.h5ad'                                   # cell2location map (mp_lymphoid01)
tsv_regions_prev   = 'lymph01_region_labels.tsv'                 # regions of mp_lymphoid01 (root_result)
fname_ref          = 'lymphoid_organ_sc.h5ad'                    # scRNA reference (root_data)
fname_regions      = 'lymph_node_regions.tsv'                    # curated region table (root_data, editable)
fname_descriptions = 'lymph_node_subregion_descriptions.tsv'     # curated subregion descriptions (root_data, editable)
visium_id          = 'V1_Human_Lymph_Node'

# ---- output files
prefix = 'lymph03_'
fig_region_heatmap = prefix + 'region_signature_heatmap.html'
fig_region_sub     = prefix + 'subregions_{region}.html'           # one per region
fig_diagnostics    = prefix + 'subregion_diagnostics.html'
fig_map            = prefix + 'subregion_map_{lib}.html'
fig_ref_heatmap    = prefix + 'subregion_ref_celltype_heatmap.html'
fig_ref_umap       = prefix + 'ref_umap_subregion_scores.png'
tsv_region_table   = prefix + 'region_table.tsv'
tsv_sub_table      = prefix + 'subregion_table.tsv'
tsv_gene_sets      = prefix + 'subregion_gene_sets.tsv.gz'
tsv_grid           = prefix + 'subcluster_resolution_grid.tsv'
tsv_ref_score      = prefix + 'subregion_ref_celltype_score.tsv'

params = dict(
    subregion_skip=('R19',),                                      # low-quality strip, not subclustered
    # subclustering (one section: no ComBat)
    sub_n_hvg=2000, sub_n_pcs=20, sub_n_neighbors=15,
    sub_resolutions=(0.2, 0.3, 0.4, 0.5, 0.6, 0.8), sub_min_size=20, sub_max_k=6,
    # gene sets
    region_sig_lfc=1.0, region_sig_padj=0.05, region_sig_min_pct=0.25,
    sub_sig_lfc=0.75, sub_sig_padj=0.05, sub_sig_min_pct=0.20,
    notpres_max_pct=0.05, notpres_min_pct_rest=0.30, notpres_padj=0.05,
    gene_universe='detected', universe_min_pct=0.01,
    # flags
    coherence_min_ratio=1.5, sub_min_sig_genes=3,
    low_depth_ratio=0.6, high_depth_ratio=2.0, border_min_frac=0.25, border_min_rel=1.25,
    # reference scores
    ref_sig_n_genes=25, ref_sig_min_genes=3,
    plotly_include_js='cdn',
)
apply_figure_style()
scl = SingleCellLymphoid(ROOT0, project, params=params)
root_data, root_figure, root_result = scl.root_data, scl.root_figure, scl.root_result
run_name = root_result / c2l_dir / map_subdir
print(scl)
print((run_name / fname_map).exists(), run_name / fname_map)""")

md("""### 1. Spots, regions and expression

The map is read without its per-cell-type expected-expression layers (**scl.read_h5ad_slots**). **scl.add_region_ids**: region_cluster of mp_lymphoid01 -> **obs['region_label']** = R00 ... R19 (and **obs['region_name']** = the mp_lymphoid01 name). **prepare_expression**: gene symbols as **var_names**, raw counts in **layers['counts']**, **X** = log1p(CP10k).""")
code("""adata_vis = scl.read_h5ad_slots(fname_map, root=run_name, skip=('layers',), verbose=True)
df_regions = pdreadcsv(tsv_regions_prev, root_result).set_index('barcode')
region_names = scl.add_region_ids(df_regions, adata_vis)
df_abund = scl.calc_df_abundance(adata_vis)                 # q05 cell abundance (cells per spot), 34 cell types

adata_expr = scl.prepare_expression(adata_vis)
df_lymph_regions = pdreadcsv(fname_regions, root_data)
print(adata_expr)
adata_expr.obs['region_label'].value_counts().sort_index().to_frame('n_spots').T""")
code("""# curated region table: id, name, compartment, description (mp_lymphoid01 name in c2l_name)
df_lymph_regions[['region_id', 'region_name', 'compartment', 'c2l_name', 'n_spots', 'description']]""")
code("""# sequencing depth and cell density per region: R19 is the low-quality strip
umis = pd.Series(np.asarray(adata_expr.layers['counts'].sum(1)).ravel(), index=adata_expr.obs_names)
df_depth = pd.DataFrame({'median_UMIs': umis, 'median_cells_per_spot': df_abund.loc[adata_expr.obs_names].sum(1)}) \\
             .groupby(adata_expr.obs['region_label'].values).median().round(0).astype(int)
df_depth.T""")

md("""### 2. Region signatures (table region)

Each region vs all other spots (Wilcoxon on log1p CP10k): signature = logFC >= 1, FDR <= 0.05, detected in >= 25% of the region's spots. Enriched cell types: mean cell2location density in the region / in all other spots. Several regions (R01, R04, R08, R09, R13) have almost no signature: they differ from the rest of the slide in cell-type proportions, not in genes that are specific at the spot level.""")
code("""df_rank_region, region_sig = scl.calc_df_region_signature(adata_expr)
df_region_table = scl.calc_df_region_table(adata_expr, df_rank_region, region_sig, df_abund, df_lymph_regions)
df_region_table[['region_id', 'region_name', 'n_spots', 'n_signature_genes', 'signature_genes', 'enriched_cell_types']]""")
code("""fig = scl.plotly_region_signature_heatmap(adata_expr, df_rank_region, n_per_region=3)
scl.save_plotly(fig, fig_region_heatmap)
fig.show()""")

md("""### 3. Subclustering each region (own UMAP)

**subcluster_adaptive** per region: genes detected in >= 2% of the region's spots, 2,000 HVGs (seurat_v3 on counts; dispersion on X when the loess fit fails on a very small region), PCA (20), kNN (15), UMAP, Leiden over **sub_resolutions**. The resolution kept is the highest at which every subregion has >= **sub_min_size** spots and there are <= **sub_max_k** subregions.""")
code("""subs, df_grid = scl.run_subregions(adata_expr)
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
code("""df_gene_sets = scl.calc_df_subregion_gene_sets(adata_expr, region_sig)
df_set_counts = df_gene_sets.groupby(['sub_region_id', 'gene_set'], sort=False).size().unstack(fill_value=0)
df_set_counts['total'] = df_set_counts.sum(1)
n_universe = df_gene_sets.groupby('sub_region_id', sort=False).size()
assert (df_set_counts['total'] == n_universe.loc[df_set_counts.index]).all()
print(f"{len(df_set_counts)} subregions; the 4 sets partition every universe")
pdwritecsv(df_gene_sets, tsv_gene_sets, root_result, verbose=True)
df_set_counts.head(10)""")

md("""### 5. Subregion table

Diagnostics: **coherence_ratio** (spatial contiguity; 1 = random), **depth_ratio** (median UMIs of the subregion / region), **border_region** / **border_frac** / **border_rel** (most frequent other region among its spots' neighbours; fraction; relative to the region's own border fraction), **status** (supported / weak + flags). Functional descriptions come from **lymph_node_subregion_descriptions.tsv** (curated); missing entries get an automatic text.""")
code("""df_descriptions = pdreadcsv(fname_descriptions, root_data)
df_sub_table = scl.calc_df_subregion_table(adata_expr, df_gene_sets, df_abund, df_descriptions=df_descriptions)

# requested columns first
first = ['region_id', 'sub_region_id', 'subregion_signature_genes', 'n_other', 'other_genes_top_expressed',
         'not_present_genes', 'functional_description']
df_sub_table = df_sub_table[first + [c for c in df_sub_table.columns if c not in first]]
print(df_sub_table.status.str.split(';').str[0].value_counts().to_dict())
print(df_sub_table.description_source.value_counts().to_dict())
df_sub_table[['sub_region_id', 'n_spots', 'coherence_ratio', 'depth_ratio', 'border_region', 'border_frac', 'border_rel',
              'n_subregion_signature', 'n_not_present', 'status']]""")
code("""fig = scl.plotly_subregion_diagnostics(df_sub_table)
scl.save_plotly(fig, fig_diagnostics)
fig.show()""")

md("""### 6. Each region: UMAP and map of its subregions

Per region: UMAP coloured by subregion, UMAP coloured by **log10 UMIs** (one section: depth replaces the section panel), and the subregions on the H&E (cropped to the region). Click a legend entry to hide / isolate a subregion in all panels; hover shows status and signature genes.""")
code("""for region, adata_sub in subs.items():
    fig = scl.plotly_region_subregions(region, adata_sub, adata_vis, df_sub_table=df_sub_table)
    scl.save_plotly(fig, fig_region_sub.format(region=region))
    fig.show()""")

md("""### 7. Map of all subregions

Every subregion in a shade of its region's colour (legend grouped by region; double-click a subregion to isolate it). Hover: status, signature genes, absent genes, enriched cell types. **R19** is grey (not subclustered).""")
code("""for lib in scl.library_ids(adata_vis):
    fig = scl.plotly_subregion_map(adata_expr, adata_vis, lib, df_sub_table=df_sub_table)
    scl.save_plotly(fig, fig_map.format(lib=lib))
    fig.show()""")

md("""### 8. Subregion signatures on the scRNA reference

Which reference cell types express each subregion's signature? The reference is the one of mp_lymphoid02 (Figure 1: **scl.umap_cell_type_plot**, UMAP by cell type).

**scl.calc_df_subregion_ref_score**: for every **supported** subregion, the top 25 subregion-signature genes (by Wilcoxon score) found in the reference (>= 3 genes); cell score = mean over the genes of the per-gene z-score of log1p(CP10k); mean per **Subset**. Rare cell types with very specific genes (VSMC 40 cells, FDC) reach high scores; compare cell types **within a row**.

Caveats: the reference has no spatial information and mixes tissues, donors and 3' / 5' chemistries; cell2location used the same reference to map the abundances.""")
code("""adata_ref = scl.read_h5ad(fname_ref, set_self=False, verbose=True)
scl.reindex_reference_var(adata_ref)                  # symbols -> var['symbol'], Ensembl IDs -> index

df_ref_score, df_cell_score, genes_used = scl.calc_df_subregion_ref_score(adata_ref, df_gene_sets, df_sub_table)
ref_top = scl.calc_df_ref_top(df_ref_score, n=3)
pdwritecsv(df_ref_score.round(4).reset_index(names='sub_region_id'), tsv_ref_score, root_result, verbose=True)
print(df_ref_score.shape, "(subregions x cell types)")
ref_top.to_frame()""")
code("""fig = scl.plotly_subregion_ref_heatmap(df_ref_score, lineage=LYMPHOID_LINEAGE, df_sub_table=df_sub_table)
scl.save_plotly(fig, fig_ref_heatmap)
fig.show()""")
md("""Reference UMAP coloured by the signature score of each scored subregion (own scale per panel, clipped at the 99th percentile; scores < 0 grey). The title gives the top reference cell type.""")
code("""fig = scl.plot_ref_umap_subregion_scores(adata_ref, df_cell_score, list(df_ref_score.index), df_ref_score=df_ref_score,
                                         ncols=6, panel_size=2.6, fname=fig_ref_umap)
plt.show()""")

md("### 9. Tables")
md("""**Table region**""")
code("""# rebuilt now that the subregions exist (n_subregions)
df_region_table = scl.calc_df_region_table(adata_expr, df_rank_region, region_sig, df_abund, df_lymph_regions)
df_region_table.insert(2, 'compartment', df_region_table.region_id.map(dict(zip(df_lymph_regions.region_id, df_lymph_regions.compartment))))
first = ['region_id', 'signature_genes', 'functional_description']
df_region_table = df_region_table[first + [c for c in df_region_table.columns if c not in first]]
pdwritecsv(df_region_table, tsv_region_table, root_result, verbose=True)
df_region_table""")
md("""**Table subregion** (with the reference cell types of section 8; '' = not scored)""")
code("""df_sub_table['ref_cell_types'] = df_sub_table.sub_region_id.map(ref_top).fillna('')
pdwritecsv(df_sub_table, tsv_sub_table, root_result, verbose=True)
df_sub_table[['region_id', 'sub_region_id', 'status', 'subregion_signature_genes', 'not_present_genes',
              'enriched_cell_types', 'ref_cell_types', 'functional_description']]""")

md("""### Notes for interpretation

- **supported** = spatially coherent (ratio >= 1.5) with >= 3 own signature genes; **weak** subregions are not distinct from the rest of their region.
- **What the subregions add.** The regions of mp_lymphoid01 are lymphocyte compositions (B follicle, GC, T zone, plasma cells). Inside them, most **supported** subregions are **non-lymphocyte structures** that the abundance clustering merged into the lymphocyte background: vessels with smooth muscle (ACTA2, TAGLN, MYH11), venules / sinus endothelium (ACKR1, VWF, CLDN5, CLEC4G, LYVE1), sinus macrophages (SIGLEC1/CD169, CD209, MARCO, STAB1), the FDC network (FDCSP, CXCL13, CR2), IgG plasma-cell clusters and cycling GC dark-zone cells. The same structure types recur in many regions (e.g. FDC subregions in R01, R08, R10, R13, R17; plasma-cell subregions in R02, R04, R09, R13, R14), so they are better read as **tissue elements spread across regions** than as parts of one region.
- **T-zone regions** (R05, R12, R16) have few supported subregions: the paracortex is homogeneous at Visium resolution, apart from scattered vessels (weak R05_3, R12_2; probably high endothelial venules).
- **border with X** (relative criterion, see critique 7) marks subregions that sit on their region's edge more than the region as a whole.
- **One section, one donor**: nothing here is replicated; anatomical words in the descriptions are inferences from genes.
- Subregion IDs and gene sets depend on the parameters; after changing them, check **description_source** for **CHECK** entries and update **lymph_node_subregion_descriptions.tsv** (generator: **_make_lymph_node_descriptions.py**).""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"display_name": "cell-cell-communication (3.12.12)", "language": "python", "name": "python3"}
nbf.write(nb, out)
print("written", out, len(C), "cells")
