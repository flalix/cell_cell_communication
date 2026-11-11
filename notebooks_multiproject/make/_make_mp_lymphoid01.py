"""Generates mp_lymphoid01_prepare.ipynb (rewrite of notebooks_ccc_lymphoid_liana/ccc01_tutorial.ipynb with
SingleCellLymphoid). Run once; then edit the notebook freely. --force overwrites."""
import sys
import nbformat as nbf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _make_lymphoid_common import old_md

out = Path(__file__).resolve().parent.parent / "mp_lymphoid01_prepare.ipynb"   # notebooks_multiproject/
if out.exists() and "--force" not in sys.argv:
    raise SystemExit(f"{out} exists: not overwritten (use --force)")
o1 = old_md("ccc01_tutorial")
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""### Human lymph node - cell2location tutorial review (mp_lymphoid01)

Rewrite of **notebooks_ccc_lymphoid_liana/ccc01_tutorial.ipynb** with the class libraries; every calculation and plot of ccc01 is kept.

#### Libs: **src/libs**

- **single_cell_lib** - Class **SingleCell**: generic Scanpy, Squidpy, cell2location and LIANA methods; **DEFAULT_PARAMS**
- **single_cell_lymphoid_lib** - Class **SingleCellLymphoid(SingleCell)**: lymph node / lymphoid-organ methods; **LYMPHOID_PARAMS**, **LYMPHOID_LINEAGE**

| section | content |
|---|---|
| 1 | Visium lymph node section: components, selected genes on the tissue, MT genes |
| 2 | scRNA reference (lymphoid_organ_sc.h5ad): components, QC statistics, samples |
| 3 | tissue x age groups; mLN: naive / memory / GC genes by age group; dot plot lineage x age group |
| 4 | gene panel on the reference UMAP by tissue x age group |
| 5 | gene filter, NB regression of the reference signatures, posterior, QC |
| 6 | cell2location mapping, posterior, QC |
| 7 | cell abundance on the slide |
| 8 | regions: neighbours, Leiden, UMAP + spatial (3 plots), interactive map |
| 9 | Figure 1: reference UMAP by cell type + panel dot plot |
| 10 | NMF co-location; posterior samples and quantiles |
| 11 | dot plot lineage x NMF factors; cell-type specific expression (plot_genes_per_cell_type) |

**Data**: **lymphoid_tissue/data/lymphoid_organ_sc.h5ad** (73,260 cells, King / James / Park 2020) and the 10x Visium section **V1_Human_Lymph_Node**. The models of ccc01 (results/lymph_nodes_analysis) are re-used: training runs only when a cache file is missing (GPU recommended: **params['accelerator']**).

Figures and tables of this notebook have the prefix **lymph01_** (ccc01 outputs are not overwritten).""")
md(o1(0).replace("### Mapping human lymph node cell types to 10X Visium with Cell2location - tutorial review",
                 "#### Background (from ccc01)"))
md(o1(1))
md(o1(2))
md(o1(3, cut="#### Contents"))

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
import squidpy as sq

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42          # editable text in PDFs
%config InlineBackend.print_figure_kwargs = {'bbox_inches': 'tight', 'dpi': 90}   # inline display only; files are saved at 300 dpi
import plotly.io as pio
pio.renderers.default = "notebook_connected"

ROOT0 = Path('/home/flavio/uv/cell_cell_communication')
ROOT_SRC = ROOT0 / 'src'
if str(ROOT_SRC) not in sys.path:
    sys.path.append(str(ROOT_SRC))

from libs.single_cell_lymphoid_lib import SingleCellLymphoid, LYMPHOID_PARAMS, LYMPHOID_LINEAGE
from libs.single_cell_lib import cmap_byr
from libs.matplotlib_figure_style import apply_figure_style
from libs.Basic import create_dir, pdwritecsv, pdreadcsv

pd.set_option('display.max_colwidth', 120)
pd.set_option('display.width', 250)
print({p: version(p) for p in ['scanpy', 'squidpy', 'anndata', 'cell2location', 'scvi-tools', 'torch']})""")

md("""## Parameters

All file names, cutoffs and model settings are set here. **params** overrides **LYMPHOID_PARAMS** / **DEFAULT_PARAMS**; **scl.calc_df_params()** lists every parameter.""")
code("""project = 'lymphoid_tissue'

# ---- input files
root_visium_raw = ROOT0 / 'notebooks_ccc_lymphoid_liana' / 'data'   # Space Ranger output of sc.datasets.visium_sge
visium_id       = 'V1_Human_Lymph_Node'
fname_ref       = 'lymphoid_organ_sc.h5ad'                          # root_data
fname_annot     = 'cell_type_annotation.tsv'                        # root_data

# ---- cell2location runs (cache-or-train: training only when the h5ad is missing)
c2l_dir    = 'lymph_nodes_analysis'          # root_result / c2l_dir
ref_subdir = 'reference_signatures'          # sc.h5ad + model.pt
map_subdir = 'cell2location_map'             # sp.h5ad + model.pt + posterior_samples_w_sf.npz

# ---- output files (figures in root_figure, tables in root_result)
prefix = 'lymph01_'
fig_genes_visium   = prefix + 'visium_genes.png'
fig_mln_dotplot    = prefix + 'mln_panel_by_celltype_age.png'
fig_panel_umap     = prefix + 'bcell_genes_umap_by_tissue_age.png'
fig_ref_history    = prefix + 'ref_training_history.png'
fig_ref_qc         = prefix + 'ref_qc.png'
fig_map_history    = prefix + 'map_training_history.png'
fig_map_qc         = prefix + 'map_qc.png'
fig_abundance      = prefix + 'celltype_abundance_8types.png'
fig_slide_3types   = prefix + 'spatial_Tnaive_Bnaive_FDC.png'
fig_regions        = prefix + 'umap_region_sample.png'
fig_regions_html   = prefix + 'regions_spatial.html'
fig_umap_dotplot   = prefix + 'celltype_umap_dotplot.png'
fig_median_mean    = prefix + 'posterior_median_vs_mean.png'
fig_factor_dotplot = prefix + 'nmf_factors_lineage_dotplot_n{n}.png'
fig_genes_celltype = prefix + 'genes_per_cell_type.png'
tsv_mln_panel      = prefix + 'mln_panel_by_celltype_age.tsv'
tsv_regions        = prefix + 'region_labels.tsv'
tsv_factor_load    = prefix + 'nmf_factor_loadings_n{n}.tsv'

# ---- cutoffs / settings (override LYMPHOID_PARAMS / DEFAULT_PARAMS)
params = dict(
    accelerator='auto',            # 'gpu' on a CUDA machine; the cached models load on CPU too
    clip_quantile=99.2,            # colour caps of the spatial plots
    c2l_N_cells_per_location=30,   # tutorial hyper-prior
    c2l_detection_alpha=20,
    c2l_max_epochs=25000,
    ref_max_epochs=250,
    region_n_neighbors=15,         # kNN on q05 abundances
    region_resolution=1.1,         # Leiden (leidenalg, seed 0)
    region_leiden_flavor='leidenalg',
    region_top_n=3,                # region names: top 3 enriched cell types
    panel_cmap_num=2,              # blue-yellow-red, as the Visium plots
    coloc_n_facts=(11, 12),
)

apply_figure_style()
scl = SingleCellLymphoid(ROOT0, project, params=params)
root_data, root_figure, root_result = scl.root_data, scl.root_figure, scl.root_result
results_folder = create_dir(root_result, c2l_dir)
ref_run_name = create_dir(results_folder, ref_subdir)
run_name = create_dir(results_folder, map_subdir)
print(scl)
print(root_data, sorted(os.listdir(root_data)))""")
code("""df_params = scl.calc_df_params()
df_params[df_params.value.astype(str) != df_params.default.astype(str)]""")

md(f"""### 1. Visium data - adata_vis

{o1(13, drop_first_line=True)}

**scl.read_visium** reads the Space Ranger output the way **sc.datasets.visium_sge** does (obs_names = barcodes), with the Ensembl IDs as **var_names** and the symbols in **var['symbol']**.""")
md(o1(20))
code("""adata_vis = scl.read_visium(root_visium_raw, visium_id, verbose=True)
scl.summary(adata_vis)""")
code("""df_comp_vis = scl.calc_df_components(adata_vis)
df_comp_vis.groupby('slot').size().to_frame('n_keys').T""")
md("### Visium array: barcodes = row x col")
code("adata_vis.obs.head(3)")
code("""print(adata_vis.var.shape[0])
adata_vis.var.head(6)""")
md(o1(22).split("<div")[0])
code("""genes_cd = np.unique([x for x in adata_vis.var.symbol if x.startswith("CD")])
len(genes_cd), genes_cd[:20]""")
code("""# Memory B Cells: CD27+, CD24+,
# Core Pan B-Cell: CD19, MS4A1 (CD20):, CD79A / CD79B, PAX5, CD22
# FDC / B: CR2 (CD21)
# Naive B Cells: IGHD (IgD), IGHM (IgM), CD27-, IL4R, FCER2 (CD23)
genes_plot = ["AICDA", "CCR7",
              "CD19", "MS4A1", "CR2", "CD22", "CD24", "CD27", "CD28",       # CD21 = CR2, CD23 = FCER2 (below)
              "CD38", "CD52", "CD53", "CD79A", "CD79B",
              "CDS2",   # not an immune marker (CDP-diacylglycerol synthase 2); use CD2 if a T/NK marker was intended
              "FCER2", "IGHD", "IGHM", "IL4R", "LEF1", "SELL", "PAX5", "TCF7"]
genes_plot = list(np.unique(genes_plot))

found_vis, _ = scl.resolve_gene_ids(genes_plot, adata_vis)
print("genes to plot", len(genes_plot), "| in the Visium data", len(found_vis))
"; ".join(found_vis)""")
md("""### Visium spatial plot per gene

Colour = counts clipped at the 99.2th percentile per gene (**clip_for_plot**: for very sparse genes the 99.2th percentile of the non-zero values).""")
code("""genes_shown, figs = scl.spatial_scatter_genes(found_vis, adata_vis, ncols=4, fname=fig_genes_visium)
plt.show()""")
md("### Mitochondria-encoded (MT) genes\n\nRemoved for spatial mapping; their counts are kept in **obsm['MT']**.")
code("""n0 = adata_vis.n_vars
adata_vis = scl.remove_mt_genes(adata_vis)
print(f"genes {n0:,} -> {adata_vis.n_vars:,}; obsm['MT'] {adata_vis.obsm['MT'].shape}")""")

md("### 2. The reference: lymphoid_organ_sc.h5ad\n\n" + o1(30).split("\n", 1)[1])
md(o1(31))
code("""adata_ref = scl.read_h5ad(fname_ref, set_self=False, verbose=True)
print(f"{os.path.getsize(root_data / fname_ref) / 1e9:.2f} GB")
adata_ref""")
code("""scl.summary(adata_ref)""")
md("""**reindex_reference_var**: the symbols (index) go to **var['symbol']**, the Ensembl IDs (**var['GeneID-2']**) become the index for a correct match with the Visium genes; **.raw** is deleted (to be removed in a future version of the tutorial). It runs once (skipped when **var['symbol']** exists).""")
code("""scl.reindex_reference_var(adata_ref)
adata_ref.var.head(5).T.head(7)""")
code("""cols = ['barcode', 'batch', 'Donor', 'ID', 'n_counts', 'n_genes', 'new_celltype', 'percent_mito',
        'CellType', 'Tissue', 'VDJsum', 'Method', 'Population', 'PrelimCellType', 'PrelimCellType_new',
        'Age', 'Sex', 'Study', 'Subset', 'Subset_Broad', 'Subset_all', 'Subset_int', 'Subset_print',
        'S_score', 'Sample', 'cell_cycle_diff', 'doublet_score', 'predicted_doublet']
adata_ref.obs[cols].head(5).T""")
code("""print("CellType:", adata_ref.obs['CellType'].nunique(), "| new_celltype:", adata_ref.obs['new_celltype'].nunique(),
      "| Subset:", adata_ref.obs['Subset'].nunique(), "| donors:", adata_ref.obs['Donor'].nunique())""")
code("""obs_cols = ['Study', 'Tissue', 'Method', 'Donor', 'Sample', 'Age', 'Sex', 'Subset_Broad', 'Subset', 'phase',
            'predicted_doublet']
for c, vc in scl.value_counts(obs_cols, adata_ref).items():
    print(f"\\n## {c} ({adata_ref.obs[c].nunique(dropna=False)} levels)")
    print(vc.to_string())""")
code("""cols = ['n_counts', 'n_genes', 'percent_mito', 'doublet_score']
adata_ref.obs[cols].describe().round(2)""")
md("### X = sparse matrix of raw counts")
code("""print("X:", scl.x_info(adata_ref))
print("raw counts:", scl.is_raw_counts(adata_ref))
print("row sums vs n_counts:", np.asarray(adata_ref.X[:5].sum(1)).ravel(), adata_ref.obs.n_counts.values[:5])
print(adata_ref.uns['regression_mod'].keys() if hasattr(adata_ref.uns['regression_mod'], 'keys') else adata_ref.uns['regression_mod'])""")
code("""adata_ref.var[['symbol', 'GeneName-2', 'n_cells', 'nonz_mean']].head()""")
md("### Sample statistics")
code("""df_sample_stats = scl.calc_df_sample_stats(adata_ref)     # also writes obs['age_group']
df_sample_stats""")

md("### 3. Tissue x age groups\n\n" + o1(55, drop_first_line=True).replace("`age_group()` in libs/single_cell_lymphoid_lib.py", "`scl.age_group()`"))
md(o1(56))
code("""np.unique(adata_ref.obs["Tissue"])""")
md(o1(59))
code("""mln_genes = ["LEF1", "CCR7", "SELL", "TCF7", "CD27", "CD28", "AICDA"]

# mLN cells; log1p(CP10k) with sc.pp.normalize_total library sizes (row sums of X)
df_mln_mean, df_mln_frac, df_mln = scl.calc_df_tissue_age_expr(mln_genes, tissue='mLN', adata=adata_ref)
print(">> df_mln: mLN cells x genes", df_mln.shape)
df_mln_mean""")
code("""df_mln.head(3)""")
code("""# fraction of mLN cells with expression > 0
df_mln_frac""")
md("""#### Dot plot lineage x age group (mLN)

Rows = reference cell types (**Subset**) grouped by lineage (**LYMPHOID_LINEAGE**; label colour = lineage), column blocks = genes, columns = age groups. Cell types are shown when they have >= 20 cells in at least 2 age groups (**panel_min_cells**). Size = % cells > 0, colour = mean log1p(CP10k) scaled to the maximum per gene (library size **obs['n_counts']**).""")
code("""lineage = LYMPHOID_LINEAGE
df_mln_long, df_mln_plot, df_mln_age = scl.calc_df_panel_celltype_age(mln_genes, tissue='mLN', adata=adata_ref,
                                                                       fname=tsv_mln_panel)
fig = scl.plot_dotplot_celltype_age(df_mln_plot, lineage=lineage, df_age=df_mln_age, fname=fig_mln_dotplot)
plt.show()
df_mln_age""")
md("### My selected genes - to observe them inside the lymph node")
code("""found_ref, ids_ref = scl.resolve_gene_ids(genes_plot, adata_ref)   # symbols -> Ensembl var_names
print(len(found_ref), 'of', len(genes_plot), 'genes found')
";".join(found_ref)""")
md(o1(68))

md("### 4. Gene panel on the reference UMAP by tissue x age group\n\n" + o1(71))
code("""apply_figure_style(sizes=(8, 7, 6))
fig = scl.plot_panel_umap_by_tissue_age(genes_plot, adata_ref, fname=fig_panel_umap, cmap_num=2)
plt.show()""")

md("### 5. Gene filter and reference signatures (NB regression)\n\n" + o1(78))
code("""n0 = adata_ref.n_vars
adata_ref, selected = scl.filter_genes_c2l(adata_ref, figsize=(3, 4))
plt.show()
print(f"genes {n0:,} -> {adata_ref.n_vars:,}")
selected[:10]""")
md(o1(80))
md(o1(90).replace("**max_epochs=250**", "**max_epochs=250** (params['ref_max_epochs'])"))
md("""**scl.fit_or_load_reference** = setup_anndata (batch **Sample**, labels **Subset**, covariate **Method**: 3' vs 5') -> train + export the posterior (1000 samples) + save, **or** load the saved model bound to adata_ref; then the posterior quantiles q05 / q50 / q95 / q0001 are exported to **varm** (this also creates **mod_reg.samples**, needed by plot_QC).""")
code("""mod_reg, adata_ref, ref_trained = scl.fit_or_load_reference(
    adata_ref, ref_run_name,
    labels_key=scl.params['ref_labels_key'],                     # cell type -> signatures
    batch_key=scl.params['ref_batch_key'],                       # 10X reaction / sample
    categorical_covariate_keys=list(scl.params['ref_covariate_keys']))   # multiplicative technical effects
mod_reg.view_anndata_setup()""")
md("### Cell type annotation")
code("""df_annot = pdreadcsv(fname_annot, root_data)
print(df_annot.shape)
df_annot[['cell_type_id', 'cell_type_name', 'marker_genes', 'description']]""")
md(o1(98))
code("""fig = scl.plot_c2l_history(mod_reg, 20, fname=fig_ref_history)
plt.show()""")
md(o1(100))
code("""list(adata_ref.varm.keys())""")
md(o1(109).replace("In this session `get_reference_signatures` returns **q50** (posterior median), because only quantiles were exported (see the comment in the training cell); the original tutorial uses **means**.",
                   "The signatures used for mapping are **q50** (posterior median, **params['c2l_signature_summary']**), as in ccc01; the original tutorial uses **means**."))
code("""df_inf_aver, summary_used = scl.get_reference_signatures(adata_ref, prefer=(scl.params['c2l_signature_summary'],))
print(f"Using {summary_used} signatures:", df_inf_aver.shape)
df_inf_aver.iloc[0:5, 0:5]""")
md(o1(105))
code("""figs = scl.plot_c2l_qc(mod_reg, summary_name="q50", figsize=(3, 3), fname=fig_ref_qc)""")

md("### 6. Cell2location: spatial mapping\n\n" + o1(112, drop_first_line=True))
md(o1(116))
md(o1(117).replace("(**N_cells_per_location=30**)", "(**N_cells_per_location=30**, params['c2l_N_cells_per_location'])"))
md("""**scl.fit_or_load_mapping**: genes shared with the signatures, setup_anndata (batch **sample**), Cell2location(N_cells_per_location, detection_alpha), training on all spots, posterior quantiles to **obsm**, expected expression per cell type in **layers** (needed by plot_genes_per_cell_type), save **or** read **sp.h5ad** + load the model and re-create **mod_c2l.samples**. The q05 abundances (\"at least this many cells\") are copied to **obs**.""")
code("""mod_c2l, adata_vis, map_trained = scl.fit_or_load_mapping(adata_vis, df_inf_aver, run_name)
print(f"{adata_vis.n_obs:,} spots x {adata_vis.n_vars:,} genes; layers: {len(adata_vis.layers)}")
mod_c2l.view_anndata_setup()""")
md("### ELBO loss history during training, first 1000 epochs removed")
code("""fig = scl.plot_c2l_history(mod_c2l, 1000, label='full data training', fname=fig_map_history)
plt.show()""")
md(o1(127))
code("""figs = scl.plot_c2l_qc(mod_c2l, summary_name="q50", figsize=(3, 3), fname=fig_map_qc)""")
md(o1(129))

md("### 7. Visualising cell abundance in spatial coordinates\n\n" + o1(130, drop_first_line=True))
code("""slide = scl.select_slide(adata_vis, visium_id)
cell_types_8 = ['B_Cycling', 'B_GC_LZ', 'T_CD4+_TfH_GC', 'FDC', 'B_naive', 'T_CD4+_naive', 'B_plasma', 'Endo']
fig = scl.plot_spatial_features(slide, cell_types_8, ncols=4, fname=fig_abundance)
plt.show()""")
md("""Several cell types in one panel (cell2location **plot_spatial**). **C2L_COLORS** = the colours plot_spatial uses, in the order of **clust_labels** (legend). FDC = follicular dendritic cells.""")
code("""C2L_COLORS = ["#F0E442", "#D55E00", "#56B4E9", "#009E73", "#5A14A5", "#C8C8C8", "#323232"]
clust_labels = ['T_CD4+_naive', 'B_naive', 'FDC']
fig = scl.plot_slide_loc_cell_types(slide, clust_labels, C2L_COLORS, figsize=(6, 6), fname=fig_slide_3types)
plt.show()""")

md("### 8. Downstream analysis: regions\n\n" + o1(133, drop_first_line=True))
md("""**scl.cluster_regions**: kNN (15 neighbours) on the q05 abundances, Leiden resolution 1.1 with the leidenalg backend (scanpy default) and seed 0 (**params['region_leiden_flavor']**; 'igraph' is faster but gives different clusters). **scl.umap_regions**: UMAP on that graph (min_dist 0.3, spread 1).""")
code("""scl.cluster_regions(adata_vis)
scl.umap_regions(adata_vis)
adata_vis.obs['region_cluster'].value_counts().sort_index().to_frame('n_spots').T""")
md("""Each region is named by its enriched cell types (**scl.region_names**): among cell types with cluster mean / global mean > 1.2, the top 3 by mean abundance (cells per spot). The same **names** are used by every region figure below.""")
code("""names = scl.region_names(adata_vis)
adata_vis.obs['region_name'] = adata_vis.obs['region_cluster'].astype(str).map(names).astype('category')
assert adata_vis.obs['region_name'].notna().all()

df_regions = adata_vis.obs[['region_cluster', 'region_name']].reset_index(names='barcode')
pdwritecsv(df_regions, tsv_regions, root_result, verbose=True)
names""")
code("""fig = scl.umap_spatial_3plot(adata_vis, names=names, fname=fig_regions)
plt.show()""")
md("### With Plotly, interactive (also saved as HTML)")
code("""fig = scl.plotly_spatial_visium(names, adata_vis)
fig.write_html(root_figure / fig_regions_html, include_plotlyjs='cdn')
fig.show()""")

md("### 9. Figure 1: reference UMAP by cell type + dot plot of the panel genes\n\n" + o1(145, drop_first_line=True))
code("""lineage = LYMPHOID_LINEAGE            # row order of the dot plot: cell types grouped by lineage
focus = ["B_GC_LZ", "B_GC_DZ", "B_Cycling", "B_preGC", "B_naive", "B_mem", "B_plasma",
         "T_CD4+_TfH_GC", "T_TIM3+", "FDC", "Macrophages_M1", "Monocytes"]
cmaps = scl.params['lineage_cmaps']     # lineage -> colormap

df_groups = scl.umap_cell_type_plot(adata_ref, genes_plot, lineage, focus, cmaps, fname=fig_umap_dotplot)
plt.show()
print(df_groups.shape)
df_groups.head(3)""")
code("""# the reference is not needed below
del adata_ref, mod_reg
import gc; gc.collect();""")

md("### 10. NMF co-location\n\n" + o1(150, drop_first_line=True))
md("""**scl.run_colocation_cached**: cell2location **run_colocation** (CoLocatedGroupsSklearnNMF, n_fact 11 and 12, 3 restarts) unless **adata_vis.uns** already holds **mod_coloc_n_fact11/12** (ccc01 saved them in sp.h5ad).""")
code("""adata_vis, res_dict, coloc_edited = scl.run_colocation_cached(adata_vis, run_name)
[k for k in adata_vis.uns if k.startswith('mod_coloc')]""")
md(o1(154))
md("### Posterior distribution and computing arbitrary quantiles\n\n" + o1(167, drop_first_line=True))
code("""ps = scl.posterior_samples_cached(mod_c2l, adata_vis, run_name)
assert ps['w_sf'].shape == (scl.params['c2l_num_samples'], adata_vis.n_obs, len(mod_c2l.factor_names_))
{k: v.shape for k, v in ps.items()}""")
md(o1(170))
code("""fig = scl.plot_posterior_median_vs_mean(mod_c2l, ps, site='w_sf', fname=fig_median_mean)
plt.show()
del ps""")

md("""### 11. Dot plot lineage x NMF factors

Relative weight of every cell type over the factors (each row sums to 1; **calc_df_coloc_loadings**). Rows = the 34 cell types grouped by lineage; columns = factors ordered by the row of their top cell type and labelled with their top-2 cell types, so each factor reads as a tissue compartment (e.g. germinal centre, T zone, medulla / plasma cells).""")
code("""df_factors = {}
for n in scl.params['coloc_n_facts']:
    fig, df_factors[n] = scl.plot_factor_dotplot(adata_vis, n, lineage=lineage, fname=fig_factor_dotplot.format(n=n),
                                                  fname_tsv=tsv_factor_load.format(n=n))
    plt.show()
df_factors[12].round(2).head(10)""")
md("### Cell-type specific expression of every gene in the spatial data (needed for NCEM)\n\n" + o1(157, drop_first_line=True))
md(o1(160))
code("""ctypes = ['T_CD4+_TfH_GC', 'T_CD4+_naive', 'B_GC_LZ']
genes = ['CD3D', 'CR2']
with mpl.rc_context({'axes.facecolor': 'black'}):
    slide = scl.select_slide(adata_vis, visium_id)
    fig, axs = scl.plot_genes_per_cell_type(slide, genes, ctypes)
    fig.savefig(root_figure / fig_genes_celltype, dpi=150, bbox_inches='tight')
    plt.show()""")
md("""**plot_genes_per_cell_type** is not part of the cell2location package (it usually needs customisation); here it is a method of **SingleCell** (libs/single_cell_lib.py), adapted from https://github.com/BayraktarLab/cell2location/blob/master/docs/notebooks/tutorial_utils.py.""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"display_name": "cell-cell-communication (3.12.12)", "language": "python", "name": "python3"}
nbf.write(nb, out)
print("written", out, len(C), "cells")
