"""Generates mp_lymphoid02_regions.ipynb (rewrite of notebooks_ccc_lymphoid_liana/ccc02_study.ipynb with
SingleCellLymphoid). Run once; then edit the notebook freely. --force overwrites."""
import sys
import nbformat as nbf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _make_lymphoid_common import old_md

out = Path(__file__).parent / "mp_lymphoid02_regions.ipynb"
if out.exists() and "--force" not in sys.argv:
    raise SystemExit(f"{out} exists: not overwritten (use --force)")
o2 = old_md("ccc02_study")
o1 = old_md("ccc01_tutorial")
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""### Human lymph node - cell2location study: B cells, mesenteric LN, ISG hotspot (mp_lymphoid02)

Rewrite of **notebooks_ccc_lymphoid_liana/ccc02_study.ipynb** with the class libraries. It continues **mp_lymphoid01_prepare.ipynb**: the cell2location map (**sp.h5ad**), the models and the region labels computed there are read, not recomputed.

#### Libs: **src/libs**

- **single_cell_lib** - Class **SingleCell**: generic Scanpy, Squidpy, cell2location and LIANA methods; **DEFAULT_PARAMS**
- **single_cell_lymphoid_lib** - Class **SingleCellLymphoid(SingleCell)**: lymph node / lymphoid-organ methods; **LYMPHOID_PARAMS**, **LYMPHOID_LINEAGE**

| section | content |
|---|---|
| 1 | read the previous calculations (cell2location map + mp_lymphoid01 regions) and the scRNA reference |
| 2 | mLN = mesenteric lymph node: naive / memory / GC genes by age group |
| 3 | B-cell study: signatures that separate the 10 B-cell subsets; marker dot plot |
| 4 | larger gene panel on the Visium section |
| 5 | reference statistics; panel by age group (dot plot lineage x age group); panel UMAP by tissue x age |
| 6 | AICDA (AID) by B subset, tissue, donor and chemistry |
| 7 | gene filter, NB regression (reference signatures), posterior, training plot, QC |
| 8 | cell2location mapping, posterior, QC |
| 9 | cell abundance on the slide |
| 10 | interferon-stimulated-gene (ISG) hotspot |
| 11 | Figure 1 (UMAP + dot plot), Figure 2 (spatial maps), Figure 3 (gene vs abundance correlation) |
| 12 | dot plot lineage x NMF factors |

Figures and tables of this notebook have the prefix **lymph02_**.""")
md(o2(0, drop_first_line=True).replace("ccc01_tutorial.ipynb", "ccc01_tutorial.ipynb (here: mp_lymphoid01_prepare.ipynb)"))

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
rcParams['pdf.fonttype'] = 42
%config InlineBackend.print_figure_kwargs = {'bbox_inches': 'tight', 'dpi': 90}   # inline display only; files are saved at 300 dpi

ROOT0 = Path('/home/flavio/uv/cell_cell_communication')
ROOT_SRC = ROOT0 / 'src'
if str(ROOT_SRC) not in sys.path:
    sys.path.append(str(ROOT_SRC))

from libs.single_cell_lymphoid_lib import SingleCellLymphoid, LYMPHOID_PARAMS, LYMPHOID_LINEAGE
from libs.matplotlib_figure_style import apply_figure_style
from libs.Basic import create_dir, pdwritecsv, pdreadcsv

pd.set_option('display.max_colwidth', 200)
pd.set_option('display.width', 250)
print({p: version(p) for p in ['scanpy', 'squidpy', 'anndata', 'cell2location', 'scvi-tools', 'torch']})""")

md("""## Parameters

**params** overrides **LYMPHOID_PARAMS** / **DEFAULT_PARAMS** (B-cell marker tiers: **marker_**, **unique_**, **shared_**; ISG hotspot: **isg_**).""")
code("""project = 'lymphoid_tissue'

# ---- input files
fname_ref      = 'lymphoid_organ_sc.h5ad'                # root_data
c2l_dir        = 'lymph_nodes_analysis'                  # root_result / c2l_dir
ref_subdir     = 'reference_signatures'
map_subdir     = 'cell2location_map'
fname_map      = 'sp.h5ad'                               # previous calculation: cell2location map (mp_lymphoid01 / ccc01)
tsv_regions_prev = 'lymph01_region_labels.tsv'                    # regions of mp_lymphoid01 (root_result)
visium_id      = 'V1_Human_Lymph_Node'

# ---- output files
prefix = 'lymph02_'
fig_bcell_dotplot  = prefix + 'bcell_subset_markers_dotplot.png'
fig_genes_visium   = prefix + 'visium_genes_{k}.png'
fig_mln_dotplot    = prefix + 'mln_panel_by_celltype_age.png'
fig_panel_umap     = prefix + 'genes_umap_by_tissue_age.png'
fig_ref_history    = prefix + 'ref_training_history.png'
fig_ref_qc         = prefix + 'ref_qc.png'
fig_map_history    = prefix + 'map_training_history.png'
fig_map_qc         = prefix + 'map_qc.png'
fig_abundance      = prefix + 'celltype_abundance_8types.png'
fig_slide_3types   = prefix + 'spatial_Tnaive_Bnaive_FDC.png'
fig_isg            = prefix + 'ifn_hotspot_summary.png'
fig_umap_dotplot   = prefix + 'celltype_umap_dotplot.png'
fig_spatial_maps   = prefix + 'celltype_spatial_maps.png'
fig_rho            = prefix + 'gene_celltype_spatial_correlation.png'
fig_factor_dotplot = prefix + 'nmf_factors_lineage_dotplot_n{n}.png'
csv_bcell_markers  = prefix + 'bcell_subset_unique_markers.csv'
tsv_mln_panel      = prefix + 'mln_panel_by_celltype_age.tsv'
csv_aicda_subset   = prefix + 'aicda_by_subset_tissue.csv'
csv_aicda_gc       = prefix + 'aicda_gc_by_donor.csv'
tsv_isg            = prefix + 'ifn_hotspot_celltype_enrichment.tsv'
tsv_rho            = prefix + 'gene_celltype_spatial_correlation.tsv'
tsv_factor_load    = prefix + 'nmf_factor_loadings_n{n}.tsv'

params = dict(
    accelerator='auto',
    clip_quantile=99.2,
    # B-cell subset markers (within-B comparison)
    marker_min_gene_frac=0.01, marker_padj=0.05, marker_min_pct_in=0.30,
    unique_max_pct_other=0.25, unique_min_log2fc=1.0,
    shared_max_pct_other=0.50, shared_min_log2fc=0.58,
    donor_min_cells=20,
    # ISG hotspot
    isg_quantile=0.97, isg_link_factor=1.6, isg_fdr=0.05,
    coloc_n_facts=(12,),
)

apply_figure_style()
scl = SingleCellLymphoid(ROOT0, project, params=params)
root_data, root_figure, root_result = scl.root_data, scl.root_figure, scl.root_result
results_folder = root_result / c2l_dir
ref_run_name = results_folder / ref_subdir
run_name = results_folder / map_subdir
print(scl)
print((run_name / fname_map).exists(), run_name / fname_map)""")

md(f"""### 1. Read the previous calculations

{o2(7, drop_first_line=True).replace("ccc01_tutorial.ipynb (results/lymph_nodes_analysis/cell2location_map/sp.h5ad)", "ccc01 / mp_lymphoid01 (results/lymph_nodes_analysis/cell2location_map/sp.h5ad)")}

The map is read **without** its 34 per-cell-type expected-expression layers (dense, about 5.6 GB; not used here): **scl.read_h5ad_slots**. The region labels are those of mp_lymphoid01.""")
code("""adata_vis = scl.read_h5ad_slots(fname_map, root=run_name, skip=('layers',), verbose=True)
scl.abundance_to_obs(adata_vis)                       # q05 abundance per cell type -> obs

df_regions = pdreadcsv(tsv_regions_prev, root_result).set_index('barcode')
adata_vis.obs['region_cluster'] = df_regions.loc[adata_vis.obs_names, 'region_cluster'].astype(str).astype('category')
adata_vis.obs['region_name'] = df_regions.loc[adata_vis.obs_names, 'region_name'].astype('category')
scl.summary(adata_vis)""")
code("""print(adata_vis.var.shape[0], "genes;", adata_vis.uns['spatial'].keys())
adata_vis.var.head(3)""")
code("""adata_vis.obs[['region_cluster', 'region_name']].value_counts().sort_index().to_frame('n_spots')""")
md(o2(19))
md("### The reference: lymphoid_organ_sc.h5ad\n\n" + o2(20).split("\n", 1)[1])
code("""adata_ref = scl.read_h5ad(fname_ref, set_self=False, verbose=True)
scl.reindex_reference_var(adata_ref)                  # symbols -> var['symbol'], Ensembl IDs -> index
scl.add_tissue_age_group(adata_ref)                   # obs['age_group']
adata_ref""")
code("""cols = ['barcode', 'batch', 'Donor', 'ID', 'n_counts', 'n_genes', 'new_celltype', 'percent_mito',
        'CellType', 'Tissue', 'VDJsum', 'Method', 'Population', 'PrelimCellType', 'PrelimCellType_new',
        'Age', 'age_group', 'Sex', 'Study', 'Subset', 'Subset_Broad', 'Subset_all', 'Subset_int', 'Subset_print',
        'S_score', 'Sample', 'cell_cycle_diff', 'doublet_score', 'predicted_doublet']
adata_ref.obs[cols].head(5).T""")

md("### 2. mLN = Mesenteric LN\n\n" + o2(65, drop_first_line=True) + "\n\n" + o1(59).split("#### mLN only")[0].split("### mLN = Mesenteric LN", 1)[1].split("The mesenteric Lymph Nodes", 1)[1].split("\n", 1)[1])
code("""mln_genes = ["LEF1", "CCR7", "SELL", "TCF7", "CD27", "CD28", "AICDA"]
df_mln_mean, df_mln_frac, df_mln = scl.calc_df_tissue_age_expr(mln_genes, tissue='mLN', adata=adata_ref)
mln_order = list(df_mln_mean.index)
print(">> df_mln: mLN cells x genes (log1p CP10k)", df_mln.shape, mln_order)

df_mln.groupby("age_group")[mln_genes].mean().reindex(mln_order).round(3)""")
code("""# fraction of mLN cells with expression > 0
df_mln_frac""")
code("""df_mln.head(3)""")

md("## 3. B-cell study\n\n" + o2(24, drop_first_line=True))
md("""**scl.prepare_subset_markers**: the 10 B subsets (38,309 cells), symbols as var_names, X = log1p(CP10k), genes detected in >= 1% of B cells, one-vs-rest Wilcoxon (**rank_genes_groups**, pts=True) within the B cells. **calc_df_group_pct_mean**: % expressing and mean CP10k per subset.""")
code("""bcell_types = ["B_GC_DZ", "B_GC_LZ", "B_GC_prePB", "B_Cycling", "B_preGC", "B_naive", "B_activated", "B_IFN", "B_mem", "B_plasma"]

adata_bcell = scl.prepare_subset_markers(adata_ref, bcell_types)
print(adata_bcell.shape, adata_bcell.obs.Subset.value_counts().reindex(bcell_types).to_dict())
print("genes tested:", adata_bcell.n_vars)

df_pct, df_mean = scl.calc_df_group_pct_mean(adata_bcell, bcell_types)
print(df_pct.shape)
df_pct.head(6)""")
md("""### Signatures: genes that are unique to one B subset

**calc_df_subset_markers** (thresholds in **params**): for every subset, % in the subset, max % in the other subsets, the next-highest subset, log2FC vs that subset (mean CP10k, pseudocount 0.05), Wilcoxon log2FC vs the rest of B and adjusted p; tiers **unique** / **enriched (shared)**; for the candidates, the donors in which the gene is still highest in that subset (**donors_top**).""")
code("""df_all, dff = scl.calc_df_subset_markers(adata_bcell, df_pct, df_mean, bcell_types, fname=csv_bcell_markers)

df_tier_summary = dff.groupby(["group", "tier"], observed=True).size().unstack(fill_value=0)
df_tier_summary""")
code("""# B cells per donor and subset (donor consistency uses pairs with >= donor_min_cells cells)
df_all.attrs['df_donor_counts']""")
code("""scl.print_subset_markers(dff, bcell_types)""")
code("""# summary table: unique genes per subset (first 10 rows of every group)
cols = ["group", "gene", "pct_in", "max_pct_other", "log2fc_vs_next", "donors_top"]

dftab = dff[dff.tier == "unique"].groupby("group", observed=True)[cols]
dftab.head(10).round(3)""")
md("""### Subsets without unique genes: best genes vs the next subset, and naive vs memory head to head""")
code("""for g in ["B_naive", "B_mem", "B_GC_LZ", "B_activated"]:
    df_b = scl.calc_df_best_vs_next(df_all, g)
    print(g, "| best vs next:", ", ".join(f"{r.gene}({r.pct_in*100:.0f}/{r.max_pct_other*100:.0f}%,{r.log2fc_vs_next:.2f} vs {r.next_mean_type})" for r in df_b.itertuples()))
    df_w = scl.wtab(g, adata_bcell.uns["rank_genes_groups"]).head(8)
    print("   top Wilcoxon vs rest-B:", ", ".join(f"{i}({row.lfc:.1f})" for i, row in df_w.iterrows()), "\\n")

df_nm = scl.calc_df_rank_genes(adata_bcell, groupby="Subset", groups=["B_naive"], reference="B_mem")
print("naive>mem:", df_nm.head(10)[["names", "logfoldchanges"]].round(2).values.tolist())
print("mem>naive:", df_nm.sort_values("scores").head(10)[["names", "logfoldchanges"]].round(2).values.tolist())""")
md("### Dot plot of the markers (4 per subset)")
code("""markers = {
    "B_GC_DZ":     ["SUGCT", "MME", "SH3TC1", "USP53"],
    "B_GC_LZ":     ["LMO2", "RGS13", "GMDS", "PLEK"],
    "B_GC_prePB":  ["FRZB", "BAIAP2L1", "ARHGAP10", "RASSF6"],
    "B_Cycling":   ["CDK1", "TOP2A", "RRM2", "UBE2C"],
    "B_preGC":     ["CCND2", "MIR155HG", "CRIP3", "LILRA4"],
    "B_naive":     ["TCL1A", "IGHD", "FCER2", "IGHM"],
    "B_activated": ["CD69", "CD83", "NR4A1", "NFKBIZ"],
    "B_IFN":       ["IFIT3", "IFIT1", "IFI44L", "MX2"],
    "B_mem":       ["CD27", "TNFRSF13B", "GPR183", "CLECL1"],
    "B_plasma":    ["DERL3", "FKBP11", "IGHG2", "PRDX4"]}

dp = scl.dotplot_bcell_markers(adata_bcell, markers, fname=fig_bcell_dotplot)
plt.show()""")
md(o2(35).replace("### ![Markers separating the 10 B-cell subsets]  (../figures/bcell_subset_markers_dotplot.png)", "")
          .replace("`figures/bcell_subset_markers_dotplot.png`", "`figures/lymph02_bcell_subset_markers_dotplot.png`")
          .replace("`results/bcell_subset_unique_markers.csv`", "`results/lymph02_bcell_subset_unique_markers.csv`"))
code("""del adata_bcell, df_all
import gc; gc.collect();""")

md("### 4. A larger gene panel on the Visium section\n\n" + o2(36, drop_first_line=True))
code("""# Memory B Cells: CD27+, CD24+,
# Core Pan B-Cell: CD19, MS4A1 (CD20):, CD79A / CD79B, PAX5, CD22
# Naive B Cells: IGHD (IgD), IGHM (IgM), CD27-, IL4R, FCER2 (CD23)
genes_plot = ["AICDA", "CCR7",
              "CD19", "MS4A1", "CR2", "CD22", "FCER2", "CD24", "CD27", "CD28",
              "CD38", "CD52", "CD53", "CD79A", "CD79B", "CDS2",
              "IGHD", "IGHM", "IL4R", "LEF1", "SELL", "PAX5", "TCF7",
              # extra: B-subset marker genes (table above), to observe in the cell-type signatures
              "DERL3", "CDK1", "IFIT3", "IFIT1", "BAIAP2L1", "ARHGAP10", "FRZB", "CCND2", "SH3TC1", "USP53", "SLC2A5", "MYADM",
              "TNFRSF13B", "CLECL1", "GPR183", "TCL1A", "RRM2", "HIST1H3B", "LILRA4", "CRIP3",
              "IGHG2", "IGHGP"]
genes_plot = list(np.unique(genes_plot))

found_vis, _ = scl.resolve_gene_ids(genes_plot, adata_vis)
print("genes", len(genes_plot), "| in the Visium map", len(found_vis), "| missing:", sorted(set(genes_plot) - set(found_vis)))""")
md("""### Visium spatial plot per gene (8 per figure)

Colour = counts clipped at the 99.2th percentile per gene (non-zero percentile for very sparse genes).""")
code("""genes_shown, figs = scl.spatial_scatter_genes(found_vis, adata_vis, ncols=4, nframe=8, fname=fig_genes_visium)
plt.show()""")

md("### 5. Reference statistics")
code("""obs_cols = ['Study', 'Tissue', 'Method', 'Donor', 'Sample', 'Age', 'Sex', 'Subset_Broad', 'Subset', 'phase',
            'predicted_doublet']
for c, vc in scl.value_counts(obs_cols, adata_ref).items():
    print(f"\\n## {c} ({adata_ref.obs[c].nunique(dropna=False)} levels)")
    print(vc.to_string())""")
code("""cols = ['n_counts', 'n_genes', 'percent_mito', 'doublet_score']
adata_ref.obs[cols].describe().round(2)""")
code("""print("X:", scl.x_info(adata_ref))
scl.calc_df_sample_stats(adata_ref)""")
md("""### Naive / memory / GC genes by age group

All cells (fraction > 0 and mean log1p(CP10k) per age group), then mLN by cell type (lineage) x age group.""")
code("""df_age_mean = scl.calc_df_gene_age_tissue_groups(mln_genes, adata_ref)
df_age_mean.pivot_table(index='gene', columns=['tissue', 'age_group'], values='mean_log1p_cp10k', sort=False)""")
code("""lineage = LYMPHOID_LINEAGE
df_mln_long, df_mln_plot, df_mln_age = scl.calc_df_panel_celltype_age(mln_genes, tissue='mLN', adata=adata_ref,
                                                                       fname=tsv_mln_panel)
fig = scl.plot_dotplot_celltype_age(df_mln_plot, lineage=lineage, df_age=df_mln_age, fname=fig_mln_dotplot)
plt.show()""")
md("### Split by age group and tissue: the larger panel on the reference UMAP\n\n" + o2(61, drop_first_line=True).replace("`age_group()` in libs/single_cell_lymphoid_lib.py", "`scl.age_group()`"))
md(o2(62))
code("""found_ref, _ = scl.resolve_gene_ids(genes_plot, adata_ref)
print(len(found_ref), 'of', len(genes_plot), 'genes found in the reference; not found:', [g for g in genes_plot if g not in found_ref])""")
code("""apply_figure_style(sizes=(8, 7, 6))
fig = scl.plot_panel_umap_by_tissue_age(genes_plot, adata_ref, fname=fig_panel_umap, cmap_num=2)
plt.show()""")

md("### 6. " + o2(76).lstrip("# ").replace("results/aicda_by_subset_tissue.csv", "results/lymph02_aicda_by_subset_tissue.csv")
                                       .replace("results/aicda_gc_by_donor.csv", "results/lymph02_aicda_gc_by_donor.csv"))
code("""aicda = scl.calc_gene_by_subset_tissue("AICDA", adata_ref, groups=bcell_types,
                                       fname_subset=csv_aicda_subset, fname_gc=csv_aicda_gc)
print(f"AICDA+ cells: {aicda['n_pos']:,}, of which in tonsil: {aicda['pct_pos_tonsil']}%")
print("tonsil GC B, % AICDA+ by chemistry:", aicda['chem_tonsil_gc'])
aicda['df_gc_share']""")
code("""aicda['df_subset']""")
code("""# % AICDA+ within GC B cells per donor x chemistry library with >= 20 GC B cells
aicda['df_gc_donor']""")

md("### 7. Gene filter and reference signatures\n\n" + o2(81) + "\n\n" + o2(82))
code("""n0 = adata_ref.n_vars
adata_ref, selected = scl.filter_genes_c2l(adata_ref, figsize=(5, 4))
plt.show()
print(f"genes {n0:,} -> {adata_ref.n_vars:,}")""")
md(o2(84))
code("""mod_reg, adata_ref, ref_trained = scl.fit_or_load_reference(
    adata_ref, ref_run_name, labels_key='Subset', batch_key='Sample', categorical_covariate_keys=['Method'])
mod_reg.view_anndata_setup()""")
md(o2(98))
code("""fig = scl.plot_c2l_history(mod_reg, 20, fname=fig_ref_history)
plt.show()""")
md(o2(100))
code("""df_inf_aver, summary_used = scl.get_reference_signatures(adata_ref, prefer=(scl.params['c2l_signature_summary'],))
print(f"Using {summary_used} signatures:", df_inf_aver.shape)
df_inf_aver.head(6).T.head(8)""")
md(o2(105))
code("""figs = scl.plot_c2l_qc(mod_reg, summary_name="q50", figsize=(3, 3), fname=fig_ref_qc)""")

md("### 8. Cell2location: spatial mapping\n\n" + o2(111, drop_first_line=True))
md(o2(117))
md(o2(118))
code("""region_obs = adata_vis.obs[['region_cluster', 'region_name']].copy()
mod_c2l, adata_vis, map_trained = scl.fit_or_load_mapping(adata_vis, df_inf_aver, run_name, read_layers=False)
adata_vis.obs[['region_cluster', 'region_name']] = region_obs.loc[adata_vis.obs_names]
mod_c2l.view_anndata_setup()""")
md("### ELBO loss history during training, first 1000 epochs removed")
code("""fig = scl.plot_c2l_history(mod_c2l, 1000, label='full data training', fname=fig_map_history)
plt.show()""")
md(o2(128))
code("""figs = scl.plot_c2l_qc(mod_c2l, summary_name="q50", figsize=(3, 3), fname=fig_map_qc)""")
md(o2(130))

md("### 9. Visualising cell abundance in spatial coordinates\n\n" + o2(131, drop_first_line=True))
code("""slide = scl.select_slide(adata_vis, visium_id)
cell_types_8 = ['B_Cycling', 'B_GC_LZ', 'T_CD4+_TfH_GC', 'FDC', 'B_naive', 'T_CD4+_naive', 'B_plasma', 'Endo']
fig = scl.plot_spatial_features(slide, cell_types_8, ncols=4, fname=fig_abundance)
plt.show()""")
code("""# cell2location plotter: several cell types in one panel; C2L_COLORS = plot_spatial colour order
C2L_COLORS = ["#F0E442", "#D55E00", "#56B4E9", "#009E73", "#5A14A5", "#C8C8C8", "#323232"]
# FDC - a follicular dendritic cell is a non-bone marrow-derived, mesenchymal stromal cell, not a classical dendritic cell
clust_labels = ['T_CD4+_naive', 'B_naive', 'FDC']
fig = scl.plot_slide_loc_cell_types(slide, clust_labels, C2L_COLORS, fname=fig_slide_3types)
plt.show()""")

md(f"""### 10. B_IFN: interferon-stimulated-gene (ISG) hotspot (IFIT1, IFIT3, ...)

**scl.run_isg_hotspot**: ISG score per spot = mean z-score of log1p(CP10k) of the ISGs found; hotspot = the largest spatially connected group of spots above the 97th percentile of the score (spots linked when closer than 1.6 x the spot spacing). Cell-type enrichment in the hotspot vs the rest of the slide: q05 abundance, Mann-Whitney U, BH FDR. Panels: ISG score, B_IFN abundance, *CXCL10*, DC_CCR7+ abundance (**params['isg_maps']**), and the log2FC bars (n.s. = FDR >= 0.05).""")
code("""ISG_GENES = ["IFIT3", "IFIT1", "IFIT2", "IFI44L", "ISG15",
             "MX1", "MX2", "IFI6", "RSAD2", "OAS1", "STAT1", "XAF1"]

df_hot = scl.run_isg_hotspot(adata_vis, ISG_GENES, fname_fig=fig_isg, fname_tsv=tsv_isg)
plt.show()""")
code("""df_hot.round(4).head(12)""")

md("### 11. Figures 1-3\n\n" + o2(141, drop_first_line=True))
code("""lineage = LYMPHOID_LINEAGE            # row order: cell types grouped by lineage
focus = ["B_GC_LZ", "B_GC_DZ", "B_Cycling", "B_preGC", "B_naive", "B_mem", "B_plasma",
         "T_CD4+_TfH_GC", "T_TIM3+", "FDC", "Macrophages_M1", "Monocytes"]
cmaps = scl.params['lineage_cmaps']

ct = adata_ref.obs["Subset"].astype(str).values
assert set(ct) == set(scl.lineage_order(lineage))
np.unique(ct)""")
md("#### Figure 1: reference UMAP by cell type + dot plot of the panel genes")
code("""dfg = scl.umap_cell_type_plot(adata_ref, genes_plot, lineage, focus, cmaps, fname=fig_umap_dotplot)
plt.show()
print(dfg.shape)
dfg.head(4)""")
md("#### Figure 2: spatial maps of the focus cell types (q05 abundance)")
code("""fig = scl.spatial_plot_bcell_types(adata_vis, focus, lineage, cmaps=cmaps, fname=fig_spatial_maps)
plt.show()""")
md("""#### Figure 3: gene (Visium) vs cell-type abundance correlation

Spearman rho across spots between the log1p(CP10k) of each panel gene and the mapped q05 abundance. Caveat: cell2location inferred the abundances from these same counts, so part of each correlation is built in; read it as consistency, not as independent validation.""")
code("""np.array(adata_vis.uns["mod"]["factor_names"])""")
code("""df_rho = scl.spatial_plot_cell_abundance(adata_vis, adata_ref, genes_plot, focus, lineage, cmaps=cmaps,
                                         fname_fig=fig_rho, fname_tsv=tsv_rho)
plt.show()
df_rho.head(6)""")

md("""### 12. Dot plot lineage x NMF factors

The NMF co-location of mp_lymphoid01 / ccc01 (12 factors, stored in **adata_vis.uns['mod_coloc_n_fact12']**). Relative weight of every cell type over the factors (rows sum to 1); rows grouped by lineage; factors ordered by the row of their top cell type and labelled with their top-2 cell types.""")
code("""adata_vis, res_dict, coloc_edited = scl.run_colocation_cached(adata_vis, run_name)    # cached: no NMF training
for n in scl.params['coloc_n_facts']:
    fig, df_load = scl.plot_factor_dotplot(adata_vis, n, lineage=lineage, cmaps=cmaps,
                                           fname=fig_factor_dotplot.format(n=n), fname_tsv=tsv_factor_load.format(n=n))
    plt.show()
df_load.round(2)""")
code("""# B-cell rows: in which compartment (factor) is each B subset?
df_b = df_load.loc[[c for c in bcell_types if c in df_load.index]]
pd.DataFrame({'top_factor': df_b.idxmax(axis=1), 'weight': df_b.max(axis=1).round(2)})""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"display_name": "cell-cell-communication (3.12.12)", "language": "python", "name": "python3"}
nbf.write(nb, out)
print("written", out, len(C), "cells")
