"""Generates mp_lymphoid04_liana_regions.ipynb and mp_mouse_brain04_liana_regions.ipynb from ONE template: the analysis
cells are identical; only the loading cell and the parameter cell differ (the test that the LIANA methods of
SingleCell, section 9, are parametrised well enough for two datasets). Usage: python _make_mp_liana_regions.py
[lymph|brain|all]. Existing notebooks are not overwritten."""
import sys
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # notebooks_multiproject/ (this script is in make/)

DS = {
 "lymph": dict(
  out="mp_lymphoid04_liana_regions.ipynb", obj="scl", prefix="lymph04_", n03="mp_lymphoid03_subregions.ipynb",
  title="Human lymph node - ligand-receptor communication in regions, subregions and between cell types (mp_lymphoid04)",
  based="Rewrite of **notebooks_ccc_lymphoid_liana/ccc03_liana_study.ipynb** (LIANA+ bivariate, cell2location-resolved sender -> receiver, reference rank_aggregate + co-localisation) with the class libraries, now **per region and per subregion**. The regions are those of **mp_lymphoid01** (Leiden on the cell2location abundances, shown on its UMAP; R00-R19) and the subregions those of **mp_lymphoid03** (each region clustered on gene expression, own UMAP).",
  data="**Data**: the cell2location map of mp_lymphoid01 (**sp.h5ad**, V1_Human_Lymph_Node, one section, 4,035 spots; read without its 34 per-cell-type layers), the mapping signatures (**reference_signatures/sc.h5ad**, 34 cell types) and the scRNA reference **lymphoid_organ_sc.h5ad** (73,260 cells, **Subset**). LR resource: LIANA **consensus** (human).",
  extra_critique="- **Reference gaps.** The reference has no fibroblastic reticular cells (FRC): CCL19 / CCL21 / CXCL12 are attributed to VSMC, Endo and DC_CCR7+, and CCL21 has no specific source in rank_aggregate.\n- **R19** (low-quality strip, mp_lymphoid03) is excluded from the group analyses (**subregion_skip**).",
  imports="from libs.single_cell_lymphoid_lib import SingleCellLymphoid, LYMPHOID_PARAMS, LYMPHOID_LINEAGE",
  params='''project = 'lymphoid_tissue'

# ---- input files
c2l_dir          = 'lymph_nodes_analysis'                 # root_result / c2l_dir
fname_map        = 'cell2location_map/sp.h5ad'            # cell2location map (mp_lymphoid01)
fname_sig        = 'reference_signatures/sc.h5ad'         # mapping signatures (means)
tsv_regions_prev = 'lymph01_region_labels.tsv'            # regions of mp_lymphoid01 (root_result)
fname_regions    = 'lymph_node_regions.tsv'               # curated region names (root_data)
tsv_sub_table    = 'lymph03_subregion_table.tsv'          # subregion table of mp_lymphoid03 (root_result)
fname_ref        = 'lymphoid_organ_sc.h5ad'               # scRNA reference (root_data)
ref_groupby, ref_symbol_col = 'Subset', None              # reference cell types; symbols are the var_names

params = dict(
    subregion_skip=('R19',),                              # low-quality strip: no subregions, no group analyses
    # subregions exactly as mp_lymphoid03
    sub_n_hvg=2000, sub_n_pcs=20, sub_n_neighbors=15, sub_resolutions=(0.2, 0.3, 0.4, 0.5, 0.6, 0.8),
    sub_min_size=20, sub_max_k=6,
    # LIANA (liana_resource: 'consensus' from LYMPHOID_PARAMS / DEFAULT_PARAMS)
    liana_bandwidth_spots=1.1, liana_cutoff=0.1, liana_nz_prop=0.05, liana_n_perms=100, liana_fdr=0.05,
    lr_group_min_auroc=0.65, lr_top_n=100, c2l_lr_n_perms=200, c2l_lr_group_n_perms=50, c2l_lr_min_z=3.0,
    ref_lr_cells_per_type=400, ref_lr_spec_rank=0.05, ref_lr_mag_rank=0.2, coloc_min=0.2, coloc_group_min=1.0,
)
apply_figure_style()
scl = SingleCellLymphoid(ROOT0, project, params=params)''',
  load='''adata_map = scl.read_h5ad_slots(fname_map, root=root_result / c2l_dir, skip=('layers',), verbose=True)
region_names_c2l = scl.add_region_ids(pdreadcsv(tsv_regions_prev, root_result).set_index('barcode'), adata_map)  # obs['region_label']
df_sig = scl.reference_signatures_symbols(fname_sig, root=root_result / c2l_dir, summary='means')   # genes x cell types
adata_ref = scl.read_h5ad(fname_ref, set_self=False, verbose=True)                                     # symbols as var_names''',
 ),
 "brain": dict(
  out="mp_mouse_brain04_liana_regions.ipynb", obj="scn", prefix="mb04_", n03="mp_mouse_brain03_subregions.ipynb",
  title="Mouse brain - ligand-receptor communication in regions, subregions and between cell types (mp_mouse_brain04)",
  based="Same analysis as **mp_lymphoid04_liana_regions.ipynb** (both notebooks are generated from one template by **_make_mp_liana_regions.py**; only the loading and parameter cells differ). The regions are those of **mp_mouse_brain01** (Leiden on the cell2location abundances of the paper map; 17 labelled regions) and the subregions those of **mp_mouse_brain03** (each region clustered on gene expression, own UMAP).",
  data="**Data**: the paper's cell2location map (**c2l_paper_map/sp_with_clusters.h5ad**; 2 sections, ST8059048 mouse 1 and ST8059051 mouse 2; 5,396 spots; no per-cell-type layers), the 59 regression signatures (**snrna/regression_signatures_sc.h5ad**) and the snRNA reference (**snrna/all_cells_20200625.h5ad** + 59-type annotation, 40,532 nuclei). LR resource: LIANA **mouseconsensus** (**NEURO_PARAMS['liana_resource']**).",
  extra_critique="- **Two sections, two mice.** The spatial kernel is built per section (sections are never connected); the bivariate and group statistics pool both sections.\n- **Neuronal resource content.** Many top pairs of the mouse resource are synaptic / adhesion pairs between neurons of one type (Nptx1->Nptxr, Rtn4->Lingo1); intracellular 'ligands' (Gnas, Arf1, Rims1, ...) are computed but not listed (**lr_display_exclude**).\n- **edge_LowQ** is excluded from the group analyses (**subregion_skip**).",
  imports="from libs.single_cell_neuro_lib import SingleCellNeuro, NEURO_PARAMS",
  params='''project = 'mouse_brain'

# ---- input files
fname_map        = 'c2l_paper_map/sp_with_clusters.h5ad'  # paper cell2location map (root_data)
fname_sig        = 'snrna/regression_signatures_sc.h5ad'  # 59 regression signatures (root_data)
fname_regions    = 'mouse_brain_regions.tsv'              # curated region names (root_data)
tsv_sub_table    = 'mb03_subregion_table.tsv'             # subregion table of mp_mouse_brain03 (root_result)
fname_ref        = 'snrna/all_cells_20200625.h5ad'        # snRNA reference, raw counts (root_data)
fname_annot      = 'snrna/snRNA_annotation_astro_subtypes_refined59_20200823.csv'
ref_groupby, ref_symbol_col = 'annotation_1', 'SYMBOL'    # reference cell types; symbols in var['SYMBOL']

params = dict(
    region_label_override={'11': 'CTX_L1', '21': 'edge_LowQ'},   # as in mp_mouse_brain01
    subregion_skip=('edge_LowQ',),                               # tissue-rim artefact: no group analyses
    # subregions exactly as mp_mouse_brain03
    sub_n_hvg=2000, sub_n_pcs=20, sub_n_neighbors=15, sub_batch='combat',
    sub_resolutions=(0.2, 0.3, 0.4, 0.5, 0.6, 0.8), sub_min_size=20, sub_max_k=6,
    # LIANA (liana_resource: 'mouseconsensus' from NEURO_PARAMS) - same values as mp_lymphoid04
    liana_bandwidth_spots=1.1, liana_cutoff=0.1, liana_nz_prop=0.05, liana_n_perms=100, liana_fdr=0.05,
    lr_group_min_auroc=0.65, lr_top_n=100, c2l_lr_n_perms=200, c2l_lr_group_n_perms=50, c2l_lr_min_z=3.0,
    ref_lr_cells_per_type=400, ref_lr_spec_rank=0.05, ref_lr_mag_rank=0.2, coloc_min=0.2, coloc_group_min=1.0,
)
apply_figure_style()
scn = SingleCellNeuro(ROOT0, project, params=params)''',
  load='''adata_map = scn.load_paper_map(fname_map)
df_regions = scn.calc_df_regions(adata_map)
scn.add_region_labels(df_regions, adata_map)                                   # obs['region_label']
df_sig = scn.signatures_from_regression(fname_sig)                             # genes (symbols) x 59 cell types
adata_ref = scn.read_h5ad(fname_ref, set_self=False, verbose=True)
adata_ref, _ = scn.add_paper_annotation(fname_annot, adata=adata_ref)          # obs['annotation_1'], annotated nuclei''',
 ),
}


def build(key):
    d = DS[key]
    o, P = d["obj"], d["prefix"]
    out = HERE / d["out"]
    if out.exists():
        print(f"{out} exists: not overwritten"); return
    nb = nbf.v4.new_notebook(); C = []
    md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
    code = lambda s: C.append(nbf.v4.new_code_cell(s.replace("{o}", o)))

    md(f"""### {d['title']}

{d['based']}

**Question**: which ligand-receptor (LR) pairs are spatially active **in each region / subregion**, and **between which cell types** do they act?

#### Libs: **src/libs**

- **single_cell_lib** - Class **SingleCell**, section **9. LIANA** (dataset-independent): spatial_weights, liana_bivariate, calc_df_lr_by_group, calc_celltype_expr, calc_df_celltype_lr, calc_df_group_celltype_links, liana_rank_aggregate_reference, calc_df_celltype_coloc, calc_df_ref_lr_support, calc_df_group_ccc + figures; **DEFAULT_PARAMS** (liana_*, lr_*, c2l_lr_*, ref_lr_*, coloc_*)
- **{'single_cell_lymphoid_lib' if key == 'lymph' else 'single_cell_neuro_lib'}** - Class **{'SingleCellLymphoid' if key == 'lymph' else 'SingleCellNeuro'}(SingleCell)**: data loading, regions, signatures; **{'LYMPHOID_PARAMS' if key == 'lymph' else 'NEURO_PARAMS'}**

| section | content |
|---|---|
| 1 | spots, regions, subregions, cell2location abundances |
| 2 | spatial weights (Gaussian kernel per section, bandwidth in spot spacings) |
| 3 | LIANA bivariate: global Moran's R and local score of every LR pair |
| 4 | LR pairs enriched in each region and subregion |
| 5 | between cell types: cell2location-resolved sender -> receiver (whole tissue) |
| 6 | between cell types inside each region / subregion |
| 7 | reference LIANA (rank_aggregate) + cell-type co-localisation |
| 8 | consensus per region / subregion |
| 9 | tables |

{d['data']}

Figures and tables have the prefix **{P}**.""")

    md(f"""### Critique of the request and how it is handled

- **What a Visium LR score can mean.** A spot (55 µm, 100 µm spacing) holds several to ~30 cells of several types. The bivariate local score says that ligand and receptor are co-expressed **in a neighbourhood** (~2 rings, ~200 µm), not that two cells touch or signal.
- **Regions vs subregions.** The same functions run on both grouping levels (**group** = region or subregion); every group is compared with the rest of the tissue (**AUROC** of the local score; 0.5 = as elsewhere) and by the fraction of significant high-high spots.
- **Statistics.** Neighbouring spots are not independent, so Mann-Whitney / permutation p-values are optimistic; groups and links are ranked by effect size (AUROC, score, z) and the cutoffs are parameters.
- **Between cell types.** Visium does not see cell types; cell2location does. The counts of each spot are split across cell types in proportion to abundance x reference signature (**calc_celltype_expr**; identical to the per-cell-type layers cell2location writes, checked on the lymph node map), the ligand is taken from the sender type and the receptor from the receiver type. Attribution therefore follows the reference: a gene absent from a reference type is never attributed to it.
- **Same-type links** (A -> A) appear wherever one type carries both ligand and receptor in the same spots (e.g. neurons of one layer); they are kept in the tables and left out of the link heatmaps.
- **Reference LIANA is not spatial.** rank_aggregate on the scRNA reference ranks source -> target pairs by expression; it is annotated with the spatial co-localisation of the two types (whole tissue and per group).
- **Uninformative pairs.** MHC / B2M and intracellular proteins listed as ligands in the resource (**lr_display_exclude**) are computed but not listed or counted as links.
{d['extra_critique']}""")

    md("## Loading packages")
    code(f"""import os, sys
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
%config InlineBackend.print_figure_kwargs = {{'bbox_inches': 'tight', 'dpi': 90}}   # inline display only; files are saved at 300 dpi
import plotly.io as pio
pio.renderers.default = "notebook_connected"

ROOT0 = Path('/home/flavio/uv/cell_cell_communication')
ROOT_SRC = ROOT0 / 'src'
if str(ROOT_SRC) not in sys.path:
    sys.path.append(str(ROOT_SRC))

{d['imports']}
from libs.matplotlib_figure_style import apply_figure_style
from libs.Basic import pdwritecsv, pdreadcsv

pd.set_option('display.max_colwidth', 120)
pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 200)
print({{p: version(p) for p in ['liana', 'scanpy', 'anndata', 'plotly']}})""")

    md("""## Parameters

**params** overrides the class defaults. The LIANA values are the same in mp_lymphoid04 and mp_mouse_brain04; dataset differences are the resource (class parameter **liana_resource**), the input files and the region / subregion settings.""")
    code(d["params"] + f"""

# ---- output files
prefix = '{P}'
h5ad_bivariate   = prefix + 'liana_bivariate.h5ad'          # cache (root_result)
tsv_ref_liana    = prefix + 'liana_reference_rank_aggregate.tsv'   # cache (root_result)
fig_lr_top       = prefix + 'lr_top_local_scores_{{lib}}.png'
fig_lr_regions   = prefix + 'lr_region_top_local_scores_{{lib}}.png'
fig_lr_region_hm = prefix + 'lr_by_region_heatmap.html'
fig_lr_sub_hm    = prefix + 'lr_by_subregion_heatmap.html'
fig_sender_recv  = prefix + 'celltype_lr_sender_receiver.png'
fig_links_region = prefix + 'celltype_links_by_region.html'
fig_links_sub    = prefix + 'celltype_links_by_subregion.html'
fig_coloc        = prefix + 'celltype_colocalisation.html'
tsv_weights      = prefix + 'spatial_weights.tsv'
tsv_inter        = prefix + 'liana_bivariate_global.tsv'
tsv_lr_region    = prefix + 'lr_by_region.tsv'
tsv_lr_sub       = prefix + 'lr_by_subregion.tsv'
tsv_ct           = prefix + 'celltype_lr_tissue.tsv.gz'
tsv_group_ct     = prefix + 'celltype_lr_by_group.tsv.gz'
tsv_links        = prefix + 'celltype_links_by_group.tsv'
tsv_ref_sup      = prefix + 'liana_reference_supported.tsv'
tsv_ccc_region   = prefix + 'ccc_consensus_by_region.tsv'
tsv_ccc_sub      = prefix + 'ccc_consensus_by_subregion.tsv'

root_data, root_figure, root_result = {o}.root_data, {o}.root_figure, {o}.root_result
print({o})
print({{k: {o}.params[k] for k in ['liana_resource', 'liana_bandwidth_spots', 'liana_nz_prop', 'lr_group_min_auroc', 'lr_display_exclude']}})""")

    md(f"""### 1. Spots, regions, subregions and cell2location abundances

**prepare_expression**: symbols as var_names (LIANA resources use symbols), raw counts in **layers['counts']**, X = log1p(CP10k). Subregions are recomputed with the parameters of **{d['n03']}** (**run_subregions** is deterministic; the subregion table of that notebook gives their status).""")
    code(d["load"] + """

df_abund = {o}.calc_df_abundance(adata_map)                 # q05 cell abundance (cells per spot)
adata_expr = {o}.prepare_expression(adata_map)
subs, _ = {o}.run_subregions(adata_expr)                     # obs['sub_region_id']
celltypes = [c for c in df_abund.columns if c in df_sig.columns]

df_region_info = pdreadcsv(fname_regions, root_data)
region_name = dict(zip(df_region_info.region_id, df_region_info.region_name))
df_sub03 = pdreadcsv(tsv_sub_table, root_result)
sub_supported = df_sub03.loc[df_sub03.status.str.startswith('supported'), 'sub_region_id'].tolist()

groups = {'region': {o}._groups(adata_expr, 'region_label'), 'subregion': {o}._groups(adata_expr, 'sub_region_id')}
print(adata_expr.shape, '|', len(celltypes), 'cell types |', groups['region'].nunique(), 'regions |',
      groups['subregion'].nunique(), 'subregions (', len(sub_supported), 'supported in 03 )')
print('sections:', {o}.library_ids(adata_expr))""")

    md("""### 2. Spatial weights

Gaussian kernel **w = exp(-d² / 2bw²)** with **bw = liana_bandwidth_spots x spot spacing**, built **per section** (block-diagonal: spots of different sections are never neighbours), self weight 1, weights < **liana_cutoff** removed. Expressing the bandwidth in spot spacings makes the kernel the same for data at different pixel scales.""")
    code("""df_w = {o}.spatial_weights(adata_expr)
W = adata_expr.obsp['spatial_connectivities']
pdwritecsv(df_w, tsv_weights, root_result)
df_w""")

    md("""### 3. LIANA bivariate: spatially co-expressed LR pairs

**li.mt.bivariate** (resource **liana_resource**): global **Moran's R** (bivariate spatial autocorrelation of ligand and receptor; permutation p, BH FDR over all tested pairs) and the **local cosine** score of every pair at every spot, with categories (1 = high-high) and local p-values. Ligand and receptor (all complex subunits) in >= **liana_nz_prop** of the spots. Cached in **root_result** (force=True recomputes).""")
    code("""df_inter, adata_lr = {o}.liana_bivariate(adata_expr, fname=h5ad_bivariate)
df_lr_sig = {o}.lr_significant(df_inter)
pdwritecsv(df_inter.reset_index(), tsv_inter, root_result)
print(f"LR pairs tested: {len(df_inter)}; Moran's R > 0 at FDR <= {{o}.params['liana_fdr']}: {len(df_lr_sig)}")
df_lr_sig[['ligand', 'receptor', 'ligand_props', 'receptor_props', 'morans', 'morans_fdr']].head(15).round(3)""")
    code("""lr_show = {o}._lr_display(df_lr_sig.index)[:8]
for lib in {o}.library_ids(adata_expr):
    fig = {o}.plot_lr_spatial(adata_lr, adata_expr, lr_show, library_id=lib, ncols=4, fname=fig_lr_top.format(lib=lib))
    plt.show()""")

    md("""### 4. LR pairs enriched in each region and subregion

**calc_df_lr_by_group**: per group, the local score of every significant pair in the group vs the rest of the tissue: **AUROC**, mean local score in / out, fraction of high-high spots (local p < 0.05) in / out. **enriched** = AUROC >= **lr_group_min_auroc** and more high-high spots than elsewhere.""")
    code("""df_lr_region = {o}.calc_df_lr_by_group(adata_lr, groups['region'], df_lr_sig.index)
df_lr_sub = {o}.calc_df_lr_by_group(adata_lr, groups['subregion'], df_lr_sig.index)
pdwritecsv(df_lr_region, tsv_lr_region, root_result)
pdwritecsv(df_lr_sub, tsv_lr_sub, root_result)

df_n = pd.DataFrame({'n_enriched': df_lr_region[df_lr_region.enriched].groupby('group').size()}) \\
         .reindex(sorted(groups['region'].dropna().unique())).fillna(0).astype(int)
df_n['region_name'] = df_n.index.map(region_name)
df_n['top_lr'] = {o}.calc_df_lr_group_top(df_lr_region)
df_n""")
    code("""fig = {o}.plotly_lr_group_heatmap(df_lr_region, n_per_group=3, group_names=region_name)
{o}.save_plotly(fig, fig_lr_region_hm)
fig.show()""")
    md("""The top enriched pair of each region on the tissue (region outlined in cyan):""")
    code("""top1 = df_lr_region[df_lr_region.enriched & df_lr_region.interaction.isin(set({o}._lr_display(df_lr_region.interaction.unique())))] \\
         .groupby('group').head(1)
reg_lab = adata_expr.obs['region_label'].astype(str).values
for lib in {o}.library_ids(adata_expr):
    fig = {o}.plot_lr_spatial(adata_lr, adata_expr, list(top1.interaction), library_id=lib, ncols=5, panel_size=3.0,
                              titles=[f"{g}: {i} (AUROC {a:.2f})" for g, i, a in zip(top1.group, top1.interaction, top1.auroc)],
                              outlines=[reg_lab == g for g in top1.group], fname=fig_lr_regions.format(lib=lib))
    plt.show()""")
    md("""Subregions: the **supported** subregions of the 03 notebook (spatially coherent, own signature).""")
    code("""df_n_sub = df_lr_sub[df_lr_sub.group.isin(sub_supported)]
print(f"{df_n_sub[df_n_sub.enriched].group.nunique()} of {len(sub_supported)} supported subregions have >= 1 enriched LR pair")
fig = {o}.plotly_lr_group_heatmap(df_lr_sub, n_per_group=2, groups=[g for g in sub_supported if g in set(df_lr_sub.group)])
{o}.save_plotly(fig, fig_lr_sub_hm)
fig.show()""")

    md("""### 5. Between cell types: cell2location-resolved sender -> receiver (whole tissue)

**calc_celltype_expr**: counts of every spot split across cell types (abundance x signature), log1p(CP10k). **calc_df_celltype_lr** for the **lr_top_n** significant pairs with the highest Moran's R:

**score[A, B] = mean over spots of (W_norm · L_A) x R_B** - the ligand carried by sender A in the spot's neighbourhood times the receptor carried by receiver B in the spot. Null: receptor spots permuted (**c2l_lr_n_perms**) -> z. **score_share** = share of the pair's total over all A, B.""")
    code("""df_lr_top = df_lr_sig.head({o}.params['lr_top_n'])[['ligand', 'receptor']]
genes = sorted({s for e in list(df_lr_top.ligand) + list(df_lr_top.receptor) for s in e.split('_')})
E, genes_E, celltypes_E = {o}.calc_celltype_expr(adata_expr, df_abund, df_sig, genes)
print('E (spots x cell types x genes):', E.shape, '| LR genes without signature:', sorted(set(genes) - set(genes_E)))

df_ct, df_group_ct = {o}.calc_df_celltype_lr(E, genes_E, celltypes_E, df_lr_top, W, groups=groups)
pdwritecsv(df_ct, tsv_ct, root_result)
pdwritecsv(df_group_ct, tsv_group_ct, root_result)
df_best = df_ct[df_ct.rank_in_lr == 1].set_index('interaction')
df_best = df_best.loc[[i for i in df_lr_top.index if i in df_best.index]]
print(df_ct.interaction.nunique(), 'LR pairs x', len(celltypes_E), 'x', len(celltypes_E), 'sender-receiver')
df_best[['sender', 'receiver', 'score', 'score_share', 'z']].head(15).round(3)""")
    code("""lr_panels = {o}._lr_display(df_best.index)[:6]
fig = {o}.plot_sender_receiver_heatmaps(df_ct, lr_panels, n_top=10, ncols=3, fname=fig_sender_recv)
plt.show()""")

    md("""### 6. Between cell types inside each region / subregion

The same score over the spots of each group; receptor spots permuted **within** the group (**c2l_lr_group_n_perms**) -> z; **ratio** = group score / whole-tissue score. **calc_df_group_celltype_links**: per group and sender -> receiver, the number of LR pairs with z >= **c2l_lr_min_z** and ratio >= 1.""")
    code("""df_links = {o}.calc_df_group_celltype_links(df_group_ct)
pdwritecsv(df_links, tsv_links, root_result)
lev = df_group_ct.drop_duplicates('group').set_index('group').level
df_links_reg = df_links[df_links.group.map(lev) == 'region']
df_links_reg[~df_links_reg.self_link].groupby('group').head(2)[['group', 'link', 'n_lr', 'lr']]""")
    code("""fig = {o}.plotly_group_celltype_links(df_links_reg, n_links=30, group_names=region_name,
                                     groups=sorted(df_links_reg.group.unique()))
{o}.save_plotly(fig, fig_links_region)
fig.show()""")
    code("""df_links_sub = df_links[df_links.group.isin(sub_supported)]
fig = {o}.plotly_group_celltype_links(df_links_sub, n_links=30, groups=[g for g in sub_supported if g in set(df_links_sub.group)])
{o}.save_plotly(fig, fig_links_sub)
fig.show()""")

    md("""### 7. Reference LIANA (rank_aggregate) + cell-type co-localisation

**liana_rank_aggregate_reference**: rank_aggregate (CellPhoneDB, Connectome, log2FC, NATMI, SCA, CellChat-like) on the reference, up to **ref_lr_cells_per_type** cells per type. Non-spatial, so each source -> target is annotated with **co-localisation** (**calc_df_celltype_coloc**): Pearson r between A at a spot and the kernel-weighted B around it (whole tissue), and each group's share of it. Specific and strong = specificity_rank <= **ref_lr_spec_rank** and magnitude_rank <= **ref_lr_mag_rank**; **spatially supported** = co-localisation r >= **coloc_min** or group share >= **coloc_group_min**.""")
    code("""df_liref = {o}.liana_rank_aggregate_reference(adata_ref, ref_groupby, symbol_col=ref_symbol_col, fname=tsv_ref_liana)
df_coloc, coloc_region = {o}.calc_df_celltype_coloc(df_abund.loc[adata_expr.obs_names, celltypes_E], W, groups['region'])
df_ref_sup = {o}.calc_df_ref_lr_support(df_liref, df_coloc, coloc_region)
pdwritecsv(df_ref_sup, tsv_ref_sup, root_result)
print(f"reference interactions: {len(df_liref)}; specific & strong: {len(df_ref_sup)}; spatially supported: {int(df_ref_sup.spatially_supported.sum())}")
df_ref_sup[df_ref_sup.spatially_supported & df_ref_sup.listed][['source', 'target', 'interaction', 'magnitude_rank', 'specificity_rank',
                                            'coloc', 'best_group', 'coloc_group']].head(15).round(3)""")
    code("""fig = {o}.plotly_coloc_heatmap(df_coloc)
{o}.save_plotly(fig, fig_coloc)
fig.show()""")

    md("""### 8. Consensus per region / subregion

**calc_df_group_ccc**: for each group, its top enriched LR pairs (section 4) with the best cell2location sender -> receiver of the pair **in the group** (section 6: z, ratio vs tissue), and the reference: **ref_same_link** = rank_aggregate finds the same LR between that sender and receiver (specific and strong), **ref_links** = best spatially supported reference source -> target of the pair.""")
    code("""df_ccc_region = {o}.calc_df_group_ccc(df_lr_region, df_group_ct[df_group_ct.level == 'region'], df_ref_sup)
df_ccc_region.insert(1, 'region_name', df_ccc_region.group.map(region_name))
pdwritecsv(df_ccc_region, tsv_ccc_region, root_result, verbose=True)
print('LR pairs with a cell-type link in the group:', int(df_ccc_region.c2l_sender.notna().sum()), 'of', len(df_ccc_region),
      '| same link in the reference:', int(df_ccc_region.ref_same_link.sum()))
df_ccc_region.round(2)""")
    code("""df_ccc_sub = {o}.calc_df_group_ccc(df_lr_sub[df_lr_sub.group.isin(sub_supported)],
                                    df_group_ct[df_group_ct.level == 'subregion'], df_ref_sup, n=3)
pdwritecsv(df_ccc_sub, tsv_ccc_sub, root_result, verbose=True)
df_ccc_sub.round(2)""")

    md(f"""### 9. Tables

All in **root_result** with prefix **{P}**: spatial weights, bivariate global table, LR by region / subregion, cell-type LR (tissue, by group), cell-type links by group, reference support, consensus by region / subregion; caches: bivariate h5ad, reference rank_aggregate.

### Notes for interpretation

- **AUROC** compares a group with the rest of the tissue; a pair can be enriched in a group only because its producers are concentrated there (composition), which is the point of the cell-type sections.
- **Sender -> receiver** calls follow the reference signatures x cell2location abundances; they say which types *can* carry ligand and receptor *where*, not that they signal.
- **z** values ignore spatial autocorrelation and are large for any co-localised pair; compare links by score, ratio and n_lr.
- **Parametrisation**: the same LIANA parameter values run unchanged on the lymph node (one section, human, 34 types) and the mouse brain (two sections, mouse, 59 types); only the resource, the files and the region settings differ.""")

    nb["cells"] = C
    nb["metadata"]["kernelspec"] = {"display_name": "cell-cell-communication (3.12.12)", "language": "python", "name": "python3"}
    nbf.write(nb, out)
    print("written", out, len(C), "cells")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for k in (DS if which == "all" else [which]):
        build(k)
