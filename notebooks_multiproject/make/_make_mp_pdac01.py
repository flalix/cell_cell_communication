"""Generates mp_pdac01_prepare.ipynb (structure of mp_heart_dvlp01_prepare.ipynb). Run once; then edit the notebook
freely (--force overwrites)."""
import sys
import nbformat as nbf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _make_pdac_common import HEADER_LIBS, IMPORTS, PAPER, md_code  # noqa: E402

out = Path(__file__).resolve().parent.parent / "mp_pdac01_prepare.ipynb"
if out.exists() and "--force" not in sys.argv:
    raise SystemExit(f"{out} exists: not overwritten (--force)")
nb = nbf.v4.new_notebook(); C = []
md, code = md_code(C)

md(f"""### PDAC and neural invasion - prepare: sections, reference, subclusters, deconvolution, regions (mp_pdac01)

{PAPER}

The authors combined **sc/snRNA-seq** (199,184 scRNA cells of tumour, adjacent normal and blood; 32,005 FFPE snRNA nuclei), **scTCR-seq**, **Visium** (21 sections of 17 patients; 81,714 spots after QC) and multiplex IHC to map PDAC tissues with **low** (8 patients) and **high** (17) neural invasion (NI): tertiary lymphoid structures (TLS) around non-invaded nerves in low-NI tumours, NLRP3+ macrophages and myCAF around invaded nerves, an endoneurial NRP2+ fibroblast population, three Schwann cell subsets (TGFBI+ Schwann cells at the leading edge of NI), and basal-like (CEACAM6) / neural-reactive (GABRP) malignant subpopulations with higher NI potential.

{HEADER_LIBS}

| section | content |
|---|---|
| 1 | the complete Visium h5ad: 21 sections, 17 patients, 3 platforms; section table; H&E gallery |
| 2 | Visium quality control per section and platform; MT genes |
| 3 | sc/sn reference: tissues, technologies, 14 major types; QC per sample |
| 4 | re-derived subclusters (ductal D01-D09, CAF F01-F08, myeloid M02-M14, Schwann S01-S03) |
| 5 | Figure 1B: reference UMAP by cell type + dot plot of 26 PDAC / NI genes |
| 6 | the 24 genes on the Visium sections (representative sections) |
| 7 | cell2location: reference signatures (cache of src/programs/run_c2l_pdac.py) |
| 8 | cell2location mapping (GPU) and RCTD (R, spacexr full mode; the paper's method) |
| 9 | cell-type abundance on the sections; cell2location vs RCTD |
| 10 | regions: Leiden on the cell-type shares; named after the paper's 12 niches |

**Data** (**pdac/data**, built by **src/programs/pdac_build_h5ad.py** from GEO GSE278687 / GSE278688 / GSE278689):

- **pdac_visium_21sections.h5ad**: 81,846 spots x 36,601 genes (union of the three panels), raw counts, hires + lowres H&E of every section; **var['in_all_panels']** = 17,368 genes measured by the three platforms.
- **pdac_sc_sn_reference.h5ad**: 231,189 cells x 30,053 genes (sc 5' v2 + snRNA Flex), raw counts, **obs['major_type']** = the 14 GEO types.
- **pdac_sections.tsv**: one row per section (patient, NI status, technique, panel, depth, matched sc / sn data, representative).

Figures and tables of this notebook have the prefix **pdac01_**.""")

md("""<div class="alert alert-warning">
<b>Differences from the paper (and from the brief)</b>
</div>

- **No h5ad is published** and GSE278694 is the SuperSeries: the 21 Visium sections are **GSE278687**, the scRNA **GSE278688**, the FFPE snRNA **GSE278689**. The h5ad files were assembled here (src/programs/pdac_build_h5ad.py).
- **Only the 14 major types are on GEO.** The paper's subclusters were **re-derived** (src/programs/pdac_reference_subclusters.py: scVI per compartment, Leiden, the paper's labels by marker scores). Recovered: D01-D09, F01/F02/F04-F08, M02-M04/M06-M14, S01-S03; **not recovered**: F03_tCAF-MME, M05_Mono-CD16. They are labels of this project, not the authors' cell assignments.
- **Three Visium platforms**: 5 fresh-frozen polyA (36,601 genes; PA03 and PA08 below 2,000 UMIs per spot), 4 FFPE v1 probe (17,943 genes) and 12 CytAssist v2 probe sections (18,085 genes). Joint analyses use the **17,368 genes of the three panels** (a gene outside a panel is 0, not 'not expressed').
- **All 5 low-NI sections are CytAssist**; every FF and FFPE-v1 section is high-NI. High vs low NI comparisons (mp_pdac02) are repeated on the 12 CytAssist sections (5 vs 7).
- **Deconvolution.** The paper used **RCTD** (full mode). This project runs RCTD (src/programs/run_rctd_pdac.R, renv R + spacexr in .R_lib) **and** cell2location (src/programs/run_c2l_pdac.py, GPU) on the same reference (tissue cells, PBMC excluded, <= 3,000 cells per type, 42 types); **abundance_source='auto'** uses cell2location when the map exists, otherwise RCTD.
- **Niches**: the paper clustered spots on expression (Harmony by section and technique, Louvain res 2) and assigned 12 niches by composition and H&E; here regions are Leiden clusters of the cell-type shares and are named after the paper's niches by composition rules (**assign_niches**). The authors' niche, RCTD and nerve annotations are not public; comparisons with the paper are qualitative (figures).""")

md("## Loading packages")
code(IMPORTS + "\nprint({p: version(p) for p in ['scanpy', 'squidpy', 'anndata', 'cell2location', 'scvi-tools', 'torch', 'liana']})")

md("""## Parameters

All parameters and file names are in the class (**src/libs/single_cell_pdac_lib.py**): **PDAC_PARAMS** (over **DEFAULT_PARAMS**) and **PDAC_FILES['inputs']** + **['pdac01']** (**scp.fname(key)**, **scp.file_path(key)**). Override in the constructor: `SingleCellPDAC(ROOT0, analysis='pdac01', params=dict(...))`.""")
code("""force=False       # True: recompute every table and redraw every figure (after changing parameters or inputs)
verbose=True

apply_figure_style()
scp = SingleCellPDAC(ROOT0, analysis='pdac01')
root_data, root_figure, root_result = scp.root_data, scp.root_figure, scp.root_result
ref_run_name = create_dir(scp.file_path('c2l_ref_dir'))
run_name     = create_dir(scp.file_path('c2l_map_dir'))
print(scp)
df_params = scp.calc_df_params()
display(df_params[df_params.value.astype(str) != df_params.default.astype(str)])
scp.calc_df_files()""")

md("""### 1. Visium data - adata_vis

21 Space Ranger outputs (GEO GSE278687) read with **sc.read_visium**; obs_names = **<section>_<barcode>**, var_names = Ensembl IDs, symbols in **var['symbol']**; obs: section, patient, **ni_status**, **technique**, GSM.""")
code("""adata_vis = scp.read_h5ad('visium', set_self=False, verbose=verbose)
print(f"{os.path.getsize(scp.file_path('visium')) / 1e9:.2f} GB")
scp.summary(adata_vis)
print(adata_vis.var[['in_FF', 'in_FFPE', 'in_CytAssist', 'in_all_panels']].sum())""")
code("""df_sections = scp.section_table(adata_vis)
df_sections""")
code("""scp.calc_df_sections_summary(df_sections, force=force, verbose=verbose)""")
code("""sections_rep = scp.representative_sections()
print(len(sections_rep), "representative sections:", sections_rep)""")

md("""#### H&E of the 21 sections

Title: section | NI status | technique, spots, median UMIs per spot. Sorted by NI status (low first), technique, section.""")
code("""fig = scp.plot_sections_gallery(adata_vis, ncols=7, panel_size=3.2, fname=scp.fname('fig_sections'), force=force, verbose=verbose)
plt.show()""")

md("""### 2. Quality control - Visium

Genes, UMIs and % mitochondrial per spot. The FF sections PA03 (median 729 UMIs) and PA08 (1,957) are shallow; the FFPE v1 and the CytAssist sections reach 10,000-24,000 UMIs. The probe panels have few or no MT probes, so % MT is comparable only within a platform.""")
code("""scp.calc_qc_metrics(adata_vis)
fig = scp.plot_qc(groupby='section', adata=adata_vis, figsize=(24, 5), fontsize_scale=1.3, fname=scp.fname('fig_qc_visium'),
                  title='Visium spots per section (ordered: low-NI, then high-NI; FF, FFPE, CytAssist)', force=force, verbose=verbose)
plt.show()""")
code("""df_qc_vis = scp.calc_df_section_qc(adata=adata_vis, df_sections=df_sections, cols=['ni_status', 'technique'], force=force, verbose=verbose)
df_qc_vis""")
code("""n0 = adata_vis.n_vars
adata_vis = scp.remove_mt_genes(adata_vis)
print(f"genes {n0:,} -> {adata_vis.n_vars:,}; obsm['MT'] {adata_vis.obsm['MT'].shape}")""")

md("""### 3. The sc/sn reference: pdac_sc_sn_reference.h5ad

GSE278688 (scRNA, 10x 5' v2: 13 tumours, 8 adjacent normal, 12 blood samples of 14 patients) + GSE278689 (FFPE snRNA, 10x Flex: 8 tumours). Symbols renamed to the Ensembl IDs of the Space Ranger features (490 GEO symbols of an older annotation dropped; none is a marker). The GEO matrices have no ribosomal genes (removed by the authors).""")
code("""adata_ref = scp.read_h5ad('reference', set_self=False, verbose=verbose)
print(f"{os.path.getsize(scp.file_path('reference')) / 1e9:.2f} GB; raw counts: {scp.is_raw_counts(adata_ref)}")
scp.summary(adata_ref)
df_ref_major = scp.calc_df_ref_major_types(adata_ref, force=force, verbose=verbose)
df_ref_major""")
code("""scp.calc_qc_metrics(adata_ref)
fig = scp.plot_qc(groupby='sample_id', adata=adata_ref, figsize=(26, 5), fontsize_scale=1.3, fname=scp.fname('fig_qc_ref'),
                  title='sc / sn cells per sample (10x library or FFPE sample)', force=force, verbose=verbose)
plt.show()""")
code("""df_qc_ref = scp.calc_df_qc_table(adata_ref, 'tsv_qc_ref', groupby='sample_id', force=force, verbose=verbose)
df_qc_ref""")

md("""### 4. Re-derived subclusters

**src/programs/pdac_reference_subclusters.py** (CPU, ~40 min; cached): tissue cells (sc + sn; PBMC excluded) of each compartment, scVI (batch = sample, covariate = technology; Schwann: PCA after ComBat by technology), Leiden, the marker-set score of every paper subcluster (**PDAC_SUBCLUSTERS**), cluster label = highest score among the labels whose anchor gene (the gene in the label) is detected in >= 10% of the cluster (**scp.annotate_clusters**). Generic states (MALAT1, heat-shock, proliferation) need a clear relative enrichment; a label that wins no cluster takes its best cluster when that cluster's label is shared.

Checks against the paper (Figure 2M, malignant cells per subcluster): D03 218 (paper 222), D04 1,519 (1,499), D06 235 (258), D09 3,301 (3,847); D08 is larger here (5,886 vs 3,267) and D05 smaller (452 vs 3,612).""")
code("""adata_ref = scp.add_subclusters(adata_ref)
df_cl = scp.read_tsv('tsv_subcluster_clusters')
print(adata_ref.obs.loc[adata_ref.obs.subcluster != '', 'subcluster'].value_counts().to_string())""")
code("""for comp in PDAC_SUBCLUSTERS:
    df_c = scp.plot_compartment_subclusters(comp, figsize=(26, 8) if comp != 'schwann' else (20, 5),
                                            fname=scp.fname('fig_subclusters', comp=comp), force=force, verbose=verbose)
    plt.show()
    display(df_c[['leiden', 'n_cells', 'label', 'score', 'z', 'second', 'anchor_pct', 'pct_sn', 'pct_normal', 'n_patients']])""")
code("""df_comp = scp.calc_df_ref_composition(adata_ref[scp.ref_tissue_cells(adata_ref)], by='tissue', fname=scp.fname('tsv_ref_comp'),
                                       force=force, verbose=verbose)
df_comp""")

md("""### 5. Figure 1B: reference UMAP + dot plot of 26 PDAC / neural-invasion genes

UMAP of a per-type subsample of the tissue cells (<= 3,000 per type; scVI, batch = sample, covariate = technology; cached in **pdac01_reference_umap.tsv**). The paper's Figure 1B integrates sc + sn of all samples with BBKNN; the layout differs, the cell types and their markers are the comparison.""")
code("""genes_plot = list(PDAC_GENES)
df_genes = scp.calc_df_gene_panel(adata_ref, adata_vis, genes=PDAC_GENES, force=force, verbose=verbose)
df_genes""")
code("""df_umap = scp.calc_reference_umap(adata_ref, fname='reference_umap', force=False, verbose=verbose)
df_dot = scp.plot_umap_celltype_dotplot(df_umap, adata_ref, genes_plot, groupby='major_type', figsize=(26, 10),
                                        fname=scp.fname('fig_umap_dotplot'), force=force, verbose=verbose)
plt.show()""")
code("""# major type with the highest mean expression of each gene (and its % expressing)
df_best_major = scp.calc_df_gene_best_group(df_dot, groupby='major_type', genes=genes_plot, fname='tsv_gene_best_major', force=force, verbose=verbose)
df_best_major""")
code("""df_dot2 = scp.plot_umap_celltype_dotplot(df_umap, adata_ref, genes_plot, groupby='cell_type', figsize=(28, 16), point_size=1.2,
                                         fname=scp.fname('fig_umap_sub_dotplot'), force=force, verbose=verbose)
plt.show()""")
code("""# cell type (subcluster) with the highest mean expression of each gene (and its % expressing)
df_best = scp.calc_df_gene_best_group(df_dot2, groupby='cell_type', genes=genes_plot, fname='tsv_gene_best_celltype', force=force, verbose=verbose)
df_best""")

md("""### 6. The 26 genes on the Visium sections

Six representative sections (PA98, PA11-2, PA40, PA06: high NI; PA72, PA12: low NI); colour = log1p(CP10k), one scale per gene shared by the sections. Six genes per figure.""")
code("""figs = scp.plot_spatial_genes(genes_plot, adata_vis, sections=sections_rep, nframe=6, panel_size=2.8, fontsize_scale=1.6,
                              fname=scp.fname('fig_genes'), force=force, verbose=verbose)
plt.show()""")

md("""### 7. cell2location: reference signatures

Reference = **scp.reference_for_deconvolution**: tissue cells, 42 cell types (14 major types with the re-derived subclusters; Platelet dropped: 2 tissue cells), <= 3,000 cells per type. NB regression (**fit_or_load_reference**): labels **cell_type**, batch **sample_id** (37 samples), covariate **technology** (sc 5' vs sn Flex). Trained by **src/programs/run_c2l_pdac.py** (GPU):

```bash
python src/programs/run_c2l_pdac.py --mode all --tag gpu --ref_epochs 250 --map_epochs 6000 --accelerator gpu > pdac/results/c2l_gpu_run.log 2>&1
```""")
code("""adata_ref_dec = scp.reference_for_deconvolution(adata_ref)
mod_reg = None
if (ref_run_name / 'sc.h5ad').exists():
    adata_ref_c2l, selected = scp.filter_genes_c2l(adata_ref_dec, figsize=(5, 4))
    plt.show()
    mod_reg, adata_ref_c2l, _ = scp.fit_or_load_reference(adata_ref_c2l, ref_run_name, labels_key='cell_type', batch_key='sample_id',
                                                          categorical_covariate_keys=['technology'])
    fig = scp.plot_c2l_history(mod_reg, 10, fname=scp.fname('fig_ref_history'), force=force, verbose=verbose); plt.show()
    figs = scp.plot_c2l_qc(mod_reg, summary_name='q50', figsize=(5, 5), fname=scp.fname('fig_ref_qc'), force=force, verbose=verbose)
    df_inf_aver, summary_used = scp.get_reference_signatures(adata_ref_c2l, prefer=(scp.params['c2l_signature_summary'],))
    print(summary_used, df_inf_aver.shape)
else:
    print(f"no cell2location reference model in {ref_run_name}: run src/programs/run_c2l_pdac.py (GPU) first")""")

md("""### 8. Spatial mapping: cell2location (GPU) and RCTD (R)

**cell2location**: N_cells_per_location = 10 (tumour stroma), detection_alpha = 20, batch = section, the 17,368 genes of the three panels, minibatches of 10,000 spots. **RCTD** (spacexr 2.2.1, full mode, default parameters, weights normalised to 1, per section; the paper's method):

```bash
R_LIBS_USER=.R_lib /home/flavio/miniforge3/envs/renv/bin/Rscript src/programs/run_rctd_pdac.R 8 > pdac/results/rctd/run_rctd.log 2>&1
```

RCTD drops spots with < 100 UMIs (mostly in PA03); they are kept with zero abundance and **obs['rctd_ok'] = False**.""")
code("""mod_c2l = None
if scp.file_path('c2l_map').exists():
    mod_c2l, adata_map, _ = scp.fit_or_load_mapping(adata_vis[:, adata_vis.var.in_all_panels.values].copy(), None, run_name)
    for k in ['q05_cell_abundance_w_sf']:
        adata_vis.obsm[k] = pd.DataFrame(adata_map.obsm[k], index=adata_map.obs_names).loc[adata_vis.obs_names]
    adata_vis.uns['mod'] = adata_map.uns['mod']
    fig = scp.plot_c2l_history(mod_c2l, 100, label='training', fname=scp.fname('fig_map_history'), force=force, verbose=verbose); plt.show()
    del adata_map
print("abundance source:", scp.abundance_source(adata_vis), "| RCTD weights:", scp.file_path('rctd_weights').exists())""")

md("""### 9. Cell-type abundance on the sections

**scp.get_abundance**: cell2location q05 (cells per spot) or RCTD weights (rows sum to 1). Regions and all later notebooks use **shares** (abundance / spot total), so both sources are on the same scale.""")
code("""df_abund = scp.get_abundance(adata_vis)
source = df_abund.attrs['source']
df_share = scp.shares(df_abund)
print(source, df_abund.shape, "| spots kept by RCTD:", int(adata_vis.obs.get('rctd_ok', pd.Series(True, index=adata_vis.obs_names)).sum()))
df_share.mean().sort_values(ascending=False).head(15).round(3)""")
code("""celltypes_show = ['D08_Ductal-CEACAM5', 'D09_Ductal-CEACAM6', 'D04_Ductal-GABRP', 'F01_myCAF-MMP11', 'F02_Progenitor-CAF-DPT',
                  'M07_Macro-NLRP3', 'M10_Macro-SPP1', 'B', 'Plasma', 'Acinar', 'S02_Schwann-TGFBI', 'S01_Schwann-ABCA8']
fig = scp.plot_spatial_celltypes(celltypes_show, adata_vis, df_share, sections=sections_rep, panel_size=2.6, cbar_label='share of the spot',
                                 fontsize_scale=1.3, fname=scp.fname('fig_abund', source=source), force=force, verbose=verbose)
plt.show()""")

md("""#### cell2location vs RCTD

Per cell type, Spearman rho over the spots RCTD kept between the cell2location share and the RCTD weight (**calc_df_c2l_vs_rctd**); also on the 14 major types (subclusters summed). Needs both results.""")
code("""if mod_c2l is not None and scp.file_path('rctd_weights').exists():
    df_cmp = scp.calc_df_c2l_vs_rctd(adata_vis, fname=scp.fname('tsv_c2l_vs_rctd'), force=force, verbose=verbose)
    display(df_cmp.sort_values('spearman'))
    df_c = scp.collapse_to_major(scp.shares(scp.get_abundance(adata_vis, 'c2l')))
    df_r = scp.collapse_to_major(scp.get_abundance(adata_vis, 'rctd'))
    ok = adata_vis.obs['rctd_ok'].values
    from scipy.stats import spearmanr
    display(pd.Series({c: spearmanr(df_c.loc[ok, c], df_r.loc[ok, c]).statistic for c in df_c.columns if c in df_r}, name='spearman (major types)').round(3))
else:
    print("comparison needs the cell2location map and the RCTD weights")""")

md("""### 10. Regions: Leiden on the cell-type shares, named after the paper's niches

**scp.cluster_regions**: kNN (**region_n_neighbors** = 30) on the shares of the 42 types, Leiden (**region_resolution** = 1.0, igraph, seed 0) jointly over the 21 sections. **assign_niches**: niche groups of the paper (TLS: B + T; endocrine; peri-vascular: endothelial + stellate; plasma; CAF; TAM; malignant ductal; non-malignant pancreas: acinar + D01; nerve: Schwann + F07); per region the group share, its enrichment over all spots and **score = share x log2(enrichment)** (groups with share >= 0.10); niche = highest score, **mixed** when the best score < 0.1, **N00_unassigned** for spots RCTD could not deconvolve (< 100 UMIs). Malignant regions are named after their dominant subtype, **N07-10_Malignant (<subtype>)** (the paper's N07-N10 numbering is not public, except N10 = D02_Ductal-APOA2). Curated names can be written in **pdac/data/pdac_regions.tsv**.

**Caveat**: the sections are not batch corrected in abundance space; a region found in one section only, or only on one platform, is a candidate technical effect (n_sections, technique mix in mp_pdac02).""")
code("""scp.cluster_regions(adata_vis, df_abund=df_share)
scp.umap_regions(adata_vis)
fmt = scp.params['region_id_fmt']
cl = adata_vis.obs['region_cluster'].astype(str)
labels = {c: fmt.format(int(c)) for c in sorted(cl.unique(), key=int)}
adata_vis.obs['region_label'] = pd.Categorical(cl.map(labels), categories=list(labels.values()))
df_niche = scp.annotate_niches(adata_vis, df_abund, abundance_source=source, force=force, verbose=verbose)
names = dict(zip(df_niche.region_label, df_niche.niche))
df_niche""")
code("""df_regions = scp.save_region_labels(adata_vis, abundance_source=source, force=force, verbose=verbose)""")
code("""fig = scp.umap_spatial_3plot(adata_vis, names=names, key='region_label', library_id=scp.params['library_main'], figsize=(24, 9),
                             fname=scp.fname('fig_regions'), force=force, verbose=verbose)
plt.show()""")
code("""# interactive version (HTML in pdac/figures): one legend drives the region UMAP and the section map
fig = scp.plotly_umap_spatial_3plot(adata_vis, names=names, key='region_label', library_id=scp.params['library_main'],
                                    fname='fig_regions_3plot_html', force=force, verbose=verbose)
fig.show()""")
code("""fig = scp.plot_regions_on_sections(adata_vis, key='region_label', sections=sections_rep, names=names, ncols=3, panel_size=4.2,
                                   legend_ncol=2, fname=scp.fname('fig_regions_sections'), force=force, verbose=verbose)
plt.show()""")
code("""# interactive version (HTML in pdac/figures): click a region to hide it, double-click to isolate it in every section
fig = scp.plotly_regions_on_sections(adata_vis, key='region_label', sections=sections_rep, names=names, ncols=3,
                                     fname='fig_regions_sections_html', force=force, verbose=verbose)
fig.show()""")
code("""# interactive: the paper's representative section PA98 (Figure 1D right); the others as HTML files
for lib in sections_rep:
    fig = scp.plotly_spatial_visium(names, adata_vis, key='region_label', library_id=lib, fontsize_title=20, fontsize_legend=12,
                                    fontsize_hover=13, marker_line_color="#222222", fname='fig_regions_html', force=force, verbose=False)
    if lib == scp.params['library_main']:
        fig.show()""")

md("""### Notes for interpretation

- **Platform and NI are confounded** (all low-NI sections are CytAssist): every high vs low statement is checked on the CytAssist sections in mp_pdac02.
- **Subclusters are re-derived**; F03_tCAF-MME and M05_Mono-CD16 are not separated (their cells are in myCAF / progenitor CAF and in the monocyte / macrophage subclusters).
- **Abundance source**: RCTD until the cell2location GPU map exists; after the GPU run rerun this notebook with **force=True** (regions are recomputed and the names reassigned).""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"display_name": "cell-cell-communication (3.12.12)", "language": "python", "name": "python3"}
nbf.write(nb, out)
print(f"written {out} ({len(C)} cells)")
