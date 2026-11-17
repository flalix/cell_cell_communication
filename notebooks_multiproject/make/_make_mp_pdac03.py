"""Generates mp_pdac03_subregions.ipynb (structure of mp_heart_dvlp03_subregions.ipynb + nerve layers, paper Figures 2L,
4A, 4B, 5K, 6A). Run once; then edit the notebook freely (--force overwrites)."""
import sys
import nbformat as nbf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _make_pdac_common import HEADER_LIBS, IMPORTS, PAPER, md_code  # noqa: E402

out = Path(__file__).resolve().parent.parent / "mp_pdac03_subregions.ipynb"
if out.exists() and "--force" not in sys.argv:
    raise SystemExit(f"{out} exists: not overwritten (--force)")
nb = nbf.v4.new_notebook(); C = []
md, code = md_code(C)

md(f"""### PDAC and neural invasion - subregions of each region and the nerve microenvironment (mp_pdac03)

{PAPER}

Part A (as **mp_heart_dvlp03**): for each region of **mp_pdac01**, the spots are clustered on **gene expression** (own UMAP, ComBat by section, the 17,368 genes of the three panels) to find **subregions**, mapped on the tissue and scored on the sc/sn reference.

Part B (new; the paper's Figures 2L, 4A, 4B, 5K, 6A): **nerve spots** (Schwann share + nerve markers), **invaded / non-invaded** nerves (malignant share in and next to the nerve), and the **juxta-neural** (rings 1-2, ~200 um) and **peri-neural** (rings 3-5) microenvironment.

{HEADER_LIBS}

| section | content |
|---|---|
| 1 | spots, regions, expression |
| 2 | region signatures (table region); heatmap; maps |
| 3 | subclustering each region (own UMAP; batch = section) |
| 4 | gene sets of each subregion (partition) |
| 5 | subregion table and diagnostics; maps |
| 6 | subregion signatures on the reference |
| 7 | B1 - nerve spots and bundles; invaded vs non-invaded |
| 8 | B2 - Figure 4B: niches by ring around invaded / non-invaded nerves |
| 9 | B3 - Figure 4A: cell types of the juxta-neural spots, invaded vs non-invaded |
| 10 | B4 - Figure 2L: malignant subtypes by ring; Figure 5K: Schwann subsets in invaded vs non-invaded nerves |
| 11 | B5 - Figure 6A: Schwann subsets by distance to the malignant niches |
| 12 | tables |

Figures and tables have the prefix **pdac03_**.""")

md("""### Critique of the analysis and how it is handled

1. **Nerves are inferred, not annotated.** The paper selected nerve spots by deconvolution **and** the H&E (object 'spatial_Nerve_manual' in the authors' code) and called invasion from the H&E. Here: Schwann share >= **nerve_min_schwann_share** (0.10) and nerve-marker z >= **nerve_min_marker_z** (1.0), bundles of >= 3 connected spots; **invaded** = malignant share (D02-D09) >= 0.20 over the bundle and its first ring. A manual table **pdac/data/pdac_nerve_annotation.tsv** (section, barcode, nerve, invaded) overrides the calls when present; check the calls on the H&E (section 7 figures).
2. **Rings are graph distances** on the hexagonal Visium lattice (1 ring = 100 um centre to centre), per section; a spot belongs to its nearest bundle.
3. **Spots are not independent**: the Wilcoxon p-values of section 9 are over spots (as the paper's Figure 4A) and optimistic; rank by effect size.
4. **Subregions** use the shared genes and ComBat by section; single-section subregions are flagged (**section_min_frac**).""")

md("## Loading packages")
code(IMPORTS + "\nprint({p: version(p) for p in ['scanpy', 'anndata', 'plotly']})")
md("## Parameters")
code("""force=False       # True: recompute every table and redraw every figure (after changing parameters or inputs)
verbose=True

apply_figure_style()
scp = SingleCellPDAC(ROOT0, analysis='pdac03')
scp.set_params(subregion_skip=tuple(scp.unassigned_regions()))     # spots without deconvolution are not subclustered
root_data, root_figure, root_result = scp.root_data, scp.root_figure, scp.root_result
print(scp)
print({k: scp.params[k] for k in ['nerve_min_schwann_share', 'nerve_min_marker_z', 'nerve_min_size', 'nerve_invaded_min_malignant', 'nerve_n_rings']})
scp.calc_df_files()""")

md("### 1. Spots, regions and expression")
code("""df_regions = scp.read_tsv('region_labels', verbose=verbose)
source = df_regions['abundance_source'].iloc[0]
adata_vis = scp.read_h5ad('visium', set_self=False, verbose=verbose)
adata_vis = scp.remove_mt_genes(adata_vis)
if source == 'c2l':
    a_map = scp.read_h5ad_slots('c2l_map', skip=('layers',))
    adata_vis.obsm['q05_cell_abundance_w_sf'] = pd.DataFrame(a_map.obsm['q05_cell_abundance_w_sf'], index=a_map.obs_names).loc[adata_vis.obs_names]
    adata_vis.uns['mod'] = a_map.uns['mod']; del a_map
d = df_regions.set_index('barcode').loc[adata_vis.obs_names]
for c in ('region_cluster', 'region_label', 'region_name', 'niche'):
    adata_vis.obs[c] = pd.Categorical(d[c].astype(str).values)
names = dict(zip(d.region_label.astype(str), d.region_name.astype(str)))
df_abund = scp.get_abundance(adata_vis, source=source)
df_share = scp.shares(df_abund)
df_region_info = scp.read_region_info(source=source)
adata_expr = scp.prepare_expression(adata_vis)
sections_rep = scp.representative_sections()
print(adata_expr)
adata_expr.obs['region_label'].value_counts().sort_index().to_frame('n_spots').T""")
code("""df_depth = scp.calc_df_region_depth(adata_expr, df_abund)
df_depth.T""")

md("""### 2. Region signatures (table region)

Each region vs all other spots (Wilcoxon on log1p CP10k): signature = logFC >= 1, FDR <= 0.05, detected in >= 25% of the region's spots.""")
code("""df_rank_region, region_sig = scp.calc_df_region_signature(adata_expr, force=force, verbose=verbose)
df_region_table = scp.calc_df_region_table(adata_expr, df_rank_region, region_sig, df_abund, df_region_info, force=force, verbose=verbose)
cols = [c for c in ['region_id', 'region_name', 'n_spots', 'median_UMIs', 'n_signature_genes', 'signature_genes', 'enriched_cell_types'] if c in df_region_table]
df_region_table[cols]""")
code("""fig = scp.plotly_region_signature_heatmap(adata_expr, df_rank_region, n_per_region=3, fontsize_title=22, fontsize_xticks=14,
                                          fontsize_yticks=13, fontsize_colorbar=14, fontsize_hover=13, height=1100, width=1400,
                                          force=force, verbose=False)
fig.show()""")
code("""for lib in sections_rep:
    fig = scp.plotly_region_map(adata_expr, adata_vis, lib, df_region_table=df_region_table, fontsize_title=20, fontsize_legend=12,
                                fontsize_hover=13, force=force, verbose=False)
    if lib == scp.params['library_main']:
        fig.show()""")

md("""### 3. Subclustering each region (own UMAP; batch = section)

**subcluster_adaptive** per region (sections with < 10 spots in the region pooled for ComBat): 2,000 HVGs, ComBat, PCA (20), kNN (15), UMAP, Leiden over **sub_resolutions**; the highest resolution with every subregion >= **sub_min_size** spots and <= **sub_max_k** subregions.""")
code("""subs, df_grid = scp.run_subregions(adata_expr, force=force, verbose=False)
df_grid[df_grid.chosen][['region_id', 'resolution', 'k', 'min_size']]""")

md("### 4. Gene sets of each subregion (partition)")
code("""df_gene_sets = scp.calc_df_subregion_gene_sets(adata_expr, region_sig, force=force, verbose=False)
df_set_counts = df_gene_sets.groupby(['sub_region_id', 'gene_set'], sort=False).size().unstack(fill_value=0)
df_set_counts['total'] = df_set_counts.sum(1)
n_universe = df_gene_sets.groupby('sub_region_id', sort=False).size()
assert (df_set_counts['total'] == n_universe.loc[df_set_counts.index]).all()
print(f"{len(df_set_counts)} subregions; the 4 sets partition every universe")
df_set_counts.head(12)""")

md("### 5. Subregion table, diagnostics and maps")
code("""df_sub_table = scp.calc_df_subregion_table(adata_expr, df_gene_sets, df_abund, df_descriptions=None, force=force, verbose=False)
print(df_sub_table.status.str.split(';').str[0].value_counts().to_dict())
cols = ['sub_region_id', 'n_spots', 'coherence_ratio', 'depth_ratio', 'border_region', 'n_subregion_signature', 'n_not_present', 'status']
df_sub_table[[c for c in cols if c in df_sub_table]].head(15)""")
code("""fig = scp.plotly_subregion_diagnostics(df_sub_table, fontsize_title=22, fontsize_axis=18, fontsize_ticks=16, fontsize_legend=12,
                                       fontsize_legend_title=16, fontsize_hover=13, grid_color="#555555", height=800, width=1800,
                                       force=force, verbose=False)
fig.show()""")
code("""for lib in sections_rep:
    fig = scp.plotly_subregion_map(adata_expr, adata_vis, lib, df_sub_table=df_sub_table, fontsize_title=20, fontsize_legend=11,
                                   fontsize_legend_title=14, fontsize_hover=13, legend_ncols=3, width=1800, height=900,
                                   force=force, verbose=False)
    if lib == scp.params['library_main']:
        fig.show()""")

md("""### 6. Subregion signatures on the reference

Top 25 subregion-signature genes of every **supported** subregion scored on the sc/sn reference (tissue cells, <= 3,000 per type): mean per-gene z-score of log1p(CP10k), mean per cell type.""")
code("""adata_ref = scp.read_h5ad('reference', set_self=False, backed='r')
adata_ref_dec = scp.reference_for_deconvolution(adata_ref)
adata_ref_sym = adata_ref_dec.copy()
adata_ref_sym.var_names = adata_ref_sym.var['symbol'].astype(str).values
adata_ref_sym.var_names_make_unique()
df_ref_score, df_cell_score, genes_used = scp.calc_df_subregion_ref_score(adata_ref_sym, df_gene_sets, df_sub_table, force=force, verbose=False)
fig = scp.plot_subregion_ref_heatmap(df_ref_score, figsize=(18, max(6, 0.22 * len(df_ref_score))), fname=scp.fname('fig_ref_heatmap'),
                                     force=force, verbose=verbose)
plt.show()
df_sub_table = scp.add_ref_cell_types(df_sub_table, df_ref_score, n=3, force=force, verbose=False)""")

md("""### 7. B1 - Nerve spots and bundles; invaded vs non-invaded

**scp.call_nerves** (cached in **pdac03_nerve_spots.tsv**). The table lists one row per bundle; the maps show nerve spots (yellow: non-invaded, red: invaded) and the rings around them on the representative sections. Compare PA98 with the paper's Figure 1E (red: invaded, blue: non-invaded nerves on the H&E).""")
code("""df_nerve = scp.call_nerves(adata_vis, df_abund, fname='nerve_spots', force=force, verbose=verbose)
df_bundles = scp.calc_df_nerve_table(df_nerve, fname='nerve_table', force=force, verbose=verbose)
print(f"nerve spots: {(df_nerve.nerve_id != '').sum():,}; bundles: {len(df_bundles)} "
      f"(invaded {int((df_bundles.status == 'invaded').sum())}, non-invaded {int((df_bundles.status == 'non-invaded').sum())})")
display(df_bundles.groupby(['ni_status', 'status']).agg(n_bundles=('nerve_id', 'size'), n_spots=('n_spots', 'sum')))
df_bundles.head(20)""")
code("""adata_vis.obs['nerve_call'] = np.where(adata_vis.obs.nerve_status == '', '', 'nerve ' + adata_vis.obs.nerve_status)
cz = {'nerve invaded': '#FF2020', 'nerve non-invaded': '#FFEE00'}
fig = scp.plot_regions_on_sections(adata_vis, key='nerve_call', sections=sections_rep, colors=cz, ncols=3, panel_size=4.5,
                                   legend_ncol=2, fname=scp.fname('fig_nerves'), force=force, verbose=verbose)
plt.show()""")
code("""lib = scp.params['library_main']
z = adata_vis.obs.nerve_zone.astype(str) + np.where(adata_vis.obs.ring_status != '', ' (' + adata_vis.obs.ring_status.astype(str) + ')', '')
adata_vis.obs['nerve_zone_status'] = np.where(adata_vis.obs.nerve_zone == '', '', z)
cz = {'nerve (invaded)': '#FF2020', 'juxta-neural (invaded)': '#FF9090', 'peri-neural (invaded)': '#FFD0D0',
      'nerve (non-invaded)': '#FFEE00', 'juxta-neural (non-invaded)': '#7FB8FF', 'peri-neural (non-invaded)': '#C8E0FF'}
fig = scp.plot_regions_on_sections(adata_vis, key='nerve_zone_status', sections=[lib], colors=cz, ncols=1, panel_size=8, legend_ncol=2,
                                   fname=scp.fname('fig_nerve_rings', lib=lib), force=force, verbose=verbose)
plt.show()""")

md("""### 8. B2 - Figure 4B: niches by ring around invaded / non-invaded nerves

Share of each niche among the spots of every ring (5 -> 1 = juxta, 0 = nerve), around invaded (dashed) and non-invaded nerves (solid). Paper: TLS, plasma and peri-vascular niches near non-invaded nerves; CAF / TAM / malignant niches near invaded nerves.""")
code("""D_niche = pd.get_dummies(adata_vis.obs['niche'].astype(str)).astype(float)
df_ring_niche = scp.calc_df_ring_profile(adata_vis, D_niche, fname='tsv_ring_niche', force=force, verbose=verbose)
fig = scp.plot_ring_profile(df_ring_niche, sorted(D_niche.columns), ncols=4, ylabel='share of the ring spots',
                            title='Niches around nerves (Figure 4B analogue)', fname=scp.fname('fig_ring_niche'), force=force, verbose=verbose)
plt.show()""")

md("""### 9. B3 - Figure 4A: cell types of the juxta-neural spots, invaded vs non-invaded

Mean share in rings 1-2 around invaded vs non-invaded nerves; two-sided Mann-Whitney over spots. Paper: around invaded nerves M07_Macro-NLRP3, myCAF (F01) and malignant cells; around non-invaded nerves TLS cells (B, Tfh), plasma cells, progenitor CAF.""")
code("""df_juxta = scp.calc_df_juxta_enrichment(adata_vis, df_share, fname='tsv_juxta', force=force, verbose=verbose)
print(df_juxta.attrs)
fig = scp.plot_juxta_enrichment(df_juxta, max_fdr=0.05, fname='fig_juxta', force=force, verbose=verbose)
plt.show()
df_juxta.head(12)""")

md("""### 10. B4 - Figure 2L: malignant subtypes by ring; Figure 5K: Schwann subsets in nerves

Paper (2L): D09_Ductal-CEACAM6 and D04_Ductal-GABRP most prevalent within 1 spot of invaded nerves. Paper (5K): S03_Schwann-SERPINA3 higher in non-invaded nerve spots.""")
code("""mal = [c for c in df_share.columns if c[:3] in ('D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09')]
df_ring_mal = scp.calc_df_ring_profile(adata_vis, df_share[mal], fname='tsv_ring_malignant', force=force, verbose=verbose)
fig = scp.plot_ring_heatmap(df_ring_mal, title='Malignant subtypes by ring (0 = nerve; Figure 2L analogue)',
                            fname='fig_ring_malignant', force=force, verbose=verbose)
plt.show()
schw = [c for c in df_share.columns if c.startswith('S0')]
nv = adata_vis.obs.nerve_ring.values == 0
df_s5k = df_share.loc[nv, schw].groupby(adata_vis.obs.loc[nv, 'nerve_status'].values).mean()
df_s5k.div(df_s5k.sum(1), axis=0).round(3)""")

md("""### 11. B5 - Figure 6A: Schwann subsets by distance to the malignant niches

Nerve spots grouped by graph distance to the nearest malignant-niche spot ('radius 1-5', farther spots in 5); share of each Schwann subset among the Schwann cells. Paper: S02_Schwann-TGFBI increases towards the tumour (leading edge of NI).""")
code("""df_sd = scp.calc_df_schwann_by_distance(adata_vis, df_abund, df_share, n_rings=5, fname='tsv_schwann_distance', force=force, verbose=verbose)
fig = scp.plot_schwann_distance(df_sd, fname='fig_schwann_distance', force=force, verbose=verbose)
plt.show()
df_sd.round(3)""")

md("### 12. Tables")
code("""df_region_table = scp.update_region_table_subregions(df_region_table, adata_expr, force=force, verbose=verbose)
cols = ['region_id', 'sub_region_id', 'status', 'subregion_signature_genes', 'not_present_genes', 'enriched_cell_types', 'ref_cell_types']
df_sub_table[[c for c in cols if c in df_sub_table]]""")

md("""### Notes for interpretation

- **Nerve calls** are a reproducible proxy of the pathologists' outlines; thresholds are parameters (**nerve_***). Overrides from the H&E go to **pdac/data/pdac_nerve_annotation.tsv**.
- **Invasion** from spot mixtures: a nerve next to dense tumour can be called invaded without perineural invasion; the paper's ENA / PNI / ENI grades need histology.
- **Schwann subsets** rest on 258 reference cells; their spot shares are small and noisy - read trends across rings, not single values.""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"display_name": "cell-cell-communication (3.12.12)", "language": "python", "name": "python3"}
nbf.write(nb, out)
print(f"written {out} ({len(C)} cells)")
