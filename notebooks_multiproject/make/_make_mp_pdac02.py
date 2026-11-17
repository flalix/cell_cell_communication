"""Generates mp_pdac02_regions.ipynb (structure of mp_heart_dvlp02_regions.ipynb; the age axis becomes the NI status).
Run once; then edit the notebook freely (--force overwrites)."""
import sys
import nbformat as nbf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _make_pdac_common import HEADER_LIBS, IMPORTS, PAPER, md_code  # noqa: E402

out = Path(__file__).resolve().parent.parent / "mp_pdac02_regions.ipynb"
if out.exists() and "--force" not in sys.argv:
    raise SystemExit(f"{out} exists: not overwritten (--force)")
nb = nbf.v4.new_notebook(); C = []
md, code = md_code(C)

md(f"""### PDAC and neural invasion - regions, niches and high vs low NI (mp_pdac02)

{PAPER}

It continues **mp_pdac01_prepare.ipynb**: the regions computed there (**pdac01_region_labels.tsv**: Leiden clusters of the cell-type shares of the 21 sections, named after the paper's 12 niches) are read, not recomputed, and studied as in the paper's Figures 1-3: which cell types define each niche, how niches and cell types differ between **low-NI** and **high-NI** tumours, the malignant subtypes and their diversity.

{HEADER_LIBS}

| section | content |
|---|---|
| 1 | spots, abundances and regions of mp_pdac01 |
| 2 | region table: size, sections, platforms, cell types, niche |
| 3 | Figure 1F: relative cell-type proportion in each niche (and region) |
| 4 | niches on the sections (Figure 1D right) |
| 5 | Figures 2E / 3D: niche share per section, low vs high NI (all sections; CytAssist only) |
| 6 | Figure 3C: PCA of the sections on niche shares |
| 7 | Figure 3B: TLS cell types and other cell types, low vs high NI |
| 8 | Figures 2C / 2J: malignant subtypes, pure spots, composition and Shannon diversity per section |
| 9 | co-occurrence of cell types across sections (Figure 3I analogue) |
| 10 | PDAC / NI genes per region (dot plot); NI-related gene score (Figure 2M on Visium) |
| 11 | interactive region maps |

Figures and tables have the prefix **pdac02_**.""")

md("""### Critique of the analysis and how it is handled

1. **21 sections, 17 patients**: sections are the replicates (PA04, PA05 and PA11 have two sections each, so p-values are slightly optimistic); effect sizes and the per-section points are shown with every test.
2. **Platform confound**: all 5 low-NI sections are CytAssist, all FF / FFPE-v1 sections high-NI. Every high vs low test is repeated on the 12 CytAssist sections (5 low vs 7 high); a difference that disappears there is not interpretable as NI biology.
3. **Ridge plots of the paper** summarise the same 5 vs 16 values per niche; here box + points (marker = platform) show them directly.
4. **Niche names are inferred** from composition (assign_niches); the paper's 12 niches were assigned with the H&E and are not public. Four malignant niches are numbered by their dominant subtype.
5. **Deconvolution source**: RCTD (the paper's method) or cell2location (GPU map); the tables carry it in **abundance_source**.""")

md("## Loading packages")
code(IMPORTS + "\nprint({p: version(p) for p in ['scanpy', 'squidpy', 'anndata', 'plotly']})")
md("## Parameters")
code("""force=False
verbose=True

apply_figure_style()
scp = SingleCellPDAC(ROOT0, analysis='pdac02')
root_data, root_figure, root_result = scp.root_data, scp.root_figure, scp.root_result
print(scp)
scp.calc_df_files()""")

md("""### 1. Spots, abundances and the regions of mp_pdac01""")
code("""df_regions = scp.read_tsv('region_labels', verbose=verbose)
source = df_regions['abundance_source'].iloc[0]
print("regions of mp_pdac01 made with:", source)
adata_vis = scp.read_h5ad('visium', set_self=False, verbose=verbose)
adata_vis = scp.remove_mt_genes(adata_vis)
if source == 'c2l':
    a_map = scp.read_h5ad_slots('c2l_map', skip=('layers',))
    adata_vis.obsm['q05_cell_abundance_w_sf'] = pd.DataFrame(a_map.obsm['q05_cell_abundance_w_sf'], index=a_map.obs_names).loc[adata_vis.obs_names]
    adata_vis.uns['mod'] = a_map.uns['mod']; del a_map
scp.calc_qc_metrics(adata_vis)
df_abund = scp.get_abundance(adata_vis, source=source)
df_share = scp.shares(df_abund)
d = df_regions.set_index('barcode').loc[adata_vis.obs_names]
for c in ('region_cluster', 'region_label', 'region_name', 'niche'):
    adata_vis.obs[c] = pd.Categorical(d[c].astype(str).values)
names = dict(zip(d.region_label.astype(str), d.region_name.astype(str)))
regions = list(adata_vis.obs['region_label'].cat.categories)
niches = sorted(adata_vis.obs['niche'].unique())
sections_rep = scp.representative_sections()
df_sections = scp.section_table(adata_vis)
print(adata_vis.shape, '|', len(regions), 'regions |', len(niches), 'niches |', source, df_abund.shape)""")

md("""### 2. Region table

Per region: spots, sections, platforms, median UMIs, top cell types by share and by enrichment, the niche (mp_pdac01).""")
code("""def _summary():
    obs = adata_vis.obs
    df_m = df_share.groupby(obs.region_label.values, observed=True).mean()
    df_e = df_m / df_share.mean(0)
    rows = []
    for r, sub in obs.groupby('region_label', observed=True):
        rows.append(dict(region_label=r, niche=names.get(r, ''), n_spots=len(sub), n_sections=sub.section.nunique(),
                         n_low_NI=int((sub.ni_status == 'low').sum()), n_high_NI=int((sub.ni_status == 'high').sum()),
                         techniques=', '.join(f"{t} {v:.0%}" for t, v in sub.technique.value_counts(normalize=True).items() if v > 0.02),
                         median_UMIs=float(sub.total_counts.median()),
                         top_share=', '.join(f"{c} ({v:.0%})" for c, v in df_m.loc[r].sort_values(ascending=False)[:3].items()),
                         top_enriched=', '.join(f"{c} ({v:.1f}x)" for c, v in df_e.loc[r].sort_values(ascending=False)[:3].items())))
    return pd.DataFrame(rows)
df_region_summary = scp._cached_any('tsv_region_summary', _summary, force=force, verbose=verbose)
df_region_summary""")

md("""### 3. Figure 1F: cell types in each niche

log2 enrichment of every cell type (share in the niche / share over all spots; clipped at 1/16). The paper's Figure 1F shows the scaled RCTD proportion per niche; the defining cell types should match: TLS (B, CD4/CD8 T), plasma, CAF, TAM, malignant ductal, non-malignant pancreas (acinar, D01), endocrine, peri-vascular, nerve (Schwann).""")
code("""df_enrich_niche = scp.plot_region_celltype_heatmap(adata_vis, df_share, key='niche', figsize=(20, 7), fname=scp.fname('fig_niche_celltype'),
                                                   force=force, verbose=verbose)
plt.show()""")
code("""df_enrich = scp.plot_region_celltype_heatmap(adata_vis, df_share, key='region_label', names=names, figsize=(20, 11),
                                             fname=scp.fname('fig_region_celltype'), force=force, verbose=verbose)
plt.show()""")

md("""### 4. Niches on the sections

The paper's representative section PA98 (Figure 1D right) and the other representative sections.""")
code("""colors = {n: PAPER_NICHES.get(n.split(' (')[0]) for n in niches}         # the paper's niche colours (myUtiliity.R)
pal = scp._region_palette(niches)
colors = {n: (c or pal[n]) for n, c in colors.items()}
fig = scp.plot_regions_on_sections(adata_vis, key='niche', sections=sections_rep, colors=colors, ncols=3, panel_size=4.2, legend_ncol=3,
                                   fname=scp.fname('fig_niches_sections'), force=force, verbose=verbose)
plt.show()""")

md("""### 5. Figures 2E / 3D: niche share per section, low vs high NI

Share of each niche among the spots of a section; box + points per NI group (marker = platform), two-sided Mann-Whitney U over sections (BH FDR in the table). Paper: TLS, plasma-enriched and peri-vascular niches higher in low-NI tissues; Malignant_ductal_4 (D02_Ductal-APOA2) in low NI.""")
code("""df_niche_sec = scp.calc_df_share_by_section(adata_vis, 'niche', fname=scp.fname('tsv_niche_by_section'), force=force, verbose=verbose)
df_t_all = scp.calc_df_ni_test(df_niche_sec)
df_t_cyt = scp.calc_df_ni_test(df_niche_sec, techniques=('CytAssist',))
scp._write_table(df_t_all, scp.fname('tsv_niche_ni_test'), force=True)
scp._write_table(df_t_cyt, scp.fname('tsv_niche_ni_test_cyt'), force=True)
df_t_all.merge(df_t_cyt[['group', 'log2_ratio_high_low', 'p']], on='group', suffixes=('', '_CytAssist')).sort_values('p')""")
code("""fig = scp.plot_ni_groups(df_niche_sec, niches, df_test=df_t_all, ncols=6, title='Niche share per section (all 21 sections)',
                         fname=scp.fname('fig_niche_ni'), force=force, verbose=verbose)
plt.show()
fig = scp.plot_ni_groups(df_niche_sec[df_niche_sec.technique == 'CytAssist'], niches, df_test=df_t_cyt, ncols=6,
                         title='Niche share per section (12 CytAssist sections: 5 low vs 7 high NI)',
                         fname=scp.fname('fig_niche_ni_cyt'), force=force, verbose=verbose)
plt.show()""")

md("""### 6. Figure 3C: PCA of the sections on niche shares

Paper: the low-NI sections form a distinct cluster. Here colour = NI, marker = platform: separation along a platform axis would be technical.""")
code("""df_pca = scp.plot_section_pca(df_niche_sec, figsize=(8, 7), title='Sections on niche shares (Figure 3C analogue)',
                              fname=scp.fname('fig_section_pca'), force=force, verbose=verbose)
plt.show()
df_pca.sort_values('PC1')""")

md("""### 7. Figure 3B: cell types, low vs high NI

Mean share of each cell type per section. Paper (ST): B01_naive-TCL1A, B03_GCB-RGS13 and T10_CD4_Tfh-like-CXCL13 (TLS components) higher in low NI; here B cells and T cells are major types (lymphoid subclusters were not re-derived). Also shown: plasma, NLRP3+ TAM (M07), myCAF (F01), the malignant subtypes and Schwann subsets.""")
code("""df_ct_sec = scp.calc_df_celltype_by_section(adata_vis, df_abund, fname=scp.fname('tsv_celltype_by_section'), force=force, verbose=verbose)
df_ct_all = scp.calc_df_ni_test(df_ct_sec, group_col='cell_type')
df_ct_cyt = scp.calc_df_ni_test(df_ct_sec, group_col='cell_type', techniques=('CytAssist',))
scp._write_table(df_ct_all, scp.fname('tsv_celltype_ni_test'), force=True)
scp._write_table(df_ct_cyt, scp.fname('tsv_celltype_ni_test_cyt'), force=True)
df_ct_all.merge(df_ct_cyt[['cell_type', 'log2_ratio_high_low', 'p']], on='cell_type', suffixes=('', '_CytAssist')).sort_values('p').head(25)""")
code("""cts = ['B', 'CD4T', 'CD8T', 'Plasma', 'M07_Macro-NLRP3', 'M10_Macro-SPP1', 'F01_myCAF-MMP11', 'F02_Progenitor-CAF-DPT',
       'D02_Ductal-APOA2', 'D04_Ductal-GABRP', 'D09_Ductal-CEACAM6', 'S02_Schwann-TGFBI']
fig = scp.plot_ni_groups(df_ct_sec, cts, group_col='cell_type', df_test=df_ct_all, ncols=6, title='Cell-type share per section (all sections)',
                         fname=scp.fname('fig_celltype_ni'), force=force, verbose=verbose)
plt.show()""")

md("""### 8. Malignant subtypes: pure spots (Figure 2C), composition and diversity (Figure 2J)

**Pure spots**: share of one malignant subtype >= 0.8 of the spot (the paper's criterion on its RCTD proportions); shown on the H&E of the section with most such spots. **Composition**: malignant subtypes (D02-D09) as a share of the malignant cells of each section, with the Shannon index. Paper: lower diversity in high-NI tissues; D02_Ductal-APOA2 in low NI.""")
code("""mal = [c for c in df_share.columns if c[:3] in ('D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09')]
pure = (df_share[mal] >= 0.8)
print(pure.sum().to_string())
for sub in ['D08_Ductal-CEACAM5', 'D09_Ductal-CEACAM6', 'D04_Ductal-GABRP', 'D02_Ductal-APOA2']:
    if sub in pure and pure[sub].any():
        lib = adata_vis.obs.loc[pure[sub].values, 'section'].value_counts().index[0]
        adata_vis.obs['_pure'] = np.where(pure[sub].values, sub, '')
        fig = scp.plot_regions_on_sections(adata_vis, key='_pure', sections=[lib], colors={sub: '#FFEE00'}, ncols=1, panel_size=6,
                                           fname=scp.fname('fig_malignant_pure', sub=sub.split('_')[0]), force=force, verbose=verbose)
        plt.show()""")
code("""df_div = scp.calc_df_malignant_diversity(adata_vis, df_abund)
scp._write_table(df_div.reset_index(names='section'), scp.fname('tsv_malignant_diversity'), force=True)
order = df_div.sort_values(['ni_status', 'shannon']).index
fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw=dict(height_ratios=[1, 3]), sharex=True)
axes[0].bar(range(len(order)), df_div.loc[order, 'shannon'], color=['#4C9BE8' if v == 'low' else '#E8504C' for v in df_div.loc[order, 'ni_status']])
axes[0].set_ylabel('Shannon')
bottom = np.zeros(len(order)); col = scp.celltype_colors(mal)
for c in mal:
    axes[1].bar(range(len(order)), df_div.loc[order, c], bottom=bottom, color=col[c], label=c, width=0.85); bottom += df_div.loc[order, c].values
axes[1].set_xticks(range(len(order))); axes[1].set_xticklabels([f"{s}\\n{df_div.loc[s, 'ni_status']}" for s in order], rotation=90)
axes[1].set_ylabel('share of malignant cells'); axes[1].legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
fig.suptitle('Malignant subtypes per section (Figure 2J analogue)')
scp.style_figure(fig, scp.params['font_color'], scp.params['bg_color'], scp.params['fontsize_scale'])
scp._savefig(fig, scp.fname('fig_malignant_comp'), force=force, verbose=verbose)
plt.show()
from scipy.stats import mannwhitneyu
for sub_ in (None, 'CytAssist'):
    d_ = df_div if sub_ is None else df_div[df_div.technique == sub_]
    print(sub_ or 'all', 'Shannon low vs high: medians', d_.groupby('ni_status').shannon.median().round(3).to_dict(),
          'p =', round(mannwhitneyu(d_[d_.ni_status == 'low'].shannon, d_[d_.ni_status == 'high'].shannon).pvalue, 4))""")

md("""### 9. Co-occurrence of cell types across sections (Figure 3I analogue)

Spearman correlation of the mean cell-type shares over the 21 sections; pairs with rho > 0.6 (the paper's threshold; the paper used sc/sn proportions per sample). Paper: M07_Macro-NLRP3 with T11_CD4_Th17-like (not separated here).""")
code("""M = df_ct_sec.pivot_table(index='section', columns='cell_type', values='share')
R = M.corr(method='spearman')
R.index.name, R.columns.name = 'cell_type_1', 'cell_type_2'
pairs = R.where(np.triu(np.ones(R.shape, bool), 1)).stack().rename('rho').reset_index()
pairs.columns = ['cell_type_1', 'cell_type_2', 'rho']
pairs = pairs[pairs.rho > 0.6].sort_values('rho', ascending=False)
scp._write_table(pairs, scp.fname('tsv_cooccurrence'), force=True)
order = scp.celltype_order(R.index)
fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(R.loc[order, order].values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=7)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7)
fig.colorbar(im, ax=ax, fraction=0.03).set_label('Spearman rho over 21 sections')
ax.set_title('Co-occurrence of cell types across sections')
scp.style_figure(fig, scp.params['font_color'], scp.params['bg_color'], scp.params['fontsize_scale'])
scp._savefig(fig, scp.fname('fig_cooccurrence'), force=force, verbose=verbose)
plt.show()
pairs.head(30)""")

md("""### 10. PDAC / NI genes per region; NI-related gene score

Dot plot of the 24 genes of **PDAC_GENES** per region (fraction of spots > 0; mean log1p(CP10k) / max over the regions). **NI score** (Figure 2M on the spots): mean z (within section) of the 34 NI-related genes of the paper (**PDAC_NI_GENES**; those measured by the three panels), averaged per region.""")
code("""order = [r for r in regions]
df_frac, df_mean, df_rel = scp.calc_df_dotplot(list(PDAC_GENES), 'region_label', adata_vis[:, adata_vis.var.in_all_panels.values], order=order)
fig = scp.plot_dotplot(df_frac, df_rel, row_labels={r: f"{r} {names.get(r, '')}" for r in order}, cmap='RdPu', figsize=(17, 11),
                       title='PDAC / NI genes per region', fname=scp.fname('fig_region_genes'), force=force, verbose=verbose)
plt.show()""")
code("""ni_score = scp.nerve_marker_score(adata_vis[:, adata_vis.var.in_all_panels.values], genes=PDAC_NI_GENES)
adata_vis.obs['NI_gene_score'] = ni_score
df_ni = adata_vis.obs.groupby('region_label', observed=True).NI_gene_score.agg(['mean', 'median', 'size']).assign(niche=lambda d: d.index.map(names))
df_ni.sort_values('mean', ascending=False).round(3)""")

md("""### 11. Interactive region maps

Hover: niche, top cell types. One HTML per representative section in **pdac/figures** (**pdac02_region_map_<section>.html**); PA98 is shown.""")
code("""hover = {r: f"{row.top_share}" for r, row in df_region_summary.set_index('region_label').iterrows()}
for lib in sections_rep:
    fig = scp.plotly_spatial_visium(names, adata_vis, key='region_label', library_id=lib, hover_extra=hover, fontsize_title=20,
                                    fontsize_legend=12, fontsize_hover=13, marker_line_color="#222222", fname='fig_region_map',
                                    force=force, verbose=False)
    if lib == scp.params['library_main']:
        fig.show()""")

md("""### Notes for interpretation

- **Confound**: read the CytAssist-only columns before the all-section ones; the platform changes depth, gene panel and the deconvolution.
- **Sections, not patients**: 21 sections of 17 patients (two sections for PA04, PA05, PA11).
- **TLS**: B / T cells are major types here; the TLS niche is defined by co-occurring B and T cells, not by GC B or Tfh states.
- **Malignant niches** are numbered by dominant subtype, not by the paper's (unpublished) assignment.""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"display_name": "cell-cell-communication (3.12.12)", "language": "python", "name": "python3"}
nbf.write(nb, out)
print(f"written {out} ({len(C)} cells)")
