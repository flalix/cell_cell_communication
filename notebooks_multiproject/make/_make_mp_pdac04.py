"""Generates mp_pdac04_liana_ccc.ipynb (structure of mp_heart_dvlp04_liana_ccc.ipynb; groups = regions, subregions, NI
status and nerve zones; CellChatDB as in the paper's COMMOT / CellChat). Run once; then edit freely (--force overwrites)."""
import sys
import nbformat as nbf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _make_pdac_common import HEADER_LIBS, IMPORTS, PAPER, md_code  # noqa: E402

out = Path(__file__).resolve().parent.parent / "mp_pdac04_liana_ccc.ipynb"
if out.exists() and "--force" not in sys.argv:
    raise SystemExit(f"{out} exists: not overwritten (--force)")
nb = nbf.v4.new_notebook(); C = []
md, code = md_code(C)

md(f"""### PDAC and neural invasion - ligand-receptor communication in regions, nerve zones, NI groups and between cell types (mp_pdac04)

{PAPER}

LIANA+ bivariate (spatial co-expression of ligand and receptor), cell-type-resolved sender -> receiver scores and the reference rank_aggregate, as in **mp_heart_dvlp04** / **mp_lymphoid04**, **per region** (mp_pdac01), **per subregion** (mp_pdac03), **per NI group** (low vs high) and, new here, **per nerve zone** (nerve, juxta- and peri-neural spots of invaded vs non-invaded nerves; mp_pdac03).

**Question**: which ligand-receptor (LR) pairs are spatially active around **invaded** vs **non-invaded** nerves, and between which cell types? The paper used **COMMOT** with **CellChatDB** (200 um range, per sample; Figures 4F / 4G: senders in non-invaded nerves, receivers in the juxta-neural niches, e.g. CXCL12 -> CXCR4), **CellChat** (TGF-beta network towards S02_Schwann-TGFBI, Figure 6) and **NicheNet** (ligands driving S02_Schwann-TGFBI: TGFB1, HGF). Here LIANA is run with the **consensus** resource and with **CellChatDB**.

{HEADER_LIBS}

| section | content |
|---|---|
| 1 | spots, regions, subregions, NI groups, nerve zones, abundances, signatures |
| 2 | spatial weights (Gaussian kernel per section) |
| 3 | LIANA bivariate (consensus; CellChatDB): global Moran's R and local scores |
| 4 | LR pairs enriched in each region and subregion |
| 5 | LR pairs enriched in each nerve zone (invaded vs non-invaded; Figure 4F / 4G analogue) |
| 6 | LR pairs by NI group (low vs high; all sections and CytAssist only) |
| 7 | between cell types: sender -> receiver (whole tissue, per group) |
| 8 | TGF-beta (and HGF) towards S02_Schwann-TGFBI; CXCL12 -> CXCR4 around non-invaded nerves |
| 9 | reference LIANA (rank_aggregate) + cell-type co-localisation |
| 10 | consensus per region / nerve zone; tables |

Figures and tables have the prefix **pdac04_**.""")

md("""### Critique of the analysis and how it is handled

- **Visium spots mix 5-15 cells**: a bivariate local score says ligand and receptor are co-expressed in a ~200 um neighbourhood (bandwidth 1.1 spot spacings, rings 1-2: the paper's COMMOT range), not that two cells signal.
- **21 sections, 3 platforms**: the spatial kernel is per section; genes = the 17,368 of the three panels. NI contrasts are repeated on the CytAssist sections.
- **Cell-type attribution** splits each spot's counts by abundance x reference signature (**calc_celltype_expr**); with 258 Schwann reference cells the S01-S03 signatures are noisy.
- **CellChatDB vs consensus**: CellChatDB carries pathway annotations (e.g. TGFb, CXCL, SEMA3) and complexes used by the paper; consensus is LIANA's default union.
- **Statistics**: neighbouring spots are not independent; groups and links are ranked by effect sizes (AUROC, z, ratio).""")

md("## Loading packages")
code(IMPORTS + "\nimport liana as li\nprint({p: version(p) for p in ['liana', 'scanpy', 'anndata', 'plotly']})")
md("## Parameters")
code("""force=False
verbose=False

apply_figure_style()
scp = SingleCellPDAC(ROOT0, analysis='pdac04')
scp.set_params(subregion_skip=tuple(scp.unassigned_regions()))     # spots without deconvolution: not a region of LR analysis
print(scp)
print({k: scp.params[k] for k in ['liana_resource', 'liana_bandwidth_spots', 'liana_nz_prop', 'lr_group_min_auroc', 'lr_display_exclude']})
print('CellChatDB in liana:', 'cellchatdb' in li.resource.show_resources())
scp.calc_df_files()""")

md("""### 1. Spots, regions, subregions, NI groups, nerve zones and abundances""")
code("""df_regions = scp.read_tsv('region_labels', verbose=verbose)
source = df_regions['abundance_source'].iloc[0]
adata_map = scp.read_h5ad('visium', set_self=False)
adata_map = scp.remove_mt_genes(adata_map)
if source == 'c2l':
    a_map = scp.read_h5ad_slots('c2l_map', skip=('layers',))
    adata_map.obsm['q05_cell_abundance_w_sf'] = pd.DataFrame(a_map.obsm['q05_cell_abundance_w_sf'], index=a_map.obs_names).loc[adata_map.obs_names]
    adata_map.uns['mod'] = a_map.uns['mod']; del a_map
d = df_regions.set_index('barcode').loc[adata_map.obs_names]
for c in ('region_cluster', 'region_label', 'region_name', 'niche'):
    adata_map.obs[c] = pd.Categorical(d[c].astype(str).values)
region_name = dict(zip(d.region_label.astype(str), d.region_name.astype(str)))
df_abund = scp.get_abundance(adata_map, source=source)
df_share = scp.shares(df_abund)
df_nerve = scp.read_tsv('nerve_spots').set_index('barcode').loc[adata_map.obs_names]
z = df_nerve.nerve_zone.fillna('').astype(str)
st = df_nerve.ring_status.fillna('').astype(str)
adata_map.obs['nerve_zone_status'] = np.where(z != '', z + ' (' + st + ')', '')

adata_ref = scp.read_h5ad('reference', set_self=False, backed='r')
adata_ref_dec = scp.reference_for_deconvolution(adata_ref)
df_sig = scp.reference_signatures_symbols(adata_ref_dec, summary='means')
print("signatures:", df_sig.attrs['source'], df_sig.shape)

adata_expr = scp.prepare_expression(adata_map)
subs, _ = scp.run_subregions(adata_expr, force=False, verbose=verbose)
celltypes = [c for c in df_abund.columns if c in df_sig.columns]
df_sub03 = scp.read_tsv('subregion_table_pdac03', verbose=verbose)
sub_supported = df_sub03.loc[df_sub03.status.str.startswith('supported'), 'sub_region_id'].tolist()
adata_expr.obs['nerve_zone_status'] = adata_map.obs.loc[adata_expr.obs_names, 'nerve_zone_status'].values
groups = {'region': scp._groups(adata_expr, 'region_label', skip=set(scp.params['subregion_skip'])),
          'subregion': scp._groups(adata_expr, 'sub_region_id'),
          'ni_status': scp._groups(adata_expr, 'ni_status'),
          'nerve_zone': scp._groups(adata_expr, 'nerve_zone_status', skip={''})}
sections_rep = scp.representative_sections()
del subs, adata_map; gc.collect()          # labels are in adata_expr.obs; frees ~6 GB before LIANA
print(adata_expr.shape, '|', len(celltypes), 'cell types |', {k: v.nunique() for k, v in groups.items()})
groups['nerve_zone'].value_counts()""")

md("""### 2. Spatial weights

Gaussian kernel **w = exp(-d^2 / 2bw^2)**, **bw = liana_bandwidth_spots x spot spacing**, per section, self weight 1, weights < **liana_cutoff** removed.""")
code("""df_w = scp.spatial_weights(adata_expr, force=force, verbose=verbose)
W = adata_expr.obsp['spatial_connectivities']
df_w.groupby('library').mean_neighbours_incl_self.first().describe()""")

md("""### 3. LIANA bivariate: consensus and CellChatDB

**li.mt.bivariate**: global Moran's R (permutation p, BH FDR) and the local cosine score of every pair at every spot; ligand and receptor (all complex subunits) in >= **liana_nz_prop** of the spots. Cached in **pdac/results**.""")
code("""df_inter, adata_lr = scp.liana_bivariate(adata_expr, resource_name='consensus', fname='liana_bivariate', fname_global='liana_global',
                                         force=force, verbose=verbose)
df_lr_sig = scp.lr_significant(df_inter)
df_inter_cc, adata_lr_cc = scp.liana_bivariate(adata_expr, resource_name='cellchatdb', fname='liana_bivariate_cellchat',
                                               fname_global='liana_global_cellchat', force=force, verbose=verbose)
df_lr_sig_cc = scp.lr_significant(df_inter_cc)
print(f"consensus: {len(df_inter)} pairs tested, {len(df_lr_sig)} significant | CellChatDB: {len(df_inter_cc)} tested, {len(df_lr_sig_cc)} significant")
df_lr_sig[['ligand', 'receptor', 'ligand_props', 'receptor_props', 'morans', 'morans_fdr']].head(20).round(3)""")
code("""lr_show = scp._lr_display(df_lr_sig.index)[:8]
for lib in [scp.params['library_main'], 'PA72']:
    fig = scp.plot_lr_spatial(adata_lr, adata_expr, lr_show, library_id=lib, ncols=4, figsize=(18, 8), fontsize_title=14,
                              fontsize_suptitle=16, fontsize_colorbar=12, fname='fig_lr_top', force=force, verbose=verbose)
    plt.show()""")

md("""### 4. LR pairs enriched in each region and subregion

**calc_df_lr_by_group**: per group, AUROC of the local score in vs out, mean score in / out, high-high fractions; **enriched** = AUROC >= **lr_group_min_auroc** and more high-high spots than elsewhere.""")
code("""df_lr_region = scp.calc_df_lr_by_group(adata_lr, groups['region'], df_lr_sig.index, fname='lr_by_region', force=force, verbose=verbose)
df_lr_sub = scp.calc_df_lr_by_group(adata_lr, groups['subregion'], df_lr_sig.index, fname='lr_by_subregion', force=force, verbose=verbose)
df_n = pd.DataFrame({'n_enriched': df_lr_region[df_lr_region.enriched].groupby('group').size()}) \\
         .reindex(sorted(groups['region'].dropna().unique())).fillna(0).astype(int)
df_n['region_name'] = df_n.index.map(region_name)
df_n['top_lr'] = scp.calc_df_lr_group_top(df_lr_region)
df_n""")
code("""fig = scp.plotly_lr_group_heatmap(df_lr_region, n_per_group=3, group_names=region_name, height=950, width=1800, fontsize_title=22,
                                  fontsize_xticks=13, fontsize_yticks=13, fontsize_colorbar=14, fontsize_hover=14,
                                  fname='fig_lr_region_hm', force=force, verbose=verbose)
fig.show()""")

md("""### 5. LR pairs by nerve zone: invaded vs non-invaded (Figure 4F / 4G analogue)

Groups = nerve, juxta-neural (rings 1-2) and peri-neural (rings 3-5) spots of invaded and non-invaded nerves (mp_pdac03), each vs the rest of the tissue; CellChatDB resource (as the paper's COMMOT). The contrast **juxta-neural (non-invaded) vs (invaded)** is read from the two AUROC columns.""")
code("""has_nerves = groups['nerve_zone'].replace('', np.nan).notna().any()
df_lr_nerve = scp.calc_df_lr_by_group(adata_lr_cc, groups['nerve_zone'], df_lr_sig_cc.index, fname='lr_by_nerve', force=force,
                                      verbose=verbose) if has_nerves else pd.DataFrame(columns=['group', 'interaction', 'auroc', 'enriched'])
print('nerve zones:', groups['nerve_zone'].value_counts().to_dict() if has_nerves else 'none (no nerve called in mp_pdac03)')
P = df_lr_nerve.pivot_table(index='interaction', columns='group', values='auroc') if len(df_lr_nerve) else pd.DataFrame()
if {'juxta-neural (non-invaded)', 'juxta-neural (invaded)'} <= set(P.columns):
    P['delta_juxta_non_minus_inv'] = P['juxta-neural (non-invaded)'] - P['juxta-neural (invaded)']
    keep = scp._lr_display(P.index)
    display(P.loc[keep].sort_values('delta_juxta_non_minus_inv', ascending=False).head(15).round(3))
    display(P.loc[keep].sort_values('delta_juxta_non_minus_inv').head(15).round(3))""")
code("""if len(df_lr_nerve):
    fig = scp.plotly_lr_group_heatmap(df_lr_nerve, n_per_group=5, height=900, width=1700, fontsize_title=22, fontsize_xticks=13,
                                      fontsize_yticks=13, fontsize_colorbar=14, fontsize_hover=14, fname='fig_lr_nerve_hm', force=force,
                                      verbose=verbose)
    fig.show()""")
code("""top_nerve = [i for i in ['CXCL12^CXCR4', 'TGFB1^TGFBR1_TGFBR2', 'SEMA3A^NRP1_PLXNA1', 'MDK^NCL', 'PTN^NCL', 'APP^CD74', 'MIF^CD74_CXCR4']
             if i in adata_lr_cc.var_names]
lib = scp.params['library_main']
reg = adata_expr.obs['nerve_zone_status'].astype(str).values
fig = scp.plot_lr_spatial(adata_lr_cc, adata_expr, top_nerve or scp._lr_display(df_lr_sig_cc.index)[:6], library_id=lib, ncols=4,
                          figsize=(18, 8), outlines=[np.char.startswith(reg.astype(str), 'nerve')] * max(1, len(top_nerve)),
                          fontsize_title=13, fontsize_suptitle=16, fontsize_colorbar=12, fname='fig_lr_nerve', force=force, verbose=verbose)
plt.show()""")

md("""### 6. LR pairs by NI group

Low vs high NI sections as groups (AUROC of the local score in the group's spots vs the other group). The ranking is repeated on the CytAssist sections only (all low-NI sections are CytAssist).""")
code("""df_lr_ni = scp.calc_df_lr_by_group(adata_lr, groups['ni_status'], df_lr_sig.index, fname='lr_by_ni', force=force, verbose=verbose)
cyt = (adata_expr.obs.technique.astype(str) == 'CytAssist').values
g_cyt = groups['ni_status'].where(cyt)
df_lr_ni_cyt = scp.calc_df_lr_by_group(adata_lr[cyt], g_cyt[cyt], df_lr_sig.index, fname=None, force=True, verbose=verbose)
keep = set(scp._lr_display(df_lr_ni.interaction.unique()))
M = pd.concat({'all': df_lr_ni.pivot_table(index='interaction', columns='group', values='auroc'),
               'CytAssist': df_lr_ni_cyt.pivot_table(index='interaction', columns='group', values='auroc')}, axis=1)
M = M.loc[[i for i in M.index if i in keep]]
display(M.sort_values(('CytAssist', 'low'), ascending=False).head(15).round(3))
display(M.sort_values(('CytAssist', 'high'), ascending=False).head(15).round(3))""")

md("""### 7. Between cell types: sender -> receiver

**calc_celltype_expr** splits each spot's counts across cell types (abundance x signature); **calc_df_celltype_lr** for the **lr_top_n** significant pairs (consensus) plus the CellChatDB pairs of section 8: score[A, B] = mean over spots of (W_norm L_A) x R_B, receptor spots permuted -> z; per group (regions, nerve zones, NI groups).""")
code("""extra = [i for i in df_lr_sig_cc.index if any(k in i for k in ('TGFB', 'HGF', 'CXCL12', 'SEMA3', 'NGF', 'GDNF', 'NRTN', 'ARTN', 'BDNF'))]
df_lr_top = pd.concat([df_lr_sig.head(scp.params['lr_top_n'])[['ligand', 'receptor']],
                       df_lr_sig_cc.loc[extra, ['ligand', 'receptor']]])
df_lr_top = df_lr_top[~df_lr_top.index.duplicated()]
genes = sorted({s for e in list(df_lr_top.ligand) + list(df_lr_top.receptor) for s in e.split('_')})
E, genes_E, celltypes_E = scp.calc_celltype_expr(adata_expr, df_abund.loc[adata_expr.obs_names, celltypes], df_sig[celltypes], genes)
print('E (spots x cell types x genes):', E.shape)
groups_ct = {'region': groups['region'], 'nerve_zone': groups['nerve_zone'], 'ni_status': groups['ni_status']}
df_ct, df_group_ct = scp.calc_df_celltype_lr(E, genes_E, celltypes_E, df_lr_top, W, groups=groups_ct, force=force, verbose=verbose)
df_best = df_ct[df_ct.rank_in_lr == 1].set_index('interaction')
df_best[['sender', 'receiver', 'score', 'score_share', 'z']].head(20).round(3)""")
code("""lr_panels = scp._lr_display(df_best.index)[:6]
fig = scp.plot_sender_receiver_heatmaps(df_ct, lr_panels, n_top=10, ncols=3, panel_size=(8, 7.0), fontsize_title=16, fontsize_ticks=11,
                                        fontsize_label=12, fontsize_colorbar=12, fname='fig_sender_recv', force=force, verbose=verbose)
plt.show()""")
code("""df_links = scp.calc_df_group_celltype_links(df_group_ct, force=force, verbose=verbose)
lev = df_group_ct.drop_duplicates('group').set_index('group').level
df_links_nerve = df_links[df_links.group.map(lev) == 'nerve_zone']
fig = scp.plotly_group_celltype_links(df_links_nerve, n_links=30, groups=sorted(df_links_nerve.group.unique()), height=1000, width=1600,
                                      fontsize_title=22, fontsize_xticks=13, fontsize_yticks=13, fontsize_colorbar=14, fontsize_hover=14,
                                      fname='fig_links_nerve', force=force, verbose=verbose)
fig.show()
df_links_nerve[~df_links_nerve.self_link].groupby('group').head(4)[['group', 'link', 'n_lr', 'lr']]""")

md("""### 8. TGF-beta and HGF towards S02_Schwann-TGFBI; CXCL12 -> CXCR4 around non-invaded nerves

Paper (Figure 6): TGFB1 from M07_Macro-NLRP3, myCAF and malignant cells induces TGFBI+ Schwann cells (CellChat TGF-beta network; NicheNet ligands TGFB1, HGF). Here: sender -> receiver scores of the TGF-beta / HGF pairs with **S02_Schwann-TGFBI** (and S01 / S03 for comparison) as receivers, whole tissue and per nerve zone. Paper (Figure 4H): CXCL12 from LUM+ fibroblasts and CXCR4+ B cells around non-invaded nerves.""")
code("""df_tgfb = scp.calc_df_tgfb_to_schwann(df_ct, top=5, fname='tsv_tgfb', force=force, verbose=verbose)
fig = scp.plot_tgfb_to_schwann(df_ct, n_senders=15, fname='fig_tgfb', force=force, verbose=verbose)
plt.show()
df_tgfb.head(20)""")
code("""cx = [i for i in df_group_ct.interaction.unique() if i.startswith('CXCL12^')]
d_c = df_group_ct[df_group_ct.interaction.isin(cx) & (df_group_ct.level == 'nerve_zone')]
d_c.sort_values('score', ascending=False).groupby('group').head(3)[['group', 'interaction', 'sender', 'receiver', 'score', 'z', 'ratio']].round(3) if len(d_c) else 'no CXCL12 pair scored'""")

md("""### 9. Reference LIANA (rank_aggregate) + cell-type co-localisation

**liana_rank_aggregate_reference** on the reference (<= **ref_lr_cells_per_type** cells per type); non-spatial, so each source -> target is annotated with the co-localisation of the two cell types (**calc_df_celltype_coloc**).""")
code("""adata_ref_sym = adata_ref_dec.copy()
df_liref = scp.liana_rank_aggregate_reference(adata_ref_sym, 'cell_type', symbol_col='symbol', force=force, verbose=verbose)
df_coloc, coloc_region = scp.calc_df_celltype_coloc(df_abund.loc[adata_expr.obs_names, celltypes_E], W, groups['region'])
df_ref_sup = scp.calc_df_ref_lr_support(df_liref, df_coloc, coloc_region, force=force, verbose=verbose)
print(f"reference interactions: {len(df_liref)}; specific & strong: {len(df_ref_sup)}; spatially supported: {int(df_ref_sup.spatially_supported.sum())}")
df_ref_sup[df_ref_sup.spatially_supported & df_ref_sup.listed][['source', 'target', 'interaction', 'magnitude_rank', 'specificity_rank',
                                                                'coloc', 'best_group', 'coloc_group']].head(20).round(3)""")
code("""fig = scp.plotly_coloc_heatmap(df_coloc, height=1100, width=1500, fontsize_title=22, fontsize_ticks=12, fontsize_colorbar=14,
                               fontsize_hover=14, fname='fig_coloc', force=force, verbose=verbose)
fig.show()""")

md("### 10. Consensus per region / nerve zone")
code("""df_ccc_region = scp.calc_df_group_ccc(df_lr_region, df_group_ct[df_group_ct.level == 'region'], df_ref_sup, group_names=region_name,
                                      fname='ccc_by_region', force=force, verbose=verbose)
if len(df_lr_nerve):
    df_ccc_nerve = scp.calc_df_group_ccc(df_lr_nerve, df_group_ct[df_group_ct.level == 'nerve_zone'], df_ref_sup, n=5,
                                         fname='ccc_by_nerve', force=force, verbose=verbose)
    display(df_ccc_nerve.round(2))
df_ccc_region.round(2).head(30)""")

md("""### Notes for interpretation

- **Nerve zones** come from the data-derived nerve calls of mp_pdac03; with H&E overrides the zones and these tables change (rerun with force=True).
- **COMMOT vs LIANA**: COMMOT solves an optimal-transport problem with competition between receptors; LIANA's bivariate score is a local co-expression. The same LR pairs can rank differently; compare directions (non-invaded vs invaded), not values.
- **Senders / receivers** follow abundances x reference signatures: which cell types *can* carry ligand and receptor *where*.""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"display_name": "cell-cell-communication (3.12.12)", "language": "python", "name": "python3"}
nbf.write(nb, out)
print(f"written {out} ({len(C)} cells)")
