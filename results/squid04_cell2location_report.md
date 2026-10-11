# cell2location deconvolution of the Visium H&E mouse brain section

**Data.** Visium H&E coronal mouse brain (squidpy `visium_hne`; 2,688 spots, raw counts from `.raw`) and the squidpy `sc_mouse_cortex` reference (Allen Smart-seq, 21,697 cells, 22 subclasses after dropping CR, n = 17).

**Method.** Signatures as per-subclass average counts (16,304 shared genes), because the NB regression model is not suited to Smart-seq data (Kleshchevnikov et al. 2020 preprint, Methods). Spatial model: cell2location 0.1.5, `N_cells_per_location=8`, `detection_alpha=20`, `detection_mean=1/20`, 30,000 iterations on an RTX 3090; ELBO plateaued by ~10,000 iterations.

Scripts: `notebooks/squid04a_c2l_spatial_gpu.py` (model fit) and `notebooks/squid04b_c2l_downstream.py` (everything below).

## 1. Cell-type maps

![Cell-type maps]({{artifact:art_7f03aee6-3619-42b9-b943-0a9f5422ae39}})

Cortex is resolved layer by layer: L2/3 IT in the outer cortex (Cortex_2, Cortex_4), L4 in Cortex_1, L6 CT and L6 IT in deep cortex (Cortex_3). Oligodendrocytes trace the fibre tracts (60 % of Fiber_tract spots). VLMC and SMC concentrate along the lateral ventricle and the midline/meningeal surface.

![Cluster composition]({{artifact:art_36ddd834-480b-469c-be19-1f89d0feaad1}})

| Cluster | Top 3 cell types (mean fraction) |
|---|---|
| Cortex_1 | L4 0.30, L5 IT 0.15, Pvalb 0.13 |
| Cortex_2 | L2/3 IT 0.31, Astro 0.14, L5 IT 0.10 |
| Cortex_3 | L6 CT 0.23, L6 IT 0.17, L4 0.10 |
| Cortex_4 | L2/3 IT 0.51, L4 0.19, Astro 0.11 |
| Fiber_tract | Oligo 0.60, Astro 0.13, Pvalb 0.04 |
| Lateral_ventricle | VLMC 0.23, Astro 0.22, Oligo 0.10 |
| Pyramidal_layer | L2/3 IT 0.42, L5 PT 0.18, Astro 0.09 |
| Striatum | L2/3 IT 0.21, Sst 0.20, Astro 0.13 |
| Thalamus_1 | L4 0.26, Pvalb 0.22, Astro 0.18 |
| Hypothalamus_1 | Sst 0.35, Astro 0.16, Vip 0.14 |

**Reference-gap caveat (expected).** The reference has no hippocampal, thalamic, hypothalamic or striatal neurons. Their spots are therefore assigned to the closest cortical type, for example hippocampal pyramidal and dentate gyrus neurons → L2/3 IT / L5 PT / L6 IT, thalamic relay neurons → L4, and striatal and hypothalamic neurons → Sst / Vip / Pvalb. Treat neuronal fractions outside cortex as "most similar cortical type", not identity. Choroid plexus and ependyma are also absent, which likely inflates VLMC/SMC at the ventricle.

## 2. Dominant type per spot

![Dominant type]({{artifact:art_c2df2cc3-2dda-44b0-9605-1fcf468fb32e}})

Median dominant fraction is 0.34; only 372 / 2,688 spots have one type ≥ 0.5 (mostly Oligo in fibre tracts and L2/3 IT in outer cortex). Most Visium spots are genuinely mixed, so fractions, not labels, are the right unit.

## 3. Detection sensitivity check

![Detection vs UMI]({{artifact:art_a3f9b35e-33c5-47fa-ac42-05547d77f4d0}})

The fitted capture efficiency `y_s` spans 0.71–1.59 × median (2.5–97.5 %), inside the α = 20 prior range and well outside α = 200. Its Spearman ρ with the smoothed observed UMI ratio is 0.36. `y_s` also follows anatomy (it is high in the pyramidal layers), so it absorbs some biological RNA-content differences as well as technical ones. Total estimated cells still tracks UMI depth closely (ρ = 0.96). The median of 12.4 cells/spot (q05) exceeds the prior of 8, likely because Smart-seq-to-Visium scaling is only approximately captured by `detection_mean=1/20`. Use **relative** abundances (fractions) for comparisons between regions.

## 4. Link to LIANA

![LR vs cell type]({{artifact:art_713617d1-9e49-4b0a-b11b-6c68152faedb}})

Local LR scores (LIANA bivariate cosine) were correlated with per-spot cell-type fractions:
- **Oligodendrocyte-associated pairs:** Gnai2–S1pr5, Gnai2–Unc5b, Ncam1–Fgfr2, Fgf1–Fgfr2, Sema4d–Plxnb3. S1pr5 is an oligodendrocyte gene, so these are coherent.
- **Cortical excitatory neurons and Lamp5:** Nptx1–Nptxr, Rims1–Slc17a7, the Rtn4/Omg/Lgi1 → Rtn4r/Lingo1 family and Lin7c–Kcnj4.
- **Subcortical pairs:** Mdk–Lrp1/Ncl/Sdc3/Ptprz1, Apoe–Ldlr and Agt–Adra2a. These correlate with Sst/Vip/Sncg/NP/L6b fractions. Because those fractions are themselves reference-gap artefacts in thalamus and hypothalamus, read these as "subcortical region" signals.

Correlations are spot-level and confounded by region. They indicate co-localisation, not signalling direction.

LIANA `rank_aggregate` on confidently dominant spots (≥ 0.5; Oligo 168, L2/3 IT 133, L4 38) ranks neuron → oligodendrocyte Psap–Gpr37 highest (specificity rank ≤ 0.01). Oligodendrocyte → neuron pairs Lgi3–Stx1a and Trf–Tfrc follow, as does neuronal Nptx1–Nptxr.

## Cell types

See `celltype_descriptions.tsv` ({{artifact:art_b727a46e-a839-477f-8dbd-78608bc20122}}) for the 22-row description table.

## Files (results/ and figures/ in ~/uv/squidpy)

`c2l_abundance_q05.tsv`, `c2l_fractions.tsv`, `c2l_cluster_by_celltype.tsv`, `lr_vs_celltype_spearman.tsv`, `liana_dominant_celltype_results.tsv`, `data/visium_hne_c2l_annot.h5ad` (all per-spot annotations).
