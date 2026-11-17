# PDAC - neural invasion (Chen et al. 2025, Cancer Cell)

Chen MM, Gao Q, Ning H, ..., Chen R, Zhang Z. *Integrated single-cell and spatial transcriptomics uncover distinct cellular
subtypes involved in neural invasion in pancreatic cancer.* Cancer Cell 43(9):1656-1676 (2025).
doi:10.1016/j.ccell.2025.06.020 - PMID 40680743 - code: github.com/PDAC-zhanglab/PDAC (R figure scripts only).

## Data provenance (pdac/data)

| item | source | notes |
|---|---|---|
| `geo/GSE278687/` | GEO GSE278687 (Visium, 21 sections) | `GSE278687_RAW.tar` (1.1 GB); per section `filtered_feature_bc_matrix.h5` + `spatial.tar.gz` (gzipped twice) |
| `geo/GSE278688/` | GEO GSE278688 (scRNA, 10x 5' v2) | combined MTX + metadata: 199,184 cells, 14 major types only |
| `geo/GSE278689/` | GEO GSE278689 (FFPE snRNA, 10x Flex) | combined MTX + metadata: 32,005 nuclei |
| `geo/*.txt` | GEO SOFT records | sample titles carry patient, preservation / technique and NI status |
| `github_PDAC-zhanglab/` | the authors' repository | figure scripts; they read local Seurat objects that are not published |
| `visium/<section>/` | unpacked from GSE278687 | Space Ranger layout (`spatial/` with hires / lowres H&E) |
| `pdac_sections.tsv` | built | 21 sections: patient, technique, panel, NI status, depth, matched sc / sn |
| `pdac_visium_21sections.h5ad` | built | 81,846 spots x 36,601 genes; `var['in_all_panels']` = 17,368 genes |
| `pdac_sc_sn_reference.h5ad` | built | 231,189 cells x 30,053 genes (Ensembl IDs, `var['symbol']`) |

GSE278694 is the **SuperSeries** (Visium + scRNA + snRNA + TCR). No h5ad, RCTD weights, niche or nerve annotations are public.

## Programs (src/programs)

```bash
# 1. GEO -> h5ad (CPU, ~3 min)
.venv/bin/python src/programs/pdac_build_h5ad.py --step all

# 2. re-derived subclusters of the reference (CPU, ~40 min; cached)
.venv/bin/python src/programs/pdac_reference_subclusters.py

# 3a. RCTD (the paper's method; R 4.5.3 of the renv conda env + spacexr 2.2.1 in the project library .R_lib)
#     inputs are exported by SingleCellPDAC.export_rctd_inputs (mp_pdac01, or the session that built them)
R_LIBS_USER=.R_lib /home/flavio/miniforge3/envs/renv/bin/Rscript src/programs/run_rctd_pdac.R 8 > pdac/results/rctd/run_rctd.log 2>&1
#     ~1,000 s per section on 8 cores (resumable: finished sections are skipped)

# 3b. cell2location (GPU)
.venv/bin/python src/programs/run_c2l_pdac.py --mode all --tag gpu --ref_epochs 250 --map_epochs 6000 --accelerator gpu \
        > pdac/results/c2l_gpu_run.log 2>&1
```

spacexr was installed once into `.R_lib` (renv itself unchanged):
`R_LIBS_USER=.R_lib .../renv/bin/Rscript -e 'remotes::install_github("dmcable/spacexr", build_vignettes=FALSE, upgrade="never", lib=Sys.getenv("R_LIBS_USER"))'`

## Notebooks (notebooks_multiproject; generators in notebooks_multiproject/make/_make_mp_pdac0*.py)

| notebook | content |
|---|---|
| mp_pdac01_prepare | sections, QC, reference, subclusters, Figure 1B, genes on sections, cell2location / RCTD, regions -> niches |
| mp_pdac02_regions | Figure 1F, niches by NI (2E / 3D), PCA (3C), cell types by NI (3B), malignant subtypes (2C / 2J), co-occurrence |
| mp_pdac03_subregions | subregions per region; nerve spots, invasion, rings: Figures 4A, 4B, 2L, 5K, 6A |
| mp_pdac04_liana_ccc | LIANA bivariate (consensus, CellChatDB) by region / subregion / NI / nerve zone; sender -> receiver; TGF-beta -> S02 Schwann |

## Differences from the paper

1. **Subclusters are re-derived** (scVI + Leiden + the paper's marker-labelled subclusters): D01-D09, F01/F02/F04-F08, M02-M04/M06-M14, S01-S03.
   Not recovered: F03_tCAF-MME, M05_Mono-CD16. Lymphoid (T01-T23, B01-B06) and endothelial (E01-E03) subsets were not re-derived.
2. **Platforms**: 5 FF polyA, 4 FFPE v1, 12 CytAssist sections; joint analyses on the 17,368 shared genes.
   **All 5 low-NI sections are CytAssist** -> every high vs low comparison is repeated on CytAssist only (5 vs 7).
3. **Niches**: Leiden on cell-type shares + rule-based naming after the paper's 12 niches (the paper: expression clusters + H&E).
   Rule (`assign_niches`): score = share x log2(enrichment) of each niche's cell types, share >= 0.10; best score < 0.1 -> `mixed (A / B)`;
   regions of spots with no deconvolution (RCTD-dropped) -> `N00_unassigned (no deconvolution)`, skipped in 03 / 04;
   the four malignant niches -> `N07-10_Malignant (<dominant subtype>)` (the paper's N07-N10 split is not reproducible from shares).
4. **Nerves**: data-derived (Schwann share + nerve markers); invasion = malignant share in / next to the bundle
   (the paper: pathologists on the H&E). Manual overrides: `pdac/data/pdac_nerve_annotation.tsv` (section, barcode, nerve, invaded).
5. **Communication**: LIANA bivariate (consensus + CellChatDB) instead of COMMOT; no NicheNet / CytoSig.
   Run per section (a joint 81,846-spot kernel would be densified to ~53 GB): pairs fixed by a global expression filter,
   pooled Moran's R = spot-weighted mean, p-values combined by Stouffer, `n_sections_sig` per pair.
6. **Cell-type names and descriptions**: `tables/pdac_lineage.tsv` (HL_ID = GEO major type, HL_SUBID, description, functional_summary),
   as `tables/heart_lineage.tsv`.

## Notebook conventions

- No file IO in the notebooks: every table is written / read inside its `calc_df_*` method and every figure inside its
  `plot_*` method (`fname=<key of PDAC_FILES>`, `force`, `verbose`). File present and `force=False` -> the table is read
  (`df.attrs` restored from `<file>.attrs.json`); otherwise computed, written and read back, so a computed and a cached
  run show the same values (verified: re-execution of 01-04 from the cache gives identical outputs).
- `force=True` in the parameter cell after changing parameters or inputs; region labels and the niche table of mp_pdac01
  are rewritten only with `force=True` (`save_region_labels`, `annotate_niches`), so 02-04 always read one clustering.

## Results of the RCTD run (2026-11-17)

- 28 regions: 6 N05_CAF-enriched, 5 N11_Non-malignant-pancreas, 7 malignant (D04 x2, D08 x2, D05, D07, Ductal-other),
  2 N04_Plasma-enriched, 1 each N01_TLS, N02_Endocrine, N03_Peri-vascular, N06_TAM-enriched, N12_Nerve, 1 mixed, 2 N00.
- High vs low NI (sections as replicates): N04_Plasma-enriched lower in high NI (p = 0.001, FDR 0.018; CytAssist only p = 0.018),
  N07-10_Malignant (D04_Ductal-GABRP) higher (p = 0.007; CytAssist p = 0.034); malignant Shannon diversity lower in high NI
  (p = 0.0001; CytAssist p = 0.0025).
- Nerves: 65 bundles; 16 invaded, all in high-NI sections; none invaded in the 5 low-NI sections.
- Juxta-neural spots of invaded vs non-invaded nerves: malignant (D09, D04), M10_Macro-SPP1, F05 / F08 CAFs up; B, Plasma,
  CD4T, F02 progenitor CAF down. Malignant enrichment is partly circular (invasion is called from the ring-1 malignant share).
- TGF-beta / HGF -> Schwann: the top senders are F01_myCAF-MMP11 and F02_Progenitor-CAF-DPT, not M07_Macro-NLRP3 as in the paper.

## Open items

- cell2location GPU run (tag gpu) -> rerun mp_pdac01 with force=True, then 02-04.
- Paper Table S1-S5 (publisher supplements) not accessible here; the authors' `spatial_st3.rds` (RCTD, niches) would allow a direct benchmark.
