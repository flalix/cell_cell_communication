# spatial_niche_lr

Niche-level ligand-receptor (LR) analysis for spatial transcriptomics (Visium or single-cell
resolution), with ligand/receptor gradients and interpretation through 50 biological programs.

| file | content |
|---|---|
| `programs.py` | 50 functional programs (9 categories), `CELL_STATES` markers, Reactome name rules, human->mouse conversion |
| `niche_lr.py` | center selection, k<=20 neighbourhood, two-direction LR scoring vs. random niches, program activity, function inference, collinear gradients, plots, multi-niche scan, LIANA cross-check |
| `lr_consensus.csv`, `lr_mouseconsensus.csv` | LIANA Consensus LR resources (human / mouse symbols) |
| `reactome/` | every Reactome pathway (human, mouse) -> program / background class; program summaries; gene catalogues |
| `squid02_niche_lr_programs.ipynb` | notebook continuing `squid01_analyze_visium_HE_data.ipynb` |

Key options (`nl.analyze_niche`):
- `center_pool` - pool the center with its nearest spots (6 = Visium ring 1). Use 0 for single cells.
- `bg_mask` - background niches: `None` = whole tissue; `cluster != own` = region vs. rest.
- `fdr` - empirical p, Gaussian tail of z beyond the background range; `fdr_emp` = purely empirical.
- gradients - 3-6 collinear cells toward the best partner; trend only if |Spearman rho| >= 0.6.

Gene expansion from Reactome needs `ReactomePathways.gmt` (gene members):
`pg.expand_with_reactome(pg.load_gmt("ReactomePathways.gmt"))`.

# heart_dvlp (developing human heart, Lázár et al. 2025)

| file | content |
|---|---|
| `heart_dvlp_download.py` | Mendeley part 1 (doi:10.17632/fhtb99mdzd.1): the 38 Space Ranger outputs, metadata, extended tables -> `heart_dvlp/data`; `--with-rds`, `--with-cellranger`, `--with-iss` for the rest |
| `export_seurat_rds.R` | Seurat RDS -> Matrix Market + CSV bundle with SeuratObject only (no Seurat / SeuratDisk; SCT assays are not touched) |
| `heart_dvlp_build_h5ad.py` | the two complete h5ad files: `hdca_heart_visium_38sections.h5ad` (+ paper clusters, stereoscope proportions) and `hdca_heart_sc_reference.h5ad` |
| `run_c2l_heart_dvlp.py` | cell2location reference (NB regression, 31 types) and mapping (38 sections) -> `heart_dvlp/results/c2l_<tag>/`; GPU: `--mode all --tag gpu --accelerator gpu [--batch_size 10000]` |
