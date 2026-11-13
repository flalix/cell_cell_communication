# Cell-cell communication: lymphoid tissue and mouse brain

## Code organisation (`src/`)

The notebooks are thin: they hold the questions, the gene and cell-type lists, and the interpretation. The computation lives in two folders under `src/`, imported after adding `src/` to `sys.path`.

### `src/libs`: the analysis classes

- **`single_cell_lib.py`** defines `SingleCell`, the generic class shared by both projects. It wraps one AnnData and the project folders (`root_data`, `root_figure`, `root_result`), and covers the full spatial workflow: reading and QC, cell2location training and mapping, region and subregion clustering, LIANA+ ligand-receptor analysis, and the figures. Every tunable value is a parameter (`DEFAULT_PARAMS`), every file is registered by a key, and method arguments left as `None` take the class values.
- **`single_cell_[project]_lib.py`** holds one subclass per dataset: `SingleCellLymphoid` (human lymph node and the lymphoid-organ reference) and `SingleCellNeuro` (mouse brain, cell2location paper Figs 2-3). Each adds the dataset's parameters (`LYMPHOID_PARAMS`, `NEURO_PARAMS`), its file registry per analysis (`LYMPHOID_FILES`, `NEURO_FILES`, e.g. `'lymph01'`, `'mb03'`), and the methods specific to its biology (B-cell subsets, tissue x age groups, astrocyte subtypes, cortical layers).
- **`cell2location_lib.py`** is the earlier function library of the `ccc01` / `ccc02` notebooks. Its methods were ported into the classes above; the file is kept unchanged so those notebooks still run.
- **`matplotlib_figure_style.py`** provides `apply_figure_style()`, the base matplotlib settings (font-size ladder, ticks, 300-dpi output) applied once at the top of each notebook.

### `src/programs`: stand-alone programs and resources

1. **`c2l_liana.py`**: functions linking a cell2location map to LIANA+ communication: cell-type-resolved expression per spot, sender -> receiver ligand-receptor scores, spatial co-localisation and niche composition. Used by `ccc03_liana_study`; its ideas became section 9 (LIANA) of `SingleCell`.
2. **`run_c2l_mouse_brain.py`**: a command-line script that runs the two long cell2location mappings of the mouse brain (Slide-seq for Fig 2H, Allen Smart-seq for Fig 2J) outside the notebook, on a GPU. It writes the caches that the notebooks then read.
3. **`niche_lr.py`**: niche-level ligand-receptor analysis around a chosen spot or cell: its nearest neighbours, two-direction LR scores tested against random niches, ligand and receptor gradients, and the biological functions the niche suggests.
4. **`programs.py`**: the curated catalogue of about 50 biological programs (gene sets in 9 categories), cell-state markers, rules to extend them from Reactome, and human -> mouse gene conversion. `niche_lr.py` uses it to interpret the niches.

Every H&E image now has black glass by default, and `plotly_spatial_visium` writes its own HTML. On the lymph node, the black-glass version also removes the fiducial-dot frame and the grey glass.

![Lymph node maps with black glass]({{artifact:765bd594-f54f-44e8-8c4e-67d5976a0215}})

![Mouse brain Fig 3E with black glass]({{artifact:9944f778-d651-49aa-aab8-85b4199fafa2}})

- **Black glass by default:** `image_bg_color="black"` is now a class parameter. All map figures use the recoloured H&E, including the scanpy, squidpy and cell2location plots, which get a temporary swap of the image. The image stored in your AnnData is unchanged afterwards; I checked.
- **Lymph-node glass detection:** this slide's glass is grey, not white, so the fixed detection threshold treated the whole slide as tissue. The threshold is now estimated from the image border: 0.49 here, 0.90 on the brain slides, which is the old value. I regenerated `lymphoid_tissue/data/hires_bg/V1_Human_Lymph_Node_bg_000000.png`, which my earlier test had written with the wrong threshold.
- **Plotly map:** `plotly_spatial_visium(..., fname='fig_regions_html', include_plotlyjs=True)` saves `mb01_brain_regions_{lib}.html` itself. I tested it: a self-contained 5.9 MB file. The notebook's `write_html` line is removed.

## Summary of everything we did

**1. Ligand–receptor table**
- **Annotated table:** I added short and summary descriptions to the 426 interactions in `most_significant_interacions_001.tsv` and corrected six rows by hand. Likely database artifacts are flagged in the descriptions. You moved the result to `tables/`.

**2. Figure style (both projects)**
- **One helper for every figure:** `style_figure` sets the background, text colour and font scale on any figure, including scanpy, squidpy and cell2location ones. Lineage colours and the values in heatmap cells keep their colours.
- **Style is now a class parameter:** `font_color`, `bg_color` and `fontsize_scale` sit in `params`, so plot arguments set to `None` take the class theme. Both projects are white on black, with font scale ×1.4.
- **Your size and font tuning, carried over to mouse brain 01/02:** figure and panel sizes, `fontsize_scale` 1.6–1.8, and the light-blue → white → light-red map `cmap_lbwr` for the 3B and 3D dot plots. `cmap_lbwr` is also the default of `plot_dotplot_celltype_age`.
- **Readable on black:** muted greys, halos around labels and the 3E title colour now follow the background. The black density panels keep their black axes.

**3. Parameters and file names in the class (all 8 notebooks)**
- **Registries:** `LYMPHOID_FILES` and `NEURO_FILES` now cover the 01/02 analyses and new inputs. Values that differ only for one notebook live in `LYMPHOID_ANALYSIS_PARAMS`.
- **Notebook setup:** each notebook builds its object with `SingleCell…(ROOT0, analysis=...)`, takes file names from `fname()` / `file_path()`, and lists its settings with `calc_df_params()` / `calc_df_files()`.
- **Also moved into the class:** section IDs, the cell2location cache tags and the reference keys. Gene panels and cell-type lists stay in the notebooks.

- **Folders:** every path (the `root_*` folders, the cell2location run folders, the cache folders) is created with `create_dir()` from `libs.Basic`.

**4. Caching, `force` and `verbose`**

- **Calc methods:** methods whose results are tables, dicts or other objects (h5ad, npz) take `force` and `verbose`. Tables (default `.tsv`) are written with `pdwritecsv()` and read with `pdreadcsv()`, both from `libs.Basic`. When `force=False` and the files exist, they are read and returned instead of recomputed.
- **Methods brought in line with this rule:** `calc_df_panel_celltype_age` (adds the age table), `calc_gene_by_subset_tissue` (all parts of its dict), `calc_df_astro_region_density` and `calc_df_marker_density_rho` (the notebooks no longer write these tables themselves), `posterior_samples_cached` (now has `force`), `train_reference` and `train_mapping` (now have `verbose`). A shared helper, `_cached_frames`, caches methods that return several tables; cached and computed runs return identical results.

- **Every figure method:** about 30 across the three libraries take `force` and `verbose`. An existing file is kept unless `force=True`.
- **Side-product tables:** written through `_write_table`, with the same keep-unless-`force` rule.
- **`calc_df_subset_markers`:** caches three files, read with `pdreadcsv`. The cached result is identical to a fresh one; on your data, 28.5 s to compute vs 0.13 s to read.
- **Reading and messages:** `add_paper_annotation` uses `pdreadcsv`. The "File already exists" message now prints only with `verbose`.

**5. Bug fixes**
- **`fit_or_load_reference`:** a second run lost the posterior samples, which broke `plot_c2l_qc`. Fixed.
- **Missing colour-bar labels:** fixed in `style_figure`, along with a crash on texts without path effects.
- **Missing imports:** `pdreadcsv` in the lymphoid and neuro libraries.
- **Notebook 04 (both projects):** the reference-key line ran before the object existed. Moved below the constructor.
- **Cell 35 (lymphoid 02):** an f-string that only ran on Python 3.12+.
- **Plotly map:** file writing moved into the method.

**Before you re-run**
- **Saved figures stay as they are until each cell runs with `force=True`.** That applies to the new theme and to the black glass.
- **The first black-glass map of each brain section writes one image** under `mouse_brain/data/hires_bg/`.
- **Reload the notebooks from disk in Jupyter** before saving, so the edits aren't overwritten.
- **Backups:** the libraries and notebooks are in the two `backup/` folders, timestamps `1113_1154` and `1113_1526`. Today's later edits have no separate backup.

**Not tested end to end:** I ran the parameter cells of all eight notebooks and rendered the main figures through the library calls, writing only to my workspace. I haven't run the notebooks from start to finish.