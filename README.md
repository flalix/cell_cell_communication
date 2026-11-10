<!--
  NOTE: The original badges pointed at scverse/squidpy's own CI, coverage and PyPI,
  so they described squidpy, not this repository. Replace OWNER/REPO below with
  your repo (e.g. flalix/cell_cell_communication) or delete the badges you don't use.
-->
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/env-uv-6340ac)](https://docs.astral.sh/uv/)
[![License](https://img.shields.io/github/license/OWNER/REPO)](LICENSE)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

# Multi-projects framework: Cell–Cell Communication (CCC) and Spatial Single-Cell Analysis in Python

A collection of Jupyter notebooks combining **Scanpy**, **Squidpy**, **cell2location** and **LIANA** to
go from single-cell / spatial transcriptomics data to cell-type maps and ligand–receptor communication programs to support different studies, see below.

**Workflow at a glance**

```
scRNA-seq reference ──► cell2location ──► cell-type abundance per spot
                                                 │
Visium (H&E / IF) ────► Scanpy / Squidpy ──► spatial graphs, neighborhoods, image features
                                                 │
                                                 ▼
                                   LIANA ──► ligand–receptor interactions & niche programs
```

---

## Projects

| Status | Folder | Notebook | Tools | Description |
|---|---|---|---|---|
| new | `notebooks_multiproject/` | `mp_mouse_brain01_prepare.ipynb` | all | complete study, new structure |
| Old | `notebooks_ccc_liana/` | `ccc01_tutorial.ipynb` | cell2location | cell2location tutorial (reproduced) |
| Old | | `ccc02_study.ipynb` | cell2location | cell2location applied study |
| Old | | `ccc03_liana_study.ipynb` | LIANA | Cell–cell communication analysis |
| Old | `notebooks_cell2loc/` | — | cell2location | Additional cell2location tutorials and studies |
| Old | `notebooks_squidpy_mouse_brain/` | `squid01_analyze_visium_HE_data.ipynb` | Squidpy, Scanpy | Mouse brain Visium, H&E image |
| Old | | `squid02_analyze_visium_fluorescence_data.ipynb` | Squidpy, Scanpy | Mouse brain Visium, fluorescence image |
| Old | | `squid03_niche_LigRec_programs.ipynb` | LIANA, Squidpy | Spatial niches and ligand–receptor programs |

### Multi-projects

| Project/Subdir | Organism | Description |
|---|---|---|
| mouse_brain | mouse | brain study |
| lymphoid_tissue | human | secondary lymphoid tissue study |
| heart | human | embrionic heart study - Univ. Sweden |
| pancreas_cancer | human | my study --> support AI Agent project |


---

## Cell-to-Cell Communication (CCC) reports

Rendered via **html-preview**:

<p><em>(Ctrl+click to open in a new tab)</em></p>
<ul>
  <li><a href="https://html-preview.github.io/?url=https://github.com/flalix/cell_cell_communication/blob/main/notebooks_ccc_liana/docs/ccc01_tutorial.html" target="_blank" rel="noopener noreferrer">01 — Cell2location Tutorial</a></li>
  <li><a href="https://html-preview.github.io/?url=https://github.com/flalix/cell_cell_communication/blob/main/notebooks_ccc_liana/docs/ccc02_study.html" target="_blank" rel="noopener noreferrer">02 — Cell2location Study</a></li>
  <li><a href="https://html-preview.github.io/?url=https://github.com/flalix/cell_cell_communication/blob/main/notebooks_ccc_liana/docs/ccc03_liana_study.html" target="_blank" rel="noopener noreferrer">03 — CCC - LIANA study</a></li>
</ul>

Open from a Linux terminal:

```bash
xdg-open "https://html-preview.github.io/?url=https://github.com/flalix/cell_cell_communication/blob/main/notebooks_ccc_liana/docs/index.html"
```

---

## Installation

The environment is managed with [uv](https://docs.astral.sh/uv/) and targets **Python 3.12**.

```bash
git clone https://github.com/OWNER/REPO.git
cd REPO
uv sync                      # creates .venv from pyproject.toml / uv.lock
uv run jupyter lab           # or open the notebooks in VS Code with the .venv kernel
```

> cell2location depends on PyTorch / scvi-tools. For GPU training, install the CUDA build of PyTorch
> that matches your driver before (or instead of) the default CPU wheel.

## Data

Tutorial datasets (e.g. the Squidpy mouse brain Visium sets and the cell2location lymph-node reference)
are downloaded automatically by the respective `*.datasets` functions on first run. Large files are not
tracked in git — see `.gitignore`.

---

## Core libraries

| Library | Role here | Source | Docs | Video |
|---|---|---|---|---|
| [Scanpy](https://scanpy.readthedocs.io) | QC, normalization, clustering, UMAP | [GitHub](https://github.com/scverse/scanpy) | [Tutorials](https://scanpy.readthedocs.io/en/stable/tutorials/index.html) | [Playlist](https://www.youtube.com/playlist?list=PL4rcQcNPLZxWWCxOpQc5-8Yp0E9MCUHXD) |
| [AnnData](https://anndata.readthedocs.io) | Core data structure | [GitHub](https://github.com/scverse/anndata) | [Tutorials](https://anndata.scverse.org/en/stable/tutorials/index.html) | [Talk](https://www.youtube.com/watch?v=68ZUcr1rSCU) |
| [Squidpy](https://squidpy.readthedocs.io) | Spatial graphs, neighborhood enrichment, image features | [GitHub](https://github.com/scverse/squidpy) | [Docs](https://squidpy.readthedocs.io/en/stable/) | [1](https://www.youtube.com/watch?v=uGpNub2yrv0), [2](https://www.youtube.com/watch?v=btqPgui8gpg) |
| [cell2location](https://cell2location.readthedocs.io) | Spatial deconvolution with a scRNA-seq reference | [GitHub](https://github.com/BayraktarLab/cell2location) | [Tutorial](https://cell2location.readthedocs.io/en/latest/notebooks/cell2location_tutorial.html) | [Talk](https://www.youtube.com/watch?v=_v3m5-O8ouA) |
| [LIANA+](https://liana-py.readthedocs.io) (Saez Lab) | Ligand–receptor inference, spatial CCC | [GitHub](https://github.com/saezlab/liana-py) | [Docs](https://liana-py.readthedocs.io) | [Talk](https://www.youtube.com/watch?v=5BHZbZE9kEc) |

> The Python package is **liana-py** (`import liana as li`). The original
> [saezlab/liana](https://github.com/saezlab/liana) repo is the R version.

---

## Citation

If you use these notebooks, please cite the underlying methods:

- **Scanpy** — Wolf, Angerer & Theis. *Genome Biology* 19, 15 (2018).
- **AnnData** — Virshup et al. *Journal of Open Source Software* 9(101), 4371 (2024).
- **Squidpy** — Palla et al. *Nature Methods* 19, 171–178 (2022).
- **cell2location** — Kleshchevnikov et al. *Nature Biotechnology* 40, 661–671 (2022).
- **LIANA** — Dimitrov et al. *Nature Communications* 13, 3224 (2022);
  **LIANA+** — Dimitrov et al. *Nature Cell Biology* 26, 1613–1622 (2024).

## Acknowledgements

Scanpy, AnnData and Squidpy are part of [scverse®](https://scverse.org/), a project fiscally sponsored by
[NumFOCUS](https://numfocus.org/). Several notebooks are adapted from the official tutorials of the libraries above.

## Author

Flavio Lichtenstein, PhD — bioinformatics & systems biology.
