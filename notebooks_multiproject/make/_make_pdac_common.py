"""Shared text and code of the mp_pdac0* notebook generators (_make_mp_pdac01.py ... _make_mp_pdac04.py)."""
import nbformat as nbf

PAPER = """**Paper**: Chen MM, Gao Q, Ning H, Chen K, Gao Y, Yu M, Liu CQ, Zhou W, Pan J, Wei L, Dou W, Zhang D, Zhu L, Zhang Q, Chen R, Zhang Z. *Integrated single-cell and spatial transcriptomics uncover distinct cellular subtypes involved in neural invasion in pancreatic cancer.* Cancer Cell. 2025 Sep 8;43(9):1656-1676.e10. doi: [10.1016/j.ccell.2025.06.020](https://doi.org/10.1016/j.ccell.2025.06.020). PMID: 40680743.

Code: [github.com/PDAC-zhanglab/PDAC](https://github.com/PDAC-zhanglab/PDAC) (R figure scripts; copy in pdac/data/github_PDAC-zhanglab) - Data: GEO SuperSeries [GSE278694](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE278694) = [GSE278687](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE278687) (Visium, 21 sections) + [GSE278688](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE278688) (scRNA) + [GSE278689](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE278689) (FFPE snRNA) + GSE300435 (scTCR); raw reads GSA HRA008568."""

HEADER_LIBS = """#### Libs: **src/libs**

- **single_cell_lib**
  - Class **SingleCell**: generic Scanpy, Squidpy, cell2location and LIANA methods
  - **DEFAULT_PARAMS**
- **single_cell_pdac_lib** (new)
  - Class **SingleCellPDAC(SingleCell)**: PDAC methods (sections, NI status and platforms, re-derived subclusters, RCTD / cell2location abundances, regions named after the paper's niches, sections as replicates for high vs low NI, nerve spots / invasion / rings)
  - **PDAC_PARAMS**, **PDAC_SUBCLUSTERS** (marker sets of the paper's subclusters), **PDAC_GENES** (26 PDAC / NI genes), **PDAC_NI_GENES**, **PDAC_NERVE_GENES**, **PAPER_NICHES**, **PDAC_FILES** (one file registry per notebook: 'pdac01' ... 'pdac04')

**Programs: src/programs**: pdac_build_h5ad.py (GEO -> h5ad), pdac_reference_subclusters.py (subclusters), run_rctd_pdac.R (RCTD), run_c2l_pdac.py (cell2location, GPU)."""

IMPORTS = """import os, sys
from pathlib import Path
from importlib.metadata import version
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import gc

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42          # editable text in PDFs
%config InlineBackend.print_figure_kwargs = {'bbox_inches': 'tight', 'dpi': 90}   # inline display only; files are saved at 300 dpi
import plotly.io as pio
pio.renderers.default = "notebook_connected"

ROOT0 = Path('/home/flavio/uv/cell_cell_communication')
ROOT_SRC = ROOT0 / 'src'
if str(ROOT_SRC) not in sys.path:
    sys.path.append(str(ROOT_SRC))

from libs.single_cell_pdac_lib import (SingleCellPDAC, PDAC_PARAMS, PDAC_SUBCLUSTERS, PDAC_GENES, PDAC_NI_GENES,
                                       PDAC_NERVE_GENES, PAPER_NICHES, PDAC_FILES)
from libs.single_cell_lib import cmap_byr, cmap_lbwr
from libs.matplotlib_figure_style import apply_figure_style
from libs.Basic import create_dir, pdwritecsv, pdreadcsv

pd.set_option('display.max_colwidth', 120)
pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 200)"""


def md_code(cells):
    """(md, code) appenders of a list of notebook cells."""
    return (lambda s: cells.append(nbf.v4.new_markdown_cell(s)),
            lambda s: cells.append(nbf.v4.new_code_cell(s)))
