"""Helpers shared by _make_mccc04.py / _make_mccc05.py: copy markdown cells of the ccc01/ccc02 notebooks."""
import json
from pathlib import Path

OLD = Path(__file__).resolve().parent.parent / "notebooks_ccc_lymphoid_liana"

REPL = [
    ("libs/cell2location_lib.py", "libs/single_cell_lymphoid_lib.py"),
    ("libs.cell2location_lib", "libs.single_cell_lymphoid_lib"),
    ("`c2l.donor_consistency`", "`scl.donor_consistency`"),
    ("(called above as `c2l.plot_genes_per_cell_type`)", "(method `scl.plot_genes_per_cell_type` of SingleCell)"),
]


def old_md(nb: str):
    cells = json.load(open(OLD / f"{nb}.ipynb"))["cells"]

    def get(i: int, cut: str = None, drop_first_line: bool = False, extra=()):
        c = cells[i]
        assert c["cell_type"] == "markdown", (nb, i, c["cell_type"])
        s = "".join(c["source"])
        if cut and cut in s:
            s = s[:s.index(cut)].rstrip()
        if drop_first_line:
            s = s.split("\n", 1)[1].lstrip()
        for a, b in list(REPL) + list(extra):
            s = s.replace(a, b)
        return s
    return get
