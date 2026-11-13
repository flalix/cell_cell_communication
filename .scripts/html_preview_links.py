#!/usr/bin/env python
"""Generate html-preview.github.io links for the HTML reports in docs folders.

Usage (from the repo root):
    python .scripts/html_preview_links.py notebooks_multiproject/docs
    python .scripts/html_preview_links.py notebooks_multiproject/docs --readme README.md

Without --readme the HTML block is printed (copy/paste into README.md).
With --readme the block replaces the text between
    <!-- previews:<folder>:start -->  and  <!-- previews:<folder>:end -->
(the markers are appended to the README the first time).

Files ignored by .gitignore (*.html) are flagged: push them with `git add -f`.
"""

import argparse
import re
import subprocess
from pathlib import Path

USER, REPO, BRANCH = "flalix", "cell_cell_communication", "main"
PREVIEW = "https://html-preview.github.io/?url=https://github.com/{u}/{r}/blob/{b}/{p}"


def title_from_name(stem: str) -> str:
    """mp_lymphoid02_regions -> '02 — Lymphoid: regions'."""
    m = re.match(r"^(?:mp_)?([a-z_]+?)(\d+)_(.*)$", stem)
    if not m:
        return stem.replace("_", " ")
    topic, num, rest = m.groups()
    return f"{num} — {topic.replace('_', ' ').title()}: {rest.replace('_', ' ')}"


def tracked(path: str) -> bool:
    out = subprocess.run(["git", "ls-files", "--error-unmatch", path],
                         capture_output=True, text=True)
    return out.returncode == 0


def build_block(folder: str) -> str:
    files = sorted(p for p in Path(folder).glob("*.html") if p.name != "index.html")
    lines = ["<p><em>(Ctrl+click to open in a new tab)</em></p>", "<ul>"]
    for p in files:
        rel = p.as_posix()
        url = PREVIEW.format(u=USER, r=REPO, b=BRANCH, p=rel)
        lines.append(f'  <li><a href="{url}" target="_blank" '
                     f'rel="noopener noreferrer">{title_from_name(p.stem)}</a></li>')
        if not tracked(rel):
            print(f"WARNING not in git (run: git add -f '{rel}')")
    lines.append("</ul>")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--readme")
    a = ap.parse_args()
    for folder in a.folders:
        block = build_block(folder.rstrip("/"))
        if not a.readme:
            print(block)
            continue
        readme = Path(a.readme)
        text = readme.read_text()
        start = f"<!-- previews:{folder.rstrip('/')}:start -->"
        end = f"<!-- previews:{folder.rstrip('/')}:end -->"
        new = f"{start}\n{block}\n{end}"
        if start in text and end in text:
            text = re.sub(re.escape(start) + r".*?" + re.escape(end),
                          lambda _: new, text, flags=re.S)
        else:
            text = text.rstrip() + f"\n\n## {folder} reports\n\n{new}\n"
        readme.write_text(text)
        print(f"updated {readme} for {folder}")


if __name__ == "__main__":
    main()
