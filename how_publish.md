# How to publish the notebooks on GitHub Pages

Site: **https://flalix.github.io/cell_cell_communication/**
Page address: `https://flalix.github.io/cell_cell_communication/notebooks_multiproject/docs/<notebook>.html`

The pages are the notebooks exported to HTML (`notebooks_multiproject/docs/*.html`). A GitHub Actions
workflow (`.github/workflows/pages.yaml`) publishes **only these HTML files**: the whole repo (~1.7 GB)
is over the 1 GB GitHub Pages limit, so "Deploy from a branch" cannot be used.

---

## A. Publish or update a notebook (every time)

1. **Run the notebook** in Jupyter: *Kernel -> Restart & Run All*.
   - The restart loads the current libraries (Plotly figures as SVG, with the Set/Reset buttons).
   - `force=False` is enough: figures are always rebuilt in the notebook; `force` only decides whether
     the standalone files in `<project>/figures/` and the cached tables are rewritten.
2. **Save the notebook** (Ctrl+S). nbconvert reads the file on disk, not the running kernel.
3. **Export to HTML into `docs/`:**
   ```bash
   cd ~/uv/cell_cell_communication
   jupyter nbconvert --to html --output-dir notebooks_multiproject/docs notebooks_multiproject/<notebook>.ipynb
   ls -lh notebooks_multiproject/docs/<notebook>.html      # must be < 100 MB (GitHub file limit)
   ```
   - No kernel restart is needed for the export itself.
4. **Commit and push:**
   ```bash
   git add -f notebooks_multiproject/docs/<notebook>.html   # -f: .gitignore has *.html (needed for a NEW page)
   git commit -m "docs: <notebook>"
   git push
   ```
5. **Wait 2-5 minutes.** The push changes `notebooks_multiproject/docs/`, so the workflow
   *Publish notebook pages* runs by itself (repo -> **Actions** tab: yellow dot -> green check).
6. **Open the page** (Ctrl+Shift+R if the browser shows an old version or a cached 404).
7. **New notebook:** add its link to `README.md`:
   ```markdown
   - [05 — Title](https://flalix.github.io/cell_cell_communication/notebooks_multiproject/docs/<notebook>.html)
   ```
   (no `blob/main/` in the address - that is only for github.com).

---

## B. One-time setup (done on 22 Nov 2026 - only for a new repo)

1. The repo must be **public** (GitHub Pages is free for public repos; ignore "Start free for 30 days").
2. Add the workflow `.github/workflows/pages.yaml` (below), commit and push.
3. On github.com: repo -> **Settings** -> **Pages** -> *Build and deployment* -> **Source = GitHub Actions**
   (not "Deploy from a branch"; no branch/folder to set). Not the "Quickstart for GitHub Pages" article.
4. First run: push a change to `notebooks_multiproject/docs/` or to `pages.yaml`, or
   Actions -> *Publish notebook pages* -> **Run workflow**.
5. `.nojekyll` at the repo root is harmless but not needed with the Actions workflow.

```yaml
name: Publish notebook pages

on:
  push:
    branches: [main]
    paths: ["notebooks_multiproject/docs/**", ".github/workflows/pages.yaml"]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4
        with:
          sparse-checkout: notebooks_multiproject/docs
      - name: Collect the HTML pages
        run: |
          mkdir -p _site/notebooks_multiproject/docs
          cp notebooks_multiproject/docs/*.html _site/notebooks_multiproject/docs/
          du -sh _site
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: _site
      - id: deployment
        uses: actions/deploy-pages@v4
```

Only `*.html` of `docs/` is published (the `.json` / `.txt` files there are not).

---

## C. Problems seen and their causes

| Symptom | Cause | Fix |
|---|---|---|
| 404 "There isn't a GitHub Pages site here" | Pages not enabled | Settings -> Pages -> Source = GitHub Actions |
| 404 for one page only | the HTML was never exported / committed | steps A.3-A.4 (`git add -f` for a new page) |
| `git commit`: "nothing to commit" | file already committed, or new `*.html` ignored | `git add -f ...html`; `git log -1 -- <file>` |
| Workflow did not run after a push | the push did not touch `docs/` or `pages.yaml` | Actions -> Run workflow, or re-export a page |
| GitHub page shows plain buttons, no "Run workflow" | browser did not load GitHub's CSS/JS (blocker) | Ctrl+Shift+R, private window, or push a change |
| Plotly figures blank in **html-preview.github.io** | html-preview never loads external scripts (plotly.js from cdn.plot.ly) | use the GitHub Pages address, not html-preview |
| "WebGL is not supported" / blank figures in long pages | Scattergl: a page keeps only ~16 WebGL contexts | libraries default to `webgl=False` (SVG); re-run the cell |
| Empty cells in the page | notebook saved without outputs | Restart & Run All, save, export again |
| Page has old figures | exported before re-running | re-run, save, export, push |

## D. Limits (GitHub)

- Published site <= 1 GB (now ~0.3 GB for 16 pages); one file <= 100 MB (largest page: pdac01, 58.6 MB).
- Soft limits: 100 GB bandwidth / month, 10 builds / hour.
