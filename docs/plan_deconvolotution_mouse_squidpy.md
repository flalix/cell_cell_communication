Found the free preprint already in your folder: "Comprehensive mapping of tissue cell architecture via integrated single cell and spatial transcriptomics - 2020.pdf." I'll now extract the methods details relevant to our run.

The preprint changes one setting in my plan. Next I'm reading the passage on the Smart-seq reference, because `sc_mouse_cortex` is probably a Smart-seq dataset too.

I found the preprint in your folder: `Comprehensive mapping of tissue cell architecture via integrated single cell and spatial transcriptomics - 2020.pdf`. Its methods section changed three settings in the plan. You've approved the revised plan, but I can't start until the reference data is downloaded.

**What the preprint changed:**

1. **Cells per spot.** For their mouse-brain Visium data, the authors set *N* = 8 cells per location, estimated from the histology image, and their nuclear segmentation counted about 10 nuclei per spot. I had proposed 30, but that came from the lymph-node analysis (*N* = 30 there). The plan now uses 8.
2. **How the cell-type signatures are built.** For a Smart-seq reference, the paper says the negative-binomial regression model "is not suitable for SmartSeq data". Instead, the authors derived signatures as per-cluster average expression. `sc_mouse_cortex` is expected to be Smart-seq, so I'll check the protocol in step 1:
   - If it is Smart-seq, I'll compute per-cluster averages.
   - If it is 10x droplet data, I'll use the usual reference model (`RegressionModel`).
3. **Sensitivity prior.** For a Smart-seq reference they set the technology-sensitivity prior to μ = 1/20. Smart-seq 4 captures about 20 times more mRNA per cell than 10x Visium does per cell (counts per spot divided by *N* = 8). In current cell2location this prior is the `detection_mean` argument.

The figures used to check their own mouse-brain Visium mapping against Allen Smart-seq cortex and hippocampus data (Fig. 2I–J and Fig. S9 in the preprint) are the closest parallel to our setup.

**Still needed from you:** `sc_mouse_cortex.h5ad` is not yet in `/home/flavio/uv/squidpy/data`. Please run:
```python
adata_sc = sq.datasets.sc_mouse_cortex()
adata_sc.write_h5ad(root_data / "sc_mouse_cortex.h5ad")
```
Reply once it's saved and I'll start at step 1.

Your studies folder also has COMMOT (2023), the scvi-tools paper and several cell–cell communication reviews. COMMOT is a spatial communication method that would make a useful comparison with the LIANA results.
