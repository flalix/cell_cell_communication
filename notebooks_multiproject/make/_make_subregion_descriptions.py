"""Writes mouse_brain/data/mouse_brain_subregion_descriptions.tsv: curated functional descriptions of the subregions of
mp_mouse_brain03 (default parameters), with the subregion's top-3 signature genes at curation time (signature_check).
Descriptions are interpretations of signature genes, enriched cell types, bordering region and QC flags."""
import sys
from pathlib import Path
ROOT0 = Path("/home/flavio/uv/cell_cell_communication")
sys.path.insert(0, str(ROOT0 / "src"))
import pandas as pd
from libs.single_cell_neuro_lib import SingleCellNeuro
from libs.Basic import pdwritecsv

D = {
 "AMY_1": "General amygdala parenchyma with few specific genes (haemoglobin = blood vessels) and lower UMI depth, bordering piriform cortex; not a distinct nucleus.",
 "AMY_2": "Amygdala portion with Nefl, S100a10, Rasgrp1 and a CA3-like excitatory signal: probably the posterior basolateral amygdala / amygdalo-hippocampal transition.",
 "AMY_3": "Amygdala spots with deep-layer cortical neurons (Etv1, Dkkl1; Ext_L5/L6B): cortex-amygdala transition next to layers 5-6.",
 "AMY_4": "Ext_Amy_1-dominated nucleus (Vgll3, Gpr101, Hopx) lacking Neurod6, present only in section ST8059051 (mouse 2): likely a posterior/medial amygdala nucleus not cut in the other section.",
 "AMY_5": "Striatum-like amygdala (Gpr88, Penk, Meis2+ inhibitory neurons): central amygdala / amygdalostriatal transition area.",
 "CTX_L1_1": "No distinct expression signature; layer 1 adjacent to L2/3. CTX_L1 is best treated as one region.",
 "CTX_L1_2": "No distinct signature; pial side of layer 1 (endothelial, LowQ_1 and astrocyte signal: meninges / vessels).",
 "CTX_L1_3": "No distinct signature; layer 1 spots with Vip interneuron signal.",
 "CTX_L1_4": "No distinct signature; layer 1 spots with microglia / progenitor signal.",
 "CTX_L2/3_1": "Lower layer 3 next to L4: Pvalb interneurons and neurofilament / fast-conduction genes (Nefm, Scn4b, Kcnab3).",
 "CTX_L2/3_2": "No distinct signature; general layer 2/3.",
 "CTX_L2/3_3": "Small signature (Pde1a, Dkk3) with a deep-layer Ext_L6 signal: probably lateral cortex where the layers are compressed.",
 "CTX_L2/3_4": "Weak: upper layer 2 (Cxcl14) with Vip interneurons.",
 "CTX_L2/3_5": "Ventrolateral layer 2/3 adjacent to piriform cortex and amygdala (Ext_Pir signal; Wfs1, Otof): lateral entorhinal / piriform transition.",
 "CTX_L4_1": "Weak: layer 4 on the layer-5 side.",
 "CTX_L4_2": "Weak: layer 4 spots rich in Pvalb interneurons (Kcnc1, Gad1).",
 "CTX_L4_3": "No distinct signature; layer 4 on the L2/3 side.",
 "CTX_L4_4": "No distinct signature; small group with oligodendrocyte signal.",
 "CTX_L5_1": "Myelin-rich layer 5 (Mbp, Plp1, Nefh) with Tshz2: deep L5 with thick myelinated bundles, probably medial (retrosplenial-like) cortex.",
 "CTX_L5_2": "Upper layer 5 (L5a) next to L4 with intratelencephalic neuron genes (Car4, C1qtnf4, Kcnip2).",
 "CTX_L5_3": "Lower layer 5 (L5b) with pyramidal-tract neuron markers (Bcl11b/Ctip2, Crym, Pcp4): the corticofugal output layer.",
 "CTX_L5_4": "Ventrolateral layer 5 adjacent to the amygdala (Ext_Amy_2 signal): cortex-amygdala / endopiriform transition.",
 "CTX_L5/6_1": "Deep layer 6 against the white matter, with myelin genes: infracortical fibre zone.",
 "CTX_L5/6_2": "Weak: core layer 5/6 with IT neurons (Foxp1).",
 "CTX_L5/6_3": "Claustrum / dorsal endopiriform nucleus: claustral markers Synpr, Lxn, Nmb with Ext_ClauPyr neurons.",
 "CTX_L5/6_4": "Interneuron-dense spots (Pvalb, Sst; Gad1, Syt2) scattered through deep layers; not a contiguous subregion.",
 "CTX_L6b_1": "Lateral layer 6b / subplate adjoining the basolateral amygdala (Ext_Amy_2, Ext_L5_3 signal).",
 "CTX_L6b_2": "Layer 6b on the white-matter edge (myelin genes, oligodendrocytes).",
 "CTX_L6b_3": "Small group without signature; striatal Meis2 signal suggests the striatal border.",
 "HAB_1": "Not habenula: neuropil spots between white matter and striatum (Meis2 inhibitory signal); part of the weakly supported HAB clusters.",
 "HAB_2": "Diencephalic GABAergic area (Gad2; Inh_2, Inh_3) bordering the hypothalamus: probably zona incerta / thalamic reticular area rather than habenula.",
 "HAB_3": "Habenula proper: habenular markers Gpr151, Gng8, Necab2 with Inh_6 habenular neurons (x283). Signature size partly inflated by higher UMI depth.",
 "HAB_4": "Low-depth edge spots (ribosomal and glial genes up, 500+ neuronal genes missing): quality artefact, not anatomy.",
 "HAB_5": "Pvalb+ Lhx8+ GABAergic spots next to white matter: globus pallidus / reticular nucleus.",
 "HAB_6": "Pitx2+ Slc17a6+ glutamatergic neurons (the Inh_5 type): subthalamic nucleus / posterior hypothalamus.",
 "HPC_CA1_1": "CA1 pyramidal layer; too small (68 spots, section ST8059048 only) to subdivide.",
 "HPC_CA3_1": "Astrocyte- and neuropil-rich spots around CA3 (Apoe, Aldoc, Glul) bordering the fimbria / alveus white matter.",
 "HPC_CA3_2": "High-UMI CA3 pyramidal spots; signature mostly general neuronal genes (depth effect).",
 "HPC_CA3_3": "CA2: CA2 markers Pcp4 and Necab2 with Ext_Hpc_CA2 neurons.",
 "HPC_CA3_4": "CA3 next to the dentate hilus (CA4): mossy-fibre zinc transporter Slc30a3, Synpr.",
 "HPC_DG_1": "Dentate molecular layer / hilus neuropil with hippocampal astrocytes (Mt3, Aldh1a1, Id3).",
 "HPC_DG_2": "Dentate granule cell layer with the neurogenic niche (Nb_1 neuroblasts); the very large signature reflects 5x UMI depth.",
 "HPC_DG_3": "Low-depth spots far from the hippocampus (border with piriform cortex), section ST8059051: misassigned edge spots, artefact.",
 "HPC_DG_4": "Low-depth spots at the ventral surface with pituitary transcripts (Gh, Prl; contamination), section ST8059048: artefact.",
 "HYPO_1": "Neuron-dense hypothalamic nuclei: Inh_3 / Inh_4 GABAergic and peptidergic neurons (Syt4, Hrh3).",
 "HYPO_2": "Low-depth ventral hypothalamus at the brain surface with blood (Hbb) and pituitary (Gh, Prl) transcripts: ventral surface / median eminence side, partly artefact.",
 "HYPO_3": "Fibre-rich hypothalamus (myelin genes): fornix / medial forebrain bundle area of the lateral hypothalamus.",
 "HYPO_4": "Mammillary region: Foxb1, Lhx1os, Calb1 with Ext_Unk_2 neurons; only in section ST8059051 (mouse 2, more posterior level).",
 "PIR_1": "Low-depth piriform spots with few specific genes (surface side).",
 "PIR_2": "Piriform-cortex transition with L2/3-like neurons (Cck, Lamp5, Egr3): insular / lateral entorhinal side.",
 "PIR_3": "Cell-dense piriform layer 2 (Penk, Chgb); signature size inflated by 3x UMI depth.",
 "PIR_4": "Weak: astrocyte-rich piriform surface (layer 1).",
 "PIR_5": "Ventromedial piriform edge with Inh_1 / Ext_Amy_1 neurons (Hap1, Moxd1, Gpr101): cortical / medial amygdala or olfactory tubercle side.",
 "STR_1": "Dorsal striatum (caudoputamen) with D1 and D2 medium spiny neurons.",
 "STR_2": "Ventral striatum next to the amygdala (Meis2_1, Ext_Amy_2 signal): nucleus accumbens / amygdalostriatal area.",
 "STR_3": "Striatum crossed by internal-capsule bundles (myelin genes) and the ventricle wall (Enpp2).",
 "STR_4": "Lateral hypothalamus, not striatum: Pmch (MCH) and Hcrt (orexin) neurons; section ST8059048 only (region label error inherited from cluster 8).",
 "STR_5": "Globus pallidus: fast-spiking Pvalb+ Kcnc1+ pallidal neurons (Inh_Meis2_4) next to the internal capsule.",
 "STR_6": "Bed nucleus of the stria terminalis: Prkcd (oval nucleus) and Tac2, Sst.",
 "THAL_1": "Lateral thalamus with reticular-nucleus Pvalb signal and fibre tracts: ventrolateral nuclei next to the reticular nucleus and internal capsule.",
 "THAL_2": "Medial / midline thalamic nuclei: Calb1+, Ext_Thal_2 neurons, medial-thalamus astrocytes.",
 "THAL_3": "Dorsal thalamic relay nuclei (Ext_Thal_1; Grin2b, Necab3).",
 "THAL_4": "Thalamic relay nuclei with Tenm2, Epha8, Kirrel3: probably posterior / sensory relay nuclei (tentative).",
 "THAL_5": "Ventral midline thalamus bordering the hypothalamus (Calb1, Cbln1; Ext_Unk_3): reuniens / zona incerta transition.",
 "THAL_6": "Thalamus next to the habenula and hypothalamus with Inh_4 and Inh_6 neurons: paraventricular / habenular transition.",
 "WM_1": "Core white matter; no signature beyond the region.",
 "WM_2": "Diencephalic white matter with diencephalic astrocytes (Slc6a11) and Inh_2 / Inh_4 neurons: fibre tracts within thalamus / hypothalamus.",
 "WM_3": "Corpus callosum-like white matter with myelinating oligodendrocytes and choroid traces (Ttr).",
 "WM_4": "White matter against cortex layer 6b; cortical neuron genes from partial volume.",
 "WM_5": "Ventricle wall: ependymal cilia genes (Rsph1, Dynlrb2), choroid plexus (Ttr) and subventricular neurogenic cells (Nb_1/Nb_2, Ccnd2).",
 "WM_6": "White-matter fascicles inside the striatum; striatal genes (Penk, Ppp1r1b, Pde10a) from partial volume.",
}

scn = SingleCellNeuro(ROOT0, "mouse_brain", params={"region_label_override": {"11": "CTX_L1", "21": "edge_LowQ"}})
m = scn.load_paper_map("c2l_paper_map/sp_with_clusters.h5ad")
df_regions = scn.calc_df_regions(m); scn.add_region_labels(df_regions, m)
ae = scn.prepare_expression(m)
subs, _ = scn.run_subregions(ae)
_, region_sig = scn.calc_df_region_signature(ae)
df_gene_sets = scn.calc_df_subregion_gene_sets(ae, region_sig)
rows = []
for (reg, s), d in df_gene_sets.groupby(["region_id", "sub_region_id"], sort=False):
    sig = d[d.gene_set == "subregion_signature"].sort_values("scores", ascending=False)
    rows.append(dict(region_id=reg, sub_region_id=s, signature_check=", ".join(scn._display_genes(sig.gene, 3)),
                     functional_description=D.get(s, "")))
df_desc = pd.DataFrame(rows)
missing = df_desc[df_desc.functional_description == ""].sub_region_id.tolist()
extra = sorted(set(D) - set(df_desc.sub_region_id))
assert not missing and not extra, (missing, extra)
pdwritecsv(df_desc, "mouse_brain_subregion_descriptions.tsv", scn.root_data, verbose=True)
