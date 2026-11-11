"""Writes lymphoid_tissue/data/lymph_node_regions.tsv (curated region table) and
lymphoid_tissue/data/lymph_node_subregion_descriptions.tsv (curated functional descriptions of the subregions of
mp_lymphoid03, default parameters), with each subregion's top-3 signature genes at curation time (signature_check).
Descriptions are interpretations of: subregion signature genes, cell2location enrichment vs the rest of the region,
the reference cell types that express the signature (calc_df_subregion_ref_score), coherence and depth.
Anatomical positions (medulla, sinus, trabecula ...) are inferred from genes, not from the H&E, and are marked
'probably' where the genes alone do not decide."""
import sys
from pathlib import Path
ROOT0 = Path("/home/flavio/uv/cell_cell_communication")
sys.path.insert(0, str(ROOT0 / "src"))
import pandas as pd
from libs.single_cell_lymphoid_lib import SingleCellLymphoid
from libs.Basic import pdwritecsv, pdreadcsv

# ------------------------------------------------------------------------------------------------ regions
# region_id: region_cluster of mp_lymphoid01 (Leiden on q05 abundances); c2l_name = mp_lymphoid01 name (top-3 cell types)
REG = [
 ("R00", "Plasma-cell cords with sinus macrophages", "medulla / plasma cells",
  "IgG plasma cells (IGHG1-4, IGKC, XBP1) with CD169+ / DC-SIGN+ macrophages (SIGLEC1, CD209): probably medullary cords."),
 ("R01", "B-rich parenchyma with sinus macrophages", "follicle-medulla transition",
  "Few specific genes (MARCO, STC2); heterogeneous: its subregions are sinus lining, activated vessels and FDC network."),
 ("R02", "Primary follicle / follicle mantle", "B follicle",
  "Naive and memory B cells on the FDC network (CXCL13, FCER2/CD23, FDCSP, CR1, BANK1)."),
 ("R03", "Perifollicular stroma and vessels", "interfollicular / sinus stroma",
  "Activated endothelium and stromal cells (SELE, CCL20, CHI3L1, MMRN1, THBS2, NTS) around memory-B areas."),
 ("R04", "Mixed plasma-cell / activated-B zone", "interfollicular",
  "Almost no expression signature (IGHG2 only); defined by cell2location composition (plasma cells, NK, ILC)."),
 ("R05", "T-cell zone (paracortex)", "T zone",
  "Naive CD4 / CD8 T cells and Treg with the T-zone chemokine CCL19 (TRAC, CD3D/E, IL7R, TCF7, CCR7, LEF1)."),
 ("R06", "Germinal centre, light zone", "germinal centre",
  "GC B cells on the FDC network (CR2, MEF2B, LRMP, RGS13, CLU, SERPINE2) with proliferation genes."),
 ("R07", "Memory-B follicle edge", "B follicle edge",
  "Memory B cells with FCRL4 and FDC genes (CXCL13, FDCSP) and cDC1 / CD4 T cells: follicle periphery."),
 ("R08", "Follicle mantle next to germinal centres", "B follicle",
  "Naive / activated B with CD23 (FCER2) and CXCL13; GC light-zone cells enriched."),
 ("R09", "T-B border", "interfollicular",
  "No expression signature vs the rest of the slide; naive T, memory B and cDC1 mixed."),
 ("R10", "Interferon-active zone (ISG hotspot rim)", "ISG / interferon",
  "Interferon-stimulated genes (ISG15, IFI6, IFIT1/3, RSAD2, CXCL10) with B_IFN and pre-GC B: rim of the ISG hotspot of mp_lymphoid02."),
 ("R11", "Germinal centre, dark zone / cycling", "germinal centre",
  "Cycling GC B cells (HMGB2, STMN1, TUBB, H2AFZ) with GC (MEF2B, RGS13) and FDC (CR2) genes."),
 ("R12", "T-cell zone periphery", "T zone",
  "Naive T (LEF1, MAL, TXK) with DC chemokines CXCL9 and CCL17: outer paracortex."),
 ("R13", "T-B border with Treg and CCR7+ DC", "interfollicular",
  "No expression signature vs the rest of the slide; Treg, DC_CCR7+ and T follicular regulatory cells enriched."),
 ("R14", "Sinus / vascular stroma", "interfollicular / sinus stroma",
  "Endothelium and activated stroma (CLDN5, SERPINE1, CCL20, CHI3L1, CXCL2, CCDC80) with monocytes and cDC2."),
 ("R15", "Vessels with smooth muscle and plasma cells", "vessels / medulla",
  "Smooth muscle and vessel wall (MYL9, ACTA2, MGP, ADIRF, C11orf96, SPARCL1) with IgG plasma cells and macrophages."),
 ("R16", "T-cell zone core", "T zone",
  "Strongest naive T signature on the slide (TRAC, TRBC1/2, CCR7, CCL19, IL7R, TCF7)."),
 ("R17", "ISG hotspot core", "ISG / interferon",
  "B_IFN (x16.6) with the full ISG programme (MX1, IFI44L, ISG15, OAS1-3, STAT1, IRF7): the hotspot of mp_lymphoid02."),
 ("R18", "ISG-high T-cell zone", "ISG / interferon",
  "ISGs together with T-zone genes (CCR7, CCL19): the part of the ISG hotspot that lies in the paracortex."),
 ("R19", "Low-quality strip (technical)", "low quality",
  "Median 945 UMIs and 3 cells per spot (other regions 15,000-24,000 UMIs); 31 of 41 spots in a strip at array rows "
  "62-65: fold / tear / permeabilisation artefact. Not subclustered."),
]

# ------------------------------------------------------------------------------------------------ subregions
D = {
 "R00_1": "No distinct signature (PDK1 only): general plasma-cell cord parenchyma of the region.",
 "R00_2": "No distinct signature; lower UMI depth (0.77), part of the region with GC / FDC signal.",
 "R00_3": "Weak (scattered, coherence 1.36) but large signature: sinus stroma with macrophages (MMP9, MARCO, STAB1, TNC, CLDN5, SERPINE1): probably medullary sinuses running along R01.",
 "R00_4": "No distinct signature; plasma-cell / M2-macrophage parenchyma of the region.",
 "R00_5": "Small vessel wall: smooth-muscle genes ACTA2, TAGLN, MYL9, C11orf96 (VSMC on the reference); very compact patch (coherence 7.0).",
 "R01_1": "No distinct signature; plasma-cell-rich part of the region.",
 "R01_2": "Sinus lining: sinus endothelium (CLEC4G, CLEC4M, LYVE1, STAB1) with CD169+ / DC-SIGN+ sinus macrophages (SIGLEC1, CD209, C1QA-C) and CCL14.",
 "R01_3": "No distinct signature; GC-adjacent part of the region.",
 "R01_4": "Activated venules and stromal cells: ACKR1, ACKR3, CLDN5, SERPINE1, CCL20, THY1, TIMP1 (Endo / VSMC on the reference).",
 "R01_5": "FDC network (FDCSP, CXCL13; FDC on the reference): the follicle part of the region, next to R02.",
 "R02_1": "No distinct signature; follicle core with naive B cells.",
 "R02_2": "Weak: follicle spots with cycling / GC genes (CKS1B, MEF2B): GC edge inside the follicle.",
 "R02_3": "Vessels and fibroblasts around the follicle (IGFBP7, EGFL7, SERPINE1, CCL20, THY1, C1R; VSMC / Endo on the reference), with monocytes.",
 "R02_4": "Weak: follicle spots on the T-zone side (AQP1; naive T and Treg enriched): follicle-T zone border.",
 "R02_5": "IgG plasma cells (IGHG1-3, IGHGP) with M2 macrophages inside or at the edge of the follicle.",
 "R03_1": "Larger vessels: smooth muscle and pericytes (MYH11, ACTA2, MYL9, RGS5), endothelium (VWF, COL4A1) with IgG plasma cells.",
 "R03_2": "Activated stroma with dendritic cells: fibroblast / vessel genes (SERPINE1, TNC, FN1, LUM, CLDN5) and DC / myeloid genes (LYZ, ITGAX); DC_CCR7+ and cDC2 enriched.",
 "R03_3": "No distinct signature; lower UMI depth (0.76).",
 "R03_4": "No distinct signature; GC-adjacent spots (B_GC_LZ, Treg, TfH enriched).",
 "R03_5": "Compact sinus segment (coherence 4.9) with CD169+ / LYVE1+ macrophages and pDC: SIGLEC1, CD209, LYVE1, CCL14, CXCL12, GZMB.",
 "R04_1": "Plasma cells (SDC1/CD138, PRDX4) with LYVE1+ macrophages: plasma-cell niche along sinuses.",
 "R04_2": "Dendritic-cell area next to FDC / GC TfH: LY75 (DEC-205), PLA2G2D, CHN1; the reference places the signature in macrophages.",
 "R04_3": "Weak (scattered, coherence 1.26) but clear endothelial signature (ACKR1, VWF, SPARCL1, HSPG2, A2M, ENG): scattered ACKR1+ venules, probably high endothelial venules.",
 "R04_4": "No distinct signature; lower UMI depth (0.85).",
 "R04_5": "IgG / IgA plasma cells (IGHG3, IGHG4, IGHA1, IGLC2, IGLC3).",
 "R05_1": "No distinct signature; T-zone core.",
 "R05_2": "No distinct signature; T-zone spots with cycling-B / FDC signal (paracortex next to follicles).",
 "R05_3": "Weak (coherence 1.43): T-zone vessels and DC (VWF, EGFL7, A2M, SPARCL1, TM4SF1, IL1B, CPVL): scattered paracortical venules, probably high endothelial venules.",
 "R06_1": "GC dark zone in S phase: DNA-replication genes (MCM2, MCM3, CDT1, CDC6, UBE2S) with B_GC_DZ / B_Cycling.",
 "R06_2": "Weak: GC spots with T / myeloid genes (TRAF3IP3, MYO1F, IRF5, NKG7) and Treg: GC border.",
 "R06_3": "GC-associated macrophages (tingible-body-like) and stroma: CTSL, PLA2G2D, ENPP2, C1QTNF1; Macrophages_M1 on the reference.",
 "R06_4": "No distinct signature; GC edge toward the mantle (B_mem, B_preGC), lower depth (0.79).",
 "R07_1": "Activated stroma and vessels with monocytes and ILC: IFITM3, TIMP1, SOD2, CCL20, SERPINE1, TNC, CLDN5, ITGAX.",
 "R07_2": "No distinct signature; follicle mantle with naive / memory B.",
 "R07_3": "Weak: plasma cells and fibroblasts (IGHGP, COL1A2).",
 "R07_4": "Weak: GC spots (HMGB2, AURKB; B_GC_LZ x4.2) inside the memory-B region.",
 "R08_1": "FDC-rich follicle centre / GC light zone: FDCSP, CR2, CLU, SERPINE2, FCAMR, LMO2 (FDC on the reference).",
 "R08_2": "Perivascular stroma with macrophages (CCL2, CXCL12, TGM2, TIMP3) and IgG plasma cells.",
 "R08_3": "No distinct signature; follicle spots on the T-zone side (naive T x2.6).",
 "R08_4": "No distinct signature; small (20 spots).",
 "R09_1": "Plasma cells (IGLC2, IGKC, IGHGP), lower depth (0.80).",
 "R09_2": "Sinus / vascular endothelium at the T-B border: CLDN5, VWF, ENG, EGFL7, CLEC4G, CD209, CCL14.",
 "R09_3": "Proliferating cells (UBE2C, CDK1) with GC LZ / TfH_GC: GC edge at the T-B border.",
 "R10_1": "FDC network (FDCSP, CXCL13, SERPINE2): follicle part of the interferon zone.",
 "R10_2": "Endothelium / sinus (PECAM1, CCL14, CXCL12, SPARCL1) with plasma cells (TXNDC5, IGHG2).",
 "R10_3": "Proliferating cells (RRM2, CCNB2, CDC25A, DHFR) next to the ISG-high T zone (R18).",
 "R10_4": "Myeloid-stromal niche of the interferon zone: LYZ, IFI30, ITGAX, CHI3L1, FUCA1 with fibroblast genes (TNC, COL3A1).",
 "R11_1": "Mantle side of the cycling GC: CXCL13, FCER2 (CD23), BANK1, FCMR (naive / memory B) with complement and fibroblast genes (C3, C7, COL1A1); flagged border with R06.",
 "R11_2": "GC dark zone: cycling GC B cells (MCM3, TFDP1; B_GC_DZ x2.4).",
 "R11_3": "No distinct signature; T / macrophage spots at the GC edge.",
 "R12_1": "Weak, scattered (coherence 0.95): plasma-cell spots (IGHG3) inside the T zone.",
 "R12_2": "Weak (coherence 1.44): T-zone venules (ACKR1, VWF, GPX3, RAMP3, A2M, CCL14, COL15A1), probably high endothelial venules.",
 "R12_3": "No distinct signature; T-zone spots next to follicles (B_Cycling, FDC).",
 "R12_4": "No distinct signature, scattered (coherence 0.85).",
 "R13_1": "FDC / GC TfH spots (FDCSP, CXCL13): follicle edge at the T-B border.",
 "R13_2": "Weak: endothelium (CCL14, COL15A1) with cDC2 and monocytes.",
 "R13_3": "IgG plasma cells (IGHG1-4, IGKC, IGLC2).",
 "R14_1": "No distinct signature; lower UMI depth (0.66).",
 "R14_2": "IgG plasma cells with smooth-muscle signal (MYL9): perivascular plasma cells.",
 "R14_3": "No distinct signature; small (28 spots) with DC_CCR7+ / B_IFN.",
 "R14_4": "Sinus macrophages and endothelium (MARCO, STAB1, LYZ, CTSD, ENPP2, VIM): probably medullary sinus.",
 "R14_5": "Weak, scattered (coherence 1.09).",
 "R15_1": "Vessel wall / fibrous stroma: smooth muscle (MYH11, TAGLN), collagens (COL1A1, COL1A2, COL3A1, COL6A2, CCN2) with IgG plasma cells: probably trabecular or hilar vessels.",
 "R15_2": "Macrophage / DC niche: MMP9, CD14, CTSC, CFP, TNFAIP2, ENPP2; DC_CCR7+ x2.6.",
 "R15_3": "Proliferating cells (TYMS, AQP3, CYCS) with B_Cycling / T_TIM3+.",
 "R16_1": "T-zone core; one cluster at every resolution: no subregions.",
 "R17_1": "Weak: ISG hotspot core without its own signature; B_mem enriched.",
 "R17_2": "FDC / GC part of the ISG hotspot (CR2, FDCSP, CLU, SERPINE2): the hotspot overlaps a follicle.",
 "R18_1": "ISG-high T zone; one cluster (43 spots).",
}

scl = SingleCellLymphoid(ROOT0, "lymphoid_tissue", params={"subregion_skip": ("R19",)})
run_name = scl.root_result / "lymph_nodes_analysis" / "cell2location_map"
adata_vis = scl.read_h5ad_slots("sp.h5ad", root=run_name, skip=("layers",))
df_regions = pdreadcsv("lymph01_region_labels.tsv", scl.root_result).set_index("barcode")
region_names = scl.add_region_ids(df_regions, adata_vis)
ae = scl.prepare_expression(adata_vis)
_, region_sig = scl.calc_df_region_signature(ae)
subs, _ = scl.run_subregions(ae)
df_gene_sets = scl.calc_df_subregion_gene_sets(ae, region_sig)

n_spots = ae.obs.region_label.value_counts()
df_reg = pd.DataFrame(REG, columns=["region_id", "region_name", "compartment", "description"])
df_reg["c2l_name"] = df_reg.region_id.map(region_names)
df_reg["region_cluster"] = df_reg.region_id.str[1:].astype(int)
df_reg["n_spots"] = df_reg.region_id.map(n_spots).astype(int)
assert set(df_reg.region_id) == set(region_names.index)
pdwritecsv(df_reg, "lymph_node_regions.tsv", scl.root_data, verbose=True)

rows = []
for (reg, s), d in df_gene_sets.groupby(["region_id", "sub_region_id"], sort=False):
    sig = d[d.gene_set == "subregion_signature"].sort_values("scores", ascending=False)
    rows.append(dict(region_id=reg, sub_region_id=s, signature_check=", ".join(scl._display_genes(sig.gene, 3)),
                     functional_description=D.get(s, "")))
df_desc = pd.DataFrame(rows)
missing = df_desc[df_desc.functional_description == ""].sub_region_id.tolist()
extra = sorted(set(D) - set(df_desc.sub_region_id))
assert not missing and not extra, (missing, extra)
pdwritecsv(df_desc, "lymph_node_subregion_descriptions.tsv", scl.root_data, verbose=True)
