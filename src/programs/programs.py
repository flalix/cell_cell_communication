"""
Biological PROGRAMS for spatial niche interpretation.

Structure
---------
PROGRAMS[category][program] -> list of core genes (curated, human symbols).
CELL_STATES[cell_type][state] -> marker genes (cell-identity / polarisation,
    kept apart from functional programs so they are not confused with them).
REACTOME_RULES[program] -> regexes matched against Reactome pathway NAMES.
    Used by `expand_with_reactome()` to grow each program with genes from every
    matching Reactome pathway (download ReactomePathways.gmt from
    https://reactome.org/download/current/ReactomePathways.gmt.zip).

Core lists draw on Reactome, MSigDB Hallmark/GO and canonical literature; they
are intentionally compact (~15-40 genes) so that scores are not dominated by
house-keeping genes. Use `expand_with_reactome` for the "most complete" sets.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 1. Functional programs (50)                                                    #
# --------------------------------------------------------------------------- #
PROGRAMS: dict[str, dict[str, list[str]]] = {
    "cell_death": {
        "apoptosis_intrinsic": ["BAX", "BAK1", "BID", "BBC3", "PMAIP1", "BCL2L11", "BCL2", "BCL2L1",
                                "MCL1", "CYCS", "APAF1", "CASP9", "CASP3", "CASP7", "DIABLO", "XIAP",
                                "BIRC2", "BIRC3", "AIFM1", "ENDOG", "TP53", "TP53I3"],
        "apoptosis_extrinsic": ["FAS", "FASLG", "TNFRSF10A", "TNFRSF10B", "TNFSF10", "TNFRSF1A", "TNF",
                                "FADD", "TRADD", "CASP8", "CASP10", "CFLAR", "CASP3", "BID"],
        "necroptosis": ["RIPK1", "RIPK3", "MLKL", "ZBP1", "TNFRSF1A", "CASP8", "CFLAR", "TLR3",
                        "TICAM1", "PGAM5", "CYLD", "OTULIN"],
        "pyroptosis_inflammasome": ["NLRP3", "NLRP1", "NLRC4", "AIM2", "PYCARD", "CASP1", "CASP4",
                                    "CASP5", "GSDMD", "GSDME", "IL1B", "IL18", "NINJ1", "P2RX7"],
        "ferroptosis": ["GPX4", "SLC7A11", "SLC3A2", "ACSL4", "LPCAT3", "ALOX15", "ALOX12", "TFRC",
                        "FTH1", "FTL", "NCOA4", "HMOX1", "AIFM2", "GCH1", "DHODH", "NFE2L2", "SAT1",
                        "CHAC1", "POR", "PEBP1", "GSS", "GCLC", "GCLM"],
        "necrosis_DAMP": ["HMGB1", "S100A8", "S100A9", "HSPA1A", "HSP90AA1", "CALR", "ANXA1",
                          "IL33", "IL1A", "SAP130", "PANX1", "ENTPD1", "NT5E"],
    },
    "stress": {
        "ER_stress_UPR": ["HSPA5", "ATF4", "ATF6", "ERN1", "XBP1", "EIF2AK3", "DDIT3", "HERPUD1",
                          "DNAJB9", "SEL1L", "HYOU1", "PDIA4", "PDIA6", "CALR", "SDF2L1", "EDEM1",
                          "ASNS", "TRIB3", "PPP1R15A", "SYVN1"],
        "ROS_oxidative_stress": ["NFE2L2", "KEAP1", "HMOX1", "NQO1", "TXN", "TXNRD1", "PRDX1", "PRDX2",
                                 "PRDX3", "SOD1", "SOD2", "CAT", "GPX1", "GPX2", "GCLM", "SRXN1",
                                 "G6PD", "NOX4", "CYBB", "MT2A"],
        "hypoxia": ["HIF1A", "EPAS1", "VEGFA", "CA9", "SLC2A1", "LDHA", "PGK1", "BNIP3", "BNIP3L",
                    "NDRG1", "ADM", "ANKRD37", "P4HA1", "P4HA2", "EGLN3", "ERO1A", "LOX", "ENO1",
                    "PDK1", "ANGPTL4"],
        "DNA_damage_response": ["ATM", "ATR", "CHEK1", "CHEK2", "TP53", "CDKN1A", "MDM2", "H2AX",
                                "MDC1", "BRCA1", "BRCA2", "RAD51", "PARP1", "XRCC1",
                                "TP53BP1", "GADD45A", "RRM2B", "SESN1", "DDB2", "XPC"],
        "proteostasis_heat_shock": ["HSF1", "HSPA1A", "HSPA1B", "HSPA8", "HSP90AA1", "HSP90AB1",
                                    "HSPB1", "DNAJB1", "BAG3", "PSMA1", "PSMB5", "PSMD1", "UBB",
                                    "UBC", "SQSTM1", "VCP"],
        "senescence_SASP": ["CDKN1A", "CDKN2A", "CDKN2B", "TP53", "RB1", "GLB1", "SERPINE1", "IL6",
                            "CXCL8", "IL1A", "IL1B", "CCL2", "MMP1", "MMP3", "IGFBP3", "IGFBP7",
                            "GDF15", "LMNB1", "HMGA1"],
    },
    "proliferation": {
        "G1_S_DNA_replication": ["CCND1", "CCND2", "CCND3", "CDK4", "CDK6", "CCNE1", "CCNE2", "CDK2",
                                 "E2F1", "E2F2", "RB1", "CDC25A", "MYC", "CDC6", "CDT1", "ORC1", "CDC45",
                                 "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "GINS2", "PCNA",
                                 "POLA1", "POLD1", "POLE", "PRIM1", "RFC4", "FEN1", "LIG1", "TYMS",
                                 "RRM1", "RRM2"],
        "G2_M_mitosis": ["CCNB1", "CCNB2", "CDK1", "CDC20", "PLK1", "AURKA", "AURKB", "BUB1",
                         "BUB1B", "TOP2A", "MKI67", "KIF11", "KIF2C", "CENPA", "CENPE", "CENPF",
                         "TPX2", "NUSAP1", "UBE2C", "HMMR", "PRC1", "CDKN3"],
        "cell_cycle_checkpoints": ["CHEK1", "CHEK2", "WEE1", "CDKN1A", "CDKN1B", "CDKN2A", "TP53",
                                   "MAD2L1", "BUB1B", "BUB3", "TTK", "MDM2", "CDC25B", "CDC25C",
                                   "GADD45A", "SFN", "RB1"],
    },
    "metabolism": {
        "glycolysis": ["HK2", "GPI", "PFKP", "PFKL", "ALDOA", "TPI1", "GAPDH", "PGK1", "PGAM1",
                       "ENO1", "PKM", "LDHA", "SLC2A1", "SLC2A3", "SLC16A3", "PFKFB3", "PDK1"],
        "OXPHOS_TCA": ["NDUFA4", "NDUFB8", "NDUFS1", "SDHA", "SDHB", "UQCRC1", "UQCRC2", "CYCS",
                       "COX4I1", "COX5A", "ATP5F1A", "ATP5F1B", "CS", "IDH3A", "OGDH", "MDH2",
                       "FH", "ACO2", "PPARGC1A"],
        "fatty_acid_lipid": ["FASN", "ACACA", "ACLY", "SCD", "SREBF1", "SREBF2", "HMGCR", "HMGCS1",
                             "LDLR", "INSIG1", "CPT1A", "ACADM", "HADHA", "PPARA", "CD36", "FABP4",
                             "FABP5", "LPL", "PLIN2", "APOE"],
        "amino_acid_glutamine": ["SLC1A5", "SLC7A5", "SLC38A2", "GLS", "GLUD1", "GOT1", "GOT2",
                                 "ASNS", "PHGDH", "PSAT1", "PSPH", "SHMT2", "MTHFD2", "ARG1",
                                 "ARG2", "IDO1", "TDO2", "KYNU"],
        "iron_metabolism": ["TFRC", "TF", "FTH1", "FTL", "SLC40A1", "HAMP", "HMOX1", "IREB2",
                            "ACO1", "STEAP3", "SLC11A2", "NCOA4", "CP", "LCN2"],
    },
    "trafficking": {
        "endocytosis": ["CLTC", "CLTA", "AP2A1", "AP2M1", "DNM1", "DNM2", "EPN1", "CAV1", "CAV2",
                        "FLOT1", "RAB5A", "RAB7A", "EEA1", "LDLR", "LRP1", "TFRC", "PICALM", "SNX1"],
        "phagocytosis_efferocytosis": ["MERTK", "AXL", "TYRO3", "GAS6", "PROS1", "MFGE8", "CD36",
                                       "ITGAV", "ITGB5", "TIMD4", "STAB1", "LRP1", "C1QA", "C1QB",
                                       "FCGR1A", "FCGR2A", "FCGR3A", "MSR1", "MARCO", "CD68",
                                       "ELMO1", "DOCK1", "RAC1", "SIRPA", "CD47"],
        "autophagy_lysosome": ["ULK1", "ATG13", "BECN1", "PIK3C3", "ATG5", "ATG7", "ATG12", "ATG16L1",
                               "MAP1LC3A", "MAP1LC3B", "GABARAP", "SQSTM1", "WIPI2", "ATG9A", "OPTN",
                               "PINK1", "PRKN", "TFEB", "LAMP1", "LAMP2", "CTSB", "CTSD", "CTSL",
                               "ATP6V0D1", "ATP6V1A", "GBA1", "HEXB", "NPC1", "NPC2", "PSAP", "GRN"],
        "secretion_exocytosis": ["SNAP23", "STX4", "VAMP7", "VAMP8", "RAB27A", "RAB27B", "SEC61A1",
                                 "SEC23A", "SEC24A", "COPA", "ARF1", "CHGA", "SCG2", "CD63", "CD81"],
    },
    "tissue_remodeling": {
        "EMT": ["VIM", "CDH2", "FN1", "SNAI1", "SNAI2", "TWIST1", "TWIST2", "ZEB1", "ZEB2", "SPARC",
                "COL1A1", "COL3A1", "COL5A1", "TAGLN", "ACTA2", "MMP2", "SERPINE1", "TGFBI",
                "ITGA5", "LAMC2", "VCAN", "POSTN", "CDH11", "PRRX1"],
        "migration_invasion": ["RAC1", "CDC42", "RHOA", "ROCK1", "ROCK2", "PAK1", "WASF2", "ACTR2",
                               "ARPC2", "CFL1", "MYH9", "VASP", "ENAH", "FSCN1", "LCP1", "PLAU",
                               "PLAUR", "MMP14", "S100A4", "CXCR4"],
        "cell_cell_adhesion": ["CDH1", "CDH3", "CTNNA1", "CTNNB1", "CTNND1", "JUP", "DSP", "DSG2",
                               "DSC2", "PKP3", "TJP1", "OCLN", "CLDN1", "CLDN4", "CLDN7", "EPCAM",
                               "F11R", "NECTIN2"],
        "ECM_integrin_adhesion": ["ITGA1", "ITGA2", "ITGA3", "ITGA5", "ITGA6", "ITGAV", "ITGB1",
                                  "ITGB3", "ITGB4", "ITGB5", "PTK2", "PXN", "TLN1", "VCL", "ILK",
                                  "LAMA3", "LAMB3", "LAMC2", "FN1", "VTN", "COL4A1", "COL4A2"],
        "ECM_remodeling_fibrosis": ["COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL6A1", "COL11A1",
                                    "FN1", "POSTN", "LOX", "LOXL2", "MMP1", "MMP2", "MMP3", "MMP9",
                                    "MMP11", "MMP14", "TIMP1", "TIMP3", "SPARC", "CTHRC1", "FAP",
                                    "PDGFRB", "THBS2"],
        "angiogenesis": ["VEGFA", "VEGFC", "KDR", "FLT1", "FLT4", "ANGPT1", "ANGPT2", "TEK", "DLL4",
                         "NOTCH1", "ESM1", "APLN", "PECAM1", "CDH5", "PDGFB", "PDGFRB", "ENG",
                         "NRP1", "HIF1A", "EGFL7"],
        "coagulation_wound": ["F3", "F2R", "F2RL1", "F5", "F10", "FGA", "FGB", "FGG", "PLG", "PLAT",
                              "PLAU", "SERPINE1", "THBD", "VWF", "TFPI", "ANXA2", "THBS1"],
    },
    "signaling": {
        "TGFb_signaling": ["TGFB1", "TGFB2", "TGFB3", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3", "SMAD4",
                           "SMAD7", "SKIL", "LTBP1", "THBS1", "SERPINE1", "BMP2", "BMP4", "ACVR1",
                           "ID1", "ID2", "ID3"],
        "WNT_signaling": ["WNT2", "WNT3A", "WNT5A", "WNT7B", "FZD1", "FZD2", "FZD7", "LRP5", "LRP6",
                          "CTNNB1", "APC", "AXIN1", "AXIN2", "LEF1", "TCF7", "TCF7L2", "LGR5",
                          "NKD1", "DKK1", "RNF43"],
        "NOTCH_signaling": ["NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4", "DLL1", "DLL3", "DLL4", "JAG1",
                            "JAG2", "RBPJ", "MAML1", "HES1", "HES5", "HEY1", "HEY2", "HEYL",
                            "NRARP", "ADAM10"],
        "Hedgehog_signaling": ["SHH", "IHH", "DHH", "PTCH1", "PTCH2", "SMO", "GLI1", "GLI2", "GLI3",
                               "SUFU", "HHIP", "KIF7"],
        "RTK_MAPK": ["EGFR", "ERBB2", "ERBB3", "MET", "HGF", "EGF", "AREG", "EREG", "KRAS", "NRAS",
                     "BRAF", "RAF1", "MAP2K1", "MAPK1", "MAPK3", "DUSP6", "SPRY2", "ETV4", "ETV5",
                     "FOS", "EGR1"],
        "PI3K_AKT_mTOR": ["PIK3CA", "PIK3CB", "PIK3R1", "AKT1", "AKT2", "PTEN", "MTOR", "RPTOR",
                          "RICTOR", "TSC1", "TSC2", "RHEB", "RPS6KB1", "EIF4EBP1", "PDK1", "FOXO1",
                          "FOXO3", "IRS1"],
        "JAK_STAT": ["JAK1", "JAK2", "JAK3", "TYK2", "STAT1", "STAT3", "STAT5A", "STAT5B", "STAT6",
                     "SOCS1", "SOCS3", "IL6", "IL6R", "IL6ST", "OSMR", "LIFR", "CISH", "PIM1"],
    },
    "immune": {
        "inflammation_TNF_NFkB": ["TNF", "TNFRSF1A", "IL1A", "IL1B", "IL1R1", "IL6", "CXCL8", "NFKB1",
                                  "NFKB2", "RELA", "RELB", "NFKBIA", "TNFAIP3", "PTGS2", "ICAM1",
                                  "CCL2", "CCL20", "CXCL1", "CXCL2", "IRAK1", "MYD88", "TLR4"],
        "interferon_type_I": ["IFNA1", "IFNB1", "IFNAR1", "IFNAR2", "IRF3", "IRF7", "ISG15", "MX1",
                              "MX2", "OAS1", "OAS2", "OAS3", "IFIT1", "IFIT2", "IFIT3", "IFI6", "IFI44L",
                              "RSAD2", "STING1", "CGAS", "DDX58", "IFIH1"],
        "interferon_type_II": ["IFNG", "IFNGR1", "IFNGR2", "STAT1", "IRF1", "CXCL9", "CXCL10",
                               "CXCL11", "IDO1", "GBP1", "GBP2", "GBP5", "CIITA", "HLA-DRA", "TAP1",
                               "PSMB9", "SOCS1", "CD274"],
        "complement": ["C1QA", "C1QB", "C1QC", "C1R", "C1S", "C2", "C3", "C4A", "C4B", "C5", "C5AR1",
                       "C3AR1", "CFB", "CFD", "CFH", "CFI", "CD55", "CD59", "CR1", "CR2", "ITGAM"],
        "antigen_presentation_MHCI": ["HLA-A", "HLA-B", "HLA-C", "HLA-E", "B2M", "TAP1", "TAP2",
                                      "TAPBP", "PSMB8", "PSMB9", "PSMB10", "ERAP1", "ERAP2",
                                      "CALR", "PDIA3", "NLRC5"],
        "antigen_presentation_MHCII": ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1",
                                       "HLA-DQB1", "HLA-DMA", "HLA-DMB", "CD74", "CIITA", "CTSS",
                                       "LGMN", "IFI30", "CD86", "CD80", "CD40"],
        "leukocyte_chemotaxis": ["CCL2", "CCL3", "CCL4", "CCL5", "CCL19", "CCL21", "CXCL1", "CXCL2",
                                 "CXCL5", "CXCL8", "CXCL12", "CXCL13", "CCR2", "CCR5", "CCR7",
                                 "CXCR1", "CXCR2", "CXCR3", "CXCR4", "CX3CL1", "CX3CR1", "SELE",
                                 "SELP", "VCAM1", "ICAM1"],
        "cytotoxic_T_NK": ["CD8A", "CD8B", "GZMA", "GZMB", "GZMH", "GZMK", "PRF1", "NKG7", "GNLY",
                           "IFNG", "FASLG", "KLRD1", "KLRK1", "NCR1", "NCAM1", "FCGR3A", "CD247",
                           "ZAP70"],
        "T_cell_exhaustion_checkpoint": ["PDCD1", "CD274", "PDCD1LG2", "CTLA4", "LAG3", "HAVCR2",
                                         "TIGIT", "PVR", "TOX", "ENTPD1", "CXCL13", "BTLA",
                                         "VSIR", "LGALS9", "IDO1"],
        "humoral_B_cell": ["MS4A1", "CD19", "CD79A", "CD79B", "CR2", "CXCR5", "CXCL13", "AICDA",
                           "BCL6", "MZB1", "JCHAIN", "XBP1", "PRDM1", "TNFRSF17", "IGHG1", "IGHA1",
                           "IGKC", "FCRL5"],
        "immunosuppression": ["FOXP3", "IL2RA", "CTLA4", "IL10", "TGFB1", "ARG1", "NOS2", "IDO1",
                              "CD274", "VSIR", "LGALS9", "PTGES", "PTGS2", "CCL22", "CCL17",
                              "ENTPD1", "NT5E", "ADORA2A", "S100A8", "S100A9"],
    },
    "neural": {
        "synaptic_neurotransmission": ["SNAP25", "SYT1", "SYP", "SYN1", "STX1A", "VAMP2", "CPLX1",
                                       "SLC17A7", "SLC17A6", "SLC32A1", "GAD1", "GAD2", "GRIN1",
                                       "GRIN2A", "GRIN2B", "GRIA1", "GRIA2", "GABRA1", "GABRB2",
                                       "DLG4", "HOMER1", "CAMK2A", "NRGN", "SLC1A2", "SLC1A3", "GLUL",
                                       "NRXN1", "NRXN2", "NRXN3", "NLGN1", "NLGN2", "NLGN3", "CBLN1", "GRID2"],
        "axon_guidance_neurodevelopment": ["SEMA3A", "SEMA4D", "NRP1", "PLXNA1", "PLXNB1", "ROBO1",
                                           "ROBO2", "SLIT1", "SLIT2", "NTN1", "DCC", "UNC5B", "EPHA4",
                                           "EPHB2", "EFNB1", "EFNA5", "L1CAM", "NCAM1", "DCX",
                                           "TUBB3", "GAP43", "BDNF", "NTRK2", "RELN"],
    },
}

# --------------------------------------------------------------------------- #
# 2. Cell states / identity markers (your format, extended)                   #
# --------------------------------------------------------------------------- #
CELL_STATES: dict[str, dict[str, list[str]]] = {
    "macrophage": {
        "M1": ["IL1B", "TNF", "CXCL9", "CXCL10", "NOS2", "CCL5"],
        "TAM": ["CD163", "MRC1", "MSR1", "TREM2", "APOE", "C1QA", "C1QB"],
        "SPP1": ["SPP1", "MARCO", "MMP9"],
        "mdsc_suppressive": ["ARG1", "IL10", "CCL17", "CCL22", "IL4", "IL13", "CD80", "CD86", "CD40"],
    },
    "endothelial": {
        "tip_angio": ["ESM1", "ANGPT2", "DLL4", "APLN", "CXCR4", "KDR"],
        "lymphatic": ["PROX1", "LYVE1", "PDPN", "CCL21", "FLT4"],
        "activated": ["VCAM1", "ICAM1", "SELE", "ACKR1"],
    },
    "humoral": {
        "TLS": ["CXCL13", "CR2", "FDCSP", "MS4A1", "CD79A", "CCL19"],
        "plasma": ["MZB1", "JCHAIN", "XBP1", "DERL3", "TNFRSF17"],
    },
    "fibroblast": {
        "myCAF": ["ACTA2", "TAGLN", "MYL9", "POSTN", "COL11A1", "THBS2"],
        "iCAF": ["IL6", "CXCL12", "PDGFRA", "CFD", "DPT", "LMNA", "CXCL1"],
        "apCAF": ["CD74", "HLA-DRA", "HLA-DRB1", "SLPI", "SAA3"],
    },
    "brain": {
        "excitatory_neuron": ["SLC17A7", "NEUROD6", "SATB2", "TBR1", "CAMK2A"],
        "inhibitory_neuron": ["GAD1", "GAD2", "SLC32A1", "PVALB", "SST", "VIP"],
        "astrocyte": ["GFAP", "AQP4", "ALDH1L1", "SLC1A3", "GJA1", "S100B"],
        "reactive_astrocyte": ["GFAP", "SERPINA3", "LCN2", "CD44", "C3", "VIM"],
        "oligodendrocyte": ["MBP", "PLP1", "MOG", "MOBP", "MAG", "OLIG2"],
        "microglia": ["P2RY12", "TMEM119", "CX3CR1", "CSF1R", "HEXB", "C1QA"],
        "DAM_microglia": ["TREM2", "APOE", "TYROBP", "CST7", "LPL", "ITGAX", "SPP1"],
    },
    "epithelial_PDAC": {
        "classical": ["GATA6", "TFF1", "LGALS4", "CLDN18", "CDH17", "REG4", "CEACAM6"],
        "basal_like": ["KRT5", "KRT6A", "S100A2", "VGLL1", "FAM83A", "SPRR1B", "LAMC2"],
    },
}

# --------------------------------------------------------------------------- #
# 3. Reactome pathway-name -> program rules (case-insensitive regexes)        #
#    Classification order: EXCLUDE (disease variants, pathogens, drugs) ->    #
#    programs (a pathway may feed several) -> CONTEXT classes (housekeeping /  #
#    non-program biology, kept so that ALL of Reactome is accounted for).     #
#    Reactome has no ferroptosis node: it is assembled from GSH/iron/lipid-    #
#    peroxidation pathways.                                                    #
# --------------------------------------------------------------------------- #
EXCLUDE_RULES: dict[str, list[str]] = {
    "disease_variant": [r"Phenylketonuria", r"neutropenia", r"deficiencies", r"variants? cause", r"aciduria", r"[a-z]+emia\b", r"Abnormal conversion", r"[Dd]isease", r"Impaired", r"Loss-of-function", r"[Ee]pilepsy", r"intolerance", r"fructosuria", r"pentosuria", r"Deletions in", r"Defects of", r"variant leads", r"Enhanced cleavage of VWF variant", r"^Defective ", r"\bcauses\b", r"^Diseases? (of|associated)", r"[Mm]utants?\b",
                        r"Loss of [Ff]unction", r"[Dd]eficiency", r"Constitutive Signaling by",
                        r"Aberrant", r"-CDG\b", r"\bCDG", r"resistant", r"Disorders? of",
                        r"in [Cc]ancer$", r"Oncogenic", r"fusions?\b", r"Variant", r"overexpressed",
                        r"amplified", r"Signaling by .*in disease", r"Glycogen storage disease",
                        r"[Ss]yndrome", r"[Dd]ystroph", r"Inhibition of .* by .*(drug|inhibitor)"],
    "infection_pathogen": [r"Virion", r"vRNA", r"cRNA", r"vRNP", r"NS1 Mediated", r"Plus-strand DNA", r"2-LTR", r"Attachment and Entry", r"^Budding$", r"Biofilm", r"[Aa]ntimicrobial", r"[Bb]acteria", r"Mtb", r"Maturation of (spike|replicase|nucleoprotein|protein [0-9A-Z]|DENV|hRSV)", r"Inhibition of (PKR|Interferon Synthesis|membrane repair|nitric oxide production)", r"Cell-Cell Fusion", r"Minus-strand", r"NEP/NS2", r"\bRev\b", r"HIV", r"SARS", r"[Vv]irus", r"[Vv]iral", r"Influenza", r"HCMV", r"Hepatitis",
                           r"Leishmania", r"Mycobacteri", r"Tuberculosis", r"Salmonella", r"Listeria",
                           r"Legionella", r"Chlamydia", r"Staphylococcus", r"Enterobacterial", r"Bacterial",
                           r"[Tt]oxin", r"Anthrax", r"Botulinum", r"Tetanus", r"Parasite", r"Plasmodium",
                           r"Infection", r"Host Interactions", r"EBV", r"KSHV", r"Dengue", r"Zika",
                           r"Ebola", r"Measles", r"RSV", r"Respiratory syncytial", r"HPV", r"Vpr", r"Vpu",
                           r"Nef", r"\bTat\b", r"Rev-", r"Pathogen"],
    "drug_xenobiotic": [r"Aryl hydrocarbon", r"Aromatic amines", r"Amine Oxidase", r"Amino Acid conjugation", r"CYP\d", r"transmembrane transport$", r"FMO", r"sulfonation", r"LGK974", r"Miscellaneous substrates", r"ADME", r"[Xx]enobiotic", r"Aspirin", r"Paracetamol", r"Phase I -", r"Phase II -",
                        r"Cytochrome P450", r"CYP2E1", r"Biological oxidations", r"Conjugation of",
                        r"Methylation of .* for excretion", r"Glucuronidation", r"[Dd]rug"],
}

REACTOME_RULES: dict[str, list[str]] = {
    # cell death
    "apoptosis_intrinsic": [r"NRIF", r"Suppression of apoptosis", r"p75 ?NTR", r"BAK", r"BAX", r"BCL2", r"SMAC", r"XIAP", r"Cell death signalling via", r"NADE", r"NRAGE", r"p75NTR", r"Intrinsic Pathway for Apoptosis", r"BH3", r"BCL-?2", r"Cytochrome c-mediated",
                            r"apoptosome", r"TP53 Regulates Transcription of .*(Death|Apopto)",
                            r"Release of apoptotic factors", r"^Apoptosis$", r"Apoptotic (execution|factor|cleavage)",
                            r"Caspase-mediated cleavage", r"Apoptosis induced", r"Breakdown of the nuclear lamina",
                            r"Regulation of Apoptosis", r"BAD", r"BID", r"Activation of caspases"],
    "apoptosis_extrinsic": [r"Extrinsic Pathway", r"Death Receptor", r"TRAIL", r"Caspase activation via",
                            r"procaspase-8", r"FasL", r"CD95", r"Ligand-dependent caspase", r"Dependence Receptor"],
    "necroptosis": [r"necroptotic", r"c-FLIP", r"RIPK3", r"CASP8 activity is inhibited", r"Regulated Necrosis", r"RIPK1", r"Necroptosis", r"MLKL"],
    "pyroptosis_inflammasome": [r"CASP4", r"CASP5", r"Pyroptosis", r"Inflammasome", r"Interleukin-1 processing", r"Interleukin-18",
                                r"Gasdermin", r"NLRP", r"Purinergic"],
    "ferroptosis": [r"Glutathione synthesis", r"Glutathione conjugation", r"Synthesis of 1[25]-eicosatetraenoic",
                    r"Iron uptake", r"Selenocysteine synthesis", r"Cystine", r"Lipid peroxid",
                    r"Synthesis of 5-eicosatetraenoic", r"Biosynthesis of .*(HETE|HPETE)"],
    "necrosis_DAMP": [r"Advanced glycosylation endproduct", r"HMGB1", r"Purinergic signaling", r"Neutrophil degranulation",
                      r"Antimicrobial peptides", r"Defensins", r"Alpha-defensins", r"Beta defensins", r"Metal sequestration"],
    # stress
    "ER_stress_UPR": [r"ER Quality Control", r"Calnexin", r"Unfolded Protein Response", r"IRE1", r"PERK", r"ATF6", r"ATF4", r"XBP1", r"ERAD",
                      r"Calnexin/Calreticulin cycle", r"N-glycan trimming in the ER"],
    "ROS_oxidative_stress": [r"BACH1", r"Tolerance of reactive oxygen", r"redox", r"HMOX1", r"MTF1", r"NADPH oxidase", r"Detoxification of Reactive Oxygen", r"chemical stress", r"NFE2L2", r"KEAP1",
                             r"ROS and RNS", r"[Oo]xidative [Ss]tress", r"Peroxiredoxin", r"Thioredoxin",
                             r"Superoxide", r"Metallothioneins", r"Response of .* to metal"],
    "hypoxia": [r"[Hh]ypoxi", r"\bHIF", r"Oxygen-dependent", r"Regulation of gene expression by Hypoxia"],
    "DNA_damage_response": [r"[Rr]eversal of alkylation", r"Abasic", r"ALKBH", r"glycosylase", r"MUTYH", r"OGG1", r"MGMT", r"NEIL", r"ICL", r"Cleavage of the damaged", r"Homologous DNA Pairing", r"damaged DNA", r"PCNA", r"BRCA", r"DNA Double[- ]Strand Break", r"DNA Damage", r"Homology Directed Repair", r"\bHDR\b",
                            r"Homologous Recombination", r"Nonhomologous End", r"Base Excision Repair",
                            r"Nucleotide Excision Repair", r"Mismatch Repair", r"Fanconi", r"DNA Repair",
                            r"Translesion", r"AP Site", r"Holliday Junction", r"D-loop", r"Single Strand Annealing",
                            r"Microhomology", r"Recognition of DNA damage", r"Dual Incision", r"Gap-filling",
                            r"Regulation of TP53", r"TP53 Regulates", r"TP53 [Aa]ctivity", r"Stabilization of p53",
                            r"Sensing of DNA Double", r"DNA strand elongation", r"Depurination", r"Depyrimidination",
                            r"Resolution of Abasic", r"Displacement of DNA glycosylase", r"Reversal of alkylation",
                            r"Processing of DNA", r"ATM", r"ATR", r"Telomere", r"telomer"],
    "proteostasis_heat_shock": [r"Amyloid", r"Aggresome", r"Attenuation phase", r"HSP90", r"HSP70", r"DUBs", r"HSF1", r"Heat Shock", r"heat stress", r"Proteasome", r"[Uu]biquitin",
                                r"Deubiquitination", r"Neddylation", r"SUMO", r"Chaperon", r"protein folding",
                                r"Protein folding", r"Prefoldin", r"CCT/TriC", r"Ovarian tumor domain",
                                r"Josephin", r"UCH proteinases", r"Ub-specific", r"\bMPAC\b", r"Cullin",
                                r"Antigen processing: Ubiquitination"],
    "senescence_SASP": [r"Senescence", r"SASP", r"DNA Damage/Telomere Stress Induced"],
    # proliferation
    "G1_S_DNA_replication": [r"G0 and Early G1", r"replication initiation", r"Cdc6", r"G1/S", r"G1 Phase", r"Cyclin D", r"Cyclin E", r"E2F", r"DNA Replication",
                             r"pre-replicative", r"Unwinding of DNA", r"Synthesis of DNA", r"Lagging Strand",
                             r"Leading Strand", r"S Phase", r"Origin", r"CDC6", r"CDT1", r"ORC", r"MCM",
                             r"Polymerase switching", r"Removal of the Flap", r"Activation of ATR in response",
                             r"Cell Cycle, Mitotic", r"^Cell Cycle$", r"PTK6 Regulates Cell Cycle",
                             r"Cyclin A", r"p27", r"p21", r"SCF\(Skp2\)"],
    "G2_M_mitosis": [r"Nuclear Lamina", r"NIMA", r"NEK", r"Centriole", r"CENPA", r"Cohesion", r"Nuclear Envelope", r"\bNE\b", r"Mitotic", r"M Phase", r"Mitosis", r"Prometaphase", r"Metaphase", r"Anaphase", r"Telophase",
                     r"Prophase", r"Cytokinesis", r"Kinesins", r"Polo-like kinase", r"PLK1", r"Separation of Sister",
                     r"Cohesin", r"Condensin", r"Centrosome", r"centrosom", r"Kinetochore", r"[Ss]pindle",
                     r"Nuclear Envelope (Breakdown|Reassembly)", r"G2 Phase", r"G2/M Transition", r"Cyclin B",
                     r"Aurora", r"AURKA", r"APC/C", r"APC-Cdc20", r"APC:Cdc20", r"Cdc20", r"Cdh1", r"Emi1",
                     r"Nek2", r"Chromosome Maintenance", r"Nuclear Pore Complex.*(Disassembly|Breakdown)",
                     r"Lamin\b", r"Golgi Cisternae Pericentriolar"],
    "cell_cycle_checkpoints": [r"Amplification\s+of\s+signal from", r"kinetochores", r"MAD2", r"Checkpoint", r"checkpoint", r"Amplification of signal from the kinetochores",
                               r"Inhibition of the proteolytic activity of APC/C", r"p53-(Dependent|Independent)",
                               r"Chk1", r"Chk2", r"Activation of the AP-1 family"],
    # metabolism
    "glycolysis": [r"ChREBP", r"Glycolysis", r"Gluconeogenesis", r"Glucose metabolism", r"Glycogen", r"Fructose",
                   r"Galactose", r"Pentose phosphate", r"Lactose", r"Glucose transport", r"Carbohydrate",
                   r"Hexose", r"PFKFB", r"Pyruvate kinase"],
    "OXPHOS_TCA": [r"OGDH", r"PDH complex", r"OADH", r"Ketone Bodies", r"Aerobic respiration", r"Malate-aspartate", r"NADPH regeneration", r"Miro GTPase", r"Citric acid cycle", r"TCA", r"Respiratory electron", r"Complex I", r"Complex III",
                   r"Complex IV", r"Cristae", r"Pyruvate metabolism", r"pyruvate dehydrogenase", r"Mitochondrial",
                   r"mitochondria", r"Electron transport", r"ATP synthesis", r"Ubiquinol", r"Uncoupling",
                   r"Mitochondrial biogenesis", r"Mitochondrial translation", r"Mitochondrial protein import",
                   r"Ketone body", r"PPARGC1A"],
    "fatty_acid_lipid": [r"[Ff]atty [Aa]cid", r"[Cc]holesterol", r"Sterol", r"SREBP", r"SREBF", r"lipoprotein",
                         r"Lipoprotein", r"Triglyceride", r"Lipid", r"lipid", r"Phospholipid", r"Glycerophospholipid", r"Phosphoinositide",
                         r"Sphingolipid", r"Ceramide", r"Eicosanoid", r"Arachidonic", r"Prostaglandin",
                         r"Leukotriene", r"Lipoxin", r"Resolvin", r"Protectin", r"Maresin", r"Steroid",
                         r"Bile acid", r"Chylomicron", r"Arachidon", r"phytan", r"LPL", r"LIPC", r"LPE\b", r"LPC\b", r"LTC4", r"CYSLTR", r"LDL", r"HDL", r"VLDL", r"Peroxisomal lipid",
                         r"beta[- ]oxidation", r"Beta-oxidation", r"Carnitine", r"PPARA", r"Acyl chain",
                         r"Plasmalogen", r"Wax", r"Ether lipid", r"Phosphatidyl",
                         r"Inositol phosphate", r"Lipophagy", r"Hydrolysis of LPC", r"Thromboxane",
                         r"Synthesis of (PA|PC|PE|PG|PI|PS|CL|DAG|TAG)\b", r"(Androgen|Estrogen|Glucocorticoid|Mineralocorticoid) biosynthesis", r"Pregnenolone", r"Vitamin D", r"Retinoid",
                         r"ABC transporters in lipid", r"Acetylcholine (synthesis|esterase)"],
    "amino_acid_glutamine": [r"BCKDH", r"BCAA", r"polyamines", r"[Aa]mino acid", r"Glutamate and glutamine", r"Glutamine", r"Serine", r"Glycine",
                             r"Threonine", r"Tryptophan", r"Kynurenine", r"Tyrosine", r"Phenylalanine",
                             r"Histidine", r"Lysine", r"Leucine", r"Isoleucine", r"Valine", r"branched-chain",
                             r"Methionine", r"Cysteine", r"Homocysteine", r"Sulfur amino", r"Proline", r"Arginine",
                             r"Urea cycle", r"Polyamine", r"Creatine", r"Asparagine", r"Aspartate", r"Alanine",
                             r"Carnitine synthesis", r"Folate", r"[Oo]ne[- ]carbon", r"Selenoamino",
                             r"Nicotinate", r"Nicotinamide", r"Purine", r"Pyrimidine", r"[Nn]ucleotide (metabolism|salvage|biosynthesis|catabolism)",
                             r"Interconversion of nucleotide", r"Metabolism of nucleotides"],
    "iron_metabolism": [r"Iron", r"iron", r"Transferrin", r"Heme", r"heme", r"Ferritin", r"Porphyrin",
                        r"Erythrocytes take up", r"Hemoglobin"],
    # trafficking
    "endocytosis": [r"[Ee]ndocytosis", r"Cargo recognition for clathrin", r"Clathrin", r"Endosomal", r"[Ee]ndosome",
                    r"Caveol", r"Macropinocytosis", r"ESCRT", r"Retrograde transport at the Trans-Golgi",
                    r"Recycling", r"recycling", r"Trafficking of .* receptor", r"Internalization",
                    r"Membrane Trafficking", r"Vesicle-mediated transport", r"RAB GEFs", r"RAB GAPs",
                    r"Rab regulation", r"RAB geranylgeranylation", r"Sorting"],
    "phagocytosis_efferocytosis": [r"phagocyt", r"PMN cells", r"[Pp]hagocytosis", r"Scaveng", r"Scavenger", r"Efferocytosis", r"Phagosom",
                                   r"Fcgamma", r"FCGR", r"Cross-presentation", r"ER-Phagosome",
                                   r"Clearance", r"Uptake of .* by", r"Dectin", r"CLEC7A"],
    "autophagy_lysosome": [r"starvation", r"[Aa]utophagy", r"Mitophagy", r"Aggrephagy", r"Pexophagy", r"Xenophagy",
                           r"Lysosom", r"lysosom", r"Glycosphingolipid (metabolism|catabolism)", r"Mucopolysaccharid",
                           r"HS-GAG degradation", r"CS/DS degradation", r"KS degradation", r"MTOR signalling",
                           r"Amino acids regulate mTORC1", r"Energy dependent regulation of mTOR", r"TFEB"],
    "secretion_exocytosis": [r"Cargo concentration in the ER", r"Cargo trafficking", r"Vesicle budding", r"COPII", r"COPI", r"ER to Golgi", r"Golgi", r"Exocytosis", r"exocytosis",
                             r"Secretion", r"secretion", r"Platelet degranulation", r"degranulation",
                             r"Insulin processing", r"Peptide hormone", r"Dense core", r"Transport to the Golgi",
                             r"Intra-Golgi", r"Kinesins", r"Trans-Golgi", r"SNARE", r"Lysosome Vesicle Biogenesis",
                             r"Exosome", r"Translocation of .* to the plasma membrane"],
    # tissue remodelling
    "EMT": [r"Epithelial-Mesenchymal", r"\bEMT\b", r"SNAI", r"TWIST", r"Regulation of CDH1", r"CDH11",
            r"Downregulation of TGF-beta receptor"],
    "migration_invasion": [r"cytoskeletal remodeling", r"Sema4D", r"RHOBTB", r"RAC1", r"Inactivation of CDC42", r"Cell Motility", r"RHO GTPase", r"Rho GTPase", r"(RHO|RAC|CDC42|RHO\w)\w* GTPase cycle", r"GTPase cycle",
                           r"Regulation of actin", r"[Aa]ctin", r"Semaphorin", r"SEMA", r"Myosin", r"Formin",
                           r"WASP", r"WAVE", r"Arp2/3", r"Podosome", r"Invadopodia", r"Lamellipodia",
                           r"Filopodia", r"Chemokine receptors", r"PTK2", r"Cell motility", r"Cell migration",
                           r"RUNX2 regulates genes involved in cell migration", r"PAK", r"ROCK", r"Smooth Muscle Contraction",
                           r"Striated Muscle Contraction", r"Muscle contraction"],
    "cell_cell_adhesion": [r"Homotypic Cell-Cell Adhesion", r"SDK interactions", r"Gap Junctions", r"Cell-cell junction", r"Cell junction", r"Adherens junction", r"Tight junction",
                           r"Gap junction", r"gap junction", r"Connexin", r"Desmosome", r"Cell-Cell communication",
                           r"Nectin", r"Cadherin", r"[Cc]ornified envelope", r"Keratinization", r"Apical junction",
                           r"Type I hemidesmosome", r"Electrical Synapses", r"Signal regulatory protein",
                           r"SIRP", r"CD47", r"Basigin", r"\bCEACAM"],
    "ECM_integrin_adhesion": [r"PINCH-ILK-PARVIN", r"ILK", r"Integrin", r"integrin", r"ECM proteoglycans", r"Laminin", r"Cell-extracellular matrix",
                              r"Focal [Aa]dhesion", r"Syndecan", r"Fibronectin", r"MET interacts with",
                              r"Non-integrin membrane-ECM", r"Dystroglycan", r"DAG1", r"O-glycosylation of TSR",
                              r"Hyaluronan", r"Glycosaminoglycan", r"Heparan", r"Chondroitin", r"Keratan",
                              r"Dermatan", r"\bGAG\b", r"Proteoglycan"],
    "ECM_remodeling_fibrosis": [r"Collagen", r"collagen", r"Degradation of the extracellular matrix",
                                r"Extracellular matrix organization", r"Elastic fibre", r"Fibrillin",
                                r"Matrix Metalloproteinase", r"\bMMP", r"Anchoring fibril", r"Assembly of collagen",
                                r"Invadopodia", r"Lysyl oxidase", r"Molecules associated with elastic"],
    "angiogenesis": [r"VEGF", r"Tie2", r"Angiopoietin", r"Sprouting", r"Angiogenesis", r"Endothelial",
                     r"Apelin", r"Nitric oxide", r"eNOS", r"NOSTRIN", r"Vasopressin", r"Prostacyclin",
                     r"Signaling by PDGF", r"Lymphangiogenesis"],
    "coagulation_wound": [r"Fibrin", r"Platelet", r"platelet", r"Hemostasis", r"Coagulation", r"coagulation",
                          r"Thrombin", r"Thrombox", r"von Willebrand", r"GP1b", r"GPVI", r"Factor [IVX0-9]", r"F9\b",
                          r"gamma-carboxylation", r"Gamma carboxylation", r"Kallikrein", r"Kinin", r"Plasminogen",
                          r"Wound", r"Megakaryocyte", r"Response to elevated platelet", r"Common Pathway of Fibrin",
                          r"Cell surface interactions at the vascular wall", r"Vitamin K"],
    # signalling
    "TGFb_signaling": [r"TGF-beta", r"TGFB", r"TGFBR", r"SMAD", r"BMP", r"Activin", r"GDF", r"Nodal", r"NODAL",
                       r"SKI/SKIL", r"Signaling by TGFB family"],
    "WNT_signaling": [r"PCP", r"TCF/LEF", r"CTNNB1", r"destruction complex", r"AXIN", r"DVL", r"PORCN", r"WNT", r"Wnt", r"TCF dependent", r"[Bb]eta-catenin", r"Frizzled", r"TCF7L2",
                      r"PCP/CE", r"Planar cell polarity", r"RSPO", r"LGR"],
    "NOTCH_signaling": [r"NOTCH", r"Notch", r"Pre-NOTCH", r"\bDLL", r"JAG", r"RBPJ"],
    "Hedgehog_signaling": [r"SMO", r"Hedgehog", r"\bHh\b", r"Hh-Np", r"\bGLI", r"Smoothened", r"PTCH", r"Cilium",
                           r"[Cc]iliary", r"Intraflagellar", r"BBSome", r"Anchoring of the basal body"],
    "RTK_MAPK": [r"p38", r"Rap1", r"Negative regulation of MET", r"PLC beta", r"PKB-mediated", r"PDE3B", r"cGMP", r"phospho-PLA2", r"CDK5", r"ARMS-mediated", r"Frs2", r"FRS", r"GAB1", r"GRB7", r"IRS", r"FLT3", r"PIP2 hydrolysis", r"Downstream signal transduction", r"MAPK6/MAPK4", r"Klotho", r"EGFR", r"ERBB", r"Signaling by MET", r"MET (activates|promotes|receptor)", r"MAPK", r"MAP kinase",
                 r"\bRAF", r"\bRAS\b", r"RAS ", r"FGFR", r"FGF", r"PDGF", r"KIT", r"Receptor Tyrosine Kinases",
                 r"\bRET\b", r"ALK", r"NTRK", r"\bTRK", r"Neurotrophin", r"NGF", r"Insulin receptor", r"IGF1R",
                 r"Insulin-like Growth Factor", r"Signaling by Insulin", r"\bSHC", r"GRB2", r"SOS",
                 r"PLC-gamma", r"PTK6", r"ROS1", r"LTK", r"DDR", r"AXL", r"MST1", r"ERK", r"\bAP-1\b",
                 r"FRS2", r"SRC", r"Leptin", r"Growth hormone", r"Prolactin", r"EPH", r"Ephrin", r"Erythropoietin",
                 r"Signaling by (VEGF|SCF|NTRKs|ALK|LTK|Leptin)", r"ABL", r"Calmodulin", r"CaMK",
                 r"DAG and IP3", r"PKC", r"PKA", r"cAMP", r"CREB", r"RSK", r"Phosphorylation of CD3 and TCR zeta"],
    "PI3K_AKT_mTOR": [r"IRS activation", r"IRS-mediated", r"PI3K", r"\bAKT", r"mTOR", r"MTORC", r"PTEN", r"PIP3", r"FOXO", r"TSC", r"AMPK", r"LKB1",
                      r"Insulin effects", r"Energy dependent regulation", r"S6K1", r"4E-BP", r"GSK3"],
    "JAK_STAT": [r"CSF3", r"G-CSF", r"JAK", r"STAT", r"Interleukin-6", r"IL-6", r"Interleukin-(3|5|7|9|11|12|15|20|21|23|27|35|37|38) ",
                 r"Interleukin-(3|5|7|9|11|12|15|20|21|23|27|35|37|38)$", r"Interleukin-(12|23|27|35) family",
                 r"Interleukin receptor SHC", r"Other interleukin", r"Interleukin-3, Interleukin-5 and GM-CSF",
                 r"Signaling by Interleukins", r"Cytokine Signaling", r"GM-CSF", r"Oncostatin", r"\bLIF\b",
                 r"Thrombopoietin", r"cytokine receptor"],
    # immune
    "inflammation_TNF_NFkB": [r"pro-inflammatory", r"Cell recruitment", r"TNF", r"TNFR", r"NF-kB", r"NF-kappa", r"NFkB", r"NFKB", r"IKK", r"Interleukin-1 ",
                              r"Interleukin-1$", r"Interleukin-1 family", r"Interleukin-17", r"Interleukin-33",
                              r"Interleukin-36", r"Interleukin-37", r"Interleukin-38", r"Toll[- ]Like", r"\bTLR",
                              r"MyD88", r"TRIF", r"TICAM1", r"TRAF6", r"TAK1", r"IRAK", r"CD14",
                              r"Innate Immune System", r"Cytosolic sensors", r"NOD1/2", r"RIP-mediated", r"MAP3K8",
                              r"\bLPS\b", r"Acute phase", r"Inflammatory", r"Prostanoid", r"C-type lectin",
                              r"CD209", r"DC-SIGN", r"Mast cell", r"FCERI mediated NF-kB", r"Ficolins",
                              r"CLEC", r"Interleukin-1 signaling", r"TRAF3", r"Alpha-protein kinase 1",
                              r"Regulation of TNFR1", r"Interleukin-18", r"Neutrophil"],
    "interferon_type_I": [r"TREX1", r"PKR", r"endogenous retroelements", r"Interferon alpha/beta", r"Interferon$", r"^Interferon Signaling", r"STING", r"cGAS",
                          r"DDX58", r"IFIH1", r"RIG-I", r"MDA5", r"ISG15", r"IRF3", r"IRF7", r"TBK1", r"IKBKE",
                          r"cytosolic DNA", r"Antiviral mechanism", r"OAS antiviral", r"TRAF3-dependent IRF",
                          r"ZBP1", r"DAI", r"IFN"],
    "interferon_type_II": [r"GBP-mediated", r"Interferon gamma", r"IFNG", r"Regulation of IFNG"],
    "complement": [r"Complement", r"complement", r"C3 and C5", r"C4 and C2 activators", r"Lectin pathway",
                   r"Classical antibody-mediated complement", r"Alternative complement", r"Terminal pathway of complement",
                   r"Ficolin", r"Collectin"],
    "antigen_presentation_MHCI": [r"MHC class I\b", r"Antigen Presentation: Folding", r"Antigen processing", r"Cross-presentation",
                                  r"ER-Phagosome", r"Endosomal/Vacuolar pathway", r"\bTAP\b", r"Peptide-loading",
                                  r"Class I MHC", r"Butyrophilin", r"BTN"],
    "antigen_presentation_MHCII": [r"MHC class II", r"Class II MHC", r"Translocation of ZAP-70 to Immunological synapse"],
    "leukocyte_chemotaxis": [r"Chemokine", r"chemokine", r"Leukocyte", r"leukocyte", r"[Tt]ransendothelial",
                             r"Cell surface interactions at the vascular wall", r"Selectin", r"PECAM1", r"\bCXCR",
                             r"\bCCR", r"Formyl peptide", r"Neutrophil", r"Diapedesis", r"Extravasation"],
    "cytotoxic_T_NK": [r"NFAT", r"Calcineurin", r"LCK", r"FYN", r"LAT", r"Killing mechanisms", r"TCR", r"T cell receptor", r"CD3", r"ZAP-70", r"Generation of second messenger",
                       r"Immunoregulatory interactions between a Lymphoid", r"DAP12", r"\bNK\b", r"Natural killer",
                       r"KIR", r"NKG2D", r"Costimulation by the CD28", r"Co-stimulation", r"ICOS", r"CD28",
                       r"Adaptive Immune System", r"Lymphoid", r"Granzyme", r"Perforin", r"Interleukin-2 ",
                       r"Interleukin-2$", r"Interleukin-2 family", r"Interleukin-15", r"Interleukin-21",
                       r"Butyrophilin", r"Gamma-delta"],
    "T_cell_exhaustion_checkpoint": [r"Co-inhibition", r"BTLA", r"PD-1", r"PD-L1", r"CD274", r"PDCD1", r"CTLA4", r"CTLA-4", r"LAG3", r"HAVCR2",
                                     r"TIGIT", r"Immune checkpoint", r"Inhibitory receptors"],
    "humoral_B_cell": [r"B Cell Receptor", r"BCR", r"Antigen activates B Cell", r"FCERI", r"FCER", r"antibody",
                       r"Antibody", r"Immunoglobulin", r"IgA", r"IgG", r"IgE", r"Plasma cell", r"Germinal center",
                       r"CD22", r"CD19", r"Scavenging of heme from plasma", r"Interleukin-4 and Interleukin-13",
                       r"FcR", r"Fc epsilon", r"Fc gamma", r"Fc receptor"],
    "immunosuppression": [r"anti-inflammatory", r"CD163", r"IL10 synthesis", r"Interleukin-10", r"IL-10", r"Interleukin-4 and Interleukin-13", r"Regulatory T",
                          r"FOXP3", r"Adenosine", r"Prostaglandin E2", r"IDO", r"Arginase", r"PD-1",
                          r"Interleukin-35", r"Interleukin-37", r"Interleukin-27", r"PTGES"],
    # neural
    "synaptic_neurotransmission": [r"TWIK", r"TASK\)", r"TALK\)", r"TRESK", r"THIK", r"NPAS4", r"Phase [0-4] -", r"cytosolic Ca\+\+", r"GABAB", r"[Kk]ainate", r"Adrenoceptor", r"Ca-dependent events", r"CaM pathway", r"Cam-PDE", r"DARPP-32", r"Kir channels", r"HCN channels", r"Inwardly rectifying", r"LGI-ADAM", r"Neurexin", r"Gap Junctions", r"Neuronal System", r"[Ss]ynap", r"Neurotransmitter", r"[Gg]lutamate (binding|Neurotransmitter)",
                                   r"Glutamatergic", r"GABA", r"Glycine receptor", r"Acetylcholine", r"[Nn]icotinic",
                                   r"Muscarinic", r"Dopamine", r"Serotonin", r"Norepinephrine", r"Adrenaline",
                                   r"Histamine", r"NMDA", r"AMPA", r"Kainate", r"Potassium Channels",
                                   r"Voltage gated", r"Ion channel", r"Ligand-gated", r"Neurexins", r"Neuroligin",
                                   r"Long-term potentiation", r"Unblocking of NMDA", r"CREB1 phosphorylation through NMDA",
                                   r"Action potential", r"Cardiac conduction", r"Presynaptic", r"Postsynaptic",
                                   r"Phototransduction", r"Olfactory", r"Sensory Perception", r"Taste", r"Hearing",
                                   r"Opioid", r"Neuropeptide", r"Orexin", r"Tachykinin", r"Endocannabinoid",
                                   r"TRP channels", r"Stimuli-sensing channels", r"Astrocytic Glutamate"],
    "axon_guidance_neurodevelopment": [r"NrCAM", r"dendrite", r"CHL1", r"NRCAM", r"CRMPs", r"Plexin", r"Neuropilin", r"growth cone", r"EPHA", r"EPHB", r"Axon guidance", r"[Aa]xon", r"ROBO", r"SLIT", r"Netrin", r"DCC", r"UNC5",
                                       r"L1CAM", r"NCAM", r"Semaphorin", r"SEMA", r"Ephrin", r"EPH", r"Growth cone",
                                       r"Neurofascin", r"CRMP", r"DSCAM", r"Reelin", r"RELN", r"Nervous system development",
                                       r"Neurogenesis", r"Neuronal", r"Myelin", r"myelin", r"Oligodendrocyte",
                                       r"Neural crest", r"Neurite", r"Synapse formation", r"Ankyrin",
                                       r"Activation of (NMDA|AMPA) receptors"],
}

# Non-program Reactome classes, so every pathway gets a label.
CONTEXT_RULES: dict[str, list[str]] = {
    "transcription_chromatin": [r"demethylate DNA", r"\bTET", r"ESR-mediated", r"[Ee]strogen", r"GPER1", r"MITF", r"BMAL", r"CLOCK", r"NOTCH2NL", r"CREB3", r"chromatin remodel", r"deamination of adenosine", r"Transcription", r"transcription", r"RNA Polymerase", r"RNA Pol", r"Chromatin",
                                r"[Hh]istone", r"HDAC", r"HAT", r"Methylation", r"methylation", r"Epigenetic",
                                r"PRC2", r"Polycomb", r"SIRT", r"Nucleosome", r"RUNX", r"Gene expression",
                                r"Gene Expression", r"Nuclear Receptor", r"Nuclear receptor", r"TFAP2", r"MECP2",
                                r"YAP1", r"TEAD", r"NR1H", r"PPARG", r"Circadian", r"BMAL1", r"MYB", r"SMARCA",
                                r"NOTCH.*transcription", r"ESR1", r"\bAR\b", r"POU5F1", r"SOX2", r"NANOG",
                                r"Germ layer", r"Pluripotent", r"DNA methylation", r"Imprinting", r"Pausing",
                                r"Elongation", r"Termination", r"CTD", r"TP53", r"MicroRNA", r"miRNA", r"microRNA",
                                r"siRNA", r"piRNA", r"lncRNA", r"Small interfering RNA", r"PIWI"],
    "translation_RNA_processing": [r"Translation", r"translation", r"[Rr]ibosom", r"rRNA", r"tRNA", r"mRNA",
                                   r"Splicing", r"splicing", r"Spliceosome", r"Capping", r"Polyadenylation",
                                   r"Nonsense-Mediated", r"Nonsense Mediated", r"NMD", r"eIF", r"Peptide chain",
                                   r"SRP-dependent", r"Selenocysteine", r"snRNP", r"snRNA", r"Exon Junction",
                                   r"Nuclear import", r"Nuclear export", r"Transport of Mature", r"Processing of",
                                   r"RNA", r"Aminoacyl", r"Major pathway of rRNA", r"AUF1", r"ARE", r"KSRP",
                                   r"HuR", r"TTP", r"Butyrate Response Factor", r"Deadenylation", r"Decapping"],
    "protein_modification": [r"Protein hydroxylation", r"Protein lipoylation", r"propeptides", r"arylsulfatases", r"glycosylation", r"Glycosylation", r"[Gg]lycan", r"N-linked", r"O-linked",
                             r"GPI", r"Asparagine N-linked", r"Sialic", r"Mannose", r"Fucosyl", r"Sulfation",
                             r"Post-translational", r"Gamma carboxylation", r"Hypusine", r"Lipidation",
                             r"Palmitoylation", r"Myristoylation", r"Prenylation", r"ADP-ribosylation",
                             r"Acetylation", r"Phosphorylation", r"Carboxyterminal post-translational",
                             r"Surfactant", r"Dolichol", r"Synthesis of .*(GDP|UDP|CMP)", r"Protein methylation",
                             r"Protein repair", r"Peptide ligand", r"Dimerization", r"Disulfide",
                             r"Blood group", r"blood group", r"Lewis", r"Glypican", r"Pre-NOTCH Processing"],
    "transport_membrane": [r"O2/CO2 exchange", r"carbon dioxide", r"ammonium transport", r"Protein localization", r"peroxisomal proteins", r"IPs? transport", r"IP[0-9] and IP[0-9] transport", r"oxygen transport", r"Miscellaneous transport", r"SLC", r"[Tt]ransport of", r"[Tt]ransporters?", r"ABC", r"Ion (homeostasis|transport)",
                           r"Aquaporin", r"Zinc", r"Copper", r"Metal ion", r"Sodium", r"Potassium", r"Calcium",
                           r"Chloride", r"Bicarbonate", r"Proton", r"Organic (anion|cation)", r"Multifunctional anion",
                           r"Cation-coupled", r"Na\+", r"Ca2\+", r"Inositol transporters", r"Vitamin B",
                           r"Cobalamin", r"Riboflavin", r"Thiamin", r"Biotin", r"Pantothenate", r"Coenzyme A",
                           r"Vitamin C", r"Vitamin E", r"[Vv]itamins", r"Peroxisomal protein import",
                           r"peroxisomal membrane protein", r"Protein import", r"Mitochondrial protein import",
                           r"Insertion of tail-anchored", r"Nuclear Pore", r"Cellular hexose transport",
                           r"Transport of small molecules", r"Iodide", r"Phosphate", r"Sulfide", r"Molybdenum",
                           r"Selenium", r"Fluoride", r"Water"],
    "GPCR_signaling": [r"Adenylate cyclase", r"Glycoprotein hormones", r"binding receptors", r"Ligand-receptor interactions", r"GPCR", r"G-protein", r"G protein", r"G alpha", r"Gα", r"G beta", r"Class A/1", r"Class B/2",
                       r"Class C/3", r"Rhodopsin", r"Opsin", r"[Rr]eceptors? bind", r"Amine ligand", r"Peptide ligand",
                       r"Hormone ligand", r"Lysosphingolipid and LPA", r"Free fatty acid receptors",
                       r"P2Y", r"Adenosine P1", r"Thrombin signalling through", r"Calcitonin", r"Glucagon",
                       r"Incretin", r"Relaxin", r"Melanocortin", r"Chemokine receptors bind", r"Formyl peptide",
                       r"Prostanoid ligand", r"Leukotriene receptors", r"Eicosanoid ligand", r"Gastrin",
                       r"Orexin", r"Vasopressin-like", r"Signal amplification", r"ADP signalling", r"Ceramide signalling",
                       r"Sphingosine 1-phosphate", r"Regulator of G-protein", r"\bRGS", r"Arrestin", r"Visual"],
    "development_differentiation": [r"Cumulus", r"Zona Pellucida", r"[Zz]ygotic", r"[Dd]evelopment", r"[Dd]ifferentiation", r"Formation of", r"specification",
                                    r"Specification", r"Gastrulation", r"Somitogenesis", r"Myogenesis", r"Adipogenesis",
                                    r"Osteoblast", r"Osteoclast", r"Chondrocyte", r"Hematopoiesis", r"Erythropoiesis",
                                    r"Spermatogenesis", r"Meiosis", r"Meiotic", r"Fertilization", r"Sperm", r"Oocyte",
                                    r"Reproduction", r"Pancreatic beta", r"Beta-cell", r"Regulation of beta-cell",
                                    r"Keratinocyte", r"Cardiogenesis", r"Kidney", r"Nephron", r"Intestine",
                                    r"Lung", r"Liver", r"mesoderm", r"endoderm", r"ectoderm", r"[Ll]ineage",
                                    r"Transcriptional regulation of granulopoiesis", r"Mesenchymal", r"Placenta",
                                    r"Pluripotency", r"Stem cell", r"Ossification", r"Tooth", r"Limb", r"Eye",
                                    r"Lens", r"Hair", r"Skin", r"Organogenesis"],
    "metabolism_other": [r"2-hydroxyglutarate", r"Glucokinase", r"ornithine decarboxylase", r"COX reactions", r"Digestion", r"[Aa]bsorption", r"Metabolism", r"metabolism", r"[Bb]iosynthesis", r"[Ss]ynthesis of", r"[Dd]egradation",
                         r"[Cc]atabolism", r"[Ss]alvage", r"Vitamin", r"Cofactor", r"cofactor", r"NAD", r"FAD",
                         r"Ubiquinone", r"Tetrahydrobiopterin", r"Molybdenum", r"Metabolic", r"Integration of energy",
                         r"Insulin secretion", r"Ethanol", r"Glyoxylate", r"Oxalate", r"Sulfur", r"Pterin",
                         r"Biogenic amines", r"Catecholamine", r"Melatonin", r"Thyroid hormone", r"Hormone",
                         r"Phosphorylation of", r"Pyrophosphate", r"pyrophosphates", r"Inositol", r"Choline",
                         r"Carnitine", r"Glutathione", r"Bile", r"Porphyrin", r"Vitamin A", r"Retinal"],
    "cellular_responses": [r"Cellular responses", r"[Ss]tress", r"Response to", r"Starvation", r"Circadian",
                           r"Signal Transduction", r"Signaling by", r"Intracellular signaling", r"Second messenger",
                           r"Immune System", r"Programmed Cell Death", r"Cell Cycle", r"Cellular Senescence"],
}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def flatten(nested: dict[str, dict[str, list[str]]], sep: str = ":") -> dict[str, list[str]]:
    """{'cat': {'prog': genes}} -> {'cat:prog': genes} (deduplicated, order kept)."""
    return {f"{c}{sep}{p}": list(dict.fromkeys(g)) for c, d in nested.items() for p, g in d.items()}


def program_category(program: str) -> str | None:
    for c, d in PROGRAMS.items():
        if program in d:
            return c
    return None


def load_gmt(path: str) -> dict[str, list[str]]:
    """Read a GMT (e.g. ReactomePathways.gmt). Returns {pathway_name: genes}."""
    out = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 2:
                out[parts[0]] = [g for g in parts[2:] if g]
    return out


_WHOLE_WORD = {"RET", "ARE", "ATR", "ATM", "BAD", "BID", "SOS", "DAI", "IDO", "HAT", "TTP", "NMD", "EMT",
               "PAK", "LIF", "ABL", "TAP", "KIR", "NK", "DCC", "DDR", "TSC", "ALK", "LTK", "BCR", "GAG", "MMP",
               "LAT", "SRC", "LPL", "MET", "ESR", "AR"}


def _compile(rules: dict[str, list[str]]) -> dict[str, list[re.Pattern]]:
    """Patterns are case-insensitive, except mixed-case tokens (IgA, FcR; whole word) and bare upper-case gene-like tokens (e.g. 'NOTCH', 'ATR'),
    which are case-sensitive with a leading word boundary (so NOTCH matches NOTCH1) and, for the
    ambiguous ones in _WHOLE_WORD, a trailing boundary as well."""
    out = {}
    for k, rs in rules.items():
        cs = []
        for r in rs:
            if re.fullmatch(r"[A-Z][A-Z0-9\-]{1,6}", r):
                cs.append(re.compile(rf"\b{r}\b" if r in _WHOLE_WORD else rf"\b{r}"))
            elif re.fullmatch(r"[A-Za-z][A-Za-z0-9]{1,5}", r) and sum(ch.isupper() for ch in r) >= 2:
                cs.append(re.compile(rf"\b{r}\b"))          # mixed-case tokens (IgA, FcR, BMAL)
            else:
                cs.append(re.compile(r, re.I))
        out[k] = cs
    return out


def load_reactome_tsv(path: str, species: str = "Homo sapiens") -> pd.DataFrame:
    """Reactome pathway list (reactome_id, pathway, species), e.g. ReactomePathways.txt."""
    df = pd.read_csv(path, sep="\t")
    df.columns = ["reactome_id", "pathway", "species"][: df.shape[1]]
    return df[df.species == species].reset_index(drop=True) if species else df


def classify_reactome(pathways, rules: dict[str, list[str]] = None,
                      exclude: dict[str, list[str]] = None,
                      context: dict[str, list[str]] = None) -> pd.DataFrame:
    """Assign every Reactome pathway a label.

    pathways : GMT dict {name: genes}, a list of names, or a DataFrame with a 'pathway' column.
    Order    : EXCLUDE (disease variants / pathogens / drugs) -> programs (multi-label)
               -> CONTEXT classes -> 'unclassified'.
    Returns a long table: pathway, [reactome_id], n_genes, level ('exclude'|'program'|'context'|
    'unclassified'), label, category.
    """
    rules = REACTOME_RULES if rules is None else rules
    exclude = EXCLUDE_RULES if exclude is None else exclude
    context = CONTEXT_RULES if context is None else context
    if isinstance(pathways, dict):
        items = [(n, None, len(g)) for n, g in pathways.items()]
    elif isinstance(pathways, pd.DataFrame):
        items = [(n, i, np.nan) for n, i in zip(pathways.pathway, pathways.get("reactome_id", [None] * len(pathways)))]
    else:
        items = [(n, None, np.nan) for n in pathways]
    cx, cr, cc = _compile(exclude), _compile(rules), _compile(context)
    hit = lambda comp, name: [k for k, rs in comp.items() if any(r.search(name) for r in rs)]
    rows = []
    for name, rid, n in items:
        name = str(name).strip().strip('"')
        ex = hit(cx, name)
        if ex:
            labs, level = ex[:1], "exclude"
        else:
            labs, level = hit(cr, name), "program"
            if not labs:
                labs, level = hit(cc, name)[:1], "context"
            if not labs:
                labs, level = ["unclassified"], "unclassified"
        for lab in labs:
            rows.append({"pathway": name, "reactome_id": rid, "n_genes": n, "level": level, "label": lab,
                         "category": program_category(lab) if level == "program" else level})
    return pd.DataFrame(rows)


def summarize_reactome(cls: pd.DataFrame) -> pd.DataFrame:
    """Pathway counts per program / class (a pathway can feed several programs)."""
    g = cls.groupby(["level", "category", "label"]).pathway.nunique().rename("n_pathways").reset_index()
    return g.sort_values(["level", "n_pathways"], ascending=[True, False]).reset_index(drop=True)


def expand_with_reactome(gmt: dict[str, list[str]],
                         programs: dict[str, dict[str, list[str]]] = PROGRAMS,
                         rules: dict[str, list[str]] = REACTOME_RULES,
                         max_pathway_size: int = 300,
                         min_support: int = 1,
                         ) -> tuple[dict[str, dict[str, list[str]]], pd.DataFrame]:
    """Build 'most complete' gene sets = core genes U genes of matched Reactome pathways.

    max_pathway_size : skip very broad pathways (e.g. 'Signal Transduction').
    min_support      : a Reactome gene is added only if it appears in >= this many
                       matched pathways (2 gives tighter, more specific sets).
    Returns (expanded nested dict, provenance table).
    """
    cls = classify_reactome(gmt, rules)
    cls = cls[(cls.level == "program") & (cls.n_genes <= max_pathway_size)].rename(columns={"label": "program"})
    expanded = {c: {} for c in programs}
    prov = []
    for cat, d in programs.items():
        for prog, core in d.items():
            pws = cls.loc[cls.program == prog, "pathway"].tolist()
            counts = defaultdict(int)
            for pw in pws:
                for g in set(gmt[pw]):
                    counts[g] += 1
            extra = sorted(g for g, n in counts.items() if n >= min_support and g not in core)
            expanded[cat][prog] = list(dict.fromkeys(core + extra))
            prov.append({"category": cat, "program": prog, "n_core": len(core),
                         "n_reactome_pathways": len(pws), "n_added": len(extra),
                         "n_total": len(expanded[cat][prog]), "pathways": "; ".join(pws)})
    return expanded, pd.DataFrame(prov)


# --------------------------------------------------------------------------- #
# Human -> mouse symbols                                                      #
# --------------------------------------------------------------------------- #
# Genes whose mouse ortholog is not simply the capitalised human symbol.
# None = no 1:1 mouse ortholog (dropped).
HUMAN2MOUSE_EXCEPTIONS: dict[str, str | list[str] | None] = {
    "HLA-A": ["H2-K1", "H2-D1"], "HLA-B": ["H2-K1", "H2-D1"], "HLA-C": ["H2-K1", "H2-D1"],
    "HLA-E": "H2-T23", "HLA-DRA": "H2-Ea", "HLA-DRB1": "H2-Eb1", "HLA-DQA1": "H2-Aa",
    "HLA-DQB1": "H2-Ab1", "HLA-DPA1": None, "HLA-DPB1": None, "HLA-DMA": "H2-DMa",
    "HLA-DMB": ["H2-DMb1", "H2-DMb2"], "CD74": "Cd74",
    "CXCL8": None, "GZMH": None, "GNLY": None, "IFNA1": "Ifna1", "MT2A": "Mt2", "C4A": "C4b", "C4B": "C4b",
    "FCGR3A": "Fcgr4", "FCGR1A": "Fcgr1", "FCGR2A": "Fcgr2b", "IGHG1": "Ighg1", "IGHA1": "Igha",
    "ATP5F1A": "Atp5a1", "ATP5F1B": "Atp5b", "GBA1": "Gba", "CASP4": "Casp4", "CASP5": None, "CASP10": None,
    "KIR2DL1": None, "CCL3": "Ccl3", "CCL4": "Ccl4", "SAA3": "Saa3", "NOS2": "Nos2", "TNFRSF10A": None,
    "TNFRSF10B": "Tnfrsf10b", "MMP1": ["Mmp1a", "Mmp1b"], "CEACAM6": None, "CXCL1": "Cxcl1", "CXCL5": "Cxcl5",
    "IFI6": None, "IFI44L": "Ifi44l", "OAS1": ["Oas1a", "Oas1g"], "OAS2": "Oas2", "MX1": "Mx1", "MX2": "Mx2",
    "GBP1": "Gbp2", "GBP5": "Gbp5", "SPRR1B": "Sprr1b", "FDCSP": None, "TFF1": "Tff1",
    "S100A8": "S100a8", "S100A9": "S100a9", "NCR1": "Ncr1", "KLRK1": "Klrk1", "KLRD1": "Klrd1",
    "ALOX15": "Alox15", "ALOX12": "Alox12", "DIABLO": "Diablo", "ACKR1": "Ackr1", "SERPINA3": "Serpina3n",
}


def human_to_mouse(genes, exceptions=HUMAN2MOUSE_EXCEPTIONS) -> list[str]:
    out = []
    for g in genes:
        m = exceptions.get(g, g[:1] + g[1:].lower()) if g in exceptions else g[:1] + g[1:].lower()
        if m is None:
            continue
        out.extend(m if isinstance(m, list) else [m])
    return list(dict.fromkeys(out))


def to_mouse(nested: dict[str, dict[str, list[str]]], var_names=None,
             report: bool = True) -> dict[str, dict[str, list[str]]]:
    """
    Convert a nested program dict to mouse symbols. 
    If var_names is given, keep only genes present in the data (and report per-program coverage).
    """
    present = set(var_names) if var_names is not None else None
    out, rows = {}, []
    for c, d in nested.items():
        out[c] = {}
        for p, g in d.items():
            m = human_to_mouse(g)
            kept = [x for x in m if present is None or x in present]
            out[c][p] = kept
            rows.append((c, p, len(g), len(m), len(kept)))
    if report and present is not None:
        rep = pd.DataFrame(rows, columns=["category", "program", "n_human", "n_mouse", "n_in_data"])
        low = rep[rep.n_in_data < 0.5 * rep.n_human]
        if len(low):
            print("programs with <50% genes found in data:\n", low.to_string(index=False))
    return out
