# Supplementary Material — ECOAI

This directory contains supplementary data for the ECOAI study, covering **both the Repellency model (XGBoost) and the Insecticidal model (RandomForest)**.

---

## Directory Structure

```
supp_material/
├── README.md                              ← This file
├── generate_supplementary.py              ← Master script (reads pipeline data, generates all outputs)
├── supplementary_material.md              ← Formatted supplementary document (Markdown)
├── plots/                                 ← High-resolution figures folder
│   ├── xgb_shap_averages.png              ← Figure S1: Repellency SHAP Lollipop Plot
│   ├── rf_shap_averages.png               ← Figure S2: Insecticidal SHAP Lollipop Plot
│   └── xgb_rf_chemical_space.png          ← Figure S3: Physicochemical Space Violins
└── tables/
    ├── Table_S1_SHAP_repellency_top30.csv     ← SHAP importance, repellency XGBoost
    ├── Table_S2_SHAP_insecticidal_top30.csv   ← SHAP importance, insecticidal RandomForest
    ├── Table_S3_repellent_compounds.csv       ← 373 repellent-active compounds
    ├── Table_S4_decoy_compounds.csv           ← 359 non-repellent (decoy) compounds
    ├── Table_S5_insecticidal_compounds.csv    ← 891 ChEMBL insecticidal compounds
    ├── Table_S6_dual_screening_results.csv    ← 2,880 dual-screened library compounds
    ├── Table_S6a_holy_grail_compounds.csv     ← 3 Holy Grail safe-repellent compounds
    ├── Table_S7_repellency_summary_statistics.csv  ← Descriptor statistics (repellent vs decoy)
    └── Table_S8_insecticide_summary_statistics.csv ← Descriptor statistics (active vs inactive insecticidal)
```

---

## Table Descriptions

### SHAP Analysis

| Table | Model | Algorithm | N Features | Source Phase |
|-------|-------|-----------|------------|--------------|
| **S1** | Repellency | XGBoost | Top 30 (27 RDKit 2D + 3 Morgan FP) | Phase 6 |
| **S2** | Insecticidal Toxicity | RandomForest | Top 30 | Phase 10 |

Both tables include rank, feature name, descriptor type, and mean |SHAP| value.
Table S1 additionally includes class-conditional means (repellent vs non-repellent) and their difference (Δ).

### Compound Lists

| Table | Dataset | N Compounds | Key Columns |
|-------|---------|-------------|-------------|
| **S3** | Repellent-active | 373 | Compound_ID, SMILES, InChIKey, Molecular_Formula, MW_Da, p_repellent, OOD_Score, Target_Species, PubChem_CID |
| **S4** | Non-repellent (decoy) | 359 | Compound_ID, Name, SMILES, InChIKey, Molecular_Formula, MW_Da, p_repellent, OOD_Score |
| **S5** | Insecticidal (ChEMBL) | 891 | Compound_ID, SMILES, InChIKey, Insecticidal_Active, ChEMBL_ID, N_Assay_Records, Best_pChEMBL, Median_pChEMBL, Target_Organisms, Source |
| **S6** | Dual screening library | 2,880 | compound_id, canonical_smiles, p_repellent, p_insecticidal, repel_lower_90, repel_upper_90, insect_lower_90, insect_upper_90, Category |
| **S6a** | Holy Grail compounds | 3 | compound_id, canonical_smiles, p_repellent, p_insecticidal, repel_lower_90, repel_upper_90, insect_lower_90, insect_upper_90, Category |

### Summary Statistics

| Table | Content |
|-------|---------|
| **S7** | Mean ± SD for molecular weight and top 27 RDKit 2D descriptors, split by repellent vs decoy class |
| **S8** | Mean ± SD for molecular weight and top RDKit 2D descriptors, split by active vs inactive insecticidal class |

---

## Data Provenance

All tables are generated **directly** from the ECOAI pipeline output files

| Source File | Content |
|-------------|---------|
| `Datasets/data/curated_molecules.parquet` | Master molecular database (3,615 compounds) |
| `Datasets/data/model_predictions.parquet` | Calibrated repellency predictions (732 labeled) |
| `Datasets/data/features_rdkit_2d.parquet` | 54 RDKit 2D descriptors for all compounds |
| `Datasets/data/insecticide_labels.parquet` | ChEMBL-derived insecticidal labels (891 compounds) |
| `experiment/phase6_interpretability/shap_feature_importance.csv` | Repellency SHAP values |
| `Datasets/data/phase10_artifacts/shap/shap_feature_importance.csv` | Insecticidal SHAP values |
| `experiment/phase11_dual_virtual_screening/artifacts/full_library_predictions.parquet` | Dual screening results |

---

## Regeneration

To regenerate all outputs from current pipeline data:

```bash
python supp_material/generate_supplementary.py
```

The script is fully deterministic and will overwrite existing outputs.

---
