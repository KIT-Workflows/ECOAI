"""
ECOAI Supplementary Material Generator
=======================================
Reads curated data from the ECOAI pipeline and produces supplementary tables (CSV) and a formatted Markdown document.

Outputs are written into the sibling `tables/` directory.

Usage:
    python generate_supplementary.py          # run from supp_material/
    python supp_material/generate_supplementary.py   # run from repo root
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ── Resolve paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "Datasets" / "data"
PHASE6_DIR = REPO_ROOT / "experiment" / "phase6_interpretability"
PHASE10_SHAP_DIR = DATA_DIR / "phase10_artifacts" / "shap"
PHASE11_DIR = REPO_ROOT / "experiment" / "phase11_dual_virtual_screening" / "artifacts"

TABLES_DIR = SCRIPT_DIR / "tables"
TABLES_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("  ECOAI Supplementary Material Generator")
print("=" * 70)
print(f"  Repo root  : {REPO_ROOT}")
print(f"  Output dir : {TABLES_DIR}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD CORE DATA
# ═══════════════════════════════════════════════════════════════════════════
print("[1/8] Loading core datasets...")

curated = pd.read_parquet(DATA_DIR / "curated_molecules.parquet")
predictions = pd.read_parquet(DATA_DIR / "model_predictions.parquet")
features_2d = pd.read_parquet(DATA_DIR / "features_rdkit_2d.parquet")
screening = pd.read_parquet(DATA_DIR / "screening_results.parquet")
insect_labels = pd.read_parquet(DATA_DIR / "insecticide_labels.parquet")
insect_preds = pd.read_parquet(DATA_DIR / "phase10_artifacts" / "insecticidal_predictions.parquet")
dual_screen = pd.read_parquet(PHASE11_DIR / "full_library_predictions.parquet")

# Merge labeled compounds with curated metadata
labeled = predictions.merge(
    curated, on="canonical_smiles", how="left", suffixes=("", "_cur")
)

print(f"  Curated molecules     : {len(curated):,}")
print(f"  Labeled (repellency)  : {len(labeled):,}")
print(f"  Insecticide labels    : {len(insect_labels):,}")
print(f"  Screening library     : {len(screening):,}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# 2. TABLE S1 — SHAP Feature Importance (Repellency Model)
# ═══════════════════════════════════════════════════════════════════════════
print("[2/8] Generating Table S1: SHAP — Repellency (XGBoost)...")

shap_rep = pd.read_csv(PHASE6_DIR / "shap_feature_importance.csv")
top30_rep = shap_rep.head(30).copy()

# Calculate class-conditional means
df_combined = pd.read_parquet(DATA_DIR / "features_combined.parquet")
df_rep_labeled = df_combined[
    (df_combined["qc_status"] == "pass") & 
    (df_combined["repellent_active"].notna())
].copy()

rep_means, nonrep_means = [], []
for feat in top30_rep["feature"]:
    if feat in df_rep_labeled.columns:
        rep_means.append(df_rep_labeled[df_rep_labeled["repellent_active"] == 1.0][feat].mean())
        nonrep_means.append(df_rep_labeled[df_rep_labeled["repellent_active"] == 0.0][feat].mean())
    else:
        rep_means.append(np.nan)
        nonrep_means.append(np.nan)

top30_rep["type"] = top30_rep["feature"].apply(
    lambda x: "Morgan FP Bit" if x.startswith("mfp_") else "RDKit 2D"
)
top30_rep["repellent_mean"] = rep_means
top30_rep["non_repellent_mean"] = nonrep_means
top30_rep["delta"] = top30_rep["repellent_mean"] - top30_rep["non_repellent_mean"]
top30_rep.insert(0, "rank", range(1, 31))

top30_rep.to_csv(TABLES_DIR / "Table_S1_SHAP_repellency_top30.csv", index=False)
print(f"  -> {len(top30_rep)} features written.")


# ═══════════════════════════════════════════════════════════════════════════
# 3. TABLE S2 — SHAP Feature Importance (Insecticidal Model)
# ═══════════════════════════════════════════════════════════════════════════
print("[3/8] Generating Table S2: SHAP — Insecticidal (RandomForest)...")

import json
shap_ins = pd.read_csv(PHASE10_SHAP_DIR / "shap_feature_importance.csv")
top30_ins = shap_ins.head(30).copy()

# Calculate class-conditional means
npz_ins = np.load(DATA_DIR / "phase10_artifacts" / "insecticide_features.npz")
X_ins = npz_ins["X"]
df_ins_meta = pd.read_parquet(DATA_DIR / "phase10_artifacts" / "insecticide_meta.parquet")

with open(DATA_DIR / "phase10_artifacts" / "feature_columns.json", "r") as f:
    feat_meta_ins = json.load(f)
feat_cols_ins = feat_meta_ins["all_cols"]

df_ins_feat = pd.DataFrame(X_ins, columns=feat_cols_ins)
df_ins_feat["insecticidal_active"] = df_ins_meta["insecticidal_active"].values

ins_active_means, ins_inactive_means = [], []
for feat in top30_ins["feature"]:
    if feat in df_ins_feat.columns:
        ins_active_means.append(df_ins_feat[df_ins_feat["insecticidal_active"] == 1.0][feat].mean())
        ins_inactive_means.append(df_ins_feat[df_ins_feat["insecticidal_active"] == 0.0][feat].mean())
    else:
        ins_active_means.append(np.nan)
        ins_inactive_means.append(np.nan)

top30_ins["type"] = top30_ins["feature"].apply(
    lambda x: "Morgan FP Bit" if x.startswith("mfp_") else "RDKit 2D"
)
top30_ins["active_mean"] = ins_active_means
top30_ins["inactive_mean"] = ins_inactive_means
top30_ins["delta"] = top30_ins["active_mean"] - top30_ins["inactive_mean"]
top30_ins.insert(0, "rank", range(1, 31))

top30_ins.to_csv(TABLES_DIR / "Table_S2_SHAP_insecticidal_top30.csv", index=False)
print(f"  -> {len(top30_ins)} features written.")


# ═══════════════════════════════════════════════════════════════════════════
# 4. TABLE S3 — Repellent-Active Compounds
# ═══════════════════════════════════════════════════════════════════════════
print("[4/8] Generating Table S3: Repellent-active compounds...")

rep_compounds = labeled[labeled["repellent_active"] == 1.0].sort_values("compound_id")
rep_out = rep_compounds[[
    "compound_id", "canonical_smiles", "inchikey", "mol_formula",
    "mol_weight", "p_repellent", "ood_score", "target_species", "sdf_PubChem CID"
]].copy()
rep_out.columns = [
    "Compound_ID", "SMILES", "InChIKey", "Molecular_Formula",
    "MW_Da", "p_repellent", "OOD_Score", "Target_Species", "PubChem_CID"
]
rep_out.to_csv(TABLES_DIR / "Table_S3_repellent_compounds.csv", index=False)
print(f"  -> {len(rep_out)} repellent compounds written.")


# ═══════════════════════════════════════════════════════════════════════════
# 5. TABLE S4 — Non-Repellent (Decoy) Compounds
# ═══════════════════════════════════════════════════════════════════════════
print("[5/8] Generating Table S4: Non-repellent (decoy) compounds...")

dec_compounds = labeled[labeled["repellent_active"] == 0.0].sort_values("compound_id")
dec_out = dec_compounds[[
    "compound_id", "mol_name", "canonical_smiles", "inchikey", "mol_formula",
    "mol_weight", "p_repellent", "ood_score"
]].copy()
dec_out.columns = [
    "Compound_ID", "Name", "SMILES", "InChIKey", "Molecular_Formula",
    "MW_Da", "p_repellent", "OOD_Score"
]
dec_out.to_csv(TABLES_DIR / "Table_S4_decoy_compounds.csv", index=False)
print(f"  -> {len(dec_out)} decoy compounds written.")


# ═══════════════════════════════════════════════════════════════════════════
# 6. TABLE S5 — Insecticidal Compounds (ChEMBL)
# ═══════════════════════════════════════════════════════════════════════════
print("[6/8] Generating Table S5: Insecticidal compounds (ChEMBL)...")

ins_out = insect_labels[[
    "compound_id", "canonical_smiles", "inchikey", "insecticidal_active",
    "molecule_chembl_id", "n_assay_records", "best_pchembl", "median_pchembl",
    "target_organisms", "source"
]].copy().sort_values("compound_id")
ins_out.columns = [
    "Compound_ID", "SMILES", "InChIKey", "Insecticidal_Active",
    "ChEMBL_ID", "N_Assay_Records", "Best_pChEMBL", "Median_pChEMBL",
    "Target_Organisms", "Source"
]
ins_out.to_csv(TABLES_DIR / "Table_S5_insecticidal_compounds.csv", index=False)
print(f"  -> {len(ins_out)} insecticidal compounds written.")


# ═══════════════════════════════════════════════════════════════════════════
# 7. TABLE S6 — Dual Screening Results (Holy Grails + Categories)
# ═══════════════════════════════════════════════════════════════════════════
print("[7/8] Generating Table S6: Dual screening results...")

dual_out = dual_screen.sort_values("p_repellent", ascending=False).copy()
dual_out.to_csv(TABLES_DIR / "Table_S6_dual_screening_results.csv", index=False)

# Also save holy grails separately
holy = dual_screen[dual_screen["Category"] == "Holy Grail (Safe Repellent)"].copy()
holy.to_csv(TABLES_DIR / "Table_S6a_holy_grail_compounds.csv", index=False)

print(f"  -> {len(dual_out)} screened compounds written.")
print(f"  -> {len(holy)} holy grail compounds written.")


# ═══════════════════════════════════════════════════════════════════════════
# 8. TABLE S7 — Dataset Summary Statistics
# ═══════════════════════════════════════════════════════════════════════════
print("[8/8] Generating Table S7: Summary statistics...")

rep_df = labeled[labeled["repellent_active"] == 1.0]
dec_df = labeled[labeled["repellent_active"] == 0.0]

# Define continuous features metadata for summary table
feat_labeled = features_2d[features_2d["compound_id"].isin(labeled["compound_id"])].copy()
feat_labeled = feat_labeled.merge(
    labeled[["compound_id", "repellent_active"]], on="compound_id"
)

# Build summary rows
rows = []

def add_stat(name, rep_vals, dec_vals, all_vals):
    rows.append({
        "Property": name,
        "Repellent_Mean": rep_vals.mean(),
        "Repellent_SD": rep_vals.std(),
        "NonRepellent_Mean": dec_vals.mean(),
        "NonRepellent_SD": dec_vals.std(),
        "Overall_Mean": all_vals.mean(),
        "Overall_SD": all_vals.std(),
    })

add_stat("Molecular Weight (Da)", rep_df["mol_weight"], dec_df["mol_weight"], labeled["mol_weight"])

# Top descriptor stats
for feat in top30_rep[top30_rep["type"] == "RDKit 2D"]["feature"]:
    if feat in feat_labeled.columns:
        add_stat(
            feat,
            feat_labeled[feat_labeled["repellent_active"] == 1.0][feat],
            feat_labeled[feat_labeled["repellent_active"] == 0.0][feat],
            feat_labeled[feat],
        )

stats_df = pd.DataFrame(rows)
stats_df.to_csv(TABLES_DIR / "Table_S7_repellency_summary_statistics.csv", index=False)
print(f"  -> {len(stats_df)} property rows written.")


# ═══════════════════════════════════════════════════════════════════════════
# 8a. TABLE S8 — Labeled Insecticidal Dataset Summary Statistics
# ═══════════════════════════════════════════════════════════════════════════
print("[8a/8] Generating Table S8: Insecticidal Summary statistics...")

ins_act_df = df_ins_feat[df_ins_feat["insecticidal_active"] == 1.0]
ins_inact_df = df_ins_feat[df_ins_feat["insecticidal_active"] == 0.0]

rows_ins = []
def add_stat_ins(name, act_vals, inact_vals, all_vals):
    rows_ins.append({
        "Property": name,
        "Active_Mean": act_vals.mean(),
        "Active_SD": act_vals.std(),
        "Inactive_Mean": inact_vals.mean(),
        "Inactive_SD": inact_vals.std(),
        "Overall_Mean": all_vals.mean(),
        "Overall_SD": all_vals.std(),
    })

# Add first row: Molecular Weight
add_stat_ins("Molecular Weight (Da)", ins_act_df["MolWt"], ins_inact_df["MolWt"], df_ins_feat["MolWt"])

# Top descriptor stats from Table S2 RDKit 2D
for feat in top30_ins[top30_ins["type"] == "RDKit 2D"]["feature"]:
    if feat in df_ins_feat.columns:
        add_stat_ins(
            feat,
            ins_act_df[feat],
            ins_inact_df[feat],
            df_ins_feat[feat]
        )

stats_ins_df = pd.DataFrame(rows_ins)
stats_ins_df.to_csv(TABLES_DIR / "Table_S8_insecticide_summary_statistics.csv", index=False)
print(f"  -> {len(stats_ins_df)} insecticide property rows written.")


# ═══════════════════════════════════════════════════════════════════════════
# 9. GENERATE FORMATTED MARKDOWN DOCUMENT
# ═══════════════════════════════════════════════════════════════════════════
print()
print("Generating supplementary_material.md ...")

lines = []
lines.append("# Supplementary Material")
lines.append("")
lines.append("## ECOAI: AI-Driven Discovery of Eco-Friendly Insect Repellent Compounds")
lines.append("")
lines.append("---")
lines.append("")

# ── S1: Repellency SHAP ──
lines.append("## Table S1. Top 30 SHAP Features — Repellency Model (XGBoost)")
lines.append("")
lines.append("Mean absolute SHAP values from TreeExplainer applied to the winning XGBoost "
             "repellency classifier. Of the top 30 features, **27 are RDKit 2D physicochemical "
             "descriptors** and **3 are Morgan fingerprint (ECFP4) structural bits**.")
lines.append("")
lines.append("### Figure S1: SHAP-Weighted Diverging Lollipop Plot — Repellency Model")
lines.append("")
lines.append("![Repellency SHAP averages](plots/xgb_shap_averages.png)")
lines.append("")
lines.append("**Figure Caption**: Top 30 features ranked by mean absolute SHAP importance. "
             "Marker area represents absolute SHAP importance (exact value in parentheses). "
             "Marker color represents descriptor class: RDKit 2D physical descriptors (🔵) vs. Morgan fingerprint bits (🟡). "
             "Horizontal position represents the Standardized Mean Difference (\\Delta Z-score) computed globally over the labeled dataset. "
             "Green stems extending rightward denote positive enrichment in repellent-active compounds (\\Delta > 0), "
             "while red stems extending leftward denote negative enrichment (\\Delta < 0).")
lines.append("")
lines.append("### Table S1 Data Averages")
lines.append("")
lines.append("| Rank | Feature | Type | Mean \\|SHAP\\| | Rep. Mean | Non-Rep. Mean | Δ |")
lines.append("|------|---------|------|-------------|-----------|---------------|---|")
for _, r in top30_rep.iterrows():
    rm = f"{r['repellent_mean']:.4f}" if pd.notna(r['repellent_mean']) else "—"
    nm = f"{r['non_repellent_mean']:.4f}" if pd.notna(r['non_repellent_mean']) else "—"
    d = f"{r['delta']:+.4f}" if pd.notna(r['delta']) else "—"
    lines.append(f"| {r['rank']} | {r['feature']} | {r['type']} | {r['mean_abs_shap']:.4f} | {rm} | {nm} | {d} |")
lines.append("")

# ── S2: Insecticidal SHAP ──
lines.append("## Table S2. Top 30 SHAP Features — Insecticidal Model (RandomForest)")
lines.append("")
lines.append("Mean absolute SHAP values from TreeExplainer applied to the winning RandomForest "
             "insecticidal classifier trained on 891 ChEMBL-derived compounds.")
lines.append("")
lines.append("### Figure S2: SHAP-Weighted Diverging Lollipop Plot — Insecticidal Model")
lines.append("")
lines.append("![Insecticidal SHAP averages](plots/rf_shap_averages.png)")
lines.append("")
lines.append("**Figure Caption**: Top 30 features ranked by mean absolute SHAP importance. "
             "Marker area represents absolute SHAP importance (exact value in parentheses). "
             "Marker color represents descriptor class: RDKit 2D physical descriptors (🔵) vs. Morgan fingerprint bits (🟡). "
             "Horizontal position represents the Standardized Mean Difference (\\Delta Z-score) computed globally over the insecticidal dataset. "
             "Green stems extending rightward denote positive enrichment in active insecticidal compounds (\\Delta > 0), "
             "while red stems extending leftward denote negative enrichment (\\Delta < 0).")
lines.append("")
lines.append("### Table S2 Data Averages")
lines.append("")
lines.append("| Rank | Feature | Type | Mean \\|SHAP\\| | Active Mean | Inactive Mean | Δ |")
lines.append("|------|---------|------|-------------|-------------|---------------|---|")
for _, r in top30_ins.iterrows():
    am = f"{r['active_mean']:.4f}" if pd.notna(r['active_mean']) else "—"
    im = f"{r['inactive_mean']:.4f}" if pd.notna(r['inactive_mean']) else "—"
    d = f"{r['delta']:+.4f}" if pd.notna(r['delta']) else "—"
    lines.append(f"| {r['rank']} | {r['feature']} | {r['type']} | {r['mean_abs_shap']:.4f} | {am} | {im} | {d} |")
lines.append("")

# ── S3/S4: Compound lists (reference to CSV) ──
lines.append("## Tables S3–S5. Full Compound Lists")
lines.append("")
lines.append("Complete compound-level data is provided as CSV files in the `tables/` directory:")
lines.append("")
lines.append("| Table | File | Records | Description |")
lines.append("|-------|------|---------|-------------|")
lines.append(f"| S3 | `Table_S3_repellent_compounds.csv` | {len(rep_out)} | "
             "Repellent-active compounds (*Aedes aegypti*) with calibrated probabilities |")
lines.append(f"| S4 | `Table_S4_decoy_compounds.csv` | {len(dec_out)} | "
             "Non-repellent (decoy/negative control) compounds |")
lines.append(f"| S5 | `Table_S5_insecticidal_compounds.csv` | {len(ins_out)} | "
             "ChEMBL-derived insecticidal compounds with bioactivity metadata |")
lines.append(f"| S6 | `Table_S6_dual_screening_results.csv` | {len(dual_out)} | "
             "Dual virtual screening predictions (repellency + toxicity) |")
lines.append("")
lines.append("### Figure S3: Physicochemical Property Space Distributions — Size, Lipophilicity, Polarity, and Shape")
lines.append("")
lines.append("![Physicochemical Property Space Distributions](plots/xgb_rf_chemical_space.png)")
lines.append("")
lines.append("**Figure Caption**: Distribution of key drug-like physicochemical properties across the repellency (n = 732) and insecticidal (n = 891) datasets. Split violin plots compare the distributions of Molecular Weight (size), MolLogP (lipophilicity), Topological Polar Surface Area (polarity), and Kappa2 (flexibility/shape) between active (🔵, 🟢) and inactive/decoy (🟡, 🔴) compounds. Black horizontal lines denote sample means (exact numeric values are listed in Table S7 for repellency, and in Table S8 for insecticidal descriptors).")
lines.append("")

# ── S7: Summary Stats ──
lines.append("## Table S7. Summary Statistics of the Labeled Repellency Dataset")
lines.append("")
lines.append(f"| Property | Repellent (n={len(rep_df)}) | Non-Repellent (n={len(dec_df)}) | Overall (n={len(labeled)}) |")
lines.append("|----------|---------------------------|-------------------------------|--------------------------|")
for _, r in stats_df.iterrows():
    lines.append(f"| {r['Property']} | {r['Repellent_Mean']:.4f} ± {r['Repellent_SD']:.4f} "
                 f"| {r['NonRepellent_Mean']:.4f} ± {r['NonRepellent_SD']:.4f} "
                 f"| {r['Overall_Mean']:.4f} ± {r['Overall_SD']:.4f} |")
lines.append("")

# ── S8: Insecticidal Summary Stats ──
lines.append("## Table S8. Summary Statistics of the Labeled Insecticidal Dataset")
lines.append("")
lines.append(f"| Property | Insecticidal Active (n={len(ins_act_df)}) | Insecticidal Inactive (n={len(ins_inact_df)}) | Overall (n={len(df_ins_feat)}) |")
lines.append("|----------|---------------------------|-------------------------------|--------------------------|")
for _, r in stats_ins_df.iterrows():
    lines.append(f"| {r['Property']} | {r['Active_Mean']:.4f} ± {r['Active_SD']:.4f} "
                 f"| {r['Inactive_Mean']:.4f} ± {r['Inactive_SD']:.4f} "
                 f"| {r['Overall_Mean']:.4f} ± {r['Overall_SD']:.4f} |")
lines.append("")

# ── Holy Grails ──
lines.append("## Table S6a. Holy Grail Compounds — Safe Repellents")
lines.append("")
lines.append("Three compounds from the LifeChemicals screening library (n = 2,880) "
             "classified as highly repellent yet non-insecticidal.")
lines.append("")
lines.append("| Compound ID | p_repellent | p_insecticidal | SMILES |")
lines.append("|-------------|-------------|----------------|--------|")
for _, r in holy.iterrows():
    lines.append(f"| {r['compound_id']} | {r['p_repellent']:.4f} | {r['p_insecticidal']:.4f} | `{r['canonical_smiles']}` |")
lines.append("")

# ── Dual screening summary ──
lines.append("## Dual Virtual Screening Category Summary")
lines.append("")
cat_counts = dual_screen["Category"].value_counts()
lines.append("| Category | Count |")
lines.append("|----------|-------|")
for cat, cnt in cat_counts.items():
    lines.append(f"| {cat} | {cnt} |")
lines.append("")
lines.append("---")
lines.append("")
lines.append("*Generated programmatically from ECOAI pipeline data by `generate_supplementary.py`.*")
lines.append("")

md_path = SCRIPT_DIR / "supplementary_material.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"  -> {len(lines)} lines written to {md_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
all_files = sorted(TABLES_DIR.glob("*"))
print(f"  Generated {len(all_files)} table files + 1 Markdown document:")
for f in all_files:
    size_kb = f.stat().st_size / 1024
    print(f"    tables/{f.name}  ({size_kb:.1f} KB)")
print(f"    supplementary_material.md")
print("=" * 70)
print("  DONE.")
