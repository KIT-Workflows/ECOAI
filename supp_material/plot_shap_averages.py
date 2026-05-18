"""
Plotting script for supplementary SHAP averages.
Generates lollipop plots for the repellency (XGBoost) and insecticidal (Random Forest) models.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# Resolve paths
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path(os.getcwd())

REPO_ROOT = SCRIPT_DIR
if SCRIPT_DIR.name == "supp_material":
    REPO_ROOT = SCRIPT_DIR.parent
elif (SCRIPT_DIR / "supp_material").exists():
    REPO_ROOT = SCRIPT_DIR
    SCRIPT_DIR = REPO_ROOT / "supp_material"

DATA_DIR = REPO_ROOT / "Datasets" / "data"
PHASE6_DIR = REPO_ROOT / "experiment" / "phase6_interpretability"
PHASE10_SHAP_DIR = DATA_DIR / "phase10_artifacts" / "shap"
SUPP_MATERIAL_DIR = REPO_ROOT / "supp_material"

# Load datasets
rep_csv = PHASE6_DIR / "shap_feature_importance.csv"
ins_csv = PHASE10_SHAP_DIR / "shap_feature_importance.csv"

if not rep_csv.exists():
    raise FileNotFoundError(f"Repellency SHAP file not found at: {rep_csv}")
if not ins_csv.exists():
    raise FileNotFoundError(f"Insecticidal SHAP file not found at: {ins_csv}")

df_rep_shap = pd.read_csv(rep_csv)
df_ins_shap = pd.read_csv(ins_csv)

# Load features and labels for repellency model
df_combined = pd.read_parquet(DATA_DIR / "features_combined.parquet")
df_rep_labeled = df_combined[
    (df_combined["qc_status"] == "pass") & 
    (df_combined["repellent_active"].notna())
].copy()

meta_cols_rep = ["compound_id", "canonical_smiles", "source_dataset", "repellent_active", "scaffold_smiles", "fold_id", "qc_status"]
feat_cols_rep = [c for c in df_combined.columns if c not in meta_cols_rep]

df_rep_feat = df_rep_labeled[feat_cols_rep].copy()
df_rep_feat = df_rep_feat.fillna(df_rep_feat.mean())
df_rep_feat_std = (df_rep_feat - df_rep_feat.mean()) / df_rep_feat.std()
df_rep_feat_std["repellent_active"] = df_rep_labeled["repellent_active"].values

# Load features and labels for insecticidal model
npz_ins = np.load(DATA_DIR / "phase10_artifacts" / "insecticide_features.npz")
X_ins = npz_ins["X"]
df_ins_meta = pd.read_parquet(DATA_DIR / "phase10_artifacts" / "insecticide_meta.parquet")

with open(DATA_DIR / "phase10_artifacts" / "feature_columns.json", "r") as f:
    feat_meta_ins = json.load(f)
feat_cols_ins = feat_meta_ins["all_cols"]

df_ins_feat = pd.DataFrame(X_ins, columns=feat_cols_ins)
df_ins_feat = df_ins_feat.fillna(df_ins_feat.mean())
df_ins_feat_std = (df_ins_feat - df_ins_feat.mean()) / df_ins_feat.std()
df_ins_feat_std["insecticidal_active"] = df_ins_meta["insecticidal_active"].values


def generate_diverging_lollipop_plot(df_shap, df_features_std, label_col, color_2d, color_mfp, output_path, active_label, inactive_label, max_features=30):
    # Select top features
    df_plot = df_shap.head(max_features).copy()
    df_plot = df_plot.iloc[::-1].reset_index(drop=True)
    
    # Calculate Standardized Mean Difference (SMD)
    smds = []
    for feat in df_plot["feature"]:
        if feat in df_features_std.columns:
            act_mean = df_features_std[df_features_std[label_col] == 1.0][feat].mean()
            inact_mean = df_features_std[df_features_std[label_col] == 0.0][feat].mean()
            smds.append(act_mean - inact_mean)
        else:
            smds.append(0.0)
            
    df_plot["smd"] = smds
    
    # Scale bubble markers based on SHAP value
    min_size, max_size = 300, 1800
    shap_min, shap_max = df_plot["mean_abs_shap"].min(), df_plot["mean_abs_shap"].max()
    if shap_max > shap_min:
        sizes = min_size + (df_plot["mean_abs_shap"] - shap_min) / (shap_max - shap_min) * (max_size - min_size)
    else:
        sizes = [1000] * len(df_plot)
        
    df_plot["bubble_size"] = sizes
    
    # Set bubble colors based on feature type
    bubble_colors = []
    for feat in df_plot["feature"]:
        if feat.startswith("mfp_"):
            bubble_colors.append(color_mfp)
        else:
            bubble_colors.append(color_2d)
            
    df_plot["bubble_color"] = bubble_colors
    
    # Configure Matplotlib styles
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "font.size": 18,
        "axes.edgecolor": "black",
        "axes.linewidth": 3.0,
        "xtick.color": "black",
        "ytick.color": "black"
    })
    
    smd_max = df_plot["smd"].abs().max()
    x_lim_val = max(smd_max * 1.35, 1.5)
    
    fig, ax = plt.subplots(figsize=(20.0, 28.0))
    
    # Centerline at zero
    ax.axvline(0, color="#000000", linestyle="--", alpha=0.5, linewidth=2.5, zorder=0)
    
    # Plot stems and markers
    for i in range(len(df_plot)):
        smd_val = df_plot.loc[i, "smd"]
        shap_val = df_plot.loc[i, "mean_abs_shap"]
        size = df_plot.loc[i, "bubble_size"]
        color = df_plot.loc[i, "bubble_color"]
        
        stem_color = "#00c853" if smd_val > 0 else "#d50000"
        
        # Guide lines
        ax.plot(
            [-x_lim_val, x_lim_val], 
            [i, i], 
            color="#b0b0b0", 
            linestyle=":", 
            linewidth=1.5, 
            alpha=0.6, 
            zorder=0
        )
        
        # Stem lines
        ax.plot(
            [0, smd_val], 
            [i, i], 
            color=stem_color, 
            linestyle="-", 
            linewidth=4.5, 
            alpha=0.8, 
            zorder=1
        )
        
        # Markers
        ax.scatter(
            smd_val, 
            i, 
            s=size, 
            color=color, 
            edgecolor="black", 
            linewidths=1.2, 
            alpha=0.95, 
            zorder=2
        )
        
        # Annotation text
        x_pad = 0.15
        ha = "left" if smd_val >= 0 else "right"
        x_pos = smd_val + x_pad if smd_val >= 0 else smd_val - x_pad
        
        sign = "+" if smd_val > 0 else ""
        label_text = f"{sign}{smd_val:.2f} (SHAP: {shap_val:.3f})"
        ax.text(
            x_pos, 
            i, 
            label_text, 
            va="center", 
            ha=ha, 
            fontsize=18.5, 
            fontweight="bold", 
            color="#000000",
            zorder=3
        )
        
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(
        df_plot["feature"], 
        fontsize=20.5, 
        fontweight="bold", 
        color="#000000"
    )
    
    ax.set_xlabel("Standardized Mean Difference (\u0394 Z-score)", fontsize=22.0, fontweight="bold", color="#000000")
    ax.set_xlim(-x_lim_val, x_lim_val)
    
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
        tick.set_fontsize(20.5)
        tick.set_color("#000000")
        
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_color("#000000")
    ax.spines["left"].set_linewidth(3.0)
    ax.spines["bottom"].set_color("#000000")
    ax.spines["bottom"].set_linewidth(3.0)
        
    ax.grid(axis="x", linestyle="--", alpha=0.3, color="#888888")
    ax.set_axisbelow(True)
    
    ax.set_ylim(-1, len(df_plot) + 1.2)
    ax.tick_params(axis="both", which="major", length=8, width=3.0, color="#000000", direction="out", left=True, bottom=True)
    
    # Enrichment labels at the top
    ax.text(-x_lim_val * 0.95, len(df_plot) + 0.1, f"\u2190 Enriched in {inactive_label}", fontsize=20.0, fontweight="bold", color="#000000", ha="left")
    ax.text(x_lim_val * 0.95, len(df_plot) + 0.1, f"Enriched in {active_label} \u2192", fontsize=20.0, fontweight="bold", color="#000000", ha="right")
    
    # Multi-column legend layout
    legend_elements = [
        Line2D([0], [0], marker="o", color="white", markerfacecolor=color_2d, markeredgecolor="black", markersize=18, label="RDKit 2D Physicochemical Descriptor"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor=color_mfp, markeredgecolor="black", markersize=18, label="Morgan ECFP4 Fingerprint Bit"),
        Line2D([0], [0], color="#00c853", linewidth=5.5, label="Higher in Active Compounds (\u0394 > 0)"),
        Line2D([0], [0], color="#d50000", linewidth=5.5, label="Lower in Active Compounds (\u0394 < 0)"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor="gray", markeredgecolor="black", markersize=21, label="High SHAP Importance (Relative Impact)"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor="gray", markeredgecolor="black", markersize=11, label="Low SHAP Importance")
    ]
    
    ax.legend(
        handles=legend_elements, 
        loc="lower center", 
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        frameon=False, 
        prop={"weight": "bold", "size": 19.0}
    )
    
    plt.tight_layout(pad=1.5)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# Generate plots
print("Generating repellency model SHAP averages plot...")
generate_diverging_lollipop_plot(
    df_shap=df_rep_shap,
    df_features_std=df_rep_feat_std,
    label_col="repellent_active",
    color_2d="#2962ff",
    color_mfp="#ffab00",
    output_path=SUPP_MATERIAL_DIR / "plots" / "xgb_shap_averages.png",
    active_label="Repellent (Active)",
    inactive_label="Decoy (Inactive)",
    max_features=30
)

print("Generating insecticidal model SHAP averages plot...")
generate_diverging_lollipop_plot(
    df_shap=df_ins_shap,
    df_features_std=df_ins_feat_std,
    label_col="insecticidal_active",
    color_2d="#2962ff",
    color_mfp="#ffab00",
    output_path=SUPP_MATERIAL_DIR / "plots" / "rf_shap_averages.png",
    active_label="Insecticidal Active",
    inactive_label="Insecticidal Inactive",
    max_features=30
)

print("All SHAP lollipop plots generated successfully.")
