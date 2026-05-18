"""
Plotting script for supplementary chemical space distributions.
Generates 2x2 violin plots comparing physical properties between active/inactive compounds.
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
SUPP_MATERIAL_DIR = REPO_ROOT / "supp_material"

# Load features and labels for repellency model
df_combined = pd.read_parquet(DATA_DIR / "features_combined.parquet")
df_rep_labeled = df_combined[
    (df_combined["qc_status"] == "pass") & 
    (df_combined["repellent_active"].notna())
].copy()

# Load features and labels for insecticidal model
npz_ins = np.load(DATA_DIR / "phase10_artifacts" / "insecticide_features.npz")
X_ins = npz_ins["X"]
df_ins_meta = pd.read_parquet(DATA_DIR / "phase10_artifacts" / "insecticide_meta.parquet")

with open(DATA_DIR / "phase10_artifacts" / "feature_columns.json", "r") as f:
    feat_meta_ins = json.load(f)
feat_cols_ins = feat_meta_ins["all_cols"]

df_ins_feat = pd.DataFrame(X_ins, columns=feat_cols_ins)
df_ins_feat["insecticidal_active"] = df_ins_meta["insecticidal_active"].values


def generate_chemical_space_plots(df_rep_labeled, df_ins_feat, output_path):
    """
    Generates a 2x2 grid of violin plots comparing the physical property space 
    (Molecular Weight, Lipophilicity, Polarity, and Shape) of the Repellency and Insecticidal datasets.
    """
    props = [
        ("Molecular Weight (Da)", "MolWt", "MolWt"),
        ("Lipophilicity (MolLogP)", "MolLogP", "MolLogP"),
        ("Polarity (TPSA, \u200b\u00c5\u00b2)", "TPSA", "TPSA"),
        ("Molecular Shape (Kappa2)", "Kappa2", "Kappa2")
    ]
    
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "font.size": 14,
        "axes.edgecolor": "black",
        "axes.linewidth": 2.0,
        "xtick.color": "black",
        "ytick.color": "black"
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 12.0))
    axes = axes.flatten()
    
    y_limits = [
        (-20, 800),    # MW y-range with -20 breathing room to allow natural tail tapering
        (-3, 8),       # LogP y-range (already has negative room)
        (-10, 250),    # TPSA y-range with -10 breathing room to allow natural tail tapering
        (-1, 25)       # Kappa2 y-range with -1 breathing room to allow natural tail tapering
    ]
    
    for idx, (label, prop_rep, prop_ins) in enumerate(props):
        ax = axes[idx]
        
        rep_decoy = df_rep_labeled[df_rep_labeled["repellent_active"] == 0.0][prop_rep].dropna()
        rep_active = df_rep_labeled[df_rep_labeled["repellent_active"] == 1.0][prop_rep].dropna()
        ins_inactive = df_ins_feat[df_ins_feat["insecticidal_active"] == 0.0][prop_ins].dropna()
        ins_active = df_ins_feat[df_ins_feat["insecticidal_active"] == 1.0][prop_ins].dropna()
        
        parts = ax.violinplot(
            [rep_decoy, rep_active, ins_inactive, ins_active],
            positions=[1, 2, 4, 5],
            showmeans=True,
            showextrema=False,
            showmedians=False
        )
        
        colors = ["#ffab00", "#2962ff", "#d50000", "#00c853"]
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_edgecolor("black")
            pc.set_linewidth(1.2)
            pc.set_alpha(0.8)
            
        parts['cmeans'].set_edgecolor("black")
        parts['cmeans'].set_linewidth(2.0)
        
        ax.set_ylabel(label, fontsize=15, fontweight="bold", color="black")
        ax.set_xticks([1.5, 4.5])
        ax.set_xticklabels(
            ["Repellency Space\n(732 Compounds)", "Insecticidal Space\n(891 Compounds)"],
            fontsize=13,
            fontweight="bold",
            color="black"
        )
        
        ax.set_ylim(y_limits[idx])
        
        for tick in ax.get_yticklabels():
            tick.set_fontweight("bold")
            tick.set_fontsize(12)
            tick.set_color("black")
            
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(2.0)
        ax.spines["bottom"].set_linewidth(2.0)
        
        ax.grid(axis="y", linestyle=":", alpha=0.5, color="#888888")
        ax.set_axisbelow(True)
        
    legend_elements = [
        Line2D([0], [0], marker="o", color="white", markerfacecolor="#ffab00", markeredgecolor="black", markersize=14, label="Repellency Decoy (Inactive)"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor="#2962ff", markeredgecolor="black", markersize=14, label="Repellent-Active"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor="#d50000", markeredgecolor="black", markersize=14, label="Insecticidal Inactive"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor="#00c853", markeredgecolor="black", markersize=14, label="Insecticidal Active")
    ]
    
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        frameon=False,
        prop={"weight": "bold", "size": 14}
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SUCCESS] Saved chemical space distribution plot to: {output_path.name}")


# Generate plot
print("Generating overall chemical space distribution plot...")
generate_chemical_space_plots(
    df_rep_labeled=df_rep_labeled,
    df_ins_feat=df_ins_feat,
    output_path=SUPP_MATERIAL_DIR / "plots" / "xgb_rf_chemical_space.png"
)
print("All chemical space plots generated successfully.")
