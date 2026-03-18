# ================================================================
# PHASE 10 — V3 MULTITASK (FUTURE / FRAMEWORK)
# ================================================================
# This phase is a FRAMEWORK for when real insecticide activity
# labels become available (from ChEMBL, PubChem, or literature).
#
# It extends the model to a multitask setup with independent
# prediction heads:
#   - p_repellent     (existing, from V1/V2)
#   - p_insecticidal  (new, requires external labels)
#
# Each head has its own calibration and uncertainty — they are
# NOT collapsed into a multi-class head.
#
# PREREQUISITES:
#   1. Obtain insecticide assay labels (% mortality, LC50, LD50)
#   2. Map labels to LifeChemicals molecules via structure matching
#   3. Define the binarization threshold for "insecticidal active"
#
# This file provides the framework and can be activated once
# real labels are acquired.
#
# Copy each "Cell N" block into a Jupyter notebook sequentially.
# ================================================================


# ================================================================
# Cell 0 — Install Dependencies
# ================================================================
# !pip install lightgbm scikit-learn pandas pyarrow numpy torch


# ================================================================
# Cell 1 — Imports and Configuration
# ================================================================
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import lightgbm as lgb

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

# ── Project Paths ──────────────────────────────────────────
PROJECT_ROOT = Path(r"G:\research\ECOAI")
DATA_DIR     = PROJECT_ROOT / "Datasets" / "data"
PHASE10_DIR  = PROJECT_ROOT / "experiment" / "phase10_v3_multitask"

# ── Placeholder for future insecticide labels ──────────────
# INSECTICIDE_LABELS_FILE = DATA_DIR / "insecticide_labels.parquet"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
PHASE10_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  PHASE 10 — V3 MULTITASK (FRAMEWORK)")
print("=" * 60)
print("  ⚠️  This phase requires external insecticide activity labels.")
print("     It provides the framework code to be activated later.")


# ================================================================
# Cell 2 — Label Acquisition Guide
# ================================================================
# This cell describes how to obtain insecticide labels.

print("\n" + "=" * 60)
print("  LABEL ACQUISITION GUIDE")
print("=" * 60)

guide = """
  Sources for insecticide activity labels:

  1. ChEMBL (www.ebi.ac.uk/chembl/)
     - Search for insecticide bioactivity data
     - Filter by target organism and assay type
     - Download structure-activity data
     - Relevant assay types: EC50, IC50, LC50, LD50, % inhibition

  2. PubChem BioAssay (pubchem.ncbi.nlm.nih.gov)
     - Search for insecticide-related bioassays
     - Download dose-response data
     - Filter for confirmed actives/inactives

  3. Literature / EPA databases
     - Pesticide registration data
     - Published SAR studies

  Label processing steps:
  a. Download assay data with SMILES/InChIKey
  b. Standardize structures (same pipeline as Phase 1)
  c. Match to LifeChemicals molecules by InChIKey
  d. Define activity threshold:
     - For LC50: active if LC50 < threshold (e.g., 10 μM)
     - For % mortality: active if mortality > 50%
  e. Save as insecticide_labels.parquet with columns:
     compound_id, inchikey, insecticidal_active (0/1),
     assay_value, assay_type, source
"""
print(guide)


# ================================================================
# Cell 3 — Multitask Model Architecture
# ================================================================
# The V3 model has TWO independent binary heads:
#   Head 1: p_repellent (from V1/V2)
#   Head 2: p_insecticidal (new)
#
# Each head has its own loss, calibration, and uncertainty.
# This avoids the pitfall of a 3-class softmax which incorrectly
# assumes mutual exclusivity.

class MultitaskFTTransformer(nn.Module):
    """
    Multitask FT-Transformer with independent binary heads.

    Architecture:
        Shared backbone: Feature tokenizer + Transformer encoder
        Head 1: p_repellent (binary sigmoid)
        Head 2: p_insecticidal (binary sigmoid)
    """
    def __init__(
        self,
        n_features: int,
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ffn: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Shared backbone (same as Phase 4 FT-Transformer)
        self.tokenizer = nn.Linear(n_features, d_token * n_features)
        self.n_features = n_features
        self.d_token = d_token

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, n_features + 1, d_token)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.ln = nn.LayerNorm(d_token)

        # Head 1: Repellent
        self.head_repellent = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, 1),
        )

        # Head 2: Insecticidal
        self.head_insecticidal = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, 1),
        )

    def encode(self, x):
        """Shared encoding."""
        batch_size = x.shape[0]

        # Simple feature tokenization
        tokens = self.tokenizer(x).view(batch_size, self.n_features, self.d_token)

        # Prepend [CLS]
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Position + Transformer
        tokens = tokens + self.pos_encoding[:, :tokens.shape[1], :]
        tokens = self.transformer(tokens)
        tokens = self.ln(tokens)

        return tokens[:, 0, :]  # [CLS] output

    def forward(self, x):
        """
        Returns dict with independent logits for each task.
        """
        cls_out = self.encode(x)

        return {
            "repellent_logit":    self.head_repellent(cls_out).squeeze(-1),
            "insecticidal_logit": self.head_insecticidal(cls_out).squeeze(-1),
        }


print("\n✅ MultitaskFTTransformer defined.")
print("   Head 1: p_repellent (binary)")
print("   Head 2: p_insecticidal (binary)")


# ================================================================
# Cell 4 — Multitask Training Loop (Template)
# ================================================================
# This is a template for training the multitask model once
# insecticide labels are available.

def train_multitask_model(
    model,
    X_train, y_rep_train, y_ins_train,
    X_val, y_rep_val, y_ins_val,
    n_epochs=100,
    lr=5e-4,
    batch_size=32,
    device="cpu",
):
    """
    Train multitask model with independent losses per head.

    Handles missing labels gracefully:
    - Molecules with y_rep = NaN are excluded from repellent loss
    - Molecules with y_ins = NaN are excluded from insecticidal loss
    - This allows training on partially labeled data
    """
    from torch.utils.data import DataLoader, TensorDataset

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    X_t = torch.FloatTensor(X_train)
    y_rep_t = torch.FloatTensor(y_rep_train)
    y_ins_t = torch.FloatTensor(y_ins_train)

    # Masks for valid labels
    rep_valid = ~torch.isnan(y_rep_t)
    ins_valid = ~torch.isnan(y_ins_t)

    # Replace NaN with 0 for computation (masked out in loss)
    y_rep_t = torch.nan_to_num(y_rep_t, 0.0)
    y_ins_t = torch.nan_to_num(y_ins_t, 0.0)

    dataset = TensorDataset(X_t, y_rep_t, y_ins_t, rep_valid.float(), ins_valid.float())
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_yr, batch_yi, mask_r, mask_i in loader:
            batch_x  = batch_x.to(device)
            batch_yr = batch_yr.to(device)
            batch_yi = batch_yi.to(device)
            mask_r   = mask_r.to(device)
            mask_i   = mask_i.to(device)

            outputs = model(batch_x)

            # Independent losses, masked by valid labels
            loss_rep = (criterion(outputs["repellent_logit"], batch_yr) * mask_r).sum()
            loss_ins = (criterion(outputs["insecticidal_logit"], batch_yi) * mask_i).sum()

            n_valid = mask_r.sum() + mask_i.sum()
            if n_valid > 0:
                loss = (loss_rep + loss_ins) / n_valid
            else:
                loss = torch.tensor(0.0, requires_grad=True)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}  Loss: {epoch_loss:.4f}")

    return model

print("✅ Multitask training template defined.")
print("   Handles missing labels via per-sample masking.")


# ================================================================
# Cell 5 — Multitask Prediction and Calibration (Template)
# ================================================================

def predict_and_calibrate_multitask(model, X, device="cpu"):
    """
    Predict with multitask model and return raw probabilities.
    Calibration should be applied separately for each head.
    """
    model.eval()
    X_t = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        outputs = model(X_t)

    return {
        "p_repellent_raw":    torch.sigmoid(outputs["repellent_logit"]).cpu().numpy(),
        "p_insecticidal_raw": torch.sigmoid(outputs["insecticidal_logit"]).cpu().numpy(),
    }

print("✅ Prediction template defined.")
print("   Each head gets independent calibration + conformal intervals.")


# ================================================================
# Cell 6 — Activation Checklist
# ================================================================

print("\n" + "=" * 60)
print("  V3 ACTIVATION CHECKLIST")
print("=" * 60)

checklist = """
  To activate Phase 10, complete these steps:

  □ 1. Obtain insecticide activity labels from ChEMBL/PubChem
  □ 2. Standardize structures with Phase 1 pipeline
  □ 3. Match labels to LifeChemicals compounds by InChIKey
  □ 4. Define binarization threshold for insecticidal_active
  □ 5. Save as Datasets/data/insecticide_labels.parquet
  □ 6. Uncomment and run the training cells above
  □ 7. Apply independent calibration per head (Phase 5 method)
  □ 8. Apply independent conformal prediction per head
  □ 9. Generate dual-task screening predictions
  □ 10. Export model_predictions_v3.parquet with:
       - compound_id
       - p_repellent (calibrated)
       - p_insecticidal (calibrated)
       - prediction_interval_repellent
       - prediction_interval_insecticidal
       - ood_score
"""
print(checklist)


# ================================================================
# Cell 7 — Save Phase 10 Status
# ================================================================

status = {
    "pipeline":    "Phase 10 — V3 Multitask",
    "status":      "framework_ready",
    "activated":   False,
    "prerequisites": {
        "insecticide_labels":   "NOT YET AVAILABLE",
        "label_sources":        ["ChEMBL", "PubChem", "EPA", "Literature"],
        "required_columns":     ["compound_id", "inchikey", "insecticidal_active",
                                  "assay_value", "assay_type", "source"],
    },
    "architecture": "MultitaskFTTransformer with independent binary heads",
    "key_design_decisions": [
        "Independent heads, NOT multi-class softmax",
        "Per-head calibration and uncertainty",
        "Handles missing labels via per-sample masking",
        "Can train on partially labeled data",
    ],
}

with open(PHASE10_DIR / "v3_status.json", "w") as f:
    json.dump(status, f, indent=2)

print(f"\n✅ Status saved: {PHASE10_DIR / 'v3_status.json'}")

print("\n" + "=" * 60)
print("  PHASE 10 — FRAMEWORK READY (awaiting insecticide labels)")
print("=" * 60)
print("  🏁 ALL 10 PHASES COMPLETE!")
print("=" * 60)
