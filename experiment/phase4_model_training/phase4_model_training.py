# ================================================================
# PHASE 4 — MODEL TRAINING (V1)
# ================================================================
# Train two model families on scaffold-split folds:
#   Model A: Gradient boosting baseline (LightGBM)
#   Model B: FT-Transformer (pretrained + fine-tuned)
#
# Compare using: PR-AUC, ROC-AUC, MCC, Brier score, calibration
# Select the best model for downstream use.
#
# Input:  Datasets/data/features_combined.parquet
#         Datasets/data/split_manifest.json
# Output: Datasets/data/model_predictions_v1.parquet
#         experiment/phase4_model_training/best_model/
#
# ================================================================


# ================================================================
# Cell 0 — Install Dependencies
# ================================================================
# !pip install lightgbm scikit-learn pandas pyarrow numpy tqdm
# !pip install torch          # for FT-Transformer
# !pip install tab-transformer-pytorch   # optional, or we build from scratch


# ================================================================
# Cell 1 — Imports and Configuration
# ================================================================
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,   # PR-AUC
    matthews_corrcoef,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
    classification_report,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── Project Paths ──────────────────────────────────────────
PROJECT_ROOT = Path(r"G:\research\ECOAI")
DATA_DIR     = PROJECT_ROOT / "Datasets" / "data"
PHASE4_DIR   = PROJECT_ROOT / "experiment" / "phase4_model_training"

# ── Input ──────────────────────────────────────────────────
INPUT_FEATURES = DATA_DIR / "features_combined.parquet"
INPUT_MANIFEST = DATA_DIR / "split_manifest.json"

# ── Output ─────────────────────────────────────────────────
OUTPUT_PREDICTIONS = DATA_DIR / "model_predictions_v1.parquet"
MODEL_DIR          = PHASE4_DIR / "models"
RESULTS_FILE       = PHASE4_DIR / "training_results.json"

# ── Constants ──────────────────────────────────────────────
RANDOM_SEED = 42
N_FOLDS = 5
np.random.seed(RANDOM_SEED)

# ── Verify ─────────────────────────────────────────────────
assert INPUT_FEATURES.exists(), f"❌ {INPUT_FEATURES} — Run Phase 3 first!"
assert INPUT_MANIFEST.exists(), f"❌ {INPUT_MANIFEST} — Run Phase 2 first!"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  PHASE 4 — MODEL TRAINING (V1)")
print("=" * 60)


# ================================================================
# Cell 2 — Load Data and Set Up Features / Labels
# ================================================================

df = pd.read_parquet(INPUT_FEATURES)
with open(INPUT_MANIFEST, "r") as f:
    manifest = json.load(f)

# ── Identify feature columns ──────────────────────────────
meta_cols = [
    "compound_id", "canonical_smiles", "source_dataset",
    "repellent_active", "scaffold_smiles", "fold_id", "qc_status",
]
feature_cols = [c for c in df.columns if c not in meta_cols]

# ── Split labeled vs. unlabeled ────────────────────────────
df_labeled   = df[df["repellent_active"].notna()].copy()
df_unlabeled = df[df["repellent_active"].isna()].copy()

print(f"\n✅ Data loaded.")
print(f"   Total molecules:     {len(df)}")
print(f"   Labeled (for CV):    {len(df_labeled)}")
print(f"   Unlabeled (screen):  {len(df_unlabeled)}")
print(f"   Feature columns:     {len(feature_cols)}")

X_labeled = df_labeled[feature_cols].values
y_labeled = df_labeled["repellent_active"].values.astype(int)
fold_ids  = df_labeled["fold_id"].values.astype(int)

print(f"   X shape: {X_labeled.shape}")
print(f"   y distribution: {dict(zip(*np.unique(y_labeled, return_counts=True)))}")


# ================================================================
# Cell 3 — Preprocessing Pipeline
# ================================================================
# Handle NaN/Inf in features. Morgan FP bits are clean (0/1),
# but RDKit 2D descriptors may have NaN or Inf.

def preprocess_features(X_train, X_val, X_all=None):
    """
    Impute NaN with median, then standardize.
    Fit on train, transform val (and optionally all data).

    Returns preprocessed arrays and fitted objects.
    """
    # Impute
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp   = imputer.transform(X_val)

    # Standardize
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_imp)
    X_val_std   = scaler.transform(X_val_imp)

    result = {
        "X_train": X_train_std,
        "X_val":   X_val_std,
        "imputer": imputer,
        "scaler":  scaler,
    }

    if X_all is not None:
        X_all_imp = imputer.transform(X_all)
        X_all_std = scaler.transform(X_all_imp)
        result["X_all"] = X_all_std

    return result

print("✅ Preprocessing pipeline defined.")


# ================================================================
# Cell 4 — Model A: LightGBM Baseline (Scaffold-Split CV)
# ================================================================
# Train LightGBM with scaffold-aware 5-fold CV.
# Collect out-of-fold (OOF) predictions for calibration.

print("\n" + "=" * 60)
print("  MODEL A — LightGBM BASELINE")
print("=" * 60)

lgb_params = {
    "objective":        "binary",
    "metric":           "binary_logloss",
    "boosting_type":    "gbdt",
    "n_estimators":     500,
    "learning_rate":    0.05,
    "max_depth":        6,
    "num_leaves":       31,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "min_child_samples": 10,
    "random_state":     RANDOM_SEED,
    "verbose":          -1,
    "n_jobs":           -1,
}

oof_preds_lgb = np.zeros(len(y_labeled))
fold_metrics_lgb = []
lgb_models = []

for fold in range(N_FOLDS):
    print(f"\n  ── Fold {fold} ──")

    train_mask = fold_ids != fold
    val_mask   = fold_ids == fold

    X_train, y_train = X_labeled[train_mask], y_labeled[train_mask]
    X_val,   y_val   = X_labeled[val_mask],   y_labeled[val_mask]

    # Preprocess
    prep = preprocess_features(X_train, X_val)

    # Train LightGBM
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        prep["X_train"], y_train,
        eval_set=[(prep["X_val"], y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # Predict probabilities
    y_pred_proba = model.predict_proba(prep["X_val"])[:, 1]
    oof_preds_lgb[val_mask] = y_pred_proba

    # Fold metrics
    y_pred_class = (y_pred_proba >= 0.5).astype(int)
    metrics = {
        "fold":    fold,
        "roc_auc": roc_auc_score(y_val, y_pred_proba),
        "pr_auc":  average_precision_score(y_val, y_pred_proba),
        "mcc":     matthews_corrcoef(y_val, y_pred_class),
        "brier":   brier_score_loss(y_val, y_pred_proba),
        "n_train": int(train_mask.sum()),
        "n_val":   int(val_mask.sum()),
    }
    fold_metrics_lgb.append(metrics)
    lgb_models.append({"model": model, "imputer": prep["imputer"], "scaler": prep["scaler"]})

    print(f"    ROC-AUC: {metrics['roc_auc']:.4f}  |  "
          f"PR-AUC: {metrics['pr_auc']:.4f}  |  "
          f"MCC: {metrics['mcc']:.4f}  |  "
          f"Brier: {metrics['brier']:.4f}")

# ── Overall OOF metrics ───────────────────────────────────
oof_class_lgb = (oof_preds_lgb >= 0.5).astype(int)
overall_lgb = {
    "model":   "LightGBM",
    "roc_auc": roc_auc_score(y_labeled, oof_preds_lgb),
    "pr_auc":  average_precision_score(y_labeled, oof_preds_lgb),
    "mcc":     matthews_corrcoef(y_labeled, oof_class_lgb),
    "brier":   brier_score_loss(y_labeled, oof_preds_lgb),
}
print(f"\n  ── LightGBM Overall (OOF) ──")
print(f"    ROC-AUC: {overall_lgb['roc_auc']:.4f}")
print(f"    PR-AUC:  {overall_lgb['pr_auc']:.4f}")
print(f"    MCC:     {overall_lgb['mcc']:.4f}")
print(f"    Brier:   {overall_lgb['brier']:.4f}")


# ================================================================
# Cell 4.2 — Model A2: XGBoost (Scaffold-Split CV)
# ================================================================
# XGBoost with similar hyperparameters to LightGBM for comparison.

from xgboost import XGBClassifier

print("\n" + "=" * 60)
print("  MODEL A2 — XGBoost")
print("=" * 60)

xgb_params = {
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    "booster":          "gbtree",
    "n_estimators":     500,
    "learning_rate":    0.05,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "min_child_weight":  5,
    "random_state":     RANDOM_SEED,
    "verbosity":        0,
    "n_jobs":           -1,
    "use_label_encoder": False,
}

oof_preds_xgb = np.zeros(len(y_labeled))
fold_metrics_xgb = []
xgb_models = []

for fold in range(N_FOLDS):
    print(f"\n  ── Fold {fold} ──")

    train_mask = fold_ids != fold
    val_mask   = fold_ids == fold

    X_train, y_train = X_labeled[train_mask], y_labeled[train_mask]
    X_val,   y_val   = X_labeled[val_mask],   y_labeled[val_mask]

    prep = preprocess_features(X_train, X_val)

    model = XGBClassifier(**xgb_params)
    model.fit(
        prep["X_train"], y_train,
        eval_set=[(prep["X_val"], y_val)],
        verbose=False,
    )

    y_pred_proba = model.predict_proba(prep["X_val"])[:, 1]
    oof_preds_xgb[val_mask] = y_pred_proba

    y_pred_class = (y_pred_proba >= 0.5).astype(int)
    metrics = {
        "fold":    fold,
        "roc_auc": roc_auc_score(y_val, y_pred_proba),
        "pr_auc":  average_precision_score(y_val, y_pred_proba),
        "mcc":     matthews_corrcoef(y_val, y_pred_class),
        "brier":   brier_score_loss(y_val, y_pred_proba),
    }
    fold_metrics_xgb.append(metrics)
    xgb_models.append({"model": model, "imputer": prep["imputer"], "scaler": prep["scaler"]})

    print(f"    ROC-AUC: {metrics['roc_auc']:.4f}  |  "
          f"PR-AUC: {metrics['pr_auc']:.4f}  |  "
          f"MCC: {metrics['mcc']:.4f}  |  "
          f"Brier: {metrics['brier']:.4f}")

oof_class_xgb = (oof_preds_xgb >= 0.5).astype(int)
overall_xgb = {
    "model":   "XGBoost",
    "roc_auc": roc_auc_score(y_labeled, oof_preds_xgb),
    "pr_auc":  average_precision_score(y_labeled, oof_preds_xgb),
    "mcc":     matthews_corrcoef(y_labeled, oof_class_xgb),
    "brier":   brier_score_loss(y_labeled, oof_preds_xgb),
}
print(f"\n  ── XGBoost Overall (OOF) ──")
print(f"    ROC-AUC: {overall_xgb['roc_auc']:.4f}")
print(f"    PR-AUC:  {overall_xgb['pr_auc']:.4f}")
print(f"    MCC:     {overall_xgb['mcc']:.4f}")
print(f"    Brier:   {overall_xgb['brier']:.4f}")


# ================================================================
# Cell 4.3 — Model A3: Random Forest (Scaffold-Split CV)
# ================================================================
# Ensemble of decorrelated decision trees. Often competitive
# with boosting on small datasets with high-dimensional features.

from sklearn.ensemble import RandomForestClassifier

print("\n" + "=" * 60)
print("  MODEL A3 — Random Forest")
print("=" * 60)

rf_params = {
    "n_estimators":     500,
    "max_depth":        None,      # grow full trees
    "min_samples_split": 5,
    "min_samples_leaf":  2,
    "max_features":     "sqrt",
    "class_weight":     "balanced",  # handle mild imbalance
    "random_state":     RANDOM_SEED,
    "n_jobs":           -1,
}

oof_preds_rf = np.zeros(len(y_labeled))
fold_metrics_rf = []

for fold in range(N_FOLDS):
    print(f"\n  ── Fold {fold} ──")

    train_mask = fold_ids != fold
    val_mask   = fold_ids == fold

    X_train, y_train = X_labeled[train_mask], y_labeled[train_mask]
    X_val,   y_val   = X_labeled[val_mask],   y_labeled[val_mask]

    prep = preprocess_features(X_train, X_val)

    model = RandomForestClassifier(**rf_params)
    model.fit(prep["X_train"], y_train)

    y_pred_proba = model.predict_proba(prep["X_val"])[:, 1]
    oof_preds_rf[val_mask] = y_pred_proba

    y_pred_class = (y_pred_proba >= 0.5).astype(int)
    metrics = {
        "fold":    fold,
        "roc_auc": roc_auc_score(y_val, y_pred_proba),
        "pr_auc":  average_precision_score(y_val, y_pred_proba),
        "mcc":     matthews_corrcoef(y_val, y_pred_class),
        "brier":   brier_score_loss(y_val, y_pred_proba),
    }
    fold_metrics_rf.append(metrics)

    print(f"    ROC-AUC: {metrics['roc_auc']:.4f}  |  "
          f"PR-AUC: {metrics['pr_auc']:.4f}  |  "
          f"MCC: {metrics['mcc']:.4f}  |  "
          f"Brier: {metrics['brier']:.4f}")

oof_class_rf = (oof_preds_rf >= 0.5).astype(int)
overall_rf = {
    "model":   "RandomForest",
    "roc_auc": roc_auc_score(y_labeled, oof_preds_rf),
    "pr_auc":  average_precision_score(y_labeled, oof_preds_rf),
    "mcc":     matthews_corrcoef(y_labeled, oof_class_rf),
    "brier":   brier_score_loss(y_labeled, oof_preds_rf),
}
print(f"\n  ── Random Forest Overall (OOF) ──")
print(f"    ROC-AUC: {overall_rf['roc_auc']:.4f}")
print(f"    PR-AUC:  {overall_rf['pr_auc']:.4f}")
print(f"    MCC:     {overall_rf['mcc']:.4f}")
print(f"    Brier:   {overall_rf['brier']:.4f}")


# ================================================================
# Cell 4.4 — Model A4: Tuned LightGBM-DART (Scaffold-Split CV)
# ================================================================
# DART boosting (Dropouts meet Multiple Additive Regression Trees)
# with deeper trees and stronger regularization. This variant
# often improves generalization on small datasets.

print("\n" + "=" * 60)
print("  MODEL A4 — LightGBM-DART (Tuned)")
print("=" * 60)

dart_params = {
    "objective":        "binary",
    "metric":           "binary_logloss",
    "boosting_type":    "dart",     # DART boosting
    "n_estimators":     400,
    "learning_rate":    0.03,       # slower learning
    "max_depth":        8,          # deeper trees
    "num_leaves":       50,         # more leaves
    "subsample":        0.7,
    "colsample_bytree": 0.7,
    "reg_alpha":        0.5,        # stronger L1
    "reg_lambda":       2.0,        # stronger L2
    "min_child_samples": 8,
    "drop_rate":        0.1,        # DART dropout rate
    "skip_drop":        0.5,        # probability of skipping dropout
    "random_state":     RANDOM_SEED,
    "verbose":          -1,
    "n_jobs":           -1,
}

oof_preds_dart = np.zeros(len(y_labeled))
fold_metrics_dart = []
dart_models = []

for fold in range(N_FOLDS):
    print(f"\n  ── Fold {fold} ──")

    train_mask = fold_ids != fold
    val_mask   = fold_ids == fold

    X_train, y_train = X_labeled[train_mask], y_labeled[train_mask]
    X_val,   y_val   = X_labeled[val_mask],   y_labeled[val_mask]

    prep = preprocess_features(X_train, X_val)

    model = lgb.LGBMClassifier(**dart_params)
    model.fit(
        prep["X_train"], y_train,
        eval_set=[(prep["X_val"], y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    y_pred_proba = model.predict_proba(prep["X_val"])[:, 1]
    oof_preds_dart[val_mask] = y_pred_proba

    y_pred_class = (y_pred_proba >= 0.5).astype(int)
    metrics = {
        "fold":    fold,
        "roc_auc": roc_auc_score(y_val, y_pred_proba),
        "pr_auc":  average_precision_score(y_val, y_pred_proba),
        "mcc":     matthews_corrcoef(y_val, y_pred_class),
        "brier":   brier_score_loss(y_val, y_pred_proba),
    }
    fold_metrics_dart.append(metrics)
    dart_models.append({"model": model, "imputer": prep["imputer"], "scaler": prep["scaler"]})

    print(f"    ROC-AUC: {metrics['roc_auc']:.4f}  |  "
          f"PR-AUC: {metrics['pr_auc']:.4f}  |  "
          f"MCC: {metrics['mcc']:.4f}  |  "
          f"Brier: {metrics['brier']:.4f}")

oof_class_dart = (oof_preds_dart >= 0.5).astype(int)
overall_dart = {
    "model":   "LightGBM-DART",
    "roc_auc": roc_auc_score(y_labeled, oof_preds_dart),
    "pr_auc":  average_precision_score(y_labeled, oof_preds_dart),
    "mcc":     matthews_corrcoef(y_labeled, oof_class_dart),
    "brier":   brier_score_loss(y_labeled, oof_preds_dart),
}
print(f"\n  ── LightGBM-DART Overall (OOF) ──")
print(f"    ROC-AUC: {overall_dart['roc_auc']:.4f}")
print(f"    PR-AUC:  {overall_dart['pr_auc']:.4f}")
print(f"    MCC:     {overall_dart['mcc']:.4f}")
print(f"    Brier:   {overall_dart['brier']:.4f}")


# ================================================================
# Cell 4.5 — Classical Models Comparison
# ================================================================
# Compare all classical baselines and select the BEST one to
# challenge the FT-Transformer in Cell 8.

print("\n" + "=" * 60)
print("  CLASSICAL BASELINES — COMPARISON")
print("=" * 60)

baseline_results = [overall_lgb, overall_xgb, overall_rf, overall_dart]
baseline_df = pd.DataFrame(baseline_results).set_index("model")

# Sort by PR-AUC descending
baseline_df = baseline_df.sort_values("pr_auc", ascending=False)
print(baseline_df.to_string(float_format="{:.4f}".format))

# Select best classical model
best_baseline_name = baseline_df.index[0]

# Map name → predictions and models for downstream use
baseline_preds_map = {
    "LightGBM":       (oof_preds_lgb,  lgb_models),
    "XGBoost":        (oof_preds_xgb,  xgb_models),
    "RandomForest":   (oof_preds_rf,   None),
    "LightGBM-DART":  (oof_preds_dart, dart_models),
}
baseline_metrics_map = {
    "LightGBM":       overall_lgb,
    "XGBoost":        overall_xgb,
    "RandomForest":   overall_rf,
    "LightGBM-DART":  overall_dart,
}

best_classical_preds, best_classical_fold_models = baseline_preds_map[best_baseline_name]
best_classical_metrics = baseline_metrics_map[best_baseline_name]

print(f"\n  🏆 Best classical model: {best_baseline_name}")
print(f"     PR-AUC: {best_classical_metrics['pr_auc']:.4f}")
print(f"     Brier:  {best_classical_metrics['brier']:.4f}")
print(f"\n  This model will challenge the FT-Transformer in Cell 8.")

# Update lgb_models to point to the best model's fold models
# (used in Cell 12 for saving and Phase 9 for screening)
if best_classical_fold_models is not None:
    lgb_models = best_classical_fold_models
    print(f"  ✅ lgb_models updated to {best_baseline_name} fold models.")



# ================================================================
# Cell 5 — Model B: FT-Transformer
# ================================================================
# NOTE: The FT-Transformer requires PyTorch. If PyTorch is not
#       available, this cell can be skipped and LightGBM used alone.
#
# Architecture:
#   1. Self-supervised pretrain on ALL molecules (masked feature 
#      reconstruction)
#   2. Freeze backbone, fine-tune classification head on labeled 
#      data only

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n" + "=" * 60)
print(f"  MODEL B — FT-TRANSFORMER")
print(f"=" * 60)
print(f"  Device: {DEVICE}")


# ── FT-Transformer Architecture ──────────────────────────
class FeatureTokenizer(nn.Module):
    """Tokenize each numerical feature into a d-dimensional embedding."""
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token))
        self.bias   = nn.Parameter(torch.zeros(n_features, d_token))

    def forward(self, x):
        # x: (batch, n_features)
        # output: (batch, n_features, d_token)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformer(nn.Module):
    """
    Feature-Tokenized Transformer for tabular data.

    Each input feature is tokenized into a d-dimensional vector,
    then processed by a stack of Transformer encoder layers.
    A [CLS] token is prepended for classification.
    """
    def __init__(
        self,
        n_features: int,
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ffn: int = 128,
        dropout: float = 0.1,
        n_classes: int = 1,   # binary → 1 logit
    ):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token

        # Feature tokenizer
        self.tokenizer = FeatureTokenizer(n_features, d_token)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

        # Positional encoding (learnable)
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

        # Layer norm
        self.ln = nn.LayerNorm(d_token)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, n_classes),
        )

        # Reconstruction head (for pretraining)
        self.recon_head = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, 1),
        )

    def encode(self, x):
        """Encode features through tokenizer and transformer. Returns CLS embedding."""
        batch_size = x.shape[0]

        # Tokenize features → (batch, n_features, d_token)
        tokens = self.tokenizer(x)

        # Prepend [CLS] token
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (batch, n_features+1, d_token)

        # Add positional encoding
        tokens = tokens + self.pos_encoding[:, :tokens.shape[1], :]

        # Transformer
        tokens = self.transformer(tokens)
        tokens = self.ln(tokens)

        return tokens

    def forward(self, x):
        """Classification forward pass. Returns logit."""
        tokens = self.encode(x)
        cls_out = tokens[:, 0, :]   # [CLS] token
        return self.head(cls_out).squeeze(-1)

    def reconstruct(self, x, mask):
        """
        Self-supervised reconstruction. Predict masked feature values.
        mask: (batch, n_features) boolean, True = masked
        """
        tokens = self.encode(x)
        feature_tokens = tokens[:, 1:, :]  # skip [CLS]
        recon = self.recon_head(feature_tokens).squeeze(-1)  # (batch, n_features)
        return recon


class TabularDataset(Dataset):
    """Simple dataset for tabular features + labels."""
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx],

print("✅ FT-Transformer architecture defined.")


# ================================================================
# Cell 6 — Self-Supervised Pretraining (Masked Feature Reconstruction)
# ================================================================
# Pretrain on ALL curated molecules (labeled + unlabeled) using
# masked feature reconstruction: randomly mask 15% of features
# and train the model to reconstruct them.

print("\n── Self-Supervised Pretraining ──")

# Prepare full feature matrix (all QC-pass molecules)
X_all = df[feature_cols].values

# Impute and scale on full data
full_imputer = SimpleImputer(strategy="median")
X_all_imp = full_imputer.fit_transform(X_all)
full_scaler = StandardScaler()
X_all_std = full_scaler.fit_transform(X_all_imp)

n_features = X_all_std.shape[1]
print(f"  Pretraining on {len(X_all_std)} molecules, {n_features} features")

# ── Hyperparameters ────────────────────────────────────────
PRETRAIN_EPOCHS   = 50
PRETRAIN_LR       = 1e-3
PRETRAIN_BATCH    = 64
MASK_RATIO        = 0.15
D_TOKEN           = 64
N_HEADS           = 4
N_LAYERS          = 3
D_FFN             = 128
DROPOUT           = 0.1

# ── Build model ────────────────────────────────────────────
ft_model = FTTransformer(
    n_features=n_features,
    d_token=D_TOKEN,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    d_ffn=D_FFN,
    dropout=DROPOUT,
    n_classes=1,
).to(DEVICE)

pretrain_dataset = TabularDataset(X_all_std)
pretrain_loader  = DataLoader(
    pretrain_dataset, batch_size=PRETRAIN_BATCH, shuffle=True,
    drop_last=True,
)

optimizer = optim.AdamW(ft_model.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS)

# ── Pretraining loop ──────────────────────────────────────
print(f"  Epochs: {PRETRAIN_EPOCHS}, Batch: {PRETRAIN_BATCH}, "
      f"Mask: {MASK_RATIO*100:.0f}%, LR: {PRETRAIN_LR}")

ft_model.train()
pretrain_losses = []

for epoch in range(PRETRAIN_EPOCHS):
    epoch_loss = 0.0
    n_batches = 0

    for (batch_x,) in pretrain_loader:
        batch_x = batch_x.to(DEVICE)

        # Create random mask
        mask = torch.rand_like(batch_x) < MASK_RATIO

        # Zero out masked features
        x_masked = batch_x.clone()
        x_masked[mask] = 0.0

        # Reconstruct
        recon = ft_model.reconstruct(x_masked, mask)

        # Loss only on masked positions
        loss = nn.MSELoss()(recon[mask], batch_x[mask])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ft_model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    scheduler.step()
    avg_loss = epoch_loss / max(n_batches, 1)
    pretrain_losses.append(avg_loss)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"    Epoch {epoch+1:>3d}/{PRETRAIN_EPOCHS}  "
              f"Loss: {avg_loss:.6f}  "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

print(f"\n  ✅ Pretraining complete. Final loss: {pretrain_losses[-1]:.6f}")

# Save pretrained backbone
pretrained_path = MODEL_DIR / "ft_transformer_pretrained.pt"
torch.save(ft_model.state_dict(), pretrained_path)
print(f"  Saved: {pretrained_path.name}")


# ================================================================
# Cell 7 — Fine-Tune FT-Transformer (Scaffold CV, Frozen Backbone)
# ================================================================
# Freeze the transformer backbone and train only the
# classification head on labeled data.

print("\n── Fine-Tuning FT-Transformer (Frozen Backbone) ──")

FINETUNE_EPOCHS = 80
FINETUNE_LR     = 5e-4
FINETUNE_BATCH  = 32

oof_preds_ft = np.zeros(len(y_labeled))
fold_metrics_ft = []

for fold in range(N_FOLDS):
    print(f"\n  ── Fold {fold} ──")

    train_mask = fold_ids != fold
    val_mask   = fold_ids == fold

    X_train, y_train = X_labeled[train_mask], y_labeled[train_mask]
    X_val,   y_val   = X_labeled[val_mask],   y_labeled[val_mask]

    # Preprocess (fit on train)
    imp = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train)
    X_val_imp   = imp.transform(X_val)
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train_imp)
    X_val_std   = sc.transform(X_val_imp)

    # Load pretrained model (fresh copy each fold)
    model = FTTransformer(
        n_features=n_features,
        d_token=D_TOKEN,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ffn=D_FFN,
        dropout=DROPOUT,
        n_classes=1,
    ).to(DEVICE)
    model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE, weights_only=True))

    # Freeze backbone (tokenizer + transformer + positional encoding)
    for name, param in model.named_parameters():
        if "head" not in name or "recon_head" in name:
            param.requires_grad = False

    # Only train classification head
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())
    if fold == 0:
        print(f"    Trainable params: {trainable_params:,} / {total_params:,}")

    # Data loaders
    train_ds = TabularDataset(X_train_std, y_train)
    val_ds   = TabularDataset(X_val_std, y_val)
    train_loader = DataLoader(train_ds, batch_size=FINETUNE_BATCH, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=FINETUNE_BATCH, shuffle=False)

    # Training
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINETUNE_LR, weight_decay=1e-3,
    )
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")
    patience = 15
    wait = 0
    best_state = None

    model.train()
    for epoch in range(FINETUNE_EPOCHS):
        epoch_loss = 0.0
        n_b = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1

        # Validation
        model.eval()
        val_loss = 0.0
        n_vb = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                logits = model(batch_x)
                val_loss += criterion(logits, batch_y).item()
                n_vb += 1
        model.train()

        avg_val_loss = val_loss / max(n_vb, 1)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if fold == 0:
                    print(f"    Early stop at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # Predict on validation
    val_preds = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(DEVICE)
            logits = model(batch_x)
            probs = torch.sigmoid(logits).cpu().numpy()
            val_preds.extend(probs)

    # Handle remaining samples not in last batch
    # Full prediction on val set
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val_std).to(DEVICE)
        all_logits = model(X_val_tensor)
        all_probs = torch.sigmoid(all_logits).cpu().numpy()

    oof_preds_ft[val_mask] = all_probs

    # Metrics
    y_pred_class = (all_probs >= 0.5).astype(int)
    metrics = {
        "fold":    fold,
        "roc_auc": float(roc_auc_score(y_val, all_probs)),
        "pr_auc":  float(average_precision_score(y_val, all_probs)),
        "mcc":     float(matthews_corrcoef(y_val, y_pred_class)),
        "brier":   float(brier_score_loss(y_val, all_probs)),
    }
    fold_metrics_ft.append(metrics)

    print(f"    ROC-AUC: {metrics['roc_auc']:.4f}  |  "
          f"PR-AUC: {metrics['pr_auc']:.4f}  |  "
          f"MCC: {metrics['mcc']:.4f}  |  "
          f"Brier: {metrics['brier']:.4f}")

# ── Overall OOF metrics ───────────────────────────────────
oof_class_ft = (oof_preds_ft >= 0.5).astype(int)
overall_ft = {
    "model":   "FT-Transformer",
    "roc_auc": float(roc_auc_score(y_labeled, oof_preds_ft)),
    "pr_auc":  float(average_precision_score(y_labeled, oof_preds_ft)),
    "mcc":     float(matthews_corrcoef(y_labeled, oof_class_ft)),
    "brier":   float(brier_score_loss(y_labeled, oof_preds_ft)),
}
print(f"\n  ── FT-Transformer Overall (OOF) ──")
print(f"    ROC-AUC: {overall_ft['roc_auc']:.4f}")
print(f"    PR-AUC:  {overall_ft['pr_auc']:.4f}")
print(f"    MCC:     {overall_ft['mcc']:.4f}")
print(f"    Brier:   {overall_ft['brier']:.4f}")


# ================================================================
# Cell 8 — Final Model Comparison & Selection
# ================================================================
# Compare ALL models (4 classical + FT-Transformer) and select
# the overall winner for downstream use.
#
# Cell 4.5 stored the best classical model info in:
#   best_baseline_name, best_classical_metrics, best_classical_preds
# The original overall_lgb, overall_xgb, etc. are UNTOUCHED.

print("\n" + "=" * 60)
print("  FINAL MODEL COMPARISON")
print("=" * 60)

# All 5 models in one table
all_model_overalls = [overall_lgb, overall_xgb, overall_rf, overall_dart, overall_ft]
comparison = pd.DataFrame(all_model_overalls).set_index("model")
comparison = comparison.sort_values("pr_auc", ascending=False)
print(comparison.to_string(float_format="{:.4f}".format))

# ── Selection criteria ─────────────────────────────────────
# Primary: PR-AUC (most relevant for imbalanced binary tasks)
# Tiebreaker: Brier score (calibration quality)

class_prevalence = y_labeled.mean()
print(f"\n  Random baseline PR-AUC: {class_prevalence:.4f}")

# Check all models beat random
for name, row in comparison.iterrows():
    if row["pr_auc"] <= class_prevalence:
        print(f"  ⚠️  {name} does NOT beat random PR-AUC baseline!")
    else:
        improvement = (row["pr_auc"] - class_prevalence) / class_prevalence * 100
        print(f"  ✅ {name}: +{improvement:.1f}% over random PR-AUC")

# Select winner: best classical (from Cell 4.5) vs. FT-Transformer
if overall_ft["pr_auc"] > best_classical_metrics["pr_auc"]:
    winner = "FT-Transformer"
    winner_preds = oof_preds_ft
    winner_metrics = overall_ft
elif overall_ft["pr_auc"] == best_classical_metrics["pr_auc"]:
    if overall_ft["brier"] < best_classical_metrics["brier"]:
        winner = "FT-Transformer"
        winner_preds = oof_preds_ft
        winner_metrics = overall_ft
    else:
        winner = best_baseline_name
        winner_preds = best_classical_preds
        winner_metrics = best_classical_metrics
else:
    winner = best_baseline_name
    winner_preds = best_classical_preds
    winner_metrics = best_classical_metrics

print(f"\n  🏆 Overall winner: {winner}")
print(f"     Best classical: {best_baseline_name} "
      f"(PR-AUC={best_classical_metrics['pr_auc']:.4f})")
print(f"     FT-Transformer: PR-AUC={overall_ft['pr_auc']:.4f}")
print(f"\n     Selected: {winner}")
print(f"     PR-AUC: {winner_metrics['pr_auc']:.4f}")
print(f"     Brier:  {winner_metrics['brier']:.4f}")


# ================================================================
# Cell 9 — Baseline Sanity Checks
# ================================================================
# Verify all trained models beat random and majority baselines.

print("\n" + "=" * 60)
print("  SANITY CHECK vs. TRIVIAL BASELINES")
print("=" * 60)

# Random baseline
random_preds = np.random.RandomState(RANDOM_SEED).rand(len(y_labeled))
random_metrics = {
    "model":   "Random",
    "roc_auc": roc_auc_score(y_labeled, random_preds),
    "pr_auc":  average_precision_score(y_labeled, random_preds),
    "mcc":     matthews_corrcoef(y_labeled, (random_preds >= 0.5).astype(int)),
    "brier":   brier_score_loss(y_labeled, random_preds),
}

# Majority baseline
majority_class = int(np.bincount(y_labeled).argmax())
majority_prob  = np.bincount(y_labeled).max() / len(y_labeled)
majority_preds = np.full(len(y_labeled), majority_prob)
majority_metrics = {
    "model":   "Majority",
    "roc_auc": 0.5,
    "pr_auc":  class_prevalence,
    "mcc":     0.0,
    "brier":   brier_score_loss(y_labeled, majority_preds),
}

# Full results table: trivial baselines + all trained models
all_results_list = [random_metrics, majority_metrics] + all_model_overalls
all_results = pd.DataFrame(all_results_list).set_index("model")
print(all_results.to_string(float_format="{:.4f}".format))

# Check each trained model vs. majority baseline
print()
for m in all_model_overalls:
    name = m["model"]
    pr_gain = m["pr_auc"] - class_prevalence
    tag = "✅ PASS" if pr_gain > 0.05 else "⚠️ MARGINAL" if pr_gain > 0 else "❌ FAIL"
    print(f"  {name:20s}: PR-AUC gain = +{pr_gain:.4f}  ({tag})")


# ================================================================
# Cell 10 — Save Out-of-Fold Predictions
# ================================================================
# Store OOF predictions from the best classical model and
# FT-Transformer for Phase 5 (calibration).

df_labeled_preds = df_labeled[["compound_id", "canonical_smiles",
                                "scaffold_smiles", "repellent_active",
                                "fold_id"]].copy()

# Save predictions from the best classical model and FT-Transformer
df_labeled_preds["p_repellent_best_classical_raw"] = best_classical_preds
df_labeled_preds["p_repellent_ft_raw"]             = oof_preds_ft
df_labeled_preds["p_repellent_winner_raw"]         = winner_preds
df_labeled_preds["selected_model"]                 = winner
df_labeled_preds["best_classical_model"]           = best_baseline_name

# Backward compatibility: Phase 5 reads "p_repellent_lgb_raw"
# We alias the best classical model's predictions to this column.
df_labeled_preds["p_repellent_lgb_raw"] = best_classical_preds

df_labeled_preds.to_parquet(OUTPUT_PREDICTIONS, index=False, engine="pyarrow")
print(f"\n✅ OOF predictions saved: {OUTPUT_PREDICTIONS}")
print(f"   Best classical: {best_baseline_name}")
print(f"   Overall winner: {winner}")


# ================================================================
# Cell 11 — Save Training Results
# ================================================================

training_results = {
    "pipeline":       "Phase 4 — Model Training (V1)",
    "random_seed":    RANDOM_SEED,
    "n_folds":        N_FOLDS,
    "selected_model": winner,
    "best_classical_model": best_baseline_name,
    "classical_models": {
        "LightGBM":       {"overall_metrics": baseline_metrics_map.get("LightGBM", {}),
                           "fold_metrics": fold_metrics_lgb},
        "XGBoost":        {"overall_metrics": overall_xgb,
                           "fold_metrics": fold_metrics_xgb},
        "RandomForest":   {"overall_metrics": overall_rf,
                           "fold_metrics": fold_metrics_rf},
        "LightGBM-DART":  {"overall_metrics": overall_dart,
                           "fold_metrics": fold_metrics_dart},
    },
    "models": {
        best_baseline_name: {
            "overall_metrics": best_classical_metrics,
        },
        "FT-Transformer": {
            "pretrain_config": {
                "epochs": PRETRAIN_EPOCHS,
                "lr": PRETRAIN_LR,
                "mask_ratio": MASK_RATIO,
                "d_token": D_TOKEN,
                "n_heads": N_HEADS,
                "n_layers": N_LAYERS,
                "d_ffn": D_FFN,
            },
            "finetune_config": {
                "epochs": FINETUNE_EPOCHS,
                "lr": FINETUNE_LR,
                "frozen_backbone": True,
            },
            "overall_metrics": overall_ft,
            "fold_metrics": fold_metrics_ft,
        },
    },
    "baselines": {
        "random": random_metrics,
        "majority": majority_metrics,
    },
}

with open(RESULTS_FILE, "w") as f:
    json.dump(training_results, f, indent=2, default=str)

print(f"✅ Training results saved: {RESULTS_FILE}")


# ================================================================
# Cell 12 — Save Best Classical Models (per fold)
# ================================================================
# Save each fold's best-classical-model for Phase 9 (screening).
# These are stored in lgb_models (updated by Cell 4.5 if needed).

print(f"\n  Saving {best_baseline_name} fold models...")

for i, model_info in enumerate(lgb_models):
    model_path = MODEL_DIR / f"lgb_fold_{i}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_info, f)

print(f"✅ Best classical models saved: {MODEL_DIR}/lgb_fold_*.pkl")
print(f"   Model type: {best_baseline_name}")
print(f"   (Filename uses lgb_fold_ prefix for backward compatibility)")


# ================================================================
# Cell 13 — Final Summary
# ================================================================

print("\n" + "=" * 60)
print("  PHASE 4 COMPLETE — Model Training (V1)")
print("=" * 60)

print(f"\n  Classical models trained:")
for m in [overall_lgb, overall_xgb, overall_rf, overall_dart]:
    tag = " 🏆 BEST" if m["model"] == best_baseline_name else ""
    print(f"    • {m['model']:20s}  PR-AUC={m['pr_auc']:.4f}  "
          f"Brier={m['brier']:.4f}{tag}")

print(f"\n  Deep learning:")
print(f"    • {'FT-Transformer':20s}  PR-AUC={overall_ft['pr_auc']:.4f}  "
      f"Brier={overall_ft['brier']:.4f}")

print(f"\n  🏆 Overall winner: {winner}")
print(f"     ROC-AUC: {winner_metrics['roc_auc']:.4f}")
print(f"     PR-AUC:  {winner_metrics['pr_auc']:.4f}")
print(f"     MCC:     {winner_metrics['mcc']:.4f}")
print(f"     Brier:   {winner_metrics['brier']:.4f}")
print(f"\n  Artifacts:")
print(f"    1. {OUTPUT_PREDICTIONS}")
print(f"    2. {RESULTS_FILE}")
print(f"    3. {MODEL_DIR}/")
print(f"\n  Ready for Phase 5 (Calibration & Uncertainty) →")
print("=" * 60)
