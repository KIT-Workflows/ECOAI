# ================================================================
# STEP 4 — CLASSICAL MODELS (Insecticidal Pipeline)
# ================================================================
# Train gradient boosting baselines (LightGBM, XGBoost, etc.)
# on scaffold-split folds. Compare Out-Of-Fold (OOF) performance
# to select the strongest baseline.
#
# Input:  implementation/artifacts/insecticide_features.npz
#         implementation/artifacts/insecticide_meta.parquet
# Output: implementation/artifacts/classical_results.json
#         implementation/artifacts/classical_oof_preds.parquet
#         implementation/models/lgb_fold_*.pkl (or best baseline)
# ================================================================


# ================================================================
# Cell 1 — Imports and Configuration
# ================================================================
import json
import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    matthews_corrcoef, brier_score_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"G:\research\ECOAI")
DATA_DIR     = PROJECT_ROOT / "Datasets" / "data"
IMPL_DIR     = PROJECT_ROOT / "experiment" / "phase10_v3" / "method1_chembl_bioactivity" / "implementation"
ARTIFACT_DIR = DATA_DIR / "phase10_artifacts"
MODEL_DIR    = IMPL_DIR / "models"

INPUT_FEATURES = ARTIFACT_DIR / "insecticide_features.npz"
INPUT_META     = ARTIFACT_DIR / "insecticide_meta.parquet"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
assert INPUT_FEATURES.exists(), f"❌ {INPUT_FEATURES} — Run step2 first!"
assert INPUT_META.exists(), f"❌ {INPUT_META} — Run step3 first!"

N_FOLDS = 5
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 65)
print("  STEP 4 — CLASSICAL MODELS")
print("=" * 65)


# ================================================================
# Cell 2 — Load Data
# ================================================================
npz = np.load(INPUT_FEATURES)
X_all = npz["X"]
df_meta = pd.read_parquet(INPUT_META)

y = df_meta["insecticidal_active"].values.astype(int)
fold_ids = df_meta["fold_id"].values.astype(int)

class_prevalence = y.mean()

print(f"\n✅ Loaded features: {X_all.shape}")
print(f"   Shape: {X_all.shape}")
print(f"   Prevalence (Random Bseline): {class_prevalence:.4f}")


# ================================================================
# Cell 3 — Shared Utilities
# ================================================================
def preprocess_features(X_train, X_val):
    imp = SimpleImputer(strategy="median")
    X_ti = imp.fit_transform(X_train)
    X_vi = imp.transform(X_val)
    sc = StandardScaler()
    return {"X_train": sc.fit_transform(X_ti), "X_val": sc.transform(X_vi),
            "imputer": imp, "scaler": sc}

def train_and_evaluate(model_name, model_fn, params, fit_params=None):
    print(f"\n" + "=" * 60)
    print(f"  MODEL — {model_name}")
    print("=" * 60)

    oof_preds = np.zeros(len(y))
    fold_metrics = []
    models = []
    
    for fold in range(N_FOLDS):
        print(f"\n  ── Fold {fold} ──")
        train_mask = (fold_ids != fold)
        val_mask = (fold_ids == fold)
        
        prep = preprocess_features(X_all[train_mask], X_all[val_mask])
        model = model_fn(**params)
        
        # Build eval set if required
        kwargs = fit_params.copy() if fit_params else {}
        if kwargs.get("eval_set"):
            kwargs["eval_set"] = [(prep["X_val"], y[val_mask])]
            
        model.fit(prep["X_train"], y[train_mask], **kwargs)
        
        yp = model.predict_proba(prep["X_val"])[:, 1]
        oof_preds[val_mask] = yp
        yc = (yp >= 0.5).astype(int)
        
        m = {
            "fold": fold,
            "roc_auc": float(roc_auc_score(y[val_mask], yp)),
            "pr_auc": float(average_precision_score(y[val_mask], yp)),
            "mcc": float(matthews_corrcoef(y[val_mask], yc)),
            "brier": float(brier_score_loss(y[val_mask], yp))
        }
        fold_metrics.append(m)
        models.append({"model": model, "imputer": prep["imputer"], "scaler": prep["scaler"]})
        
        print(f"    ROC-AUC: {m['roc_auc']:.4f}  |  PR-AUC: {m['pr_auc']:.4f}  |  "
              f"MCC: {m['mcc']:.4f}  |  Brier: {m['brier']:.4f}")

    oof_c = (oof_preds >= 0.5).astype(int)
    overall = {
        "model": model_name,
        "roc_auc": float(roc_auc_score(y, oof_preds)),
        "pr_auc": float(average_precision_score(y, oof_preds)),
        "mcc": float(matthews_corrcoef(y, oof_c)),
        "brier": float(brier_score_loss(y, oof_preds))
    }
    print(f"\n  ── {model_name} Overall ──")
    print(f"     ROC-AUC: {overall['roc_auc']:.4f}")
    print(f"     PR-AUC:  {overall['pr_auc']:.4f}")
    print(f"     Brier:   {overall['brier']:.4f}")
    
    return oof_preds, overall, fold_metrics, models


# ================================================================
# Cell 4 — LightGBM Baseline
# ================================================================
lgb_params = {
    "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
    "n_estimators": 500, "learning_rate": 0.05, "max_depth": 6,
    "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "min_child_samples": 10,
    "random_state": RANDOM_SEED, "verbose": -1, "n_jobs": -1,
}
oof_lgb, ov_lgb, fm_lgb, m_lgb = train_and_evaluate(
    "LightGBM", lgb.LGBMClassifier, lgb_params,
    fit_params={"eval_set": True, "callbacks": [lgb.early_stopping(50, verbose=False)]}
)


# ================================================================
# Cell 5 — XGBoost Baseline
# ================================================================
xgb_params = {
    "objective": "binary:logistic", "eval_metric": "logloss", "booster": "gbtree",
    "n_estimators": 500, "learning_rate": 0.05, "max_depth": 6,
    "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1,
    "reg_lambda": 1.0, "min_child_weight": 5,
    "random_state": RANDOM_SEED, "verbosity": 0, "n_jobs": -1,
}
oof_xgb, ov_xgb, fm_xgb, m_xgb = train_and_evaluate(
    "XGBoost", XGBClassifier, xgb_params,
    fit_params={"eval_set": True, "verbose": False}
)


# ================================================================
# Cell 6 — Random Forest Baseline
# ================================================================
rf_params = {
    "n_estimators": 500, "max_depth": None, "min_samples_split": 5,
    "min_samples_leaf": 2, "max_features": "sqrt",
    "class_weight": "balanced", "random_state": RANDOM_SEED, "n_jobs": -1,
}
oof_rf, ov_rf, fm_rf, m_rf = train_and_evaluate(
    "RandomForest", RandomForestClassifier, rf_params
)


# ================================================================
# Cell 7 — LightGBM-DART Baseline
# ================================================================
dart_params = dict(lgb_params)
dart_params.update({
    "boosting_type": "dart",
    "learning_rate": 0.03, "max_depth": 8, "num_leaves": 50,
    "drop_rate": 0.1, "skip_drop": 0.5,
})
oof_dart, ov_dart, fm_dart, m_dart = train_and_evaluate(
    "LightGBM-DART", lgb.LGBMClassifier, dart_params,
    fit_params={"eval_set": True, "callbacks": [lgb.early_stopping(50, verbose=False)]}
)


# ================================================================
# Cell 8 — Comparison & Winner Selection
# ================================================================
print("\n" + "=" * 60)
print("  CLASSICAL BASELINES — COMPARISON")
print("=" * 60)

all_results = [ov_lgb, ov_xgb, ov_rf, ov_dart]
df_results = pd.DataFrame(all_results).set_index("model")
df_results = df_results.sort_values("pr_auc", ascending=False)
print(df_results.to_string(float_format="{:.4f}".format))

# Select best
best_name = df_results.index[0]
print(f"\n  🏆 Best classical: {best_name} (PR-AUC={df_results.loc[best_name, 'pr_auc']:.4f})")

# Map of things to save
model_map = {
    "LightGBM": (oof_lgb, fm_lgb, m_lgb, ov_lgb),
    "XGBoost": (oof_xgb, fm_xgb, m_xgb, ov_xgb),
    "RandomForest": (oof_rf, fm_rf, m_rf, ov_rf),
    "LightGBM-DART": (oof_dart, fm_dart, m_dart, ov_dart),
}
best_oof, best_fm, best_models, best_ov = model_map[best_name]

# Save OOF predictions
df_preds = df_meta[["compound_id", "inchikey", "canonical_smiles", "fold_id"]].copy()
df_preds["p_insecticidal_raw"] = best_oof
preds_path = ARTIFACT_DIR / "classical_oof_preds.parquet"
df_preds.to_parquet(preds_path, index=False)
print(f"  ✅ Saved predictions: {preds_path}")

# Save fold models
for i, mi in enumerate(best_models):
    path = MODEL_DIR / f"classical_fold_{i}.pkl"
    with open(path, "wb") as f:
        pickle.dump(mi, f)
print(f"  💾 Saved {N_FOLDS} {best_name} fold models.")

# Save results JSON
results = {
    "best_classical_model": best_name,
    "best_metrics": best_ov,
    "random_baseline": class_prevalence,
    "all_models": {
        "LightGBM": {"overall": ov_lgb, "fold": fm_lgb},
        "XGBoost": {"overall": ov_xgb, "fold": fm_xgb},
        "RandomForest": {"overall": ov_rf, "fold": fm_rf},
        "LightGBM-DART": {"overall": ov_dart, "fold": fm_dart},
    }
}
res_path = ARTIFACT_DIR / "classical_results.json"
with open(res_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n  → Next: Run step5_ft_transformer.py (Colab/GPU advised) OR proceed to step6")
print("=" * 65)
