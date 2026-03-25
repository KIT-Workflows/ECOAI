# ================================================================
# STEP 5 — FT-TRANSFORMER (Insecticidal Pipeline)
# ================================================================
# Train the Tabular Deep Learning model (FT-Transformer).
# Step A: Self-supervised pretraining (Masked Feature Reconstruction)
# Step B: Fine-tuning on scaffold folds (Frozen Backbone)
#
# Best run on a GPU environment (like Google Colab).
#
# Input:  implementation/artifacts/insecticide_features.npz
#         implementation/artifacts/insecticide_meta.parquet
# Output: implementation/artifacts/ft_transformer_results.json
#         implementation/artifacts/ft_transformer_oof_preds.parquet
#         implementation/models/ft_transformer_pretrained.pt
# ================================================================


# ================================================================
# Cell 1 — Imports and Configuration
# ================================================================
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    matthews_corrcoef, brier_score_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import sys

warnings.filterwarnings("ignore")

# ── Colab Compatibility ────────────────────────────────────
if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Google Drive mounted.")
    
    # Try multiple common drive root patterns
    potential_roots = [
        Path("/content/drive/MyDrive/ECOAI"),
        Path("/content/drive/My Drive/ECOAI"), # Older Colab space pattern
    ]
    
    PROJECT_ROOT = None
    for pr in potential_roots:
        if pr.exists():
            PROJECT_ROOT = pr
            print(f"✅ Found project root: {PROJECT_ROOT}")
            break
            
    if PROJECT_ROOT is None:
        print("⚠️  Project root not found in standard locations. Searching Drive...")
        # Emergency search for the folder named ECOAI
        import os
        for root, dirs, files in os.walk("/content/drive/"):
            if "ECOAI" in dirs:
                PROJECT_ROOT = Path(root) / "ECOAI"
                print(f"✅ AUTO-FOUND project root: {PROJECT_ROOT}")
                break
                
    if PROJECT_ROOT is None:
        print("\n" + "!" * 60)
        print("  ❌ CRITICAL: 'ECOAI' folder not found in Google Drive!")
        print("  Please make sure you uploaded the 'ECOAI' folder to MyDrive.")
        print("!" * 60)
        raise FileNotFoundError("Could not locate ECOAI directory in Google Drive")
else:
    PROJECT_ROOT = Path(r"G:\research\ECOAI")

DATA_DIR     = PROJECT_ROOT / "Datasets" / "data"
IMPL_DIR     = PROJECT_ROOT / "experiment" / "phase10_v3" / "method1_chembl_bioactivity" / "implementation"
ARTIFACT_DIR = DATA_DIR / "phase10_artifacts"
MODEL_DIR    = IMPL_DIR / "models"

INPUT_FEATURES = ARTIFACT_DIR / "insecticide_features.npz"
INPUT_META     = ARTIFACT_DIR / "insecticide_meta.parquet"

# ── Robust Existence Check ─────────────────────────────────
if not INPUT_FEATURES.exists():
    print("\n" + "!" * 60)
    print("  ❌ FILE STILL NOT FOUND!")
    print(f"  Final Path checked: {INPUT_FEATURES}")
    print("!" * 60)
    
    # Trace the path to reveal the exact failure point
    parts = INPUT_FEATURES.parts
    current = Path(parts[0])
    print(f"\n  Checking path hierarchy step-by-step:")
    for part in parts[1:]:
        current = current / part
        exists = "✅" if current.exists() else "❌"
        print(f"    {exists} {current}")
    
    print("\n💡 TIP: Ensure your 'phase10_artifacts' folder is fully synced inside 'Datasets/data/'.")
    raise FileNotFoundError(f"Missing: {INPUT_FEATURES.name}")

assert INPUT_META.exists(), f"❌ {INPUT_META} not found!"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5
RANDOM_SEED = 42

print("=" * 65)
print("  STEP 5 — FT-TRANSFORMER MODEL")
print("=" * 65)
print(f"  Device: {DEVICE}")


# ================================================================
# Cell 2 — Load Data
# ================================================================
npz = np.load(INPUT_FEATURES)
X_all = npz["X"]
df_meta = pd.read_parquet(INPUT_META)

y = df_meta["insecticidal_active"].values.astype(int)
fold_ids = df_meta["fold_id"].values.astype(int)

n_features = X_all.shape[1]
print(f"\n✅ Features loaded: {X_all.shape}")


# ================================================================
# Cell 3 — FT-Transformer Architecture Definition
# ================================================================
class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token))
        self.bias   = nn.Parameter(torch.zeros(n_features, d_token))
    def forward(self, x):
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

class FTTransformer(nn.Module):
    def __init__(self, n_features, d_token=64, n_heads=4, n_layers=3, d_ffn=128, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        self.pos_encoding = nn.Parameter(torch.randn(1, n_features + 1, d_token))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_ffn,
            dropout=dropout, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_ffn), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ffn, 1))
        self.recon_head = nn.Sequential(
            nn.Linear(d_token, d_ffn), nn.GELU(), nn.Linear(d_ffn, 1))

    def encode(self, x):
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_encoding[:, :tokens.shape[1], :]
        return self.ln(self.transformer(tokens))

    def forward(self, x):
        return self.head(self.encode(x)[:, 0, :]).squeeze(-1)

    def reconstruct(self, x, mask):
        return self.recon_head(self.encode(x)[:, 1:, :]).squeeze(-1)

class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        if self.y is not None: return self.X[idx], self.y[idx]
        return self.X[idx],


# ================================================================
# Cell 4 — Self-Supervised Pretraining
# ================================================================
print("\n" + "=" * 60)
print("  PHASE 5.A — SELF-SUPERVISED PRETRAINING")
print("=" * 60)

# Impute and scale entire dataset for pretraining
full_imp = SimpleImputer(strategy="median")
X_all_imp = full_imp.fit_transform(X_all)
full_sc = StandardScaler()
X_all_std = full_sc.fit_transform(X_all_imp)

PRETRAIN_EPOCHS = 50; PRETRAIN_LR = 1e-3; PRETRAIN_BATCH = 64
MASK_RATIO = 0.15; D_TOKEN = 64; N_HEADS = 4; N_LAYERS = 3; D_FFN = 128; DROPOUT = 0.1

ft_model = FTTransformer(n_features, D_TOKEN, N_HEADS, N_LAYERS, D_FFN, DROPOUT).to(DEVICE)

loader = DataLoader(TabularDataset(X_all_std), batch_size=PRETRAIN_BATCH, shuffle=True, drop_last=True)
opt = optim.AdamW(ft_model.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PRETRAIN_EPOCHS)

print(f"  Pretraining {PRETRAIN_EPOCHS} epochs (Mask: {MASK_RATIO*100:.0f}%)")
ft_model.train()
for epoch in range(PRETRAIN_EPOCHS):
    eloss, nb = 0.0, 0
    for (bx,) in loader:
        bx = bx.to(DEVICE)
        mask = torch.rand_like(bx) < MASK_RATIO
        x_masked = bx.clone()
        x_masked[mask] = 0.0
        recon = ft_model.reconstruct(x_masked, mask)
        loss = nn.MSELoss()(recon[mask], bx[mask])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ft_model.parameters(), 1.0)
        opt.step(); eloss += loss.item(); nb += 1
    sched.step()
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"    Epoch {epoch+1:>3d}/{PRETRAIN_EPOCHS}  Loss: {eloss/max(nb,1):.6f}")

pretrained_path = MODEL_DIR / "ft_transformer_pretrained.pt"
torch.save(ft_model.state_dict(), pretrained_path)
print(f"  ✅ Pretrained weights saved: {pretrained_path.name}")


# ================================================================
# Cell 5 — Fine-Tuning (Scaffold-Split CV)
# ================================================================
print("\n" + "=" * 60)
print("  PHASE 5.B — FINE-TUNING (Frozen Backbone)")
print("=" * 60)

FINETUNE_EPOCHS = 80; FINETUNE_LR = 5e-4; FINETUNE_BATCH = 32

oof_preds_ft = np.zeros(len(y))
fold_metrics_ft = []
finetuned_models = []

for fold in range(N_FOLDS):
    print(f"\n  ── Fold {fold} ──")
    train_mask = (fold_ids != fold)
    val_mask = (fold_ids == fold)
    
    # Preprocess
    imp = SimpleImputer(strategy="median")
    Xti = imp.fit_transform(X_all[train_mask])
    Xvi = imp.transform(X_all[val_mask])
    sc = StandardScaler()
    Xts = sc.fit_transform(Xti)
    Xvs = sc.transform(Xvi)

    # Load frozen model
    model = FTTransformer(n_features, D_TOKEN, N_HEADS, N_LAYERS, D_FFN, DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE, weights_only=True))
    
    # Freeze backbone
    for name, param in model.named_parameters():
        if "head" not in name or "recon_head" in name:
            param.requires_grad = False
            
    if fold == 0:
        tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Trainable params: {tp:,}")

    train_loader = DataLoader(TabularDataset(Xts, y[train_mask]), batch_size=FINETUNE_BATCH, shuffle=True, drop_last=True)
    val_loader = DataLoader(TabularDataset(Xvs, y[val_mask]), batch_size=FINETUNE_BATCH, shuffle=False)

    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=FINETUNE_LR, weight_decay=1e-3)
    crit = nn.BCEWithLogitsLoss()
    
    best_vl, patience, wait, best_st = float("inf"), 15, 0, None
    model.train()
    
    for epoch in range(FINETUNE_EPOCHS):
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            loss = crit(model(bx), by)
            opt.zero_grad(); loss.backward(); opt.step()
            
        model.eval()
        vl = sum(crit(model(bx.to(DEVICE)), by.to(DEVICE)).item() for bx, by in val_loader) / max(len(val_loader), 1)
        model.train()
        
        if vl < best_vl:
            best_vl = vl
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience: break

    model.load_state_dict(best_st)
    model.eval()
    finetuned_models.append({"model_st": best_st, "imputer": imp, "scaler": sc})
    
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.FloatTensor(Xvs).to(DEVICE))).cpu().numpy()
    oof_preds_ft[val_mask] = probs

    yc = (probs >= 0.5).astype(int)
    m = {
        "fold": fold,
        "roc_auc": float(roc_auc_score(y[val_mask], probs)),
        "pr_auc": float(average_precision_score(y[val_mask], probs)),
        "mcc": float(matthews_corrcoef(y[val_mask], yc)),
        "brier": float(brier_score_loss(y[val_mask], probs))
    }
    fold_metrics_ft.append(m)
    print(f"    ROC-AUC: {m['roc_auc']:.4f}  |  PR-AUC: {m['pr_auc']:.4f}  |  "
          f"MCC: {m['mcc']:.4f}  |  Brier: {m['brier']:.4f}")

oof_c = (oof_preds_ft >= 0.5).astype(int)
overall_ft = {
    "model": "FT-Transformer",
    "roc_auc": float(roc_auc_score(y, oof_preds_ft)),
    "pr_auc": float(average_precision_score(y, oof_preds_ft)),
    "mcc": float(matthews_corrcoef(y, oof_c)),
    "brier": float(brier_score_loss(y, oof_preds_ft))
}
print(f"\n  ── Overall ──")
print(f"     ROC-AUC: {overall_ft['roc_auc']:.4f}")
print(f"     PR-AUC:  {overall_ft['pr_auc']:.4f}")
print(f"     Brier:   {overall_ft['brier']:.4f}")


# ================================================================
# Cell 6 — Save Results
# ================================================================
df_preds = df_meta[["compound_id", "inchikey", "canonical_smiles", "fold_id"]].copy()
df_preds["p_insecticidal_raw"] = oof_preds_ft
preds_path = ARTIFACT_DIR / "ft_transformer_oof_preds.parquet"
df_preds.to_parquet(preds_path, index=False)
print(f"  ✅ Saved predictions: {preds_path}")

# Save fold models
for i, mi in enumerate(finetuned_models):
    path = MODEL_DIR / f"ft_transformer_fold_{i}.pt"
    torch.save(mi["model_st"], path)
    
results = {
    "model": "FT-Transformer",
    "overall": overall_ft,
    "fold": fold_metrics_ft,
    "config": {
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "mask_ratio": MASK_RATIO,
        "finetune_epochs": FINETUNE_EPOCHS,
        "frozen_backbone": True,
        "d_token": D_TOKEN, "n_heads": N_HEADS, "n_layers": N_LAYERS
    }
}
res_path = ARTIFACT_DIR / "ft_transformer_results.json"
with open(res_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n  → Next: Run step6_model_selection.py to compare Classical vs. FT-Transformer")
print("=" * 65)
