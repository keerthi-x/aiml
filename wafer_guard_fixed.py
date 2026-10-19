#!/usr/bin/env python3
"""WAFER-GUARD · Fab-7 Excursion Detector — Production-Repaired Pipeline (wafer_guard_fixed.py)"""

# ============================================================================
# # WAFER-GUARD · Fab-7 Excursion Detector — Production-Repaired Pipeline & Forensic Audit (`wafer_guard_fixed.ipynb`)
# 
# **Release Board Technical Review & Repaired Reference Implementation**
# 
# This notebook repairs all silent defects identified in the pre-release `bnypi.ipynb` submission and introduces a **Physics-Guided, Observability-Masked Supervised Autoencoder (`WaferGuardSAE`)** designed specifically for inline semiconductor metrology with multi-generation probe-station telemetry.
# 
# ### Summary of Repaired Silent Defects
# 1. **Strict Physical & Statistical Causality (`t' > t` Forbidden)**:
#    - Sorts all lots chronologically by `timestamp` prior to partitioning (`Train [70%] → Temporal Validation [10%] → Chronological Test Holdout [20%]`).
#    - Eliminates **pre-split `SMOTE`**, **pre-split `SimpleImputer`**, and **pre-split `StandardScaler`**, which previously leaked future evaluation statistics into training and fabricated a **50.0% synthetic test set** (~93% synthetic minority test points).
#    - Evaluates strictly on the untouched chronological test population at **natural defect prevalence (~6.6%)**, with decision threshold $\tau^*$ calibrated exclusively on the temporal validation split.
# 2. **Corrected Tensor Mechanics & Gradient Fidelity (`[B, F]` Alignment)**:
#    - Fixes the silent Rank-3 PyTorch broadcasting bug in `recon_loss = torch.mean((reconstructed - batch_x) ** 2)` where `reconstructed` (`[B, F] = [64, 590]`) was subtracted from unsqueezed `batch_x` (`[B, 1, F] = [64, 1, 590]`), producing a `[64, 64, 590]` all-pairs cross-batch tensor that forced the decoder gradient $\nabla_{\hat{x}_i}\mathcal{L}_{\text{recon}} = \frac{2}{BF}(\hat{x}_i - \bar{x}_{\text{batch}})$ toward the mini-batch mean vector.
# 3. **Domain & Statistical Invariant Preservation (Probe Station PS-L Tool-Matching & Zero-Variance Pruning)**:
#    - Isolates **Legacy Probe Station PS-L** lots (`sensor_0 == 0.0` sentinel and polymorphic `sensor_1` categorical firmware error codes `100.0`, `200.0`, `300.0`).
#    - Fits continuous imputation and scaling for `sensor_0` and `sensor_1` **strictly on modern-station training lots (`~is_psl`)**, encodes PS-L firmware states as dedicated unscaled one-hot channels, and gates PS-L continuous `sensor_0`/`sensor_1` channels via an **Observability Mask $M \in \{0,1\}^{B \times F_{\text{cont}}}$** so categorical codes are **never included in continuous scaling and never overwritten with fabricated continuous values**.
#    - Prunes invariant/zero-variance continuous sensors (`std == 0` or all-`NaN`) learned strictly on the training split.
# ============================================================================

# ============================================================================
# ## 0 · Environment & Reproducibility
# ============================================================================

import os
import math
import random
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    brier_score_loss,
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 1337
def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HP = dict(
    batch_size=64,
    epochs=45,
    lr=1e-3,
    weight_decay=1e-4,
    hidden=256,
    latent=32,
    lambda_clf=1.5,
    focal_gamma=1.5,
    train_frac=0.70,
    val_frac=0.10,
    test_frac=0.20,
)
print(f"PyTorch {torch.__version__} | device={DEVICE} | seed={SEED}")


# ============================================================================
# ## 1 · Data Ingestion & Forensic Audit of Legacy / Invariant Telemetry
# 
# We load `secom_telemetry_modified.csv`, parse `timestamp`, and run automated forensic diagnostics to quantify:
# 1. Natural defect prevalence (`label == 1`) and temporal ordering.
# 2. **Legacy Probe Station PS-L** prevalence (`sensor_0 == 0.0`) and exact correspondence with polymorphic categorical error codes in `sensor_1` (`100.0`, `200.0`, `300.0`).
# 3. **Invariant / Zero-Variance Channels**: sensors with zero variance or 100% missingness that dilute autoencoder MSE gradients when kept via `keep_empty_features=True`.
# 4. **Tensor Broadcasting Proof**: mathematical and empirical verification of how `[64, 590] - [64, 1, 590]` corrupts reconstruction gradients into `[64, 64, 590]`.
# ============================================================================

DATA_PATH = "secom_telemetry_modified.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find {DATA_PATH} in current working directory ({os.getcwd()}). "
        "Place secom_telemetry_modified.csv alongside wafer_guard_fixed.ipynb before running."
    )

df_raw = pd.read_csv(DATA_PATH)
SENSOR_COLS = [c for c in df_raw.columns if c.startswith("sensor_")]
assert len(SENSOR_COLS) == 590, f"Expected 590 sensor columns, found {len(SENSOR_COLS)}"

# Parse timestamps and inspect chronological order
df_raw["timestamp_dt"] = pd.to_datetime(df_raw["timestamp"])
is_monotonic_originally = df_raw["timestamp_dt"].is_monotonic_increasing

# Sort strictly by timestamp ascending (ties broken deterministically by original index)
df = df_raw.sort_values("timestamp_dt", kind="mergesort").reset_index(drop=True)

X_all_raw = df[SENSOR_COLS].to_numpy(dtype=np.float64)
y_all = (df["label"].to_numpy() == 1).astype(np.int64)

# --- Forensic Check 1: Legacy Probe Station PS-L Sentinel & Categorical Error Codes ---
s0 = df["sensor_0"].to_numpy(dtype=np.float64)
s1 = df["sensor_1"].to_numpy(dtype=np.float64)
PSL_ERROR_CODES = (100.0, 200.0, 300.0)

is_psl_s0 = (s0 == 0.0)
is_psl_s1 = np.isin(s1, PSL_ERROR_CODES)
psl_exact_match = np.array_equal(is_psl_s0, is_psl_s1)

print("=== FORENSIC AUDIT SUMMARY: DATASET & TELEMETRY INVARIANTS ===")
print(f"Total wafers: {len(df):,} | Sensors: {len(SENSOR_COLS)} | Originally chronological: {is_monotonic_originally}")
print(f"Time span: {df['timestamp_dt'].min()} -> {df['timestamp_dt'].max()}")
print(f"Natural defect rate: {y_all.mean():.2%} ({y_all.sum()} fail / {(1 - y_all).sum()} pass)")
print(f"Overall missing cells: {np.isnan(X_all_raw).mean():.2%}")
print("-" * 72)
print(f"Legacy PS-L wafers (sensor_0 == 0.0): {is_psl_s0.sum():,} ({is_psl_s0.mean():.2%})")
print(f"Categorical sensor_1 in {PSL_ERROR_CODES}: {is_psl_s1.sum():,} ({is_psl_s1.mean():.2%})")
print(f"Exact equivalence (sensor_0 == 0.0 <=> sensor_1 in {PSL_ERROR_CODES}): {psl_exact_match}")
for code in PSL_ERROR_CODES:
    mask_c = is_psl_s0 & (s1 == code)
    defect_c = y_all[mask_c].mean() if mask_c.sum() > 0 else 0.0
    print(f"  - PS-L Firmware Code {code:5.1f}: n={mask_c.sum():4d} | defect rate={defect_c:.2%}")
modern_mask = ~is_psl_s0
print(f"  - Modern Station (~PS-L)  : n={modern_mask.sum():4d} | defect rate={y_all[modern_mask].mean():.2%}")

# Show distortion caused by naive continuous scaling on sensor_0 and sensor_1
naive_s0_mean, naive_s0_std = np.nanmean(s0), np.nanstd(s0)
clean_s0_mean, clean_s0_std = np.nanmean(s0[modern_mask]), np.nanstd(s0[modern_mask])
naive_s1_mean, naive_s1_std = np.nanmean(s1), np.nanstd(s1)
clean_s1_mean, clean_s1_std = np.nanmean(s1[modern_mask]), np.nanstd(s1[modern_mask])
print("-" * 72)
print(f"sensor_0 moments -> Naive (with 0.0 sentinel): mean={naive_s0_mean:.2f}, std={naive_s0_std:.2f} "
      f"| Clean modern-only: mean={clean_s0_mean:.2f}, std={clean_s0_std:.2f}")
print(f"sensor_1 moments -> Naive (with 100/200/300): mean={naive_s1_mean:.2f}, std={naive_s1_std:.2f} "
      f"| Clean modern-only: mean={clean_s1_mean:.2f}, std={clean_s1_std:.2f}")

# --- Forensic Check 2: Tensor Broadcasting Proof ([64, 590] vs [64, 1, 590]) ---
demo_B, demo_F = 64, 590
demo_batch_x = torch.randn(demo_B, 1, demo_F)
demo_recon = torch.randn(demo_B, demo_F, requires_grad=True)
buggy_diff = demo_recon - demo_batch_x
buggy_loss = torch.mean(buggy_diff ** 2)
buggy_loss.backward()
grad_buggy = demo_recon.grad.clone()

# Analytical batch-mean gradient: (2 / (B * F)) * (recon_i - mean_j(batch_x_j))
batch_mean_x = demo_batch_x.squeeze(1).mean(dim=0, keepdim=True)
grad_analytical_mean = (2.0 / (demo_B * demo_F)) * (demo_recon.detach() - batch_mean_x)
max_grad_discrepancy = torch.max(torch.abs(grad_buggy - grad_analytical_mean)).item()

print("-" * 72)
print(f"Broadcast bug shape: reconstructed {tuple(demo_recon.shape)} - batch_x {tuple(demo_batch_x.shape)} "
      f"-> {tuple(buggy_diff.shape)} ({buggy_diff.numel():,} elements vs expected {demo_recon.numel():,})")
print(f"Max diff between buggy gradient and batch-mean collapse gradient: {max_grad_discrepancy:.2e}")


# ============================================================================
# ## 2 · Causality-Strict Temporal Partitioning & Regime-Gated Preprocessor
# 
# ### Why Naive `SimpleImputer` Over `NaN`-Replaced PS-L Codes Still Violates the Spec
# The specification states:
# > *"Identify and handle these categorical codes appropriately. Do not include them in continuous-feature scaling or impute fabricated continuous values over them."*
# 
# If one simply replaces `sensor_0 == 0.0` and `sensor_1 in {100.0, 200.0, 300.0}` with `np.nan` and passes the matrix to `SimpleImputer(strategy="mean")`, the imputer overwrites every PS-L wafer with the **modern station's continuous mean** — imputing a fabricated continuous value over a categorical regime and forcing the autoencoder MSE loss to reconstruct that fabricated value.
# 
# ### Our Solution: `FabTelemetryPreprocessor` with Observability Masking
# 1. **Strict Chronological Split**:
#    - First **70%** of wafers by `timestamp` $\rightarrow$ **Train** (`t <= t_train_end`)
#    - Next **10%** of wafers by `timestamp` $\rightarrow$ **Validation** (`t_train_end < t <= t_val_end`) for threshold $\tau^*$ calibration and early stopping
#    - Final **20%** of wafers by `timestamp` $\rightarrow$ **Test Holdout** (`t > t_val_end`) at untouched natural defect prevalence (~6.6%).
# 2. **Dual-Path Tool-Matching (`sensor_0` & `sensor_1`)**:
#    - Detects PS-L legacy wafers per row: `is_psl = (sensor_0 == 0.0) | np.isin(sensor_1, [100.0, 200.0, 300.0])`.
#    - Builds 4 explicit, unscaled binary regime features: `[is_psl, psl_err_100, psl_err_200, psl_err_300]`.
#    - Fits continuous median/mean imputation and `StandardScaler` for `sensor_0` and `sensor_1` **exclusively on modern-station (`~is_psl`) training lots**.
#    - On PS-L lots, sets standardized continuous `sensor_0` and `sensor_1` to `0.0` (gated off, never imputed with fabricated modern continuous values) and sets their entries in the **Observability Mask $M \in \{0, 1\}^{N \times F_{\text{cont}}}$** to `0.0`.
# 3. **Training-Only Invariant Sensor Pruning**:
#    - Identifies continuous sensors with zero variance (`std <= 1e-8`) or 100% missingness **on `X_train` only** and removes them across all splits.
# ============================================================================

class FabTelemetryPreprocessor:
    """
    Causality-safe, regime-gated preprocessor for Fab-7 wafer telemetry.
    - Separates Probe Station PS-L sentinel (sensor_0 == 0.0) and categorical firmware
      error codes (sensor_1 in {100.0, 200.0, 300.0}) from continuous metrology channels.
    - Never includes PS-L sentinel/error codes in continuous imputation/scaling.
    - Never imputes fabricated continuous values over PS-L sensor_0/sensor_1; instead,
      gates their continuous channels to 0.0 and records an observability mask M in {0, 1}.
    - Prunes zero-variance / all-NaN invariant channels using training statistics only.
    """
    PSL_CODES = (100.0, 200.0, 300.0)

    def __init__(self, impute_strategy: str = "median", clip_z: float = 10.0):
        self.impute_strategy = impute_strategy
        self.clip_z = clip_z
        self.active_cont_indices_: Optional[np.ndarray] = None
        self.cont_means_: Optional[np.ndarray] = None
        self.cont_stds_: Optional[np.ndarray] = None
        self.n_raw_features_: int = 0
        self.n_invariant_dropped_: int = 0

    @staticmethod
    def _detect_psl_and_codes(X_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s0 = X_raw[:, 0]
        s1 = X_raw[:, 1]
        is_psl = (s0 == 0.0) | np.isin(s1, FabTelemetryPreprocessor.PSL_CODES)
        cat_feats = np.column_stack([
            is_psl.astype(np.float32),
            (is_psl & (s1 == 100.0)).astype(np.float32),
            (is_psl & (s1 == 200.0)).astype(np.float32),
            (is_psl & (s1 == 300.0)).astype(np.float32),
        ])
        return is_psl, cat_feats

    def fit(self, X_train_raw: np.ndarray) -> "FabTelemetryPreprocessor":
        X = np.array(X_train_raw, dtype=np.float64, copy=True)
        self.n_raw_features_ = X.shape[1]
        is_psl, _ = self._detect_psl_and_codes(X)

        # Mask out PS-L sentinel/categorical entries in sensor_0 and sensor_1 so they
        # NEVER participate in continuous moment estimation
        X[is_psl, 0] = np.nan
        X[is_psl, 1] = np.nan

        # Identify non-empty, non-constant continuous channels strictly on X_train
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            col_stds = np.nanstd(X, axis=0)
            col_all_nan = np.isnan(X).all(axis=0)

        active_mask = (~col_all_nan) & (col_stds > 1e-8)
        # Always keep sensor_0 and sensor_1 in the active list if modern stations have variance
        self.active_cont_indices_ = np.where(active_mask)[0]
        self.n_invariant_dropped_ = int((~active_mask).sum())

        X_active = X[:, self.active_cont_indices_]
        if self.impute_strategy == "median":
            self.cont_means_ = np.nanmedian(X_active, axis=0)
        else:
            self.cont_means_ = np.nanmean(X_active, axis=0)

        # Compute standard deviation after filling genuine dropouts on training set
        X_filled = np.where(np.isnan(X_active), self.cont_means_, X_active)
        self.cont_stds_ = np.std(X_filled, axis=0)
        self.cont_stds_ = np.where(self.cont_stds_ <= 1e-8, 1.0, self.cont_stds_)
        return self

    def transform(self, X_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          X_full: [N, F_cont + 4] concatenated standardized continuous features + unscaled PS-L one-hot features
          obs_mask: [N, F_cont] float32 mask (1.0 where continuous value was genuinely observed, 0.0 where missing or PS-L gated)
          cat_feats: [N, 4] explicit PS-L regime features
        """
        assert self.active_cont_indices_ is not None, "Preprocessor must be fit before transform."
        X = np.array(X_raw, dtype=np.float64, copy=True)
        is_psl, cat_feats = self._detect_psl_and_codes(X)

        # Mark PS-L sensor_0 and sensor_1 as non-continuous before extracting active columns
        X[is_psl, 0] = np.nan
        X[is_psl, 1] = np.nan

        X_active = X[:, self.active_cont_indices_]
        # Observability mask is 1.0 ONLY for genuinely measured continuous sensor cells
        obs_mask = (~np.isnan(X_active)).astype(np.float32)

        # Standardize observed cells; unobserved cells (missing dropouts or PS-L gated sensor_0/1)
        # become 0.0 in standardized space (no fabricated continuous value is reconstructed)
        X_imputed = np.where(np.isnan(X_active), self.cont_means_, X_active)
        X_scaled = (X_imputed - self.cont_means_) / self.cont_stds_
        if self.clip_z is not None:
            X_scaled = np.clip(X_scaled, -self.clip_z, self.clip_z)

        # Explicitly zero out gated/unobserved entries so PS-L sensor_0/sensor_1 carry 0.0
        X_scaled = (X_scaled * obs_mask).astype(np.float32)

        X_full = np.hstack([X_scaled, cat_feats]).astype(np.float32)
        return X_full, obs_mask, cat_feats

# --- Chronological Train / Val / Test Split (Strictly Respecting Timestamp Causality) ---
N_total = len(df)
n_train = int(N_total * HP["train_frac"])
n_val = int(N_total * HP["val_frac"])

X_train_raw, y_train = X_all_raw[:n_train], y_all[:n_train]
X_val_raw,   y_val   = X_all_raw[n_train:n_train + n_val], y_all[n_train:n_train + n_val]
X_test_raw,  y_test  = X_all_raw[n_train + n_val:], y_all[n_train + n_val:]

t_train_max = df["timestamp_dt"].iloc[n_train - 1]
t_val_min, t_val_max = df["timestamp_dt"].iloc[n_train], df["timestamp_dt"].iloc[n_train + n_val - 1]
t_test_min = df["timestamp_dt"].iloc[n_train + n_val]

assert t_train_max <= t_val_min and t_val_max <= t_test_min, "Temporal causality violation detected!"

preprocessor = FabTelemetryPreprocessor(impute_strategy="median", clip_z=10.0)
preprocessor.fit(X_train_raw)

X_train, M_train, C_train = preprocessor.transform(X_train_raw)
X_val,   M_val,   C_val   = preprocessor.transform(X_val_raw)
X_test,  M_test,  C_test  = preprocessor.transform(X_test_raw)

F_cont = M_train.shape[1]
F_cat = C_train.shape[1]
F_total = X_train.shape[1]

print("=== CAUSALITY-SAFE CHRONOLOGICAL PARTITION & PREPROCESSING ===")
print(f"Train : {X_train.shape[0]:4d} lots | defect rate: {y_train.mean():.2%} ({y_train.sum()} fail) | up to {t_train_max}")
print(f"Val   : {X_val.shape[0]:4d} lots | defect rate: {y_val.mean():.2%} ({y_val.sum()} fail) | {t_val_min} -> {t_val_max}")
print(f"Test  : {X_test.shape[0]:4d} lots | defect rate: {y_test.mean():.2%} ({y_test.sum()} fail) | from {t_test_min}")
print(f"Active continuous sensors: {F_cont} (dropped {preprocessor.n_invariant_dropped_} invariant/constant channels fit on Train)")
print(f"Regime categorical features: {F_cat} ([is_psl, err_100, err_200, err_300]) -> Total input dim: {F_total}")


# ============================================================================
# ## 3 · Causality-Safe Training Rebalancing & DataLoaders
# 
# To handle the ~6.6% minority class **without leaking across splits or corrupting the normal autoencoder manifold**:
# 1. Validation (`X_val, y_val`) and Test (`X_test, y_test`) are **never** resampled; they remain at natural fab defect prevalence.
# 2. On the training set only, we apply `SMOTE` within each tool regime (so SMOTE never interpolates between a modern station wafer and a legacy PS-L wafer to create fractional categorical error codes!), and we flag synthetic SMOTE samples with `is_real_normal = 0` so the autoencoder's reconstruction branch trains exclusively on **real, physical Pass (`y == 0`) wafers**.
# ============================================================================

def custom_causal_smote(
    X_tr: np.ndarray,
    M_tr: np.ndarray,
    y_tr: np.ndarray,
    f_cont: int,
    k_neighbors: int = 5,
    seed: int = SEED,
    target_ratio: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Custom from-scratch regime-aware k-NN minority oversampler (General Guideline #4).
    - Computes exact L2 pairwise distances among minority training lots without external library wrappers.
    - Synthesizes minority samples strictly inside X_train while respecting Probe Station PS-L
      discrete one-hot boundaries (never creating fractional firmware error codes).
    - Flags synthetic samples with `is_real_pass = 0` so the autoencoder normal manifold is trained
      strictly on real physical Pass (`y == 0`) wafers.
    """
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_tr == 1)[0]
    neg_idx = np.where(y_tr == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    n_target = int(round(n_neg * target_ratio))
    n_syn = max(0, n_target - n_pos)

    if n_syn == 0 or n_pos < 2:
        is_real_pass = (y_tr == 0).astype(np.float32)
        return X_tr.copy(), M_tr.copy(), y_tr.astype(np.float32), is_real_pass

    X_pos = X_tr[pos_idx]
    M_pos = M_tr[pos_idx]
    sq_dists = np.sum((X_pos[:, None, :] - X_pos[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(sq_dists, np.inf)
    k_eff = min(k_neighbors, n_pos - 1)
    nn_indices = np.argsort(sq_dists, axis=1)[:, :k_eff]

    base_i = rng.randint(0, n_pos, size=n_syn)
    nbr_col = rng.randint(0, k_eff, size=n_syn)
    nbr_i = nn_indices[base_i, nbr_col]
    lambdas = rng.uniform(0.0, 1.0, size=(n_syn, 1)).astype(np.float32)

    X_syn = X_pos[base_i] + lambdas * (X_pos[nbr_i] - X_pos[base_i])
    M_syn = ((M_pos[base_i] + lambdas * (M_pos[nbr_i] - M_pos[base_i])) >= 0.5).astype(np.float32)

    # Project synthetic categorical tail to valid discrete PS-L states
    cat_tail = X_syn[:, f_cont:]
    is_psl_syn = (cat_tail[:, 0] >= 0.5).astype(np.float32)
    discrete_codes = np.zeros((n_syn, 3), dtype=np.float32)
    for i in range(n_syn):
        if is_psl_syn[i] > 0.5:
            c_idx = int(np.argmax(cat_tail[i, 1:4]))
            discrete_codes[i, c_idx] = 1.0
            X_syn[i, 0] = 0.0
            X_syn[i, 1] = 0.0
            M_syn[i, 0] = 0.0
            M_syn[i, 1] = 0.0
    X_syn[:, f_cont:] = np.column_stack([is_psl_syn, discrete_codes])

    X_res = np.vstack([X_tr, X_syn]).astype(np.float32)
    M_res = np.vstack([M_tr, M_syn]).astype(np.float32)
    y_res = np.concatenate([y_tr, np.ones(n_syn, dtype=y_tr.dtype)]).astype(np.float32)
    is_real_pass = np.zeros(len(y_res), dtype=np.float32)
    is_real_pass[:len(y_tr)] = (y_tr == 0).astype(np.float32)
    return X_res, M_res, y_res, is_real_pass

X_train_bal, M_train_bal, y_train_bal, real_pass_train = custom_causal_smote(
    X_train, M_train, y_train, f_cont=F_cont, k_neighbors=5, seed=SEED, target_ratio=0.35
)

def make_causal_loader(
    X: np.ndarray,
    M: np.ndarray,
    y: np.ndarray,
    real_pass: Optional[np.ndarray],
    shuffle: bool,
) -> DataLoader:
    # Keep 2D [N, F] tensors so batch dimensions explicitly match [B, F] throughout
    Xt = torch.from_numpy(X).float()
    Mt = torch.from_numpy(M).float()
    yt = torch.from_numpy(y).float()
    if real_pass is None:
        rpt = (yt == 0).float()
    else:
        rpt = torch.from_numpy(real_pass).float()
    g = torch.Generator().manual_seed(SEED)
    return DataLoader(
        TensorDataset(Xt, Mt, yt, rpt),
        batch_size=HP["batch_size"],
        shuffle=shuffle,
        generator=g if shuffle else None,
        drop_last=False,
    )

train_loader = make_causal_loader(X_train_bal, M_train_bal, y_train_bal, real_pass_train, shuffle=True)
val_loader   = make_causal_loader(X_val,       M_val,       y_val,       None,            shuffle=False)
test_loader  = make_causal_loader(X_test,      M_test,      y_test,      None,            shuffle=False)

print(f"After custom causal regime-aware SMOTE on Train: {X_train_bal.shape} | train defect rate: {y_train_bal.mean():.2%}")
print(f"Real normal (Pass) wafers for autoencoder manifold: {int(real_pass_train.sum()):,}")
print(f"Val defect rate (natural): {y_val.mean():.2%} | Test defect rate (natural): {y_test.mean():.2%}")


# ============================================================================
# ## 4 · Model — Physics-Guided Observability-Masked Supervised Autoencoder (`WaferGuardSAE`)
# 
# ### Architectural Improvements Over `WaferSAE`
# 1. **Strict Shape Invariance (`[B, F_cont]`)**: `forward()` accepts `[B, F]` or `[B, 1, F]`, flattens to `[B, F]`, splits continuous channels `x_cont` (`[B, F_cont]`) from PS-L regime channels `x_cat` (`[B, 4]`), and guarantees `reconstructed.shape == x_cont.shape == (B, F_cont)`.
# 2. **Observability-Masked Normal-Manifold Reconstruction**:
#    - The decoder reconstructs only continuous metrology channels `x_cont` (`[B, F_cont]`).
#    - Reconstruction loss is masked by $M_{i,j} \in \{0, 1\}$ (ignoring missing dropouts and PS-L gated `sensor_0`/`sensor_1`) and weighted by `is_real_pass` so the autoencoder learns the **normal Pass wafer manifold** rather than memorizing synthetic defective wafers.
# 3. **Residual Excursion Feature Fusion**:
#    - For every wafer, we compute a 3-dimensional **Excursion Residual Descriptor** from the masked reconstruction error $e_{i,j} = M_{i,j}(\hat{x}_{i,j} - x_{i,j})^2$:
#      1. Mean masked reconstruction MSE per wafer
#      2. Max single-sensor masked squared error (captures localized single-chamber excursions)
#      3. Top-10 sensor mean squared error (captures multi-sensor subsystem excursions)
#    - The classification head fuses `[z, x_cat, excursion_stats]` (`latent + 4 + 3` dims) to predict defect logits.
# ============================================================================

class WaferGuardSAE(nn.Module):
    def __init__(self, d_cont: int, d_cat: int, d_hid: int, d_lat: int):
        super().__init__()
        self.d_cont = d_cont
        self.d_cat = d_cat
        d_in = d_cont + d_cat

        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.LayerNorm(d_hid),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_hid, d_hid // 2),
            nn.LayerNorm(d_hid // 2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(d_hid // 2, d_lat),
            nn.LayerNorm(d_lat),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_lat + d_cat, d_hid // 2),
            nn.GELU(),
            nn.Linear(d_hid // 2, d_hid),
            nn.GELU(),
            nn.Linear(d_hid, d_cont),
        )
        # Classifier head fuses latent representation z, explicit PS-L regime indicators,
        # and 3 physics-guided reconstruction residual statistics (mean MSE, max sensor error, top-10 sensor error)
        self.head = nn.Sequential(
            nn.Linear(d_lat + d_cat + 3, d_lat),
            nn.LayerNorm(d_lat),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(d_lat, 1),
        )

    def forward(self, x: torch.Tensor, obs_mask: Optional[torch.Tensor] = None):
        x_flat = x.flatten(start_dim=1)                  # [B, F_cont + F_cat]
        x_cont = x_flat[:, :self.d_cont]                 # [B, F_cont]
        x_cat  = x_flat[:, self.d_cont:]                 # [B, F_cat]

        if obs_mask is None:
            obs_mask = torch.ones_like(x_cont)
        else:
            obs_mask = obs_mask.flatten(start_dim=1)

        z = self.encoder(x_flat)                         # [B, d_lat]
        reconstructed = self.decoder(torch.cat([z, x_cat], dim=1))  # [B, F_cont]

        # Compute per-wafer masked reconstruction residual descriptors (detached for stability)
        sq_err = ((reconstructed.detach() - x_cont) ** 2) * obs_mask
        valid_counts = torch.clamp(obs_mask.sum(dim=1, keepdim=True), min=1.0)
        mean_err = sq_err.sum(dim=1, keepdim=True) / valid_counts
        max_err, _ = sq_err.max(dim=1, keepdim=True)
        k = min(10, self.d_cont)
        topk_err = torch.topk(sq_err, k=k, dim=1).values.mean(dim=1, keepdim=True)

        excursion_feats = torch.log1p(torch.cat([mean_err, max_err, topk_err], dim=1))
        logits = self.head(torch.cat([z, x_cat, excursion_feats], dim=1)).squeeze(-1)
        return reconstructed, logits, z, x_cont


def masked_normal_recon_loss(
    reconstructed: torch.Tensor,
    target_cont: torch.Tensor,
    obs_mask: torch.Tensor,
    is_real_pass: torch.Tensor,
) -> torch.Tensor:
    # Explicit shape assertion prevents silent [B, B, F] broadcasting
    assert reconstructed.shape == target_cont.shape == obs_mask.shape, (
        f"Shape mismatch: {reconstructed.shape} vs {target_cont.shape} vs {obs_mask.shape}"
    )
    sample_weights = is_real_pass.unsqueeze(1)           # [B, 1] -> only real Pass wafers train normal manifold
    effective_mask = obs_mask * sample_weights           # [B, F_cont]
    denom = torch.clamp(effective_mask.sum(), min=1.0)
    return torch.sum(((reconstructed - target_cont) ** 2) * effective_mask) / denom


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float = 2.0,
    gamma: float = 1.5,
) -> torch.Tensor:
    assert logits.shape == targets.shape, f"Logit/target shape mismatch: {logits.shape} vs {targets.shape}"
    bce_raw = F_nn.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = pos_weight * targets + 1.0 * (1.0 - targets)
    focal_term = ((1.0 - p_t) ** gamma) * alpha_t * bce_raw
    return focal_term.mean()


seed_everything(SEED)
model = WaferGuardSAE(F_cont, F_cat, HP["hidden"], HP["latent"]).to(DEVICE)
print(f"WaferGuardSAE parameters: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================================
# ## 5 · Training with Causal Validation Monitoring & Threshold Calibration
# 
# During training:
# - Every gradient step explicitly verifies `reconstructed.shape == x_cont.shape == [B, F_cont]`.
# - Model checkpoints and decision threshold $\tau^*$ are selected **exclusively on the chronological validation split (`val_loader`)** using Precision-Recall curve optimization (never peeking at `test_loader`).
# ============================================================================

@torch.no_grad()
def predict_loader(net: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    net.eval()
    all_probs, all_targets, all_recon_err = [], [], []
    for xb, mb, yb, _ in loader:
        xb, mb = xb.to(DEVICE), mb.to(DEVICE)
        reconstructed, logits, _, x_cont = net(xb, mb)
        probs = torch.sigmoid(logits).cpu().numpy()
        sq_err = (((reconstructed - x_cont) ** 2) * mb).sum(dim=1) / torch.clamp(mb.sum(dim=1), min=1.0)
        all_probs.append(probs)
        all_targets.append(yb.numpy())
        all_recon_err.append(sq_err.cpu().numpy())
    return (
        np.concatenate(all_probs),
        np.concatenate(all_targets).astype(int),
        np.concatenate(all_recon_err),
    )


def find_optimal_threshold_causal(val_probs: np.ndarray, val_y: np.ndarray) -> Tuple[float, float]:
    precisions, recalls, thresholds = precision_recall_curve(val_y, val_probs)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1s = (2 * precisions[:-1] * recalls[:-1]) / np.maximum(precisions[:-1] + recalls[:-1], 1e-12)
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


seed_everything(SEED)
model = WaferGuardSAE(F_cont, F_cat, HP["hidden"], HP["latent"]).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HP["epochs"])

history = {"total": [], "recon": [], "clf": [], "val_ap": [], "val_f1": []}
best_val_score = -1.0
best_state = None

for epoch in range(1, HP["epochs"] + 1):
    model.train()
    acc = dict(total=0.0, recon=0.0, clf=0.0)
    n_batches = 0

    for batch_x, batch_m, batch_y, batch_rp in train_loader:
        batch_x = batch_x.to(DEVICE)
        batch_m = batch_m.to(DEVICE)
        batch_y = batch_y.to(DEVICE)
        batch_rp = batch_rp.to(DEVICE)

        reconstructed, logits, _, x_cont = model(batch_x, batch_m)

        recon_loss = masked_normal_recon_loss(reconstructed, x_cont, batch_m, batch_rp)
        clf_loss   = focal_bce_with_logits(logits, batch_y, pos_weight=2.0, gamma=HP["focal_gamma"])
        loss       = recon_loss + HP["lambda_clf"] * clf_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        acc["total"] += loss.item()
        acc["recon"] += recon_loss.item()
        acc["clf"]   += clf_loss.item()
        n_batches += 1

    sched.step()
    for k in ("total", "recon", "clf"):
        history[k].append(acc[k] / n_batches)

    val_probs, val_targets, _ = predict_loader(model, val_loader)
    val_ap = average_precision_score(val_targets, val_probs) if val_targets.sum() > 0 else 0.0
    _, val_best_f1 = find_optimal_threshold_causal(val_probs, val_targets)
    history["val_ap"].append(val_ap)
    history["val_f1"].append(val_best_f1)

    composite_val = 0.6 * val_best_f1 + 0.4 * val_ap
    if composite_val >= best_val_score:
        best_val_score = composite_val
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if epoch == 1 or epoch % 5 == 0 or epoch == HP["epochs"]:
        print(
            f"epoch {epoch:02d}/{HP['epochs']} | total {history['total'][-1]:.4f} "
            f"| recon {history['recon'][-1]:.4f} | clf {history['clf'][-1]:.4f} "
            f"| val AP {val_ap:.4f} | val F1* {val_best_f1:.4f}"
        )

if best_state is not None:
    model.load_state_dict(best_state)


# ============================================================================
# ## 6 · Honest Evaluation at Natural Defect Prevalence (~6.6%)
# 
# We calibrate the decision threshold $\tau^*$ **strictly on the chronological validation split** and evaluate on the untouched chronological test split at both:
# - The validation-calibrated threshold $\tau^*$ (production operating point)
# - The uncalibrated default threshold $\tau = 0.50$ (for direct diagnostic comparison)
# ============================================================================

val_probs, val_y_true, val_recon_err = predict_loader(model, val_loader)
tau_star, val_f1_star = find_optimal_threshold_causal(val_probs, val_y_true)

test_probs, test_y_true, test_recon_err = predict_loader(model, test_loader)

def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return dict(
        threshold=float(threshold),
        prevalence=float(y_true.mean()),
        f1=float(f1_score(y_true, preds, zero_division=0)),
        precision=float(precision_score(y_true, preds, zero_division=0)),
        recall=float(recall_score(y_true, preds, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.0,
        pr_auc=float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.0,
        brier=float(brier_score_loss(y_true, probs)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )

metrics_tau_star = compute_metrics(test_y_true, test_probs, tau_star)
metrics_tau_050  = compute_metrics(test_y_true, test_probs, 0.50)

print("=== FINAL EVALUATION ON CHRONOLOGICAL TEST SET (NATURAL DEFECT PREVALENCE) ===")
print(f"Test Population : N={len(test_y_true)} wafers | Natural Defect Prevalence = {metrics_tau_star['prevalence']:.2%} "
      f"({test_y_true.sum()} fail / {(1 - test_y_true).sum()} pass)")
print(f"Causal Validation Threshold tau* = {tau_star:.4f} (Validation F1 = {val_f1_star:.4f})")
print("-" * 78)
print(f"[Calibrated tau*={tau_star:.3f}] F1: {metrics_tau_star['f1']:.4f} | Precision: {metrics_tau_star['precision']:.4f} "
      f"| Recall: {metrics_tau_star['recall']:.4f} | ROC-AUC: {metrics_tau_star['roc_auc']:.4f} | PR-AUC (AP): {metrics_tau_star['pr_auc']:.4f}")
print(f"  Confusion Matrix -> TN={metrics_tau_star['tn']}  FP={metrics_tau_star['fp']}  FN={metrics_tau_star['fn']}  TP={metrics_tau_star['tp']}")
print(f"[Default   tau =0.500] F1: {metrics_tau_050['f1']:.4f} | Precision: {metrics_tau_050['precision']:.4f} "
      f"| Recall: {metrics_tau_050['recall']:.4f} | ROC-AUC: {metrics_tau_050['roc_auc']:.4f} | PR-AUC (AP): {metrics_tau_050['pr_auc']:.4f}")
print(f"  Confusion Matrix -> TN={metrics_tau_050['tn']}  FP={metrics_tau_050['fp']}  FN={metrics_tau_050['fn']}  TP={metrics_tau_050['tp']}")


# ============================================================================
# ## 7 · Reproducible 5-Stage Ablation Study (Quantifying Every Silent Bug)
# 
# To satisfy the audit requirement (*"For each defect give: (a) exact location, (b) mechanism, (c) reproducible evidence from your own runs — numbers, shapes, ablations"*), we execute a controlled 5-stage ablation from the original buggy pipeline to the repaired production pipeline:
# 
# | Stage | Pipeline Configuration | What It Isolates |
# |---|---|---|
# | **Stage 0 (Buggy Original)** | Pre-split Impute/Scale/SMOTE, random split, 50% synthetic test set, `[B, B, F]` broadcast bug, naive PS-L | Reproduces the illusory `F1 >= 0.90` release-gate pass caused by synthetic test leakage |
# | **Stage 1 (Fix Causality & Natural Test)** | Chronological split (`timestamp`), fit Impute/Scale/SMOTE on Train only, evaluate on natural (~6.6%) Test | Exposes the true performance collapse once synthetic test leakage is removed |
# | **Stage 2 (+ Fix Tensor Broadcasting)** | Fix `(reconstructed - batch_x)` from `[64, 64, 590]` to `[64, 590]` per-sample MSE | Quantifies the gain from restoring sample-specific autoencoder reconstruction gradients |
# | **Stage 3 (+ Fix Invariant Blindness)** | Add Probe Station PS-L regime gating (`sensor_0`/`sensor_1`) without fabricated imputation + drop constant sensors | Quantifies the gain from proper tool-matching and zero-variance channel pruning |
# | **Stage 4 (Full `WaferGuardSAE`)** | Add Observability-Masked Normal Reconstruction + Excursion Residual Fusion + Causal Threshold $\tau^*$ | Full repaired model performance at natural prevalence |
# ============================================================================

class LegacyWaferSAE(nn.Module):
    """Original WaferSAE architecture from bnypi.ipynb for controlled ablation stages 0-3."""
    def __init__(self, d_in: int, d_hid: int = 256, d_lat: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.LayerNorm(d_hid), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(d_hid, d_hid // 2), nn.LayerNorm(d_hid // 2), nn.GELU(),
            nn.Linear(d_hid // 2, d_lat),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_lat, d_hid // 2), nn.GELU(),
            nn.Linear(d_hid // 2, d_hid), nn.GELU(),
            nn.Linear(d_hid, d_in),
        )
        self.head = nn.Linear(d_lat, 1)

    def forward(self, x: torch.Tensor):
        x = x.flatten(start_dim=1)
        z = self.encoder(x)
        return self.decoder(z), self.head(z).squeeze(-1), z


def run_legacy_stage(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    fix_broadcast: bool,
    epochs: int = 40,
) -> Dict[str, float]:
    seed_everything(SEED)
    Xt_tr = torch.from_numpy(X_tr).float().unsqueeze(1)   # [N, 1, F] as in original bnypi.ipynb
    yt_tr = torch.from_numpy(y_tr).float()
    Xt_te = torch.from_numpy(X_te).float().unsqueeze(1)
    yt_te = torch.from_numpy(y_te).float()

    g = torch.Generator().manual_seed(SEED)
    tr_loader = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=64, shuffle=True, generator=g, drop_last=True)
    te_loader = DataLoader(TensorDataset(Xt_te, yt_te), batch_size=64, shuffle=False)

    net = LegacyWaferSAE(X_tr.shape[1], 256, 32).to(DEVICE)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bce_fn = nn.BCEWithLogitsLoss()

    last_recon = 0.0
    for _ in range(epochs):
        net.train()
        for bx, by in tr_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            recon, logits, _ = net(bx)
            target = bx.flatten(start_dim=1) if fix_broadcast else bx
            r_loss = torch.mean((recon - target) ** 2)
            c_loss = bce_fn(logits, by)
            loss = r_loss + c_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            last_recon = r_loss.item()
        scheduler.step()

    net.eval()
    probs_list = []
    with torch.no_grad():
        for bx, _ in te_loader:
            _, logits, _ = net(bx.to(DEVICE))
            probs_list.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(probs_list)
    m = compute_metrics(y_te, probs, 0.50)
    m["train_recon_mse"] = float(last_recon)
    return m

# --- Stage 0: Exact Original Buggy Pipeline (from bnypi.ipynb) ---
seed_everything(SEED)
X_raw_orig = df_raw[SENSOR_COLS].to_numpy(dtype=np.float64)
y_raw_orig = (df_raw["label"].to_numpy() == 1).astype(np.int64)
imp_0 = SimpleImputer(strategy="mean", keep_empty_features=True)
scl_0 = StandardScaler()
X_scaled_0 = scl_0.fit_transform(imp_0.fit_transform(X_raw_orig))
smote_0 = SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=SEED)
X_bal_0, y_bal_0 = smote_0.fit_resample(X_scaled_0, y_raw_orig)
X_tr_0, X_te_0, y_tr_0, y_te_0 = train_test_split(
    X_bal_0, y_bal_0, test_size=0.20, stratify=y_bal_0, random_state=SEED
)
stage0_metrics = run_legacy_stage(X_tr_0, y_tr_0, X_te_0, y_te_0, fix_broadcast=False)

# --- Stage 1: Fix Causality & Natural Test Split (still has broadcast bug & naive PS-L) ---
imp_1 = SimpleImputer(strategy="mean", keep_empty_features=True)
scl_1 = StandardScaler()
X_tr_1_s = scl_1.fit_transform(imp_1.fit_transform(X_train_raw))
X_te_1_s = scl_1.transform(imp_1.transform(X_test_raw))
smote_1 = SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=SEED)
X_tr_1_bal, y_tr_1_bal = smote_1.fit_resample(X_tr_1_s, y_train)
stage1_metrics = run_legacy_stage(X_tr_1_bal, y_tr_1_bal, X_te_1_s, y_test, fix_broadcast=False)

# --- Stage 2: + Fix Tensor Broadcasting ([B, B, F] -> [B, F]) ---
stage2_metrics = run_legacy_stage(X_tr_1_bal, y_tr_1_bal, X_te_1_s, y_test, fix_broadcast=True)

# --- Stage 3: + Fix Invariant Blindness (PS-L Tool-Matching & Zero-Variance Pruning) ---
stage3_metrics = run_legacy_stage(X_train_bal, y_train_bal, X_test, y_test, fix_broadcast=True)

# --- Stage 4: Full Repaired WaferGuardSAE (Masked Normal Recon + Excursion Fusion + Causal tau*) ---
stage4_metrics = dict(metrics_tau_star)
stage4_metrics["train_recon_mse"] = float(history["recon"][-1])

ablation_rows = [
    ("Stage 0: Original Buggy (50% Synthetic Test)", stage0_metrics),
    ("Stage 1: + Causal Split & Natural Test (~6.6%)", stage1_metrics),
    ("Stage 2: + Fix [B,B,F] Tensor Broadcast Bug", stage2_metrics),
    ("Stage 3: + Fix PS-L Sentinel/Codes & Constant Cols", stage3_metrics),
    ("Stage 4: + Masked Normal SAE & Causal Threshold", stage4_metrics),
]

ablation_df = pd.DataFrame([
    {
        "Stage": name,
        "Test Prev.": f"{m['prevalence']:.2%}",
        "Threshold": f"{m['threshold']:.3f}",
        "F1": round(m["f1"], 4),
        "Precision": round(m["precision"], 4),
        "Recall": round(m["recall"], 4),
        "ROC-AUC": round(m["roc_auc"], 4),
        "PR-AUC (AP)": round(m["pr_auc"], 4),
        "TN / FP / FN / TP": f"{m['tn']} / {m['fp']} / {m['fn']} / {m['tp']}",
    }
    for name, m in ablation_rows
])
print("=== 5-STAGE CONTROLLED ABLATION TABLE ===")
print(ablation_df.to_string(index=False))


# ============================================================================
# ## 8 · Diagnostic Plots, Release Gate Verdict & Automated `AUDIT.md` Generation
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# Panel 1: Training loss curves
epochs_arr = np.arange(1, len(history["total"]) + 1)
axes[0].plot(epochs_arr, history["total"], label="Total Loss", lw=2, color="#1f4e78")
axes[0].plot(epochs_arr, history["recon"], label="Masked Normal Recon MSE", lw=2, color="#2e8b57")
axes[0].plot(epochs_arr, history["clf"], label="Focal Classification Loss", lw=2, color="#d95f02")
axes[0].set_title("WaferGuardSAE Training Convergence")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend(frameon=True)
axes[0].grid(alpha=0.25)

# Panel 2: Precision-Recall Curve on Validation vs Test (Natural Prevalence)
val_p, val_r, _ = precision_recall_curve(val_y_true, val_probs)
test_p, test_r, _ = precision_recall_curve(test_y_true, test_probs)
axes[1].plot(val_r, val_p, label=f"Validation PR (AP={average_precision_score(val_y_true, val_probs):.3f})", lw=2, color="#7570b3")
axes[1].plot(test_r, test_p, label=f"Test PR (AP={metrics_tau_star['pr_auc']:.3f})", lw=2, color="#e7298a")
axes[1].axhline(metrics_tau_star["prevalence"], ls="--", color="gray", label=f"Natural Base Rate ({metrics_tau_star['prevalence']:.1%})")
axes[1].scatter([metrics_tau_star["recall"]], [metrics_tau_star["precision"]], color="black", zorder=5, label=f"Causal tau*={tau_star:.2f}")
axes[1].set_title("Precision-Recall at Natural Prevalence")
axes[1].set_xlabel("Recall (Excursion Catch Rate)")
axes[1].set_ylabel("Precision (1 - False Alarm Rate)")
axes[1].legend(frameon=True, fontsize=8.5)
axes[1].grid(alpha=0.25)

# Panel 3: Ablation progression (PR-AUC and F1 on Natural Test vs Illusory Stage 0)
stage_labels = ["S0\n(Buggy 50%)", "S1\n(Causal)", "S2\n(+Shape)", "S3\n(+PS-L)", "S4\n(Full SAE)"]
f1_vals = [m["f1"] for _, m in ablation_rows]
ap_vals = [m["pr_auc"] for _, m in ablation_rows]
x_pos = np.arange(len(stage_labels))
axes[2].bar(x_pos - 0.18, f1_vals, width=0.35, label="F1 Score", color="#2b5c8f")
axes[2].bar(x_pos + 0.18, ap_vals, width=0.35, label="PR-AUC (AP)", color="#e07a5f")
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(stage_labels)
axes[2].set_ylim(0, 1.05)
axes[2].set_title("5-Stage Ablation: Illusory vs Honest Metrics")
axes[2].set_ylabel("Metric Value")
axes[2].legend(frameon=True)
axes[2].grid(axis="y", alpha=0.25)

plt.tight_layout()
os.makedirs("artifacts", exist_ok=True)
plt.savefig("artifacts/audit_diagnostics.png", dpi=180, bbox_inches="tight")
plt.show()

# --- Release Gate Assessment ---
GATE_F1 = 0.90
verdict = "PASSED" if metrics_tau_star["f1"] >= GATE_F1 else "HOLD / NO-GO FOR UNGATED RELEASE"
print(f"Release gate (F1 >= {GATE_F1:.2f} at natural {metrics_tau_star['prevalence']:.2%} prevalence): "
      f"F1 = {metrics_tau_star['f1']:.4f} -> {verdict}")

torch.save(model.state_dict(), "artifacts/wafer_sae_fixed.pt")
joblib.dump({"preprocessor": preprocessor, "tau_star": tau_star, "hp": HP}, "artifacts/preprocessor_fixed.joblib")
print("Saved repaired model and causality-safe preprocessor to ./artifacts/")


# --- Auto-Generate Production AUDIT.md Populated with Exact Run Metrics ---
audit_md_text = f"""# WAFER-GUARD · Fab-7 Excursion Detector — Pre-Release Technical Audit (`AUDIT.md`)

**Reviewer:** Final Technical Reviewer, Fab-7 ML Infrastructure Release Board  
**Target Pipeline:** `bnypi.ipynb` → Repaired in `wafer_guard_fixed.ipynb`  
**Dataset:** `secom_telemetry_modified.csv` ({len(df):,} wafer lots, {len(SENSOR_COLS)} inline metrology sensors, natural defect rate **{y_all.mean():.2%}** [{y_all.sum()} fail / {(1 - y_all).sum()} pass])  
**Release Board Verdict:** **{verdict}** (Reported `F1 >= 0.90` in `bnypi.ipynb` was an artifact of pre-split SMOTE test contamination and a 50% synthetic evaluation set; true chronological test performance at natural **{metrics_tau_star['prevalence']:.2%}** defect prevalence is **F1 = {metrics_tau_star['f1']:.4f}**, **PR-AUC = {metrics_tau_star['pr_auc']:.4f}**, **ROC-AUC = {metrics_tau_star['roc_auc']:.4f}** at causal validation threshold $\\tau^* = {tau_star:.4f}$).

## 1. Executive Summary & Release Recommendation

The candidate notebook (`bnypi.ipynb`) clears the `F1 >= 0.90` release gate only because **three compounding silent defects** invalidate both its training dynamics and its holdout evaluation:
1. **Severe Data & Temporal Leakage (`Cell 6` & `Cell 8`)**: Imputation, standardization, and `SMOTE` oversampling are fit on the **entire dataset prior to splitting**, `timestamp` is ignored in favor of a random shuffle split ($t' > t$ acausality), and the model is evaluated on an **artificially balanced 50.0% test set** where **{100 * (1 - (y_all.sum() * 0.2) / max(y_te_0.sum(), 1)):.1f}%** of defective test wafers are synthetic `SMOTE` interpolations of training defects.
2. **Corrupted Loss Gradients via Silent Rank-3 Tensor Broadcasting (`Cell 10`, `Cell 12`, `Cell 14`)**: `make_loader` unsqueezes inputs to `[B, 1, F] = [64, 1, 590]`, whereas `WaferSAE.forward` flattens inputs and outputs `reconstructed` of shape `[64, 590]`. Subtracting `reconstructed - batch_x` silently broadcasts into a **3D tensor of shape `[64, 64, 590]`** ($2,416,640$ elements instead of $37,760$), computing all-pairs cross-wafer MSE and forcing every wafer's reconstruction toward the mini-batch mean vector $\\bar{{x}}_{{\\text{{batch}}}}$.
3. **Invariant Blindness — Legacy Probe Station PS-L Polymorphism & Constant Channels (`Cell 6`, `Cell 8`, `Cell 14`)**: For the **{is_psl_s0.sum():,} ({is_psl_s0.mean():.2%})** wafers routed through legacy **Probe Station PS-L**, `sensor_0` emits a `0.0` sentinel and `sensor_1` switches from a continuous measurement to discrete categorical firmware error codes (`100.0`, `200.0`, `300.0`). The baseline pipeline blindly includes these sentinel/categorical values in continuous `SimpleImputer` and `StandardScaler` fits, distorting channel moments (`sensor_0` std inflated from `{clean_s0_std:.2f}` to `{naive_s0_std:.2f}`; `sensor_1` std inflated from `{clean_s1_std:.2f}` to `{naive_s1_std:.2f}`), while also retaining **{preprocessor.n_invariant_dropped_} zero-variance/empty sensors** and training the autoencoder reconstruction loss on synthetic defective wafers.

## 2. Forensic Audit of the Three Silent Bugs & Exact Repairs

| Severity | Defect Pillar | Exact Location in `bnypi.ipynb` | Failure Mechanism & Mathematical Proof | Production Repair in `wafer_guard_fixed.ipynb` |
|---|---|---|---|---|
| **P0 (Critical)** | **1. Data Leakage & Causality Violations** | `Cell 6` (ignores `timestamp`) & `Cell 8` (`imputer.fit_transform(X_raw)`, `scaler.fit_transform(X_imp)`, `smote.fit_resample(X_scaled, y_raw)` before `train_test_split`) | (a) **Pre-split SMOTE**: synthesizes minority points across the full dataset via $k=5$ NN before splitting, placing linear combinations of training defects into `X_test` and inflating test defect prevalence from **{y_all.mean():.2%}** to **{y_te_0.mean():.2%}**. (b) **Temporal acausality**: random stratified splitting places future lots ($t' > t$) into training despite fab drift. (c) **Moment leakage**: evaluation population means/variances leak into training normalization. | Sort strictly by `timestamp` ascending (`70% Train -> 10% Causal Val -> 20% Chronological Test`). Fit `FabTelemetryPreprocessor` and regime-aware `SMOTE` **strictly on Train**. Leave Validation and Test untouched at natural prevalence (**{metrics_tau_star['prevalence']:.2%}**). |
| **P0 (Critical)** | **2. Corrupted Loss Gradients (Tensor Broadcasting)** | `Cell 10` (`Xt.unsqueeze(1)` $\\rightarrow$ `[B, 1, F]`), `Cell 12` (`x.flatten(1)` $\\rightarrow$ `reconstructed` `[B, F]`), `Cell 14` (`torch.mean((reconstructed - batch_x) ** 2)`) | PyTorch aligns trailing axes of `[64, 590]` and `[64, 1, 590]` to broadcast `(reconstructed - batch_x)` into **`[64, 64, 590]`** (`[B, B, F]`). This computes $\\mathcal{{L}}_{{\\text{{recon}}}}^{{\\text{{buggy}}}} = \\frac{{1}}{{B^2 F}}\\sum_{{i=1}}^B \\sum_{{j=1}}^B \\|\\hat{{x}}_i - x_j\\|_2^2$, whose gradient $\\nabla_{{\\hat{{x}}_i}}\\mathcal{{L}} = \\frac{{2}}{{BF}}(\\hat{{x}}_i - \\bar{{x}}_{{\\text{{batch}}}})$ drives the decoder to predict the constant batch mean $\\bar{{x}}_{{\\text{{batch}}}}$ (verified max analytical discrepancy `{max_grad_discrepancy:.2e}`). | Keep tensors aligned at `[B, F_cont]`, assert `reconstructed.shape == target_cont.shape == obs_mask.shape`, and compute **Observability-Masked Normal-Manifold MSE** strictly over genuinely observed continuous cells ($M_{{i,j}}=1$) on real `Pass` (`y=0`) wafers. |
| **P1 (High)** | **3. Invariant Blindness (PS-L Tool-Matching & Constant Sensors)** | `Cell 6` & `Cell 8` (`SimpleImputer(keep_empty_features=True)` + `StandardScaler()` across all 590 columns) | (a) **PS-L Sentinel & Polymorphic Error Codes**: `sensor_0 == 0.0` (sentinel) and `sensor_1 in {{100.0, 200.0, 300.0}}` (categorical firmware codes) corrupt continuous mean/std and impose false ordinality. Crucially, naively replacing them with `NaN` before `SimpleImputer` still violates the spec by imputing fabricated modern-station continuous values over legacy wafers. (b) **Constant Sensors**: **{preprocessor.n_invariant_dropped_}** training channels have zero variance or 100% `NaN`, diluting MSE gradients. | Implement `FabTelemetryPreprocessor`: extract explicit one-hot regime indicators (`[is_psl, err_100, err_200, err_300]`), fit continuous `sensor_0`/`sensor_1` stats strictly on modern training lots (`~is_psl`), gate PS-L `sensor_0`/`sensor_1` continuous channels to `0.0` with observability mask $M_{{i,0}}=M_{{i,1}}=0$ (zero fabricated imputation), and drop all **{preprocessor.n_invariant_dropped_}** invariant training channels. |

## 3. True Evaluation Metrics & 5-Stage Controlled Ablation

| Pipeline Stage | Test Prevalence | Decision Threshold | F1 Score | Precision | Recall | ROC-AUC | PR-AUC (AP) | Confusion (`TN / FP / FN / TP`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Stage 0: Original Buggy (`bnypi.ipynb`)** | {stage0_metrics['prevalence']:.2%} (Synthetic) | 0.500 | {stage0_metrics['f1']:.4f} | {stage0_metrics['precision']:.4f} | {stage0_metrics['recall']:.4f} | {stage0_metrics['roc_auc']:.4f} | {stage0_metrics['pr_auc']:.4f} | `{stage0_metrics['tn']} / {stage0_metrics['fp']} / {stage0_metrics['fn']} / {stage0_metrics['tp']}` |
| **Stage 1: + Causal Chronological Split & Natural Test** | {stage1_metrics['prevalence']:.2%} (Natural) | 0.500 | {stage1_metrics['f1']:.4f} | {stage1_metrics['precision']:.4f} | {stage1_metrics['recall']:.4f} | {stage1_metrics['roc_auc']:.4f} | {stage1_metrics['pr_auc']:.4f} | `{stage1_metrics['tn']} / {stage1_metrics['fp']} / {stage1_metrics['fn']} / {stage1_metrics['tp']}` |
| **Stage 2: + Fix `[B, B, F]` Broadcast Bug** | {stage2_metrics['prevalence']:.2%} (Natural) | 0.500 | {stage2_metrics['f1']:.4f} | {stage2_metrics['precision']:.4f} | {stage2_metrics['recall']:.4f} | {stage2_metrics['roc_auc']:.4f} | {stage2_metrics['pr_auc']:.4f} | `{stage2_metrics['tn']} / {stage2_metrics['fp']} / {stage2_metrics['fn']} / {stage2_metrics['tp']}` |
| **Stage 3: + Fix PS-L Tool-Matching & Constant Channels** | {stage3_metrics['prevalence']:.2%} (Natural) | 0.500 | {stage3_metrics['f1']:.4f} | {stage3_metrics['precision']:.4f} | {stage3_metrics['recall']:.4f} | {stage3_metrics['roc_auc']:.4f} | {stage3_metrics['pr_auc']:.4f} | `{stage3_metrics['tn']} / {stage3_metrics['fp']} / {stage3_metrics['fn']} / {stage3_metrics['tp']}` |
| **Stage 4a: Repaired `WaferGuardSAE` (Default $\\tau=0.500$)** | {metrics_tau_050['prevalence']:.2%} (Natural) | 0.500 | {metrics_tau_050['f1']:.4f} | {metrics_tau_050['precision']:.4f} | {metrics_tau_050['recall']:.4f} | {metrics_tau_050['roc_auc']:.4f} | {metrics_tau_050['pr_auc']:.4f} | `{metrics_tau_050['tn']} / {metrics_tau_050['fp']} / {metrics_tau_050['fn']} / {metrics_tau_050['tp']}` |
| **Stage 4b: Repaired `WaferGuardSAE` (Causal Val $\\tau^*={tau_star:.3f}$)** | **{metrics_tau_star['prevalence']:.2%} (Natural)** | **{tau_star:.3f}** | **{metrics_tau_star['f1']:.4f}** | **{metrics_tau_star['precision']:.4f}** | **{metrics_tau_star['recall']:.4f}** | **{metrics_tau_star['roc_auc']:.4f}** | **{metrics_tau_star['pr_auc']:.4f}** | **`{metrics_tau_star['tn']} / {metrics_tau_star['fp']} / {metrics_tau_star['fn']} / {metrics_tau_star['tp']}`** |

## 4. Residual Production Risks & Release Board Sign-Off

1. **Non-Stationary Chamber & Recipe Drift**: Because fab telemetry drifts over time (chamber cleaning cycles, target erosion, preventive maintenance), static standardization moments ($\\mu, \\sigma$) fit on historical windows will gradually shift. Production deployment should use an exponentially weighted moving average (EWMA) causal normalizer updated exclusively on verified `Pass` lots.
2. **Severe Class Scarcity (~6.6% Base Rate)**: With only ~104 total defective lots across 590 sensors ($p > 5n_{{\\text{{defect}}}}$), deep encoders remain susceptible to variance. Fusing our **Observability-Masked Reconstruction Residuals** (mean MSE, max-sensor excursion, top-10 sensor MSE) with Regularized Gradient Boosted Trees or an ensemble over rolling temporal folds is recommended before enabling automated lot-hold triggers.
"""

with open("AUDIT.md", "w", encoding="utf-8") as f:
    f.write(audit_md_text)
print("Successfully generated AUDIT.md with live measured evaluation and ablation metrics.")

