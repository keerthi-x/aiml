#!/usr/bin/env python3
"""
Automated Verification & Regression Test Suite for WAFER-GUARD (Fab-7 Excursion Detector)
Run via:
    python3 -m unittest discover -v -s . -p "test_wafer_guard.py"
or:
    python3 test_wafer_guard.py
"""

import os
import unittest
import numpy as np
import pandas as pd


class CustomFabTelemetryPreprocessor:
    """
    Custom from-scratch causality-safe, regime-gated preprocessor (Guideline #4).
    - Zero reliance on sklearn SimpleImputer or StandardScaler.
    - Isolates Probe Station PS-L sentinel (sensor_0 == 0.0) and categorical error codes
      (sensor_1 in {100.0, 200.0, 300.0}).
    - Fits continuous moments strictly on modern-station (~is_psl) training lots.
    - Gates PS-L continuous sensor_0/sensor_1 to 0.0 with Observability Mask M = 0.0
      (never imputing fabricated continuous values over categorical codes).
    - Prunes zero-variance / all-NaN invariant channels using training statistics only.
    """
    PSL_CODES = (100.0, 200.0, 300.0)

    def __init__(self, impute_strategy: str = "median", clip_z: float = 10.0, drift_window: int = 45):
        self.impute_strategy = impute_strategy
        self.clip_z = clip_z
        self.drift_window = drift_window
        self.active_cont_indices_ = None
        self.cont_Operation_center_ = None
        self.cont_scale_ = None
        self.snr_weights_ = None
        self.n_invariant_dropped_ = 0

    @staticmethod
    def detect_psl_regime(X_raw: np.ndarray):
        s0 = X_raw[:, 0]
        s1 = X_raw[:, 1]
        is_psl = (s0 == 0.0) | np.isin(s1, CustomFabTelemetryPreprocessor.PSL_CODES)
        cat_feats = np.column_stack([
            is_psl.astype(np.float32),
            (is_psl & (s1 == 100.0)).astype(np.float32),
            (is_psl & (s1 == 200.0)).astype(np.float32),
            (is_psl & (s1 == 300.0)).astype(np.float32),
        ])
        return is_psl, cat_feats

    def fit(self, X_train_raw: np.ndarray, y_train: np.ndarray = None):
        X = np.array(X_train_raw, dtype=np.float64, copy=True)
        is_psl, _ = self.detect_psl_regime(X)

        # Exclude PS-L sentinel/categorical entries from continuous statistics
        X[is_psl, 0] = np.nan
        X[is_psl, 1] = np.nan

        col_all_nan = np.isnan(X).all(axis=0)
        col_stds = np.nanstd(X, axis=0)
        active_mask = (~col_all_nan) & (col_stds > 1e-8)

        self.active_cont_indices_ = np.where(active_mask)[0]
        self.n_invariant_dropped_ = int((~active_mask).sum())

        X_act = X[:, self.active_cont_indices_]
        if self.impute_strategy == "median":
            self.cont_Operation_center_ = np.nanmedian(X_act, axis=0)
        else:
            self.cont_Operation_center_ = np.nanmean(X_act, axis=0)

        X_filled = np.where(np.isnan(X_act), self.cont_Operation_center_, X_act)
        q75 = np.percentile(X_filled, 75, axis=0)
        q25 = np.percentile(X_filled, 25, axis=0)
        iqr_scale = (q75 - q25) / 1.349
        std_scale = np.std(X_filled, axis=0)
        self.cont_scale_ = np.where(iqr_scale > 1e-6, iqr_scale, np.where(std_scale > 1e-8, std_scale, 1.0))

        if y_train is not None and np.sum(y_train == 1) > 1 and np.sum(y_train == 0) > 1:
            X_z = np.clip((X_filled - self.cont_Operation_center_) / self.cont_scale_, -self.clip_z, self.clip_z)
            pos_m = X_z[y_train == 1].mean(axis=0)
            neg_m = X_z[y_train == 0].mean(axis=0)
            pos_v = X_z[y_train == 1].var(axis=0) + 1e-4
            neg_v = X_z[y_train == 0].var(axis=0) + 1e-4
            t_stat = np.abs(pos_m - neg_m) / np.sqrt(pos_v / np.sum(y_train == 1) + neg_v / np.sum(y_train == 0))
            self.snr_weights_ = np.clip(t_stat / (np.median(t_stat) + 1e-6), 0.1, 3.0).astype(np.float32)
        else:
            self.snr_weights_ = np.ones(len(self.active_cont_indices_), dtype=np.float32)
        return self

    def transform(self, X_raw: np.ndarray):
        X = np.array(X_raw, dtype=np.float64, copy=True)
        is_psl, cat_feats = self.detect_psl_regime(X)
        X[is_psl, 0] = np.nan
        X[is_psl, 1] = np.nan

        X_act = X[:, self.active_cont_indices_]
        obs_mask = (~np.isnan(X_act)).astype(np.float32)

        X_imp = np.where(np.isnan(X_act), self.cont_Operation_center_, X_act)
        X_scaled = (X_imp - self.cont_Operation_center_) / self.cont_scale_
        if self.clip_z is not None:
            X_scaled = np.clip(X_scaled, -self.clip_z, self.clip_z)

        # Gate unobserved / PS-L channels strictly to 0.0 (zero fabricated continuous imputation)
        X_scaled = (X_scaled * obs_mask).astype(np.float32)
        X_full = np.hstack([X_scaled, cat_feats]).astype(np.float32)
        return X_full, obs_mask, cat_feats


def custom_causal_smote(X_tr: np.ndarray, M_tr: np.ndarray, y_tr: np.ndarray, f_cont: int, k: int = 5, target_ratio: float = 0.35, seed: int = 1337):
    """
    Custom from-scratch k-NN minority oversampler (Guideline #4).
    Synthesizes minority samples strictly within the training set using exact L2 nearest neighbors
    and preserves discrete Probe Station PS-L regime flags without cross-regime contamination.
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
    # Compute exact pairwise squared Euclidean distances among positive training wafers
    sq_dists = np.sum((X_pos[:, None, :] - X_pos[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(sq_dists, np.inf)
    k_eff = min(k, n_pos - 1)
    nn_indices = np.argsort(sq_dists, axis=1)[:, :k_eff]

    base_i = rng.randint(0, n_pos, size=n_syn)
    nbr_col = rng.randint(0, k_eff, size=n_syn)
    nbr_i = nn_indices[base_i, nbr_col]
    lambdas = rng.uniform(0.0, 1.0, size=(n_syn, 1)).astype(np.float32)

    X_syn = X_pos[base_i] + lambdas * (X_pos[nbr_i] - X_pos[base_i])
    M_syn = ((M_pos[base_i] + lambdas * (M_pos[nbr_i] - M_pos[base_i])) >= 0.5).astype(np.float32)

    # Enforce discrete PS-L one-hot invariants on synthetic samples
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

    X_out = np.vstack([X_tr, X_syn]).astype(np.float32)
    M_out = np.vstack([M_tr, M_syn]).astype(np.float32)
    y_out = np.concatenate([y_tr, np.ones(n_syn, dtype=y_tr.dtype)]).astype(np.float32)
    is_real_pass = np.zeros(len(y_out), dtype=np.float32)
    is_real_pass[:len(y_tr)] = (y_tr == 0).astype(np.float32)
    return X_out, M_out, y_out, is_real_pass


class TestWaferGuardAuditAndRepair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidates = [
            "secom_telemetry_modified.csv",
            "downloads/secom_telemetry_modified.csv",
            "../downloads/secom_telemetry_modified.csv",
        ]
        cls.data_path = next((p for p in candidates if os.path.exists(p)), None)
        if cls.data_path is not None:
            cls.df_raw = pd.read_csv(cls.data_path)
            cls.sensor_cols = [c for c in cls.df_raw.columns if c.startswith("sensor_")]
            cls.df_raw["timestamp_dt"] = pd.to_datetime(cls.df_raw["timestamp"])
            cls.df = cls.df_raw.sort_values("timestamp_dt", kind="mergesort").reset_index(drop=True)
            cls.X_all = cls.df[cls.sensor_cols].to_numpy(dtype=np.float64)
            cls.y_all = (cls.df["label"].to_numpy() == 1).astype(np.int64)
        else:
            # Deterministic synthetic fallback if run outside dataset directory
            rng = np.random.RandomState(1337)
            cls.X_all = rng.normal(2500.0, 75.0, size=(200, 590))
            cls.X_all[:20, 0] = 0.0
            cls.X_all[:20, 1] = rng.choice([100.0, 200.0, 300.0], size=20)
            cls.X_all[:, 5] = 100.0
            cls.y_all = np.zeros(200, dtype=np.int64)
            cls.y_all[30:43] = 1

    def test_01_temporal_causality_and_split_boundaries(self):
        """Verify strict chronological ordering (t_train <= t_val <= t_test) and natural prevalence."""
        if self.data_path is not None:
            n_total = len(self.df)
            self.assertEqual(n_total, 1567)
            self.assertEqual(int(self.y_all.sum()), 104)
            n_tr = int(n_total * 0.70)
            n_va = int(n_total * 0.10)
            t_tr_max = self.df["timestamp_dt"].iloc[n_tr - 1]
            t_va_min = self.df["timestamp_dt"].iloc[n_tr]
            t_va_max = self.df["timestamp_dt"].iloc[n_tr + n_va - 1]
            t_te_min = self.df["timestamp_dt"].iloc[n_tr + n_va]
            self.assertLessEqual(t_tr_max, t_va_min)
            self.assertLessEqual(t_va_max, t_te_min)

    def test_02_probe_station_psl_sentinel_and_error_codes(self):
        """Verify PS-L sentinel (sensor_0==0.0) & error codes (100/200/300) bypass continuous scaling & imputation."""
        prep = CustomFabTelemetryPreprocessor(impute_strategy="median", clip_z=10.0)
        n_tr = int(len(self.X_all) * 0.70)
        prep.fit(self.X_all[:n_tr], self.y_all[:n_tr])
        X_full, obs_mask, cat_feats = prep.transform(self.X_all)

        is_psl = (self.X_all[:, 0] == 0.0)
        if self.data_path is not None:
            self.assertEqual(int(is_psl.sum()), 146)
            self.assertEqual(int(self.y_all[is_psl].sum()), 0)

        # Verify PS-L wafers have 0.0 on continuous sensor_0 and sensor_1 (no fabricated continuous imputation)
        np.testing.assert_allclose(X_full[is_psl, 0], 0.0, atol=1e-7)
        np.testing.assert_allclose(X_full[is_psl, 1], 0.0, atol=1e-7)
        np.testing.assert_allclose(obs_mask[is_psl, 0], 0.0, atol=1e-7)
        np.testing.assert_allclose(obs_mask[is_psl, 1], 0.0, atol=1e-7)
        # Verify one-hot codes sum to 1.0 on every PS-L wafer and 0.0 on modern wafers
        np.testing.assert_allclose(cat_feats[is_psl, 0], 1.0, atol=1e-7)
        np.testing.assert_allclose(cat_feats[is_psl, 1:4].sum(axis=1), 1.0, atol=1e-7)
        np.testing.assert_allclose(cat_feats[~is_psl].sum(axis=1), 0.0, atol=1e-7)

    def test_03_invariant_zero_variance_pruning(self):
        """Verify zero-variance / constant training sensors are pruned without test leakage."""
        prep = CustomFabTelemetryPreprocessor()
        n_tr = int(len(self.X_all) * 0.70)
        prep.fit(self.X_all[:n_tr], self.y_all[:n_tr])
        self.assertGreater(prep.n_invariant_dropped_, 0)
        if self.data_path is not None:
            self.assertEqual(prep.n_invariant_dropped_, 122)
            self.assertEqual(len(prep.active_cont_indices_), 468)

    def test_04_tensor_broadcast_prevention_and_gradient_proof(self):
        """Prove [B, F] - [B, 1, F] broadcasts to [B, B, F] and collapses gradients to the batch mean."""
        rng = np.random.RandomState(42)
        B, F = 64, 590
        recon = rng.randn(B, F)
        batch_x_unsqueezed = rng.randn(B, 1, F)

        buggy_diff = recon - batch_x_unsqueezed
        self.assertEqual(buggy_diff.shape, (B, B, F))
        self.assertEqual(buggy_diff.size, B * B * F)

        # Analytical gradient of mean((recon[None, :, :] - batch_x[:, None, :])^2) w.r.t. recon
        grad_buggy = (2.0 / (B * B * F)) * np.sum(recon[None, :, :] - batch_x_unsqueezed, axis=0)
        batch_mean = np.mean(batch_x_unsqueezed.squeeze(1), axis=0, keepdims=True)
        grad_batch_mean = (2.0 / (B * F)) * (recon - batch_mean)
        np.testing.assert_allclose(grad_buggy, grad_batch_mean, atol=1e-12)

        # Aligned [B, F] gradient depends on individual sample x_i
        aligned_target = batch_x_unsqueezed.reshape(B, F)
        self.assertEqual(recon.shape, aligned_target.shape)

    def test_05_custom_causal_smote_regime_integrity(self):
        """Verify CustomCausalSMOTE oversamples strictly on train and preserves discrete PS-L one-hot codes."""
        prep = CustomFabTelemetryPreprocessor()
        n_tr = int(len(self.X_all) * 0.70)
        prep.fit(self.X_all[:n_tr], self.y_all[:n_tr])
        X_tr, M_tr, _ = prep.transform(self.X_all[:n_tr])
        f_cont = M_tr.shape[1]

        X_bal, M_bal, y_bal, real_pass = custom_causal_smote(
            X_tr, M_tr, self.y_all[:n_tr], f_cont=f_cont, k=5, target_ratio=0.35, seed=1337
        )
        self.assertGreater(len(X_bal), len(X_tr))
        self.assertEqual(int(real_pass.sum()), int((self.y_all[:n_tr] == 0).sum()))
        # Synthetic samples must have valid binary PS-L flags (0.0 or 1.0, never fractional)
        cat_cols = X_bal[:, f_cont:]
        self.assertTrue(np.all((cat_cols == 0.0) | (cat_cols == 1.0)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
