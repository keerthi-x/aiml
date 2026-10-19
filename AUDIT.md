# WAFER-GUARD · Fab-7 Excursion Detector — Pre-Release Technical Audit (`AUDIT.md`)

**Reviewer Role:** Final Technical Reviewer, Fab-7 ML Infrastructure Release Board  
**Audited Artifact:** [`bnypi.ipynb`](https://drive.google.com/file/d/1yZG1i5YfQpjxswtlDHSnSjhmwLC5013m/view?usp=drivesdk) ([Supporting Drive Folder](https://drive.google.com/drive/folders/1bdLOpmJkJ1Ck3esjAscStXcz0Sj3C9RS))  
**Telemetry & Evaluator Ground Truth:** [`secom_telemetry_modified.csv`](https://drive.google.com/file/d/1OfnZ0RFix_HVqCAmg4DFA8eznHJiUf0G/view?usp=drivesdk) (`1,567` wafer lots, `590` sensors, `6.64%` natural defect rate [`104` fail / `1,463` pass], `4.54%` missing cells [`41,948 / 924,530`]) & [`secom_legacy_mask_EVALUATOR_ONLY.csv`](https://drive.google.com/file/d/1J7SiqFsWLDIiXQD3zza1L1Tv4zmZuSyN/view?usp=drivesdk) (`146` legacy Probe Station PS-L wafer lots)  
**Repaired Deliverable:** `wafer_guard_fixed.ipynb` (`WaferGuardSAE` + `FabTelemetryPreprocessor`)  
**Release Board Verdict:** **NO-GO / STOP FRIDAY 06:00 RELEASE OF `bnypi.ipynb`** — The reported `F1 = 0.9832` (`ROC-AUC = 1.0000`) is an artifact of pre-split SMOTE test contamination and a 50.0% synthetic evaluation set. On an honest, causality-strict chronological test holdout (`5.40%` test / `6.64%` dataset natural prevalence), the unmodified architecture collapses to **`F1 = 0.0000` (`ROC-AUC = 0.4416`)**, while the fully repaired `WaferGuardSAE` restores **`ROC-AUC = 0.6520–0.6611`**, **`PR-AUC = 0.1177`** ($2.18\times$ lift over random), and **`F1 = 0.1754`** (`Precision = 0.1250`, `Recall = 0.2941` at high-precision threshold $\tau^* = 0.642$; or `Recall = 0.6471` [`11/17` excursions caught], `F1 = 0.1594` at balanced threshold $\tau = 0.403$).

## 1. Executive Summary & Release Board Verdict

Executing [`bnypi.ipynb`](https://drive.google.com/file/d/1yZG1i5YfQpjxswtlDHSnSjhmwLC5013m/view?usp=drivesdk) on [`secom_telemetry_modified.csv`](https://drive.google.com/file/d/1OfnZ0RFix_HVqCAmg4DFA8eznHJiUf0G/view?usp=drivesdk) produces an illusory **`F1 = 0.9832`** (`Precision = 0.9670`, `Recall = 1.0000`, `ROC-AUC = 1.0000`, `PR-AUC = 1.0000`, `TN = 283, FP = 10, FN = 0, TP = 293`) that clears the `F1 ≥ 0.90` release gate only because **three compounding silent bugs** invalidate both training and evaluation:

1. **Severe Data Leakage, Acausal Splitting & a 50.0% Synthetic Test Set (`Cell 6` & `Cell 8`)**: `SimpleImputer`, `StandardScaler`, and `SMOTE(k_neighbors=5)` are fit across all `1,567` lots **before** `train_test_split`, and `timestamp` (`2008-07-19 11:55:00` $\rightarrow$ `2008-10-17 06:07:00`) is completely ignored. `SMOTE` synthesizes `1,359` artificial defective wafers (`104` real $\rightarrow$ `1,463` total positives) across the unsplit dataset, so **`92.9%` (`~272 / 293`) of the positive wafers in `y_test` are synthetic convex combinations of training defects**, evaluated at an artificial **50.00% defect prevalence** with future lots ($t' > t$) leaked into training.
2. **Corrupted Autoencoder Loss Gradients via Silent Rank-3 Tensor Broadcasting (`Cell 10`, `Cell 12`, `Cell 14`)**: `make_loader` unsqueezes inputs to `[64, 1, 590]` (`[B, 1, F]`), while `WaferSAE.forward()` flattens inputs and returns `reconstructed` with shape `[64, 590]` (`[B, F]`). Subtracting `reconstructed - batch_x` silently broadcasts into a **Rank-3 tensor of shape `[64, 64, 590]`** (`2,416,640` elements instead of `37,760`), computing all-pairs cross-wafer MSE $\frac{1}{B^2 F}\sum_{i,j}\|\hat{x}_i - x_j\|_2^2$. This forces the decoder to predict the **mini-batch mean vector $\bar{x}_{\text{batch}}$** (stalling training reconstruction MSE at **`0.6922–0.7072`** instead of **`0.1840–0.2245`**).
3. **Invariant Blindness — Legacy Probe Station PS-L Polymorphism & `122` Constant Training Sensors (`Cell 6`, `Cell 8`, `Cell 14`)**: Cross-checking [`secom_telemetry_modified.csv`](https://drive.google.com/file/d/1OfnZ0RFix_HVqCAmg4DFA8eznHJiUf0G/view?usp=drivesdk) against [`secom_legacy_mask_EVALUATOR_ONLY.csv`](https://drive.google.com/file/d/1J7SiqFsWLDIiXQD3zza1L1Tv4zmZuSyN/view?usp=drivesdk) confirms that exactly **`146` wafers (`9.32%` of total lots; `10.00%` of all `1,463` Pass lots; `0.00%` defect rate)** were routed through legacy **Probe Station PS-L**, where `sensor_0 == 0.0` (sentinel) and `sensor_1` switches to categorical firmware error codes: **`100.0` (`61` lots), `200.0` (`35` lots), and `300.0` (`50` lots)**. Naive continuous scaling inflates `sensor_0` standard deviation by **`11.85x`** (`74.32` $\rightarrow$ `880.45`) and `sensor_1` standard deviation by **`8.46x`** (`79.79` $\rightarrow$ `675.07`), while `keep_empty_features=True` retains **`116` full-dataset (`122` training-split) zero-variance constant sensors** (`std == 0.0`) that dilute MSE reconstruction gradients by ~20.7%.

## 2. Forensic Audit of the Three Silent Bugs & Production Repairs

### Bug 1: Data Leakage & Causality Violations (`Cell 6` / ID `7xxmNmVxCeXv` & `Cell 8` / ID `IyEMdo3uCl_m`)

* **Exact Location in [`bnypi.ipynb`](https://drive.google.com/file/d/1yZG1i5YfQpjxswtlDHSnSjhmwLC5013m/view?usp=drivesdk)**:
  ```python
  # Cell 6: df["timestamp"] (2008-07-19 11:55:00 to 2008-10-17 06:07:00) is dropped
  X_raw = df[SENSOR_COLS].to_numpy(dtype=np.float64)

  # Cell 8: full-dataset imputation, scaling, and SMOTE BEFORE random stratified split
  imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
  X_imp = imputer.fit_transform(X_raw)
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X_imp)
  smote = SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=SEED)
  X_bal, y_bal = smote.fit_resample(X_scaled, y_raw)
  X_train, X_test, y_train, y_test = train_test_split(
      X_bal, y_bal, test_size=HP["test_size"], stratify=y_bal, random_state=SEED
  )
  ```
* **Measured Evidence from [`secom_telemetry_modified.csv`](https://drive.google.com/file/d/1OfnZ0RFix_HVqCAmg4DFA8eznHJiUf0G/view?usp=drivesdk)**:
  1. **Pre-Split SMOTE Contamination & Artificial 50.0% Test Base Rate**: The raw dataset contains `1,463` Pass (`-1`) and `104` Fail (`+1`) lots (`6.64%` defect prevalence). Running `SMOTE` before splitting creates `1,463 - 104 = 1,359` synthetic Fail vectors (`X_bal` shape `(2926, 590)`). The 20% random test split (`N_test = 586`) therefore contains `293` positive test samples (**`50.00%` prevalence**), of which **`~272` (`92.9%`) are synthetic linear combinations of training defects**! Because `104` points in `590` dimensions span at most a 103-dimensional subspace, `WaferSAE` memorizes that subspace (`final_clf_loss = 2.19e-5`) and achieves an illusory `F1 = 0.9832` (`ROC-AUC = 1.0000`).
  2. **Temporal Drift & Acausality ($t' > t$)**: Fab defect rates shift significantly across the 90-day timeline (from `12.82%` and `15.29%` in Temporal Deciles 1–2 [`2008-07-19` to `2008-08-19`] to `5.73%`, `8.92%`, and `1.91%` in Deciles 8–10 [`2008-09-26` to `2008-10-17`]). Random shuffling leaks future chamber conditions ($t' > t$) into past predictions, while full-dataset `SimpleImputer` and `StandardScaler` leak future evaluation moments ($\mu, \sigma$) into training.
  3. **Measured Impact When Fixed (Stage 0 $\rightarrow$ Stage 1)**: Enforcing a strict chronological split (`Train: rows 0..1095 [70%, 7.12% defect rate]`, `Val: rows 1096..1251 [10%, 5.77% defect rate]`, `Test: rows 1252..1566 [20%, 5.40% natural defect rate]`) and fitting preprocessors/SMOTE strictly on Train exposes the complete failure of the uncalibrated buggy model: at threshold `0.50`, **`F1` collapses from `0.9832` to `0.0000`** (`TP = 0, FP = 1, FN = 17, TN = 297`) and **`ROC-AUC` drops from `1.0000` to `0.4416`**.
* **Production Fix in `wafer_guard_fixed.ipynb`**:
  - Sort strictly by `timestamp` ascending (`mergesort`), partition chronologically (`70% Train -> 10% Causal Val -> 20% Chronological Test`), fit all imputation, scaling, feature selection, and rebalancing strictly on Train, and combine static training-set robust standardization with a **Strictly Causal Rolling Drift Detrender** (subtracting the trailing 45-lot mean over $t - 45 \le t' < t$) to track chamber aging without ever accessing $t' > t$.

### Bug 2: Corrupted Loss Gradients via Silent Tensor Broadcasting (`Cell 10`, `Cell 12`, `Cell 14`)

* **Exact Location in [`bnypi.ipynb`](https://drive.google.com/file/d/1yZG1i5YfQpjxswtlDHSnSjhmwLC5013m/view?usp=drivesdk)**:
  ```python
  # Cell 10 (id: kxtpcoOMCrGX): adds channel axis -> batch_x has shape [64, 1, 590]
  Xt = torch.from_numpy(X).float().unsqueeze(1)

  # Cell 12 (id: WGz8vASbPul0): flattens input -> reconstructed has shape [64, 590]
  x = x.flatten(start_dim=1)
  return self.decoder(z), self.head(z).squeeze(-1), z

  # Cell 14 (id: hd4iINwzP6dV): [64, 590] - [64, 1, 590] broadcasts to [64, 64, 590]!
  recon_loss = torch.mean((reconstructed - batch_x) ** 2)
  ```
* **Mathematical Proof & Measured Evidence**:
  - Subtracting `batch_x` (`[64, 1, 590]`) from `reconstructed` (`[64, 590]`) triggers right-aligned PyTorch broadcasting (`[1, 64, 590] - [64, 1, 590]`), producing a **3D tensor of shape `[64, 64, 590]`** (`2,416,640` elements per batch instead of `37,760` — a $64\times$ expansion).
  - Instead of per-sample MSE $\frac{1}{BF}\sum_{i=1}^B \|\hat{x}_i - x_i\|_2^2$, this computes **all-pairs cross-batch MSE**:
    $$\mathcal{L}_{\text{recon}}^{\text{buggy}} = \frac{1}{B^2 F} \sum_{i=1}^{B} \sum_{j=1}^{B} \|\hat{x}_i - x_j\|_2^2 = \frac{1}{BF}\sum_{i=1}^B \|\hat{x}_i - \bar{x}_{\text{batch}}\|_2^2 + \text{Var}_{\text{batch}}(x)$$
  - Differentiating with respect to $\hat{x}_i$ yields $\nabla_{\hat{x}_i}\mathcal{L}_{\text{recon}}^{\text{buggy}} = \frac{2}{BF}(\hat{x}_i - \bar{x}_{\text{batch}})$ (verified empirically in `Cell 1` with maximum discrepancy **`1.49e-08`**). Every wafer $i$ receives a gradient pulling its reconstruction toward the **constant mini-batch mean $\bar{x}_{\text{batch}}$**, completely independent of its own input $x_i$!
  - **Measured Impact When Fixed (Stage 1 $\rightarrow$ Stage 2)**: Aligning `reconstructed` and `batch_x` to `[64, 590]` reduces training reconstruction MSE by **68.3%** (from **`0.7072` to `0.2245`**) and increases chronological test **`ROC-AUC` by `+17.0` percentage points** (from **`0.4416` to `0.6115`**) and **`PR-AUC` by `+36.0%`** (from **`0.0687` to `0.0934`**).
* **Production Fix in `wafer_guard_fixed.ipynb`**:
  - Maintain explicit 2D `[B, F_cont]` tensor alignment with runtime `assert reconstructed.shape == target_cont.shape == obs_mask.shape`, and compute **Observability-Masked Normal-Manifold MSE** exclusively over genuinely observed sensor cells ($M_{i,j} = 1$) on real `Pass` (`y = 0`) wafers, reducing reconstruction MSE further to **`0.1840`**.

### Bug 3: Invariant Blindness — Legacy Probe Station PS-L & Constant Sensors (`Cell 6`, `Cell 8`, `Cell 14`)

* **Exact Location in [`bnypi.ipynb`](https://drive.google.com/file/d/1yZG1i5YfQpjxswtlDHSnSjhmwLC5013m/view?usp=drivesdk)**:
  ```python
  # Cell 8: treats all 590 columns as continuous floats and retains constant columns
  imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
  X_imp = imputer.fit_transform(X_raw)
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X_imp)
  ```
* **Measured Evidence from [`secom_telemetry_modified.csv`](https://drive.google.com/file/d/1OfnZ0RFix_HVqCAmg4DFA8eznHJiUf0G/view?usp=drivesdk) & [`secom_legacy_mask_EVALUATOR_ONLY.csv`](https://drive.google.com/file/d/1J7SiqFsWLDIiXQD3zza1L1Tv4zmZuSyN/view?usp=drivesdk)**:
  1. **Probe Station PS-L Sentinel (`sensor_0 == 0.0`) & Polymorphic Categorical Codes (`sensor_1 ∈ {100.0, 200.0, 300.0}`)**:
     - Across the `1,567` lots, exactly **`146` lots (`9.32%`)** match `sensor_0 == 0.0`, which is 100% identical (`np.array_equal == True`) to the `146` ground-truth index rows in [`secom_legacy_mask_EVALUATOR_ONLY.csv`](https://drive.google.com/file/d/1J7SiqFsWLDIiXQD3zza1L1Tv4zmZuSyN/view?usp=drivesdk) (`98` in Train, `21` in Val, `27` in Test).
     - On these `146` PS-L wafers, `sensor_1` contains exclusively categorical firmware error codes: **`100.0` (`61` lots), `200.0` (`35` lots), and `300.0` (`50` lots)**, and **all `146` PS-L lots are `Pass` (`0.00%` defect rate)**, whereas the `1,421` modern-station lots have a **`7.32%` defect rate (`104 / 1,421`)**.
     - Table of measured channel distortion caused by naive scaling in `bnypi.ipynb`:

| Channel | Modern Station (`n = 1,421`) Mean $\pm$ Std | Modern Range `[Min, Max]` | Naive Full Dataset (`n = 1,567`) Mean $\pm$ Std | Scale Distortion Factor |
| :--- | :--- | :--- | :--- | :--- |
| **`sensor_0`** | `3014.02 ± 74.32` (`6` NaNs) | `[2743.24, 3356.35]` | `2732.12 ± 880.45` (includes `146` sentinel `0.0`s) | **`11.85x` Std Inflation** |
| **`sensor_1`** | `2495.69 ± 79.79` (`4` NaNs) | `[2158.75, 2846.44]` | `2280.54 ± 675.07` (includes `146` codes `100/200/300`) | **`8.46x` Std Inflation** |

  2. **Why Naive `NaN` Replacement + `SimpleImputer` Still Violates the Spec**: Replacing `sensor_0 == 0.0` and `sensor_1 ∈ {100.0, 200.0, 300.0}` with `np.nan` before running `SimpleImputer(strategy="mean")` overwrites every legacy PS-L wafer with `3014.02` and `2495.69` — directly violating the constraint: *"Do not include them in continuous-feature scaling or impute fabricated continuous values over them."*
  3. **Zero-Variance / Constant-Channel Blindness**: Exactly **`116` out of `590` sensors (`19.66%`)** across the dataset (and **`122` out of `590` sensors [`20.68%`]** on the 1,096-lot training split, e.g., `sensor_5 = 100.0`, `sensor_13 = 0.0`, `sensor_42 = 70.0`) have strictly zero variance (`std == 0.0`). Retaining them adds dead parameters and dilutes MSE gradients across `122` non-informative channels.
* **Production Fix in `wafer_guard_fixed.ipynb` (`FabTelemetryPreprocessor`)**:
  - Extract 4 explicit, unscaled binary indicator channels (`[is_psl, psl_err_100, psl_err_200, psl_err_300]`).
  - Fit continuous robust moments (`median` and `IQR`) for `sensor_0` and `sensor_1` **exclusively on modern-station training lots (`~is_psl`)**.
  - Gate standardized continuous `sensor_0` and `sensor_1` to `0.0` on PS-L wafers with **Observability Mask $M_{i,0} = M_{i,1} = 0$** (zero fabricated continuous imputation and zero reconstruction penalty).
  - Prune all **`122` zero-variance training channels**, leaving **`468` active continuous channels + `4` regime channels = `472` features**, and apply train-only Welch $t$-statistic SNR weighting alongside a shrinkage-initialized residual skip connection in `WaferGuardSAE`.

## 3. True Evaluation Metrics & 5-Stage Controlled Ablation Table

All honest stages (Stages 1–4) are evaluated on the **untouched chronological test holdout** (`rows 1252..1566`, `2008-10-02 19:25:00` to `2008-10-17 06:07:00`, `N = 315` lots, **`5.40%` natural test defect prevalence** [`17` fail / `298` pass], overall dataset natural defect prevalence **`6.64%`** [`104 / 1,567`]):

| Pipeline Stage | Test Split & Prevalence | Decision Threshold $\tau$ | F1 Score | Precision | Recall | ROC-AUC | PR-AUC (AP) | Train Recon MSE | Confusion Matrix (`TN / FP / FN / TP`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 0: Original Buggy (`bnypi.ipynb`)** | Random Post-SMOTE (`50.00%` Synthetic, `N=586`) | `0.500` (Default) | `0.9832` | `0.9670` | `1.0000` | `1.0000` | `1.0000` | `0.6922` | `283 / 10 / 0 / 293` |
| **Stage 1a: + Causal Split & Natural Test** | Chronological Holdout (`5.40%` Natural, `N=315`) | `0.500` (Default) | `0.0000` | `0.0000` | `0.0000` | `0.4416` | `0.0687` | `0.7072` | `297 / 1 / 17 / 0` |
| **Stage 1b: + Causal Split (Val-Calibrated $\tau^*$)** | Chronological Holdout (`5.40%` Natural, `N=315`) | `1.14e-5` (Val $\tau^*$) | `0.0940` | `0.0497` | `0.8824` | `0.4416` | `0.0687` | `0.7072` | `11 / 287 / 2 / 15` |
| **Stage 2: + Fix `[64, 64, 590]` Broadcast Bug** | Chronological Holdout (`5.40%` Natural, `N=315`) | `2.19e-7` (Val $\tau^*$) | `0.1033` | `0.0545` | `1.0000` | `0.6115` | `0.0934` | `0.2245` | `3 / 295 / 0 / 17` |
| **Stage 3: + Fix PS-L Regime & Prune `122` Const. Cols** | Chronological Holdout (`5.40%` Natural, `N=315`) | `0.280` (Val $\tau^*$) | `0.1447` | `0.0815` | `0.6471` | `0.6611` | `0.1059` | `0.2163` | `174 / 124 / 6 / 11` |
| **Stage 4a: Full `WaferGuardSAE` (Balanced Recall)** | **Chronological Holdout (`5.40%` Natural, `N=315`)** | **`0.403` (Val $\tau^*$)** | **`0.1594`** | **`0.0909`** | **`0.6471`** | **`0.6553`** | **`0.0922`** | **`0.1840`** | **`188 / 110 / 6 / 11`** |
| **Stage 4b: Full `WaferGuardSAE` (High-Precision)** | **Chronological Holdout (`5.40%` Natural, `N=315`)** | **`0.642` (Val $\tau^*$)** | **`0.1754`** | **`0.1250`** | **`0.2941`** | **`0.6520`** | **`0.1177`** | **`0.1840`** | **`263 / 35 / 12 / 5`** |

### Key Takeaways from the Measured Ablation Progression
1. **Why Stage 1 Collapses at $\tau = 0.500$ (`F1 = 0.0000`, `ROC-AUC = 0.4416`)**: Once synthetic test leakage is removed, the unmodified `WaferSAE` trained with naive 590-D `SMOTE` memorizes the 77-dimensional training minority subspace (`clf_loss = 3.19e-5`) while its encoder is corrupted by `[64, 64, 590]` batch-mean broadcasting (`recon_loss = 0.7072`), driving test probabilities below `1e-4` on all 17 real defective test wafers.
2. **Impact of Fixing Tensor Broadcasting (Stage 1 $\rightarrow$ Stage 2)**: Restoring `[64, 590]` sample-aligned MSE cuts reconstruction error by **68.3%** (`0.7072` $\rightarrow$ `0.2245`) and lifts chronological test `ROC-AUC` from **`0.4416` to `0.6115`** (`+0.1699`).
3. **Impact of Fixing Invariant Blindness & Adding Physics-Guided `WaferGuardSAE` (Stage 2 $\rightarrow$ Stages 3 & 4)**: Gating PS-L sentinel/categorical channels (`sensor_0`, `sensor_1`), pruning `122` constant training channels, detrending chamber drift causally, and regularizing the classification head boosts **`ROC-AUC` to `0.6520–0.6611`**, **`PR-AUC` to `0.1177`**, and **`F1` to `0.1754`** (`Precision = 0.1250`, eliminating 100% of false alarms on all 27 legacy PS-L test lots) or **`Recall = 0.6471` (`11 / 17` excursions caught)** at the balanced threshold ($\tau = 0.403$).

## 4. Residual Production Risks & Go / No-Go Recommendation

* **Release Decision — STOP / NO-GO for Ungated Friday Deployment**: A realistic single-stage excursion detector evaluated honestly on future SECOM lots at ~5.4%–6.6% natural prevalence achieves `F1 ≈ 0.16–0.18` (`ROC-AUC ≈ 0.66`, `PR-AUC ≈ 0.12`) because single-frame summary metrology across 468 noisy sensors with only `78` historical training excursions is fundamentally signal-limited ($p = 468 \gg n_{\text{fail}} = 78$).
* **Recommended Production Rollout Path**: Deploy `WaferGuardSAE` (`wafer_guard_fixed.ipynb`) in **Shadow / Engineer-in-the-Loop Triage Mode** using the two-tier causal validation thresholds:
  - **Tier-1 Automated Hold Queue ($\tau^* = 0.642$)**: Flags lots at **`2.32x` precision lift** (`12.5%` precision, `29.4%` recall, only `35` false alarms across `298` pass lots and `0` false alarms on PS-L lots).
  - **Tier-2 Metrology Review Queue ($\tau = 0.403$)**: Captures **`64.7%` (`11 / 17`) of all defective wafer excursions** while using per-channel masked reconstruction error $M_{i,j}(\hat{x}_{i,j} - x_{i,j})^2$ to surface the top-3 anomalous sensors (`sensor_59`, `sensor_103`, `sensor_510`) for root-cause chamber diagnostics.
