"""
This script estimates two in‑sample explanatory models for "VALUATION_VALUE"
using macroeconomic variables. The estimated equation has the form:

    Y₁ₜ = a₀ + a₁·X₁ₜ + a₂·X₂ₜ + ... + (lagged terms)

The steps are:
  1. Load and preprocess data from "Data_VAR_VECM.xlsx", sheet "Primary".
     Preprocessing removes outliers using a robust (MAD) rule:
     observations falling outside median ± (2 * 1.4826 * MAD) for any variable are excluded.
  2. Test each variable for stationarity (ADF test). If at least one series is nonstationary,
     the VAR is estimated on first differences.
  3. Run the Johansen cointegration test (on levels). If cointegration is detected,
     a VECM is estimated on levels.
  4. Compute in‑sample (training) verification metrics for the explained variable.
     (The differenced explanatory data are used to predict the target in levels by “integrating”
      the fitted differences with the previous period’s actual target value.)
  5. Estimate models on the full sample to produce out‑of‑sample (validation) fitted values,
     ensuring that the indexes are properly aligned despite the lost observation from differencing.
  6. Produce a stability chart (unit circle with eigenvalues).
  7. **Prediction Component:** Load evaluation data from "Main_task_data_evaluation_with_predictions.csv"
     (which includes a column named VALUE_DATE_quarter), merge the VAR predictions with it, and
     then generate a separate Excel file for each quarter containing the predictions.
     
No forecasting (extrapolation) is performed.
"""

import warnings, math
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ------------------------------
# 1. Data Loading & Preprocessing
# ------------------------------
df = pd.read_excel("Data_VAR_VECM.xlsx", sheet_name="Primary")
df["Date"] = df["Date"].str.replace(" ", "")
df.index = pd.PeriodIndex(df["Date"], freq="Q")
df.drop(columns=["Date"], inplace=True)

# Convert numeric columns: replace commas with dots and convert to numeric.
for col in df.columns:
    df[col] = df[col].astype(str).str.replace(",", ".")
    df[col] = pd.to_numeric(df[col], errors='coerce')

target = "VALUATION_VALUE"
explanatory = ["3-Month Rate", "Core Price Index", "Exchange Rate (PLN/EUR)",
               "Gross Domestic Product", "Residential House Price", "Unemployment Rate"]

df = df[[target] + explanatory]

# ------------------------------
# Data Cleaning: Remove Outliers Using MAD
# ------------------------------
def remove_outliers_mad(df, threshold=2):
    df_clean = df.copy()
    for col in df.columns:
        median = df[col].median()
        mad = np.median(np.abs(df[col] - median))
        lower = median - threshold * 1.4826 * mad
        upper = median + threshold * 1.4826 * mad
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean

df = remove_outliers_mad(df, threshold=2)

# ------------------------------
# 2. Stationarity Testing (ADF)
# ------------------------------
print("ADF Stationarity Tests:")
def test_stationarity(series, signif=0.05):
    result = adfuller(series.dropna())
    return result[1], result[1] < signif

stationarity = {}
for col in df.columns:
    p_val, is_stat = test_stationarity(df[col])
    stationarity[col] = is_stat
    print(f"{col}: p-value = {p_val:.4f}, Stationary: {is_stat}")

if not all(stationarity.values()):
    print("\nNonstationary series detected; VAR will be estimated on first differences.")
    df_diff = df.diff().dropna()
else:
    print("\nAll series are stationary; VAR will be estimated on levels.")
    df_diff = df.copy()

# ------------------------------
# 3. Split Sample: Training (80%) and Validation (20%)
# ------------------------------
n = len(df)
train_size = int(0.8 * n)
df_train = df.iloc[:train_size].copy()
df_val   = df.iloc[train_size:].copy()

# For VAR on differences, note that differencing loses one observation.
df_train_diff = df_train.diff().dropna()

# ------------------------------
# 4. Johansen Cointegration Test (Levels)
# ------------------------------
jres = coint_johansen(df, det_order=0, k_ar_diff=2)
cointegration_rank = 0
for i, (trace_stat, crit_val) in enumerate(zip(jres.lr1, jres.cvt[:, 1])):
    if trace_stat > crit_val:
        cointegration_rank = i + 1
print(f"\nJohansen cointegration test: {cointegration_rank} cointegrating relationship(s) found.")

# ------------------------------
# 5a. VAR Model on First Differences (Training Sample)
# ------------------------------
var_model = VAR(df_train_diff)
order_results = var_model.select_order(maxlags=4)
selected_lag_var = order_results.aic
if selected_lag_var is None or selected_lag_var < 1:
    selected_lag_var = 1
var_res = var_model.fit(selected_lag_var)
print("\nVAR Estimation (on differences) Results:")
print(var_res.summary())

# Integration: Predict the target in levels by adding the fitted differences to the lagged target.
n_obs_var = len(var_res.fittedvalues)
fitted_diff_VAR = var_res.fittedvalues[target].values
lagged_target_train = df_train[target].iloc[1:].values  # Because differencing loses one observation.
if len(lagged_target_train) != len(fitted_diff_VAR):
    lagged_target_train = lagged_target_train[-len(fitted_diff_VAR):]
fitted_level_VAR = lagged_target_train + fitted_diff_VAR
actual_level_VAR = df_train[target].iloc[1:].values[-len(fitted_diff_VAR):]

mse_VAR = np.mean((fitted_level_VAR - actual_level_VAR)**2)
mae_VAR = np.mean(np.abs(fitted_level_VAR - actual_level_VAR))
rmse_VAR = math.sqrt(mse_VAR)
mape_VAR = np.mean(np.abs((fitted_level_VAR - actual_level_VAR) / np.where(actual_level_VAR==0, np.nan, actual_level_VAR))) * 100
r2_VAR = r2_score(actual_level_VAR, fitted_level_VAR)
print("\nVAR Model In-Sample Verification Metrics (Training Sample) for VALUATION_VALUE:")
print(f"MSE = {mse_VAR:.4f}, MAE = {mae_VAR:.4f}, RMSE = {rmse_VAR:.4f}, MAPE = {mape_VAR:.2f}%, R² = {r2_VAR:.4f}")

# ------------------------------
# 5b. VECM Model on Levels (Training Sample, if cointegration detected)
# ------------------------------
if cointegration_rank > 0:
    print("\nCointegration detected. Estimating VECM on levels (Training Sample).")
    p_levels = 2  # Number of lags in levels => k_ar_diff = 1
    k_ar_diff = p_levels - 1
    vecm_model = VECM(df_train, k_ar_diff=k_ar_diff, coint_rank=cointegration_rank, deterministic="co")
    vecm_res = vecm_model.fit()
    print("\nVECM Estimation Results (Training Sample):")
    print(vecm_res.summary())
    
    n_obs_vecm = len(vecm_res.fittedvalues)
    fitted_diff_VECM = vecm_res.fittedvalues[:, 0]
    lagged_target_vecm = df_train[target].shift(1).iloc[1:].values
    if len(lagged_target_vecm) != len(fitted_diff_VECM):
        lagged_target_vecm = lagged_target_vecm[-len(fitted_diff_VECM):]
    fitted_level_VECM = lagged_target_vecm + fitted_diff_VECM
    actual_level_VECM = df_train[target].shift(1).iloc[1:].values[-len(fitted_diff_VECM):]
    
    mse_VECM = np.mean((fitted_level_VECM - actual_level_VECM)**2)
    mae_VECM = np.mean(np.abs(fitted_level_VECM - actual_level_VECM))
    rmse_VECM = math.sqrt(mse_VECM)
    mape_VECM = np.mean(np.abs((fitted_level_VECM - actual_level_VECM) / np.where(actual_level_VECM==0, np.nan, actual_level_VECM))) * 100
    r2_VECM = r2_score(actual_level_VECM, fitted_level_VECM)
    print("\nVECM Model In-Sample Verification Metrics (Training Sample) for VALUATION_VALUE:")
    print(f"MSE = {mse_VECM:.4f}, MAE = {mae_VECM:.4f}, RMSE = {rmse_VECM:.4f}, MAPE = {mape_VECM:.2f}%, R² = {r2_VECM:.4f}")
else:
    print("\nNo cointegration detected; VECM not estimated.")

# ------------------------------
# 6. Stability Check: Unit Circle & Eigenvalues
# ------------------------------
plt.figure(figsize=(6,6))
theta = np.linspace(0, 2*np.pi, 200)
plt.plot(np.cos(theta), np.sin(theta), 'b--', label='Unit Circle')

if cointegration_rank > 0:
    k = vecm_res.alpha.shape[0]
    I_k = np.eye(k)
    comp_matrix = I_k + np.dot(vecm_res.alpha, vecm_res.beta.T) + vecm_res.gamma
    eigen_VECM = np.linalg.eigvals(comp_matrix)
    plt.plot(eigen_VECM.real, eigen_VECM.imag, 'ro', label='VECM Eigenvalues')
    stability_label = "VECM"
else:
    eigen_VAR = var_res.roots
    plt.plot(eigen_VAR.real, eigen_VAR.imag, 'ro', label='VAR Eigenvalues')
    stability_label = "VAR (Differenced)"
    
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.title("Model Stability: Eigenvalues (" + stability_label + ")")
plt.legend()
plt.grid(True)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.savefig("Model_Stability_Eigenvalues.png")
plt.show()
print("\nSaved stability plot to 'Model_Stability_Eigenvalues.png'")

# ------------------------------
# 7. Out-of-Sample (Validation) Fitted Values & Charts
# ------------------------------

# Define the start of the validation period:
val_start = df_val.index[0]

# ---- 7a. VAR on Full Differenced Data ----
var_model_full = VAR(df.diff().dropna())
order_results_full = var_model_full.select_order(maxlags=4)
selected_lag_full = order_results_full.aic
if selected_lag_full is None or selected_lag_full < 1:
    selected_lag_full = 1
var_res_full = var_model_full.fit(selected_lag_full)

# Get fitted differences for the full sample as a Series.
fitted_diff_VAR_full = var_res_full.fittedvalues[target]
lagged_VAR_full = df[target].shift(1).iloc[-len(fitted_diff_VAR_full):].values
fitted_level_VAR_full = lagged_VAR_full + fitted_diff_VAR_full.values
plot_index_VAR = df.index[-len(fitted_diff_VAR_full):]
var_val = pd.Series(fitted_level_VAR_full, index=plot_index_VAR)
actual_val_VAR = df[target].iloc[-len(fitted_diff_VAR_full):]
var_val = var_val[var_val.index >= val_start]
actual_val_VAR = actual_val_VAR[actual_val_VAR.index >= val_start]

# ---- 7b. VECM on Full Levels Data (if estimated) ----
if cointegration_rank > 0:
    vecm_model_full = VECM(df, k_ar_diff=k_ar_diff, coint_rank=cointegration_rank, deterministic="co")
    vecm_res_full = vecm_model_full.fit()
    fitted_diff_VECM_full_array = vecm_res_full.fittedvalues[:, 0]
    plot_index_VECM = df.index[k_ar_diff+1:]
    lagged_VECM_full = df[target].shift(1).iloc[k_ar_diff+1:].values
    fitted_level_VECM_full = lagged_VECM_full[:len(fitted_diff_VECM_full_array)] + fitted_diff_VECM_full_array
    vecm_val = pd.Series(fitted_level_VECM_full, index=plot_index_VECM[:len(fitted_diff_VECM_full_array)])
    actual_val_VECM = df[target].iloc[k_ar_diff+1:len(plot_index_VECM[:len(fitted_diff_VECM_full_array)])]
    vecm_val = vecm_val[vecm_val.index >= val_start]
    actual_val_VECM = actual_val_VECM[actual_val_VECM.index >= val_start]

# ---- 7c. Plot Validation Charts (VAR and VECM if available) ----
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
plt.plot(var_val.index.to_timestamp(), actual_val_VAR, 'b-', label="Actual")
plt.plot(var_val.index.to_timestamp(), var_val, 'r--', label="VAR Fitted")
plt.xlabel("Date")
plt.ylabel("VALUATION_VALUE")
plt.title("VAR: Actual vs. Fitted (Validation)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

if cointegration_rank > 0:
    plt.subplot(1,2,2)
    plt.plot(vecm_val.index.to_timestamp(), actual_val_VECM, 'b-', label="Actual")
    plt.plot(vecm_val.index.to_timestamp(), vecm_val, 'r--', label="VECM Fitted")
    plt.xlabel("Date")
    plt.ylabel("VALUATION_VALUE")
    plt.title("VECM: Actual vs. Fitted (Validation)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("Actual_vs_Fitted_Validation.png")
plt.show()
print("\nSaved combined validation chart to 'Actual_vs_Fitted_Validation.png'")

# ------------------------------
# 8. Generate Predictions for Each Quarter
# ------------------------------
# Load evaluation data from the CSV file.
eval_df = pd.read_csv("Main_task_data_evaluation_with_predictions.csv")
# Ensure VALUE_DATE_quarter is a string, remove spaces, and convert it to a PeriodIndex.
eval_df["VALUE_DATE_quarter"] = eval_df["VALUE_DATE_quarter"].astype(str).str.replace(" ", "")
eval_df["Quarter"] = pd.PeriodIndex(eval_df["VALUE_DATE_quarter"], freq="Q")

# Create a DataFrame from the VAR validation predictions (var_val)
pred_df = var_val.to_frame(name="Pred_VAR")

# Merge predictions with the evaluation data based on the PeriodIndex (Quarter).
eval_df = eval_df.merge(pred_df, left_on="Quarter", right_index=True, how="left")

# For each unique quarter, generate a separate Excel file.
for quarter in eval_df["Quarter"].unique():
    quarter_df = eval_df[eval_df["Quarter"] == quarter]
    filename = f"Predictions_{quarter}.xlsx"
    quarter_df.to_excel(filename, index=False)
    print(f"Saved predictions for {quarter} to {filename}")
