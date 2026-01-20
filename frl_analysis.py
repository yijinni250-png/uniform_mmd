import pandas as pd
import numpy as np
from scipy.optimize import nnls
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load Data
file_name = 'sdr_fairness_results_full_opt.csv'
try:
    df = pd.read_csv(file_name)
except FileNotFoundError:
    print(f"Error: File '{file_name}' not found.")
    # For testing purposes only, create dummy data if file is missing
    # df = pd.DataFrame(...) 
    exit()

# 2. Define Sets (As given in the original file)
calibration_widths = [100, 400, 1600, 2000]
evaluation_widths = [20, 800, 1200]

# 3. Precompute Feature Terms (Theoretical Logic)
n_train = 128
delta = 0.05
nu = 1.0
# Term B constant calculation
term_B_const = (8.0 * np.sqrt(2.0) * nu * np.sqrt(np.log(2.0/delta)/n_train)) + (4.0 * nu / n_train)
df['Term_B_Logic'] = term_B_const

# ---------------------------------------------------------
# 4. Per-Seed Fitting and Evaluation
# ---------------------------------------------------------
seeds = df['Seed'].unique()
results = []
test_predictions = []

print(f"Processing {len(seeds)} seeds...")

for seed in seeds:
    # Filter data for this seed
    df_seed = df[df['Seed'] == seed]
    
    # Split into Calibration (Train) and Evaluation (Test)
    df_cal = df_seed[df_seed['Width'].isin(calibration_widths)]
    df_eval = df_seed[df_seed['Width'].isin(evaluation_widths)].copy()
    
    if df_cal.empty or df_eval.empty:
        continue

    # Prepare Calibration Data
    X_train = df_cal[['Comp', 'Term_B_Logic']].values
    y_train = df_cal['Gap'].values
    
    # Fit Model (Non-negative Least Squares)
    coeffs, _ = nnls(X_train, y_train)
    c1, c2 = coeffs
    
    # Predict on Test Set
    X_test = df_eval[['Comp', 'Term_B_Logic']].values
    y_test = df_eval['Gap'].values
    y_pred = X_test @ coeffs
    
    # Calculate Metrics for this Seed
    # Note: R2 is calculated on the specific test set for this seed
    # (Handling case where r2 might be undefined if y_test is constant)
    if np.var(y_test) > 0:
        r2 = r2_score(y_test, y_pred)
    else:
        r2 = np.nan # Undefined if variance is 0
        
    mae = mean_absolute_error(y_test, y_pred)
    
    # Store Seed-Level Aggregate Metrics
    results.append({
        'Seed': seed,
        'R2': r2,
        'MAE': mae,
        'C1': c1,
        'C2': c2
    })
    
    # Store Per-Point Predictions for Table Generation
    df_eval['Predicted'] = y_pred
    df_eval['Abs_Error'] = np.abs(df_eval['Gap'] - df_eval['Predicted'])
    df_eval['Rel_Error'] = (df_eval['Predicted'] - df_eval['Gap']) / df_eval['Gap'] * 100
    test_predictions.append(df_eval)

# ---------------------------------------------------------
# 5. Aggregate and Report Results
# ---------------------------------------------------------
results_df = pd.DataFrame(results)
all_preds_df = pd.concat(test_predictions)

# A. Overall Model Performance (Mean +/- Std of R2)
mean_r2 = results_df['R2'].mean()
std_r2 = results_df['R2'].std()
mean_mae = results_df['MAE'].mean()
std_mae = results_df['MAE'].std()

print("\n" + "="*60)
print("OVERALL MODEL PERFORMANCE (Test Set)")
print(f"Calibration Widths: {calibration_widths}")
print(f"Test Widths:        {evaluation_widths}")
print("-" * 60)
print(f"Test R^2: {mean_r2:.4f} ± {std_r2:.4f}")
print(f"Test MAE: {mean_mae:.4f} ± {std_mae:.4f}")
print("="*60)

# B. Detailed Table by Width (Test Set)
# We group by Width and calculate Mean/Std for Actual, Pred, and Errors
table_stats = all_preds_df.groupby('Width').agg({
    'Gap': ['mean', 'std'],
    'Predicted': ['mean', 'std'],
    'Abs_Error': ['mean', 'std'],
    'Rel_Error': ['mean', 'std']
}).reset_index()

print("\nDETAILED PREDICTIVE RESULTS (Test Widths)")
print("-" * 115)
header = f"{'Width':<8} | {'Actual Gap':<20} | {'Predicted':<20} | {'Abs Error':<20} | {'Rel Error (%)':<20}"
print(header)
print("-" * 115)

for _, row in table_stats.iterrows():
    w = int(row['Width'])
    # Extract Mean and Std
    act_m, act_s = row['Gap']['mean'], row['Gap']['std']
    pred_m, pred_s = row['Predicted']['mean'], row['Predicted']['std']
    abs_m, abs_s = row['Abs_Error']['mean'], row['Abs_Error']['std']
    rel_m, rel_s = row['Rel_Error']['mean'], row['Rel_Error']['std']
    
    # Format String
    line = (f"{w:<8} | "
            f"{act_m:.4f} ± {act_s:.4f}   | "
            f"{pred_m:.4f} ± {pred_s:.4f}   | "
            f"{abs_m:.4f} ± {abs_s:.4f}   | "
            f"{rel_m:.2f} ± {rel_s:.2f}")
    print(line)

print("-" * 115)