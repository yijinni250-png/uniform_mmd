import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.optimize import nnls
from collections import defaultdict

# --- Configuration ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_REPEATS = 50 
CSV_FILENAME = "gan_simulation_raw.csv"

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# --- 1. Kernel (Your Original Code) ---
class MMD_Kernel:
    def __init__(self, sigma=10.0): 
        self.sigma = sigma 
    
    def compute_gram(self, f_x, f_y):
        f_x_sq = f_x.pow(2).sum(1).view(-1, 1)
        f_y_sq = f_y.pow(2).sum(1).view(1, -1)
        dist = f_x_sq + f_y_sq - 2 * f_x.mm(f_y.t())
        return torch.exp(-dist / (2 * self.sigma**2))

def mmd_loss_func(f_x, f_y, kernel):
    K_xx = kernel.compute_gram(f_x, f_x)
    K_yy = kernel.compute_gram(f_y, f_y)
    K_xy = kernel.compute_gram(f_x, f_y)
    m, n = f_x.size(0), f_y.size(0)
    xx = (K_xx.sum() - torch.diag(K_xx).sum()) / (m * (m - 1))
    yy = (K_yy.sum() - torch.diag(K_yy).sum()) / (n * (n - 1))
    xy = K_xy.mean()
    return xx + yy - 2 * xy

# --- 2. Baselines ---
def run_wild_bootstrap(f_x, f_y, kernel, n_boot=200):
    f_x, f_y = f_x.detach(), f_y.detach()
    m, n = f_x.size(0), f_y.size(0)
    Z = torch.cat([f_x, f_y], dim=0)
    K = kernel.compute_gram(Z, Z)
    w = torch.cat([torch.ones(m)/m, -torch.ones(n)/n]).to(device)
    stats = []
    for _ in range(n_boot):
        xi = torch.randn(m + n).to(device)
        v = w * xi
        val = v.unsqueeze(0) @ K @ v.unsqueeze(1)
        stats.append(abs(val.item()))
    return np.percentile(stats, 95)

def run_asymptotic_test(f_x, f_y, kernel, n_sim=1000):
    f_x, f_y = f_x.detach(), f_y.detach()
    m, n = f_x.size(0), f_y.size(0)
    N = m + n
    Z = torch.cat([f_x, f_y], dim=0)
    K = kernel.compute_gram(Z, Z)
    H = torch.eye(N).to(device) - torch.ones(N, N).to(device) / N
    K_centered = H @ K @ H
    try:
        evals, _ = torch.linalg.eigh(K_centered.cpu()) 
        evals = evals[evals > 1e-5]
    except: return 0.0
    scale = (1.0/m + 1.0/n)
    weights = evals.numpy() * (1.0/N)
    chi_sq = np.random.chisquare(1, size=(n_sim, len(weights)))
    null_dist = np.dot(chi_sq - 1, weights) * scale
    return np.percentile(np.abs(null_dist), 95)

# --- 3. Complexity ---
def compute_joint_complexity(f_z):
    N = f_z.size(0)
    B = 50 
    norms = []
    for _ in range(B):
        xi = torch.randn(N, 1).to(device)
        weighted_sum = (f_z * xi).sum(dim=0)
        norms.append(weighted_sum.norm().item())
    return np.mean(norms) / N

# --- 4. Deep Discriminator (Your Original Code) ---
class DeepDiscriminator(nn.Module):
    def __init__(self, x_dim, feature_dim, hidden_width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden_width),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_width, hidden_width),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_width, feature_dim)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x): return self.net(x)

# --- 5. Main Experiment & Storage ---
if __name__ == "__main__":
    m, n = 200, 200
    x_dim, feature_dim = 20, 10
    delta = 0.05
    kernel = MMD_Kernel(sigma=10.0)
    widths = [10, 50, 100, 200, 400, 600, 800, 1000]
    
    all_raw_records = []

    print(f"Starting GAN experiment with {NUM_REPEATS} repeats...")

    for seed in range(NUM_REPEATS):
        set_seed(42 + seed)
        if (seed + 1) % 5 == 0: print(f"Processing Seed {seed+1}/{NUM_REPEATS}...")
        
        X = torch.randn(m, x_dim).to(device)
        Y = torch.randn(n, x_dim).to(device)
        
        for width in widths:
            D = DeepDiscriminator(x_dim, feature_dim, width).to(device)
            optimizer = torch.optim.Adam(D.parameters(), lr=0.005, weight_decay=0.01)
            
            # Maximize MMD
            for _ in range(150):
                optimizer.zero_grad()
                loss = -mmd_loss_func(D(X), D(Y), kernel)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
                optimizer.step()
                
            with torch.no_grad():
                f_x = D(X)
                f_y = D(Y)
                max_mmd = abs(mmd_loss_func(f_x, f_y, kernel).item())
                
                # Baselines
                boot_val = run_wild_bootstrap(f_x, f_y, kernel)
                asym_val = run_asymptotic_test(f_x, f_y, kernel)
                
                # Complexity
                f_joint = torch.cat([f_x, f_y], dim=0)
                mean_comp = compute_joint_complexity(f_joint)
                sum_comp = mean_comp * (m + n)
                
                all_raw_records.append({
                    'Seed': seed,
                    'Width': width,
                    'MaxMMD': max_mmd,
                    'Boot': boot_val,
                    'Asym': asym_val,
                    'Comp': sum_comp
                })

    # --- SAVE TO CSV (Added Step) ---
    df = pd.DataFrame(all_raw_records)
    df.to_csv(CSV_FILENAME, index=False)
    print(f"\n[Storage] Full simulation data saved to '{CSV_FILENAME}'")

    # --- ANALYSIS SECTION ---
    print("\n" + "="*60)
    print("SECTION 5.3: PREDICTIVE CALIBRATION ANALYSIS")
    print("="*60)

    # 1. Define Anchor Split
    calibration_widths = [10, 400, 1000]
    evaluation_widths = [50, 100, 200, 600, 800]

    # 2. Add Theoretical Term B
    n_total = m + n
    term_B_const = (8.0 * np.sqrt(2.0) * np.sqrt(np.log(2.0/delta)/n_total)) + (4.0/n_total)
    df['Term_B'] = term_B_const

    # 3. Non-Negative Calibration (NNLS)
    df_calib = df[df['Width'].isin(calibration_widths)]
    df_eval = df[df['Width'].isin(evaluation_widths)]

    X_train = df_calib[['Comp', 'Term_B']].values
    y_train = df_calib['MaxMMD'].values
    
    coeffs, _ = nnls(X_train, y_train)
    c1, c2 = coeffs

    # 4. Prediction
    X_test = df_eval[['Comp', 'Term_B']].values
    y_test = df_eval['MaxMMD'].values
    y_pred = X_test @ coeffs

    # 5. Metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Calibrated Constants (on w={calibration_widths}):")
    print(f"  c1: {c1:.6f}, c2: {c2:.6f}")
    print("-" * 60)
    print(f"Performance on Unseen Widths {evaluation_widths}:")
    print(f"  Test R2 Score: {r2:.4f}")
    print(f"  Test MSE:      {mse:.6f}")
    print("-" * 60)

    # 6. Table
    summary = df_eval.copy()
    summary['Predicted'] = y_pred
    table = summary.groupby('Width').agg({'MaxMMD': 'mean', 'Predicted': 'mean'})
    table['Rel_Error_%'] = ((table['Predicted'] - table['MaxMMD']) / (table['MaxMMD'] + 1e-9)) * 100
    
    print("\n--- Summary Table ---")
    print(table)
    print("="*60)