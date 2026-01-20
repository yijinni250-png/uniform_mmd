import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from collections import defaultdict
from scipy.stats import norm

# --- Configuration ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_REPEATS = 50 

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# --- 1. Kernel ---
class MMD_Kernel:
    def __init__(self, sigma=None):
        self.sigma = sigma 
    
    def compute_gram(self, f_x, f_y):
        if self.sigma is None:
            with torch.no_grad():
                n_samples = f_x.size(0)
                if n_samples > 100:
                    f_x_sub = f_x[:100]
                else:
                    f_x_sub = f_x
                pdist = torch.cdist(f_x_sub, f_x_sub)
                sigma_est = pdist.median()
                if sigma_est == 0: sigma_est = 1.0
            self.sigma = sigma_est.item()

        f_x_sq = f_x.pow(2).sum(1).view(-1, 1)
        f_y_sq = f_y.pow(2).sum(1).view(1, -1)
        dist = f_x_sq + f_y_sq - 2 * f_x.mm(f_y.t())
        return torch.exp(-dist / (2 * self.sigma**2))

def mmd_loss_func(f_x, f_y, kernel):
    m, n = f_x.size(0), f_y.size(0)
    if m <= 1 or n <= 1: return torch.tensor(0.0).to(device)
    
    K_xx = kernel.compute_gram(f_x, f_x)
    K_yy = kernel.compute_gram(f_y, f_y)
    K_xy = kernel.compute_gram(f_x, f_y)
    
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
    
    # Vectorized Bootstrap
    rademacher = torch.randint(0, 2, (n_boot, m + n)).float().to(device) * 2 - 1
    v = w.unsqueeze(0) * rademacher 
    Kv = K @ v.T 
    val = (v * Kv.T).sum(dim=1).abs()
    
    return np.percentile(val.cpu().numpy(), 95)

def run_asymptotic_test(f_x, f_y, kernel):
    """
    Non-Degenerate Asymptotic Test.
    Estimates the variance of the linear influence function.
    """
    f_x, f_y = f_x.detach(), f_y.detach()
    m, n = f_x.size(0), f_y.size(0)
    
    # 1. Compute Kernel Blocks
    K_xx = kernel.compute_gram(f_x, f_x)
    K_yy = kernel.compute_gram(f_y, f_y)
    K_xy = kernel.compute_gram(f_x, f_y)
    
    # 2. Estimate Linear Terms (Row Means)
    mu_xx = K_xx.mean(dim=1) 
    mu_xy = K_xy.mean(dim=1) 
    
    mu_yy = K_yy.mean(dim=1)
    mu_yx = K_xy.mean(dim=0) 
    
    g_x = mu_xx - mu_xy
    g_y = mu_yy - mu_yx
    
    # 3. Total Linear Variance
    var_x = torch.var(g_x, unbiased=True)
    var_y = torch.var(g_y, unbiased=True)
    
    linear_variance = 4.0 * (var_x / m + var_y / n)
    
    if linear_variance.item() <= 1e-12:
        return 0.0
        
    final_std = torch.sqrt(linear_variance).item()
    return norm.ppf(0.95, loc=0, scale=final_std)

# --- 3. Complexity (Updated with Optimization) ---
def compute_gaussian_complexity_opt(f_z, n_samples=50, opt_steps=20, lr=1.0):
    """
    Computes Gaussian Complexity by explicitly solving the maximization problem:
    G(F) = (1/N) * E_xi [ sup_{||w||<=1} sum_{i=1}^N xi_i * (w^T f_z_i) ]
    Uses Projected Gradient Ascent.
    """
    # 1. Detach f_z to ensure we don't backprop into the encoder during this calculation
    f_z = f_z.detach()
    N, d = f_z.shape
    
    # 2. Generate Gaussian Noise samples (N, B)
    xi = torch.randn(N, n_samples).to(f_z.device)
    
    # 3. Initialize weights 'w' (d, B) - normalized to unit sphere
    w = torch.randn(d, n_samples, device=f_z.device)
    w = w / (w.norm(dim=0, keepdim=True) + 1e-8)
    
    # 4. Projected Gradient Ascent (no autograd needed)
    for _ in range(opt_steps):
        # Forward: Project data onto w (N, d) @ (d, B) -> (N, B)
        projections = f_z @ w
        
        # Gradient of score w.r.t. w: d/dw [sum_i xi_i * (w^T f_i)]
        # = sum_i xi_i * f_i = f_z^T @ xi
        grad = f_z.T @ xi  # (d, B)
        
        # Gradient ascent step
        w = w + lr * grad
        
        # Project back to unit sphere
        norms = w.norm(dim=0, keepdim=True)
        w = w / torch.clamp(norms, min=1e-8)

    # 5. Compute Final Value
    final_projections = f_z @ w
    weighted_sums = (final_projections * xi).sum(dim=0)
    complexity = weighted_sums.mean().item() / N
    
    return complexity

# --- 4. Data Generation ---
def generate_data(n=100, dim=2000):
    S = torch.randint(0, 2, (n, 1)).float()
    noise = torch.randn(n, dim)
    X = noise + S * 0.5 
    logits = (X * (S * 2 - 1)).sum(dim=1, keepdim=True) * 0.1
    Y_prob = torch.sigmoid(logits)
    Y = (torch.rand(n, 1) < Y_prob).float()
    return X.to(device), S.to(device), Y.to(device)

# --- 5. Models ---
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_width, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_width, hidden_width),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_width, latent_dim)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
    def forward(self, x): return self.net(x)

class Predictor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 1)
    def forward(self, z): 
        return self.fc(z)

# --- 6. Main Experiment ---
if __name__ == "__main__":
    n_train, n_test = 128, 5000
    x_dim, latent_dim = 2000, 16
    lambda_mmd = 1.0 
    
    # Theory Constants
    l_const, nu = 0.1715, 1.0
    coef_term1_raw = (12.0 * np.sqrt(2 * np.pi) * l_const) / n_train
    term2_raw_val = (8.0 * np.sqrt(2.0) * nu * np.sqrt(np.log(2.0/0.05)/n_train)) + (4.0 * nu / n_train)

    widths = [20, 100, 400, 800, 1200, 2000]
    
    # --- STORAGE ---
    all_raw_records = []  
    aggregated_results = defaultdict(lambda: defaultdict(list))
    all_r2_calibrated = []
    all_r2_boot = []
    all_r2_asym = []

    print(f"Starting experiment with {NUM_REPEATS} repeats...")

    for seed_idx in range(NUM_REPEATS):
        set_seed(42 + seed_idx)
        if (seed_idx + 1) % 5 == 0:
            print(f"Processing Seed {seed_idx+1}/{NUM_REPEATS}...")

        X_tr, S_tr, Y_tr = generate_data(n_train, x_dim)
        idx0_tr, idx1_tr = (S_tr==0).squeeze(), (S_tr==1).squeeze()
        X_te, S_te, Y_te = generate_data(n_test, x_dim)
        idx0_te, idx1_te = (S_te==0).squeeze(), (S_te==1).squeeze()
        
        current_run_data = []

        for w in widths:
            h = Encoder(x_dim, w, latent_dim).to(device)
            pred = Predictor(latent_dim).to(device)
            optimizer = torch.optim.SGD(list(h.parameters()) + list(pred.parameters()), lr=0.05, momentum=0.9)
            kernel = MMD_Kernel(sigma=None)

            # Train
            for epoch in range(300):
                optimizer.zero_grad()
                z = h(X_tr)
                y_hat = pred(z)
                loss_cls = F.binary_cross_entropy_with_logits(y_hat, Y_tr)
                z0, z1 = z[idx0_tr], z[idx1_tr]
                if epoch % 50 == 0: kernel.compute_gram(z0, z1)
                mmd_val = mmd_loss_func(z0, z1, kernel)
                loss = loss_cls + lambda_mmd * mmd_val
                loss.backward()
                torch.nn.utils.clip_grad_norm_(h.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(pred.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Evaluate
            with torch.no_grad():
                z_tr, z_te = h(X_tr), h(X_te)
                emp_mmd = mmd_loss_func(z_tr[idx0_tr], z_tr[idx1_tr], kernel).item()
                true_mmd = mmd_loss_func(z_te[idx0_te], z_te[idx1_te], kernel).item()
                gap = abs(true_mmd - emp_mmd)
                
                boot_val = run_wild_bootstrap(z_tr[idx0_tr], z_tr[idx1_tr], kernel)
                asym_val = run_asymptotic_test(z_tr[idx0_tr], z_tr[idx1_tr], kernel)
                
                f_joint = torch.cat([z_tr[idx0_tr], z_tr[idx1_tr]], dim=0)
                
                # --- UPDATED: Use Optimization-based Gaussian Complexity ---
                mean_comp = compute_gaussian_complexity_opt(f_joint, n_samples=50, opt_steps=20)
                # -----------------------------------------------------------
                
                sum_comp = mean_comp * n_train 
                theory_full = (coef_term1_raw * sum_comp) + term2_raw_val

                record = {
                    'Seed': seed_idx + 1,
                    'Width': w, 
                    'Gap': gap, 
                    'Boot': boot_val, 
                    'Asym': asym_val, 
                    'Theory': theory_full, 
                    'Comp': sum_comp
                }
                current_run_data.append(record)
                all_raw_records.append(record)
        
        # Per-Seed Calibration
        df_run = pd.DataFrame(current_run_data)
        reg = LinearRegression(fit_intercept=True)
        reg.fit(df_run[['Comp']], df_run['Gap'])
        df_run['Calibrated'] = reg.predict(df_run[['Comp']])
        
        # Backfill calibrated data
        for i, row in df_run.iterrows():
            global_idx = len(all_raw_records) - len(widths) + i
            all_raw_records[global_idx]['Calibrated'] = row['Calibrated']

        all_r2_calibrated.append(r2_score(df_run['Gap'], df_run['Calibrated']))
        all_r2_boot.append(r2_score(df_run['Gap'], df_run['Boot']))
        all_r2_asym.append(r2_score(df_run['Gap'], df_run['Asym']))
        
        for _, row in df_run.iterrows():
            w_key = int(row['Width'])
            aggregated_results[w_key]['Gap'].append(row['Gap'])
            aggregated_results[w_key]['Boot'].append(row['Boot'])
            aggregated_results[w_key]['Asym'].append(row['Asym'])
            aggregated_results[w_key]['Calibrated'].append(row['Calibrated'])
            aggregated_results[w_key]['Theory'].append(row['Theory'])

    # --- SAVE FULL RESULTS ---
    df_all = pd.DataFrame(all_raw_records)
    csv_filename = "sdr_fairness_results_full_opt.csv"
    df_all.to_csv(csv_filename, index=False)
    print(f"\n[SAVED] Full records saved to '{csv_filename}' ({len(df_all)} rows).")

    # --- SUMMARY TABLE ---
    plot_data = defaultdict(list)
    for w in widths:
        res = aggregated_results[w]
        plot_data['Width'].append(w)
        for key in ['Gap', 'Boot', 'Asym', 'Calibrated', 'Theory']:
            plot_data[f'{key}_M'].append(np.mean(res[key]))
            plot_data[f'{key}_S'].append(np.std(res[key]))

    print("\n" + "="*90)
    print(f"TREND TRACKING ANALYSIS (Structural Validity - {NUM_REPEATS} runs)")
    print("-" * 90)
    print(f"Proposed Calibrated Bound R2: {np.mean(all_r2_calibrated):.4f} ± {np.std(all_r2_calibrated):.4f}")
    print(f"Wild Bootstrap Baseline R2:   {np.mean(all_r2_boot):.4f} ± {np.std(all_r2_boot):.4f}")
    print(f"Asymptotic Baseline R2:       {np.mean(all_r2_asym):.4f} ± {np.std(all_r2_asym):.4f}")
    print("="*90)

    print(f"\n{'Width':<6} | {'Gap (Mean ± Std)':<18} | {'Boot (Mean ± Std)':<18} | {'Asym (Mean ± Std)':<18} | {'Calib (Mean ± Std)':<18}")
    print("-" * 110)
    for i, w in enumerate(widths):
        print(f"{w:<6} | {plot_data['Gap_M'][i]:.4f} ± {plot_data['Gap_S'][i]:.4f} | "
              f"{plot_data['Boot_M'][i]:.4f} ± {plot_data['Boot_S'][i]:.4f} | "
              f"{plot_data['Asym_M'][i]:.4f} ± {plot_data['Asym_S'][i]:.4f} | "
              f"{plot_data['Calibrated_M'][i]:.4f} ± {plot_data['Calibrated_S'][i]:.4f}")

    # --- PLOTTING ---
    plt.figure(figsize=(10, 7))
    plt.rcParams['font.family'] = 'serif'
    def plot_with_std(x, y_mean, y_std, label, color, fmt):
        plt.plot(x, y_mean, fmt, color=color, linewidth=2, label=label)
        plt.fill_between(x, np.array(y_mean) - np.array(y_std), 
                         np.array(y_mean) + np.array(y_std), color=color, alpha=0.15)

    x_vals = plot_data['Width']
    plot_with_std(x_vals, plot_data['Gap_M'], plot_data['Gap_S'], 'Gen. Gap', 'red', 'o-')
    plot_with_std(x_vals, plot_data['Calibrated_M'], plot_data['Calibrated_S'], 'Calibrated Bound', 'blue', '--')
    plot_with_std(x_vals, plot_data['Boot_M'], plot_data['Boot_S'], 'Wild Bootstrap', 'gray', 'v--')
    plot_with_std(x_vals, plot_data['Asym_M'], plot_data['Asym_S'], 'Asymptotic (Gaussian)', 'black', ':')

    plt.xlabel('Encoder Width ($w$)')
    plt.ylabel('Generalization Gap')
    plt.title(f'Fairness Gap vs. Baselines ({NUM_REPEATS} runs)')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('fairness_gap_final_opt.pdf', format='pdf', bbox_inches='tight')
    print("Experiment complete. Plot saved.")
