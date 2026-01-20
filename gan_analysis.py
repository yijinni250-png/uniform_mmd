import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import defaultdict

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_REPEATS = 50 
FILE_NAME = "gan_simulation_raw_50runs.csv"

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

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

def compute_joint_complexity(f_z):
    N = f_z.size(0)
    B = 50 
    xi = torch.randn(N, B).to(device)
    weighted_sum = f_z.t() @ xi
    norms = weighted_sum.norm(dim=0) 
    return norms.mean().item() / N

class DeepDiscriminator(nn.Module):
    def __init__(self, x_dim, feature_dim, hidden_width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden_width), nn.LeakyReLU(0.2),
            nn.Linear(hidden_width, hidden_width), nn.LeakyReLU(0.2),
            nn.Linear(hidden_width, feature_dim)
        )
    def forward(self, x): return self.net(x)

if __name__ == "__main__":
    m, n, x_dim, feature_dim = 200, 200, 20, 10
    kernel = MMD_Kernel(sigma=10.0)
    widths = [10, 50, 100, 200, 400, 600, 800, 1000]
    all_raw_records = []

    for seed in range(NUM_REPEATS):
        set_seed(42 + seed)
        X, Y = torch.randn(m, x_dim).to(device), torch.randn(n, x_dim).to(device)
        print(f"Running Seed {seed+1}/{NUM_REPEATS}")
        
        for width in widths:
            D = DeepDiscriminator(x_dim, feature_dim, width).to(device)
            optimizer = torch.optim.Adam(D.parameters(), lr=0.005, weight_decay=0.01)
            for _ in range(150):
                optimizer.zero_grad()
                loss = -mmd_loss_func(D(X), D(Y), kernel)
                loss.backward()
                optimizer.step()
                
            with torch.no_grad():
                f_x, f_y = D(X), D(Y)
                max_mmd = abs(mmd_loss_func(f_x, f_y, kernel).item())
                comp = compute_joint_complexity(torch.cat([f_x, f_y], dim=0)) * (m + n)
                all_raw_records.append({'Seed': seed, 'Width': width, 'MaxMMD': max_mmd, 'Comp': comp})

    pd.DataFrame(all_raw_records).to_csv(FILE_NAME, index=False)
    print(f"Simulation data stored in {FILE_NAME}")