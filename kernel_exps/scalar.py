#!/usr/bin/env python3

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use('Agg')


##############################################################################
# 1. DATASET (Scalar => input=2)
##############################################################################
class GaussianBumpDatasetScalar(Dataset):
    def __init__(
        self,
        num_samples=1000,
        grid_size=28,
        sigma=1.0,
        holdout_center=True,
        only_holdout=False,
        heldout_range=[6, 22]
    ):
        super().__init__()
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.sigma = sigma
        self.holdout_center = holdout_center
        self.only_holdout = only_holdout
        self.lb, self.ub = heldout_range

        xv, yv = np.meshgrid(np.arange(grid_size), np.arange(grid_size))

        self.data = []
        self.targets = []

        if self.only_holdout:
            x = np.random.randint(self.lb, self.ub+1, size=self.num_samples)
            y = np.random.randint(self.lb, self.ub+1, size=self.num_samples)
        elif self.holdout_center:
            x_list, y_list = [], []
            while len(x_list) < self.num_samples:
                batch_size = min(self.num_samples*2, 10000)
                x_b = np.random.randint(0, self.grid_size, size=batch_size)
                y_b = np.random.randint(0, self.grid_size, size=batch_size)
                valid = ~(
                    (x_b >= self.lb) & (x_b <= self.ub) &
                    (y_b >= self.lb) & (y_b <= self.ub)
                )
                x_list.extend(x_b[valid])
                y_list.extend(y_b[valid])
            x = np.array(x_list[:self.num_samples])
            y = np.array(y_list[:self.num_samples])
        else:
            x = np.random.randint(0, self.grid_size, size=self.num_samples)
            y = np.random.randint(0, self.grid_size, size=self.num_samples)

        for i in range(self.num_samples):
            # build 28x28
            xx, yy = x[i], y[i]
            big_2d = 1.0 - np.exp(
                -(((xv - xx)**2 + (yv - yy)**2)/(2*self.sigma*self.sigma))
            )
            self.data.append(np.array([xx, yy], dtype=np.float32))
            self.targets.append(big_2d[None, :, :].astype(np.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx],    dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )

##############################################################################
# 2. CNN MODEL
##############################################################################


class CNNDecoderScalar(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, n_hidden_layers=4):
        super().__init__()
        layers = []
        layers.append(nn.ConvTranspose2d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=7, stride=1, padding=0
        ))
        layers.append(nn.ReLU())
        current_size = 7
        upsample_needed = max(0, int(np.log2(28/current_size)))

        for _ in range(n_hidden_layers):
            if current_size < 28 and upsample_needed > 0:
                layers.append(nn.ConvTranspose2d(
                    hidden_size, hidden_size, kernel_size=4, stride=2, padding=1
                ))
                current_size *= 2
                upsample_needed -= 1
            else:
                layers.append(nn.ConvTranspose2d(
                    hidden_size, hidden_size, kernel_size=3, stride=1, padding=1
                ))
            layers.append(nn.ReLU())

        layers.append(nn.ConvTranspose2d(
            hidden_size, 1, kernel_size=3, stride=1, padding=1
        ))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)  # => (B,2,1,1)
        return self.decoder(x)            # => (B,1,28,28)


##############################################################################
# 3. TRAIN
##############################################################################
def train_model(model, loader, epochs=15, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        run_loss = 0.0
        for x_2, y_2d in loader:
            x_2, y_2d = x_2.to(device), y_2d.to(device)
            opt.zero_grad()
            out_2d = model(x_2)
            loss = crit(out_2d, y_2d)
            loss.backward()
            opt.step()
            run_loss += loss.item()
        avg_loss = run_loss / len(loader)
        print(f"[Epoch {ep+1}/{epochs}] Loss={avg_loss:.4f}")
    return model

##############################################################################
# 4. Pairwise Matrices: Cosine, L2, & Pixel-Overlap
##############################################################################


def compute_cosine_matrix(feats):
    N = feats.shape[0]
    cos_mat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        fi = feats[i]
        norm_i = np.linalg.norm(fi)+1e-12
        for j in range(N):
            fj = feats[j]
            dotval = fi@fj
            norm_j = np.linalg.norm(fj)+1e-12
            cos_mat[i, j] = dotval/(norm_i*norm_j)
    return cos_mat


def compute_l2_matrix(feats):
    N = feats.shape[0]
    dist_mat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        fi = feats[i]
        for j in range(N):
            diff = fi - feats[j]
            dist_mat[i, j] = np.sqrt(np.sum(diff*diff))
    return dist_mat


def compute_overlap_matrix(feats, threshold=0.3):
    N = feats.shape[0]
    overlap_mat = np.zeros((N, N), dtype=np.float32)
    bin_masks = (feats < threshold).astype(np.float32)
    for i in range(N):
        m_i = bin_masks[i]
        for j in range(N):
            m_j = bin_masks[j]
            both = m_i*m_j
            overlap_count = np.sum(both)
            overlap_mat[i, j] = overlap_count / m_i.shape[0]  # /784
    return overlap_mat


def reorder_and_plot_mats(cos_mat, dist_mat, overlap_mat,
                          coords, lb=6, ub=22, out_path="pairwise_mat.pdf"):
    is_ood = []
    for (xx, yy) in coords:
        inside = (xx >= lb and xx <= ub and yy >= lb and yy <= ub)
        is_ood.append(inside)
    is_ood = np.array(is_ood)
    idx_id = np.where(~is_ood)[0]
    idx_ood = np.where(is_ood)[0]

    new_order = np.concatenate([idx_id, idx_ood], axis=0)
    reorder_cos = cos_mat[new_order][:, new_order]
    reorder_dist = dist_mat[new_order][:, new_order]
    reorder_overlap = overlap_mat[new_order][:, new_order]
    boundary = len(idx_id)
    N = cos_mat.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Cos
    im0 = axes[0].imshow(reorder_cos, origin='lower', cmap='viridis')
    axes[0].set_title("Cosine similarity")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    rectA = patches.Rectangle((0, 0), boundary, boundary,
                              fill=False, edgecolor='red', linestyle='dashed', linewidth=2)
    rectB = patches.Rectangle((boundary, boundary), N-boundary, N-boundary,
                              fill=False, edgecolor='red', linestyle='dashed', linewidth=2)
    axes[0].add_patch(rectA)
    axes[0].add_patch(rectB)
    axes[0].text(boundary/4, boundary/2, "ID block",
                 color='white', fontsize=10)
    axes[0].text(boundary+(N-boundary)/4, boundary+(N-boundary)/2,
                 "OOD block", color='white', fontsize=10)

    # Dist
    im1 = axes[1].imshow(reorder_dist, origin='lower', cmap='magma')
    axes[1].set_title("L2 distance")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    rectC = patches.Rectangle((0, 0), boundary, boundary,
                              fill=False, edgecolor='yellow', linestyle='dashed', linewidth=2)
    rectD = patches.Rectangle((boundary, boundary), N-boundary, N-boundary,
                              fill=False, edgecolor='yellow', linestyle='dashed', linewidth=2)
    axes[1].add_patch(rectC)
    axes[1].add_patch(rectD)
    axes[1].text(boundary/4, boundary/2, "ID block",
                 color='black', fontsize=10)
    axes[1].text(boundary+(N-boundary)/4, boundary+(N-boundary)/2,
                 "OOD block", color='black', fontsize=10)

    # Overlap
    im2 = axes[2].imshow(reorder_overlap, origin='lower', cmap='cividis')
    axes[2].set_title("Pixel Overlap")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    rectE = patches.Rectangle((0, 0), boundary, boundary,
                              fill=False, edgecolor='white', linestyle='dashed', linewidth=2)
    rectF = patches.Rectangle((boundary, boundary), N-boundary, N-boundary,
                              fill=False, edgecolor='white', linestyle='dashed', linewidth=2)
    axes[2].add_patch(rectE)
    axes[2].add_patch(rectF)
    axes[2].text(boundary/4, boundary/2, "ID block",
                 color='white', fontsize=10)
    axes[2].text(boundary+(N-boundary)/4, boundary+(N-boundary)/2,
                 "OOD block", color='white', fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[reorder_and_plot_mats] => saved {out_path}")

##############################################################################
# 5. OOD 16x16 SUBPLOTS
##############################################################################


def visualize_ood_region(model, out_dir, lb=6, ub=22):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    fig, axes = plt.subplots(16, 16, figsize=(16, 16),
                             sharex=True, sharey=True)

    with torch.no_grad():
        for row_idx, y_ in enumerate(range(lb, ub)):
            for col_idx, x_ in enumerate(range(lb, ub)):
                inp_2 = torch.tensor(
                    [x_, y_], dtype=torch.float32, device=device).unsqueeze(0)
                out_2d = model(inp_2)  # => shape(1,1,28,28)
                out_np = out_2d.cpu().numpy()[0, 0]

                ax = axes[row_idx, col_idx]
                ax.imshow(out_np, cmap='gray', origin='lower')
                ax.set_xticks([])
                ax.set_yticks([])

                rect = patches.Rectangle((6, 6), 16, 16,
                                         fill=False, edgecolor='red', linestyle='dashed', linewidth=1)
                ax.add_patch(rect)
                ax.plot([x_], [y_], marker='x', color='red', markersize=2)

    out_fig = os.path.join(out_dir, "ood_16x16.pdf")
    plt.savefig(out_fig, bbox_inches='tight')
    plt.close()
    print(f"[visualize_ood_region] => saved {out_fig}")


##############################################################################
# 6. MAIN
##############################################################################
def main():
    result_dir = "result_scalar"
    os.makedirs(result_dir, exist_ok=True)

    # (A) ID dataset => skip [6..22]
    train_ds = GaussianBumpDatasetScalar(
        num_samples=2000,
        grid_size=28,
        sigma=1.0,
        holdout_center=True,
        only_holdout=False,
        heldout_range=[6, 22]
    )
    train_loader = DataLoader(train_ds, batch_size=32,
                              shuffle=True, num_workers=2)

    # (B) OOD dataset => only [6..22]
    ood_ds = GaussianBumpDatasetScalar(
        num_samples=500,
        grid_size=28,
        sigma=1.0,
        holdout_center=False,
        only_holdout=True,
        heldout_range=[6, 22]
    )
    ood_loader = DataLoader(ood_ds, batch_size=32,
                            shuffle=False, num_workers=2)

    # (C) Model => train => measure OOD
    model = CNNDecoderScalar(input_size=2, hidden_size=64, n_hidden_layers=4)
    train_model(model, train_loader, epochs=15, lr=1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    crit = nn.MSELoss()
    tot_loss = 0.0
    count = 0
    for (x_2, y_2d) in ood_loader:
        x_2, y_2d = x_2.to(device), y_2d.to(device)
        with torch.no_grad():
            pred_2d = model(x_2)
            loss_val = crit(pred_2d, y_2d).item()
        tot_loss += loss_val*x_2.size(0)
        count += x_2.size(0)
    ood_mse = tot_loss/count
    print(f"[OOD MSE] => {ood_mse:.4f}")

    # (D) Build all 784 => feats => cos, dist, overlap => reorder => plot
    coords_1d = np.arange(28)
    xv, yv = np.meshgrid(coords_1d, coords_1d)
    coords = np.stack([xv.ravel(), yv.ravel()], axis=-1).astype(np.float32)
    feats_all = []

    with torch.no_grad():
        bs = 64
        outs_list = []
        for start_idx in range(0, coords.shape[0], bs):
            end_idx = min(start_idx+bs, coords.shape[0])
            batch_coords = coords[start_idx:end_idx]
            inp_t = torch.tensor(
                batch_coords, dtype=torch.float32, device=device)
            out_2d = model(inp_t)  # => shape(b,1,28,28)
            out_flat = out_2d.view(out_2d.size(0), -1).cpu().numpy()
            outs_list.append(out_flat)
    feats = np.concatenate(outs_list, axis=0)  # shape(784,784)

    cos_mat = compute_cosine_matrix(feats)
    dist_mat = compute_l2_matrix(feats)
    overlap_mat = compute_overlap_matrix(feats, threshold=0.3)

    pairwise_out = os.path.join(result_dir, "pairwise_mat.pdf")
    reorder_and_plot_mats(cos_mat, dist_mat, overlap_mat,
                          coords, lb=6, ub=22, out_path=pairwise_out)

    # (E) 16x16 subplots for OOD
    visualize_ood_region(model, result_dir, lb=6, ub=22)

    print("Done. All results in =>", result_dir)


if __name__ == "__main__":
    main()
