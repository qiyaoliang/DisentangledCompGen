#!/usr/bin/env python3

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import math

##############################################################################
# 1. DATASET: Gaussian-Encoded => input dim=56 => [gx(28), gy(28)]
##############################################################################


class GaussianBumpDataset_Encoded(Dataset):
    """
    56-d input => [gaussian_x(28), gaussian_y(28)].
    Output => (1,28,28) bump at (x,y).
    We skip or only hold out region [lb..ub]^2.
    """

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

        # (A) Decide how to sample x,y
        if self.only_holdout:
            # Only sample from [lb..ub]^2
            xs = np.random.randint(self.lb, self.ub + 1, size=self.num_samples)
            ys = np.random.randint(self.lb, self.ub + 1, size=self.num_samples)
        elif self.holdout_center:
            # Exclude [lb..ub]^2
            xs, ys = [], []
            while len(xs) < self.num_samples:
                batch_size = min(self.num_samples * 2, 10000)
                x_b = np.random.randint(0, grid_size, size=batch_size)
                y_b = np.random.randint(0, grid_size, size=batch_size)
                valid = ~(
                    (x_b >= self.lb) & (x_b <= self.ub) &
                    (y_b >= self.lb) & (y_b <= self.ub)
                )
                xs.extend(x_b[valid])
                ys.extend(y_b[valid])
            xs = np.array(xs[: self.num_samples])
            ys = np.array(ys[: self.num_samples])
        else:
            # Sample anywhere in [0..grid_size-1]
            xs = np.random.randint(0, grid_size, size=self.num_samples)
            ys = np.random.randint(0, grid_size, size=self.num_samples)

        # (B) Build the 56-d input + 28×28 GT
        for i in range(self.num_samples):
            x_i, y_i = xs[i], ys[i]

            gx = np.exp(-((np.arange(grid_size) - x_i) ** 2) /
                        (2 * sigma * sigma))
            gy = np.exp(-((np.arange(grid_size) - y_i) ** 2) /
                        (2 * sigma * sigma))
            bump_56 = np.concatenate([gx, gy], axis=0).astype(np.float32)

            big_2d = 1.0 - np.exp(
                -(((xv - x_i) ** 2 + (yv - y_i) ** 2) / (2 * sigma * sigma))
            )
            self.data.append(bump_56)  # shape (56,)
            self.targets.append(big_2d[None, :, :].astype(
                np.float32))  # shape (1,28,28)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )

##############################################################################
# 2. CNN MODEL
##############################################################################


class CNNDecoderGaussian(nn.Module):
    """
    Input => shape(56,1,1).
    We'll do transposed conv => final => (1,28,28).
    """

    def __init__(self, input_size=56, hidden_size=64, n_hidden_layers=4):
        super().__init__()
        layers = []
        # from (56,1,1)=>(hidden_size,7,7)
        layers.append(nn.ConvTranspose2d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=7,
            stride=1,
            padding=0
        ))
        layers.append(nn.ReLU())
        current_size = 7

        upsample_needed = max(0, int(np.log2(28 / current_size)))

        for _ in range(n_hidden_layers):
            if current_size < 28 and upsample_needed > 0:
                layers.append(nn.ConvTranspose2d(
                    hidden_size, hidden_size,
                    kernel_size=4, stride=2, padding=1
                ))
                current_size *= 2
                upsample_needed -= 1
            else:
                layers.append(nn.ConvTranspose2d(
                    hidden_size, hidden_size,
                    kernel_size=3, stride=1, padding=1
                ))
            layers.append(nn.ReLU())

        layers.append(nn.ConvTranspose2d(
            hidden_size, 1, kernel_size=3, stride=1, padding=1
        ))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        # x => shape(B,56) => (B,56,1,1)
        x = x.unsqueeze(-1).unsqueeze(-1)
        return self.decoder(x)

##############################################################################
# 3. TRAIN
##############################################################################


def train_model(model, loader, epochs=15, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        run_loss = 0.0
        for x_56, y_2d in loader:
            x_56, y_2d = x_56.to(device), y_2d.to(device)
            opt.zero_grad()
            out_2d = model(x_56)  # => shape(B,1,28,28)
            loss = crit(out_2d, y_2d)
            loss.backward()
            opt.step()
            run_loss += loss.item()
        avg_loss = run_loss / len(loader)
        print(f"[Epoch {ep+1}/{epochs}] Loss={avg_loss:.4f}")
    return model

##############################################################################
# 4. FACTORIZED vs 2D RADIAL KERNEL
##############################################################################


def factorized_kernel_matrix(coordsA, coordsB, lx, ly):
    """
    coordsA, coordsB => shape(N,2). factorized => exp(-(dx^2)/(2lx^2)) * exp(-(dy^2)/(2ly^2)).
    """
    Na = coordsA.shape[0]
    Nb = coordsB.shape[0]
    K = np.zeros((Na, Nb), dtype=np.float32)
    for i in range(Na):
        x_i, y_i = coordsA[i]
        for j in range(Nb):
            x_j, y_j = coordsB[j]
            dx = x_i - x_j
            dy = y_i - y_j
            valx = math.exp(-(dx * dx) / (2 * lx * lx))
            valy = math.exp(-(dy * dy) / (2 * ly * ly))
            K[i, j] = valx * valy
    return K


def radial_kernel_matrix(coordsA, coordsB, l_):
    """
    2D radial => exp(-(r^2)/(2 l_^2)).
    """
    Na = coordsA.shape[0]
    Nb = coordsB.shape[0]
    K = np.zeros((Na, Nb), dtype=np.float32)
    for i in range(Na):
        x_i, y_i = coordsA[i]
        for j in range(Nb):
            x_j, y_j = coordsB[j]
            rr = (x_i - x_j)**2 + (y_i - y_j)**2
            K[i, j] = math.exp(-rr / (2 * l_ * l_))
    return K


def fit_kernel_and_eval(coords_train, feats_train,
                        coords_probe, feats_probe,
                        is_factorized=True):
    """
    grid search over length scales => solve alpha => measure MSE => separate ID vs OOD
    """
    best_err = float('inf')
    best_params = None
    best_alpha = None
    # small grid
    list_l = [0.5, 1.0, 2.0, 4.0]

    def build_kernel(A, B, p):
        if is_factorized:
            lx, ly = p
            return factorized_kernel_matrix(A, B, lx, ly)
        else:
            (l_,) = p
            return radial_kernel_matrix(A, B, l_)

    for lx in list_l:
        if is_factorized:
            for ly in list_l:
                param = (lx, ly)
                Ktrain = build_kernel(coords_train, coords_train, param)
                alpha, _, _, _ = np.linalg.lstsq(
                    Ktrain, feats_train, rcond=None)
                Kprobe = build_kernel(coords_probe, coords_train, param)
                recon = Kprobe @ alpha
                error = np.mean((feats_probe - recon)**2)
                if error < best_err:
                    best_err = error
                    best_params = param
                    best_alpha = alpha
        else:
            param = (lx,)
            Ktrain = build_kernel(coords_train, coords_train, param)
            alpha, _, _, _ = np.linalg.lstsq(Ktrain, feats_train, rcond=None)
            Kprobe = build_kernel(coords_probe, coords_train, param)
            recon = Kprobe @ alpha
            error = np.mean((feats_probe - recon)**2)
            if error < best_err:
                best_err = error
                best_params = param
                best_alpha = alpha

    # measure ID vs OOD
    lb, ub = 6, 22
    is_ood = []
    for (xx, yy) in coords_probe:
        inside = (xx >= lb and xx <= ub and yy >= lb and yy <= ub)
        is_ood.append(inside)
    is_ood = np.array(is_ood)
    # final reconstruction
    Kprobe_final = build_kernel(coords_probe, coords_train, best_params)
    recon_final = Kprobe_final @ best_alpha
    all_err = np.mean((feats_probe - recon_final)**2, axis=1)
    id_err = np.mean(all_err[~is_ood])
    ood_err = np.mean(all_err[is_ood])

    return best_err, best_params, best_alpha, id_err, ood_err, recon_final

##############################################################################
# 5. HELPER: build 56-d bump from (x,y)
##############################################################################


def build_encoded_bump(x_int, y_int, grid_size=28, sigma=1.0):
    gx = np.exp(-((np.arange(grid_size) - x_int)**2) / (2*sigma*sigma))
    gy = np.exp(-((np.arange(grid_size) - y_int)**2) / (2*sigma*sigma))
    return np.concatenate([gx, gy], axis=0).astype(np.float32)

##############################################################################
# 6. MAIN
##############################################################################


def main():
    out_dir = "kernel_gaussian"
    os.makedirs(out_dir, exist_ok=True)

    # (A) Build dataset => skip region [6..22]
    train_ds = GaussianBumpDataset_Encoded(
        num_samples=1500,
        grid_size=28,
        sigma=1.0,
        holdout_center=True,
        only_holdout=False,
        heldout_range=[6, 22]
    )
    train_loader = DataLoader(train_ds, batch_size=32,
                              shuffle=True, num_workers=2)

    # (B) Build & train model
    model = CNNDecoderGaussian(
        input_size=56, hidden_size=64, n_hidden_layers=4)
    model = train_model(model, train_loader, epochs=15, lr=1e-3)

    # (C) Gather train coords & feats
    def recover_xy(vec_56):
        half = 28
        gx = vec_56[:half]
        gy = vec_56[half:]
        x_ = np.argmax(gx)
        y_ = np.argmax(gy)
        return (x_, y_)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    all_train_coords = []
    all_train_feats = []

    with torch.no_grad():
        for (inp_56, tar_2d) in train_loader:
            inp_56 = inp_56.to(device)
            outs = model(inp_56)  # shape(b,1,28,28)
            out_flat = outs.view(
                outs.size(0), -1).cpu().numpy()  # shape(b,784)
            in_cpu = inp_56.cpu().numpy()                       # shape(b,56)

            for b in range(in_cpu.shape[0]):
                x_, y_ = recover_xy(in_cpu[b])
                all_train_coords.append((x_, y_))
                all_train_feats.append(out_flat[b])

    all_train_coords = np.array(all_train_coords, dtype=np.float32)
    all_train_feats = np.array(
        all_train_feats,  dtype=np.float32)  # shape(N,784)

    # (D) Build a full probe coords => [0..27]^2 => 784
    coords_1d = np.arange(28)
    xv, yv = np.meshgrid(coords_1d, coords_1d)
    coords_probe = np.stack([xv.ravel(), yv.ravel()],
                            axis=-1).astype(np.float32)  # shape(784,2)

    # Evaluate model => feats_probe => shape(784,784)
    # (IMPORTANT) Build 56-d input for each (x,y)
    bump_56_all = []
    for i in range(coords_probe.shape[0]):
        xx, yy = coords_probe[i]
        bump_56 = build_encoded_bump(int(xx), int(yy), grid_size=28, sigma=1.0)
        bump_56_all.append(bump_56)
    bump_56_all = np.array(bump_56_all, dtype=np.float32)  # shape(784,56)

    feats_list = []
    bs = 64
    with torch.no_grad():
        for start_idx in range(0, bump_56_all.shape[0], bs):
            end_idx = min(start_idx+bs, bump_56_all.shape[0])
            batch_56 = torch.tensor(
                bump_56_all[start_idx:end_idx], device=device)
            outs_2d = model(batch_56)  # => shape(?,1,28,28)
            out_flat = outs_2d.view(outs_2d.size(0), -1).cpu().numpy()
            feats_list.append(out_flat)

    feats_probe = np.concatenate(feats_list, axis=0)  # shape(784,784)

    # (E) Fit factorized vs. radial
    fact_err, fact_params, fact_alpha, fact_id, fact_ood, fact_recon = fit_kernel_and_eval(
        all_train_coords, all_train_feats,
        coords_probe, feats_probe,
        is_factorized=True
    )
    rad_err, rad_params, rad_alpha, rad_id, rad_ood, rad_recon = fit_kernel_and_eval(
        all_train_coords, all_train_feats,
        coords_probe, feats_probe,
        is_factorized=False
    )

    print(
        f"[Factorized-Kernel] total={fact_err:.4f}, ID={fact_id:.4f}, OOD={fact_ood:.4f}, param={fact_params}")
    print(
        f"[Radial-Kernel]    total={rad_err:.4f},  ID={rad_id:.4f},  OOD={rad_ood:.4f}, param={rad_params}")

    # (F) 2D bar plot
    bars_labels = ['total', 'ID', 'OOD']
    fact_vals = [fact_err, fact_id, fact_ood]
    rad_vals = [rad_err,  rad_id,  rad_ood]
    x_positions = np.array([0, 1, 2])
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x_positions - width/2, fact_vals, width=width,
            color='blue',   label='Factorized kernel')
    plt.bar(x_positions + width/2, rad_vals,  width=width,
            color='orange', label='2D radial kernel')
    plt.xticks(x_positions, bars_labels)
    plt.ylabel("Mean Sq. Error (kernel vs. net output)")
    plt.title("Kernel vs. Net Output Reconstruction Error (Gaussian-Encoded)")
    plt.legend()
    out_bar = os.path.join(out_dir, "kernel_bar_plot.pdf")
    plt.savefig(out_bar, bbox_inches='tight')
    plt.close()

    # (G) 2D Heatmap of OOD region error => shape(16,16)
    def build_ood_heatmap(coords, net_fe, ker_fe, lb=6, ub=22):
        """
        net_fe, ker_fe => shape(784,784)
        We'll measure sum((net - ker)^2) over the 784 pixels => shape(784,) => reshape(28,28).
        Then slice [lb..ub-1] => 16x16 region.
        """
        all_err = np.sum((net_fe - ker_fe)**2, axis=1)  # shape(784,)
        err_2d = all_err.reshape(28, 28)
        slice_ = err_2d[lb:ub, lb:ub]  # shape(16,16)
        return slice_

    fact_oodmap = build_ood_heatmap(
        coords_probe, feats_probe, fact_recon, lb=6, ub=22)
    rad_oodmap = build_ood_heatmap(
        coords_probe, feats_probe, rad_recon,  lb=6, ub=22)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("OOD region error (Factorized)")
    plt.imshow(fact_oodmap, origin='lower', cmap='magma')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title("OOD region error (Radial)")
    plt.imshow(rad_oodmap, origin='lower', cmap='magma')
    plt.colorbar()
    out_oodmap = os.path.join(out_dir, "kernel_ood_heatmap.pdf")
    plt.savefig(out_oodmap, bbox_inches='tight')
    plt.close()

    # (H) Compare a few OOD coords
    lb, ub = 6, 22
    chosen = []
    for _ in range(4):
        xx = np.random.randint(lb, ub)
        yy = np.random.randint(lb, ub)
        chosen.append((xx, yy))

    def idx_of(x_, y_):
        # consistent with coords_probe ordering => index = y*28 + x
        return int(y_*28 + x_)

    net_imgs = []
    fact_imgs = []
    rad_imgs = []
    for (xx, yy) in chosen:
        i = idx_of(xx, yy)
        net_img_1d = feats_probe[i]      # shape(784,)
        net_img_2d = net_img_1d.reshape(28, 28)
        fact_img_1d = fact_recon[i]       # shape(784,)
        fact_img_2d = fact_img_1d.reshape(28, 28)
        rad_img_1d = rad_recon[i]
        rad_img_2d = rad_img_1d.reshape(28, 28)

        net_imgs.append(net_img_2d)
        fact_imgs.append(fact_img_2d)
        rad_imgs.append(rad_img_2d)

    plt.figure(figsize=(9, 3*len(chosen)))
    for idx, (xx, yy) in enumerate(chosen):
        net_2d = net_imgs[idx]
        fact_2d = fact_imgs[idx]
        rad_2d = rad_imgs[idx]

        # row => 3 columns => net, factorized, radial
        plt.subplot(len(chosen), 3, 3*idx + 1)
        plt.imshow(net_2d, origin='lower', cmap='gray')
        plt.title(f"Net Output (x={xx}, y={yy})")
        plt.axis('off')

        plt.subplot(len(chosen), 3, 3*idx + 2)
        plt.imshow(fact_2d, origin='lower', cmap='gray')
        plt.title("Factorized Kernel Recon")
        plt.axis('off')

        plt.subplot(len(chosen), 3, 3*idx + 3)
        plt.imshow(rad_2d, origin='lower', cmap='gray')
        plt.title("Radial Kernel Recon")
        plt.axis('off')

    out_cmp = os.path.join(out_dir, "kernel_ood_imagecmp.pdf")
    plt.savefig(out_cmp, bbox_inches='tight')
    plt.close()
    print(f"Saved final image comparisons => {out_cmp}")

    print("Done. All results in =>", out_dir)


if __name__ == "__main__":
    main()
