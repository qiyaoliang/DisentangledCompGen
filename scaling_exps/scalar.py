#!/usr/bin/env python3
# scaling_ood_area_subsampling_deep_equierror_scalar.py

import os
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import matplotlib.patches as patches

##############################################################################
# 1. DATASET: ScalarInputDatasetHoldout
##############################################################################


class ScalarInputDatasetHoldout(Dataset):
    """
    NxN 2D 'Gaussian bump' images, but the INPUT is just 2D scalar (x,y).
    We skip or only sample a K×K subregion (center or corner).
      - If holdout_center=True & only_holdout=False => ID => skip region
      - If only_holdout=True => OOD => only that region

    For each sample:
      - input => (2,) = (x, y) in [0..N)
      - target => (1, N, N) = a Gaussian bump at (x,y).
    """

    def __init__(
        self,
        num_samples=1000,
        N=28,
        sigma=1.0,
        K=10,
        holdout_mode='center',   # 'center' or 'corner'
        holdout_center=True,
        only_holdout=False
    ):
        super().__init__()
        self.num_samples = num_samples
        self.N = N
        self.sigma = sigma
        self.K = K
        self.holdout_mode = holdout_mode
        self.holdout_center = holdout_center
        self.only_holdout = only_holdout

        # bounding box in continuous space
        if holdout_mode == 'center':
            mid = N / 2.0
            self.lb = mid - (K / 2.0)
            self.ub = self.lb + K
        elif holdout_mode == 'corner':
            self.lb = N - K
            self.ub = N
        else:
            raise ValueError("holdout_mode must be 'center' or 'corner'.")

        # We'll build the NxN mesh for generating the target images
        self.xv, self.yv = np.meshgrid(np.arange(N), np.arange(N))

        self.inputs = []
        self.targets = []
        self._generate_samples()

    def _in_holdout(self, x, y):
        """
        Check if real-valued x,y is inside the [lb..ub) bounding box in both dims.
        """
        return (x >= self.lb) and (x < self.ub) and (y >= self.lb) and (y < self.ub)

    def _generate_samples(self):
        xs, ys = [], []
        attempts = 0
        max_attempts = self.num_samples * 20

        while len(xs) < self.num_samples and attempts < max_attempts:
            x_c = np.random.uniform(0, self.N)
            y_c = np.random.uniform(0, self.N)
            inside = self._in_holdout(x_c, y_c)

            # OOD => only inside
            if self.only_holdout and not self.holdout_center:
                if inside:
                    xs.append(x_c)
                    ys.append(y_c)
            # ID => skip region
            elif self.holdout_center and not self.only_holdout:
                if not inside:
                    xs.append(x_c)
                    ys.append(y_c)
            else:
                # fallback => sample anywhere
                xs.append(x_c)
                ys.append(y_c)
            attempts += 1

        xs = xs[:self.num_samples]
        ys = ys[:self.num_samples]

        for i in range(len(xs)):
            x_i, y_i = xs[i], ys[i]
            # input => shape=(2,)
            in_xy = np.array([x_i, y_i], dtype=np.float32)

            # target => NxN bump
            bump_2d = 1.0 - np.exp(
                -(((self.xv - x_i)**2 + (self.yv - y_i)**2) / (2*self.sigma**2))
            )
            bump_2d = bump_2d.astype(np.float32)[None, ...]  # shape=(1,N,N)

            self.inputs.append(in_xy)
            self.targets.append(bump_2d)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x_in = torch.tensor(
            self.inputs[idx],  dtype=torch.float32)   # shape=(2,)
        y_out = torch.tensor(
            self.targets[idx], dtype=torch.float32)   # shape=(1,N,N)
        return (x_in, y_out)


##############################################################################
# 2. DEEPER CNN with final interpolation
##############################################################################

class CNNDecoderWithActivations(nn.Module):
    """
    Deeper CNN-based decoder that:
      - Takes (B, 2) scalar inputs => (B,2,1,1).
      - Expands step by step with transposed conv.
      - Forces final => (B,1,N,N).
      - Stores post-ReLU activations in self.activations.
    """

    def __init__(self, input_size=2, base_hidden_size=128, extra_layers=4, N=28):
        super().__init__()
        self.N = N
        self.activations = []

        layers = []
        # 1) Expand from (B,2,1,1) => (B, base_hidden_size,7,7)
        layers.append(nn.ConvTranspose2d(
            in_channels=input_size,
            out_channels=base_hidden_size,
            kernel_size=7,
            stride=1,
            padding=0
        ))
        layers.append(nn.ReLU())
        current_size = 7
        hidden_size = base_hidden_size

        # 2) extra_layers expansions
        for _ in range(extra_layers):
            if current_size < N:
                layers.append(nn.ConvTranspose2d(
                    hidden_size, hidden_size, kernel_size=4, stride=2, padding=1
                ))
                current_size *= 2
            else:
                layers.append(nn.ConvTranspose2d(
                    hidden_size, hidden_size, kernel_size=3, stride=1, padding=1
                ))
            layers.append(nn.ReLU())

        # final => (B,1,??,??)
        layers.append(nn.ConvTranspose2d(
            hidden_size, 1, kernel_size=3, stride=1, padding=1
        ))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        # x => shape (B,2)
        self.activations = []
        x = x.unsqueeze(-1).unsqueeze(-1)  # => (B,2,1,1)
        for layer in self.decoder:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                self.activations.append(x.detach().clone())
        # force final => (B,1,N,N)
        x = F.interpolate(x, size=(self.N, self.N), mode='nearest')
        return x

##############################################################################
# 3. TRAIN & EVAL
##############################################################################


def train_cnn(model, loader, num_epochs=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            opt.zero_grad()
            preds = model(x_batch)
            loss = crit(preds, y_batch)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss/len(loader)
        print(f"Epoch {epoch+1}/{num_epochs}, TrainLoss={avg_loss:.4f}")


def eval_ood_mean_std(model, loader):
    """
    Return (mean_mse, std_mse) across OOD samples by computing per-sample MSE.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    crit = nn.MSELoss(reduction='none')
    all_errors = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            # shape => (B,1,N,N)
            loss_map = crit(preds, y_batch)   # => (B,1,N,N)
            sample_loss = loss_map.view(x_batch.size(0), -1).mean(dim=1)
            all_errors.extend(sample_loss.cpu().numpy())
    all_errors = np.array(all_errors)
    return np.mean(all_errors), np.std(all_errors)


def sample_4_ood_images(model, loader, how_many=4):
    """
    Return up to 'how_many' (inputs, gts, preds) from the OOD loader
    for visualization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    x_out, y_out, p_out = [], [], []
    count = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            for i in range(x_batch.size(0)):
                if count < how_many:
                    x_out.append(x_batch[i].cpu().numpy())
                    y_out.append(y_batch[i].cpu().numpy())
                    p_out.append(preds[i].cpu().numpy())
                    count += 1
                else:
                    break
            if count >= how_many:
                break
    return x_out, y_out, p_out

##############################################################################
# 4. MAIN
##############################################################################


def run_experiment_area_subsampling_deep_equierror_scalar():
    out_dir = "figs_deeper_scalar"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    Ns = [24, 28, 32, 36, 40]
    ps = [0.2, 0.3, 0.4, 0.5, 0.6]
    holdout_modes = ["center"]  # or ["center","corner"]

    results_mean = {m: np.zeros((len(Ns), len(ps))) for m in holdout_modes}
    results_std = {m: np.zeros((len(Ns), len(ps))) for m in holdout_modes}

    c_id = 1000
    c_ood = 1000

    # We'll define a custom dataset that uses scalar (x,y) inputs
    # => "ScalarInputDatasetHoldout"
    class ScalarInputDatasetHoldout(Dataset):
        """
        NxN 2D 'Gaussian bump' images with (x,y) in [0..N).
        K×K holdout region => ID or OOD
        input => shape(2,)
        target => shape(1,N,N)
        """

        def __init__(self, num_samples=1000, N=28, sigma=1.0, K=10,
                     holdout_mode='center', holdout_center=True, only_holdout=False):
            super().__init__()
            self.num_samples = num_samples
            self.N = N
            self.sigma = sigma
            self.K = K
            self.holdout_mode = holdout_mode
            self.holdout_center = holdout_center
            self.only_holdout = only_holdout

            if holdout_mode == "center":
                mid = N/2.0
                self.lb = mid-(K/2.0)
                self.ub = self.lb+K
            else:
                self.lb = N-K
                self.ub = N

            self.xv, self.yv = np.meshgrid(np.arange(N), np.arange(N))

            self.inputs = []
            self.targets = []
            self._gen_samples()

        def _in_holdout(self, x, y):
            return (x >= self.lb) and (x < self.ub) and (y >= self.lb) and (y < self.ub)

        def _gen_samples(self):
            xs, ys = [], []
            attempts = 0
            max_attempts = self.num_samples*20
            while len(xs) < self.num_samples and attempts < max_attempts:
                x_c = np.random.uniform(0, self.N)
                y_c = np.random.uniform(0, self.N)
                inside = self._in_holdout(x_c, y_c)

                if self.only_holdout and not self.holdout_center:
                    if inside:
                        xs.append(x_c)
                        ys.append(y_c)
                elif self.holdout_center and not self.only_holdout:
                    if not inside:
                        xs.append(x_c)
                        ys.append(y_c)
                else:
                    xs.append(x_c)
                    ys.append(y_c)
                attempts += 1

            xs = xs[:self.num_samples]
            ys = ys[:self.num_samples]
            for i in range(len(xs)):
                x_i, y_i = xs[i], ys[i]
                in_xy = np.array([x_i, y_i], dtype=np.float32)
                # build NxN bump
                bump_2d = 1.0 - np.exp(
                    -(((self.xv - x_i)**2 + (self.yv - y_i)**2)/(2*self.sigma**2))
                )
                bump_2d = bump_2d.astype(np.float32)[None, ...]
                self.inputs.append(in_xy)
                self.targets.append(bump_2d)

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            x_in = torch.tensor(self.inputs[idx], dtype=torch.float32)
            y_out = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x_in, y_out

    for mode in holdout_modes:
        print(f"===== holdout_mode={mode} =====")
        for iN, N in enumerate(Ns):
            for ip, p in enumerate(ps):
                K = int(np.floor(p*N))
                if K < 1 or K >= N:
                    results_mean[mode][iN, ip] = np.nan
                    results_std[mode][iN, ip] = np.nan
                    continue

                train_num = int(c_id*(N-K)*(N-K)/(N*N))
                ood_num = int(c_ood*K*K/(N*N))

                print(f"--- Building dataset: N={N}, p={p:.2f}, K={K}, mode={mode}, "
                      f"train_num={train_num}, ood_num={ood_num}")

                # ID => skip K×K
                train_ds = ScalarInputDatasetHoldout(
                    num_samples=train_num,
                    N=N,
                    sigma=1.0,
                    K=K,
                    holdout_mode=mode,
                    holdout_center=True,
                    only_holdout=False
                )
                train_loader = DataLoader(
                    train_ds, batch_size=32, shuffle=True)

                # OOD => only K×K
                ood_ds = ScalarInputDatasetHoldout(
                    num_samples=ood_num,
                    N=N,
                    sigma=1.0,
                    K=K,
                    holdout_mode=mode,
                    holdout_center=False,
                    only_holdout=True
                )
                if len(ood_ds) == 0:
                    results_mean[mode][iN, ip] = np.nan
                    results_std[mode][iN, ip] = np.nan
                    continue
                ood_loader = DataLoader(ood_ds, batch_size=32, shuffle=False)

                print(f"   => ID size={len(train_ds)}, OOD size={len(ood_ds)}")
                if len(ood_ds) < 4:
                    print("   => too few OOD samples => skip\n")
                    results_mean[mode][iN, ip] = np.nan
                    results_std[mode][iN, ip] = np.nan
                    continue

                # Build deeper net => input_size=2
                net = CNNDecoderWithActivations(
                    input_size=2,
                    base_hidden_size=128,
                    extra_layers=8,  # or however many
                    N=N
                )

                # train
                train_cnn(net, train_loader, num_epochs=50, lr=1e-3)
                # evaluate => mean & std across OOD samples
                mean_err, std_err = eval_ood_mean_std(net, ood_loader)
                print(f"N={N}, p={p:.2f}, K={K}, mode={mode} => "
                      f"mean={mean_err:.4f}, std={std_err:.4f}\n")

                results_mean[mode][iN, ip] = mean_err
                results_std[mode][iN, ip] = std_err

                # Also sample 4 images for each model
                x_out, y_out, p_out = sample_4_ood_images(
                    net, ood_loader, how_many=4)
                fig, axes = plt.subplots(2, 4, figsize=(12, 6))
                axes = axes.flatten()

                # bounding box
                if mode == "center":
                    mid = N//2
                    lb = mid-(K//2)
                    ub = lb+K
                else:
                    lb = N-K
                    ub = N

                for iimg in range(len(x_out)):
                    gt_img = y_out[iimg][0]
                    pd_img = p_out[iimg][0]
                    axes[2*iimg].imshow(gt_img, cmap='gray', origin='lower')
                    axes[2*iimg].set_title(f"GT OOD (N={N}, p={p:.2f})")
                    axes[2*iimg].axis('off')
                    rect_gt = patches.Rectangle((lb, lb), K, K,
                                                fill=False, edgecolor='red', linestyle='dashed', linewidth=2)
                    axes[2*iimg].add_patch(rect_gt)

                    axes[2*iimg+1].imshow(pd_img, cmap='gray', origin='lower')
                    axes[2*iimg+1].set_title("Pred OOD")
                    axes[2*iimg+1].axis('off')
                    rect_pd = patches.Rectangle((lb, lb), K, K,
                                                fill=False, edgecolor='red', linestyle='dashed', linewidth=2)
                    axes[2*iimg+1].add_patch(rect_pd)

                fig.suptitle(
                    f"N={N}, p={p:.2f}, K={K}, mode={mode} => 4 OOD samples")
                out_fig_name = f"{out_dir}/ood_{mode}_N{N}_p{p:.2f}_samples.png"
                plt.savefig(out_fig_name, bbox_inches='tight')
                plt.close()
                print(f"Saved 4-sample figure => {out_fig_name}")

    # Save results
    data_out = {
        'Ns': Ns,
        'ps': ps,
        'results_mean': results_mean,
        'results_std':  results_std
    }
    out_pickle = os.path.join(
        out_dir, "results_dict_area_deep_equi_scalar.pkl")
    with open(out_pickle, "wb") as f:
        pickle.dump(data_out, f)
    print(f"Stored data in {out_pickle}")

    # produce contour with equi-error lines
    Ns_array = np.array(Ns)
    ps_array = np.array(ps)
    X, Y = np.meshgrid(Ns_array, ps_array, indexing='ij')
    for mode in holdout_modes:
        data_2d = results_mean[mode]
        valid_vals = data_2d[~np.isnan(data_2d)]
        if valid_vals.size == 0:
            print(f"No valid data for mode={mode}, skipping contour.")
            continue
        vmin, vmax = np.min(valid_vals), np.max(valid_vals)

        plt.figure(figsize=(7, 5))
        cf = plt.contourf(X, Y, data_2d, levels=20, cmap='viridis')
        cbar = plt.colorbar(cf)
        cbar.set_label("OOD MSE (mean)")

        # overlay equi-error lines
        line_levels = np.linspace(vmin, vmax, 5)
        cont_lines = plt.contour(X, Y, data_2d, levels=line_levels, colors='k')
        plt.clabel(cont_lines, inline=True, fontsize=8, fmt="%.4f")

        plt.xlabel("N (image size)")
        plt.ylabel("p (fraction of holdout, K=p*N)")
        plt.title(f"OOD MSE vs. (N,p), mode={mode}, deeper net + scalar input")

        out_fig_equi = f"{out_dir}/ood_mse_contour_{mode}_deep_equi_scalar.png"
        plt.savefig(out_fig_equi, bbox_inches='tight')
        plt.close()
        print(f"Saved equi-error contour => {out_fig_equi}")

    print("Done! All results are in 'figs_deeper_scalar/'.")


##############################################################################
# 5. MAIN
##############################################################################

if __name__ == "__main__":
    run_experiment_area_subsampling_deep_equierror_scalar()
