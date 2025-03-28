#!/usr/bin/env python3
# cnn_gaussian_experiment_multi_runs.py

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

##############################################################################
# 1. DATASET: 56-dim input, 28x28 output
##############################################################################


class GaussianBumpDatasetSquare(Dataset):
    """
    Each sample:
      - 56-dim input => [gaussian_x(28), gaussian_y(28)]
      - 28x28 target => 2D Gaussian bump at (x,y).

    We can hold out region [lb..ub]^2 from training or sample only that region
    for OOD testing.
    """

    def __init__(
        self,
        num_samples=1000,
        grid_size=28,
        sigma=1.0,
        holdout_center=True,
        only_holdout=False,
        heldout_range=[10, 18],
    ):
        super().__init__()
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.sigma = sigma
        self.holdout_center = holdout_center
        self.only_holdout = only_holdout
        self.lb, self.ub = heldout_range

        # For building 2D Gaussian outputs
        self.xv, self.yv = np.meshgrid(
            np.arange(grid_size), np.arange(grid_size)
        )

        self.data = []
        self.targets = []
        self._generate()

    def _generate(self):
        if self.only_holdout:
            # Only sample inside [lb..ub]^2
            xs = np.random.randint(self.lb, self.ub + 1, size=self.num_samples)
            ys = np.random.randint(self.lb, self.ub + 1, size=self.num_samples)
        elif self.holdout_center:
            # Exclude [lb..ub]^2
            xs, ys = [], []
            while len(xs) < self.num_samples:
                batch_size = min(self.num_samples * 2, 10000)
                x_b = np.random.randint(0, self.grid_size, size=batch_size)
                y_b = np.random.randint(0, self.grid_size, size=batch_size)
                valid = ~(
                    (x_b >= self.lb) & (x_b <= self.ub) &
                    (y_b >= self.lb) & (y_b <= self.ub)
                )
                xs.extend(x_b[valid])
                ys.extend(y_b[valid])
            xs = np.array(xs[: self.num_samples])
            ys = np.array(ys[: self.num_samples])
        else:
            # Sample anywhere in [0..grid_size-1]^2
            xs = np.random.randint(0, self.grid_size, size=self.num_samples)
            ys = np.random.randint(0, self.grid_size, size=self.num_samples)

        for i in range(self.num_samples):
            x_i, y_i = xs[i], ys[i]

            # 1D Gaussians => each 28 dim => total 56
            gx = np.exp(-((np.arange(self.grid_size) - x_i) ** 2) /
                        (2 * self.sigma ** 2))
            gy = np.exp(-((np.arange(self.grid_size) - y_i) ** 2) /
                        (2 * self.sigma ** 2))
            bump_56 = np.concatenate([gx, gy], axis=0)

            # 2D Gaussian => shape (28,28)
            gauss_2d = 1.0 - np.exp(
                -(
                    ((self.xv - x_i) ** 2 + (self.yv - y_i) ** 2)
                    / (2 * self.sigma ** 2)
                )
            )

            self.data.append(bump_56.astype(np.float32))
            self.targets.append(gauss_2d.astype(np.float32))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(0),
        )


##############################################################################
# 2. CNN MODEL with variable intermediate sizes
##############################################################################


class CNNDecoderWithActivations(nn.Module):
    """
    Takes (B, 56) => unsqueezed -> (B,56,1,1).
    Upsampling logic:
      - Start with kernel_size=7 => (B, hidden, 7,7)
      - Possibly do stride=2 => (14,14), then again => (28,28)
      - Final => (B,1,28,28)
    """

    def __init__(self, input_size=56, hidden_size=64, n_hidden_layers=4):
        super().__init__()
        layers = []
        self.activations = []

        # First layer => (B,56,1,1) -> (B, hidden, 7,7)
        layers.append(
            nn.ConvTranspose2d(
                input_size, hidden_size,
                kernel_size=7, stride=1, padding=0
            )
        )
        layers.append(nn.ReLU())
        current_size = 7

        # How many times we do stride=2 => upsample
        upsample_needed = max(0, int(np.log2(28 / current_size)))

        for _ in range(n_hidden_layers):
            if current_size < 28 and upsample_needed > 0:
                # 2× upsample => e.g. (7,7)->(14,14)
                layers.append(
                    nn.ConvTranspose2d(
                        hidden_size, hidden_size,
                        kernel_size=4, stride=2, padding=1
                    )
                )
                current_size *= 2
                upsample_needed -= 1
            else:
                # keep same size => stride=1
                layers.append(
                    nn.ConvTranspose2d(
                        hidden_size, hidden_size,
                        kernel_size=3, stride=1, padding=1
                    )
                )
            layers.append(nn.ReLU())

        # Final => (B,1,28,28)
        layers.append(
            nn.ConvTranspose2d(
                hidden_size, 1, kernel_size=3, stride=1, padding=1
            )
        )
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)  # => (B,56,1,1)
        self.activations = []
        for layer in self.decoder:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                self.activations.append(x.clone().detach())
        return x


##############################################################################
# 3. TRAINING / EVAL
##############################################################################


def train_model(model, dataloader, num_epochs=50, learning_rate=1e-3):
    """
    Train for 50 epochs.
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Loss: {running_loss/len(dataloader):.4f}")


def eval_ood_mse(model, dataloader):
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    total_loss, total_count = 0.0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            loss = criterion(preds, targets)
            total_loss += loss.item() * inputs.size(0)
            total_count += inputs.size(0)

    return total_loss / total_count


##############################################################################
# 4. ADD HELPER FUNCTION TO DRAW BOUNDING BOX
##############################################################################

def add_ood_box(ax, lb, ub, color='red'):
    """
    Adds a dashed rectangle from (lb, lb) to (ub, ub).
    """
    width = ub - lb
    rect = patches.Rectangle(
        (lb, lb), width, width,
        fill=False, edgecolor=color, linestyle='dashed', linewidth=1.5
    )
    ax.add_patch(rect)


##############################################################################
# 5. MAIN EXPERIMENT with MULTIPLE TRIALS + FINAL VISUALIZATION
##############################################################################


def run_experiment_cnn_gaussian_multi():
    # Output directory
    fig_dir = "fig_cnn_gaussian"
    os.makedirs(fig_dir, exist_ok=True)

    # 1) Prepare train / OOD sets
    lb, ub = 6, 22

    train_ds = GaussianBumpDatasetSquare(
        num_samples=2000,
        grid_size=28,
        sigma=1.0,
        holdout_center=True,
        only_holdout=False,
        heldout_range=[lb, ub]
    )
    train_loader = DataLoader(train_ds, batch_size=32,
                              shuffle=True, num_workers=2)

    ood_ds = GaussianBumpDatasetSquare(
        num_samples=500,
        grid_size=28,
        sigma=1.0,
        holdout_center=False,
        only_holdout=True,
        heldout_range=[lb, ub]
    )
    ood_loader = DataLoader(ood_ds, batch_size=32,
                            shuffle=False, num_workers=2)

    # 2) Depth range & multiple trials
    depths = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_runs = 1
    mean_ood_list = []
    std_ood_list = []

    # We'll store the "final model" from the *last* run of each depth
    final_models = {}

    for depth in depths:
        print("=" * 60)
        print(f"Depth = {depth}: running {num_runs} trials")
        ood_results = []

        # We'll keep a reference to the final model from the last run
        final_model_for_this_depth = None

        for run_idx in range(num_runs):
            print(f"  Trial {run_idx+1}/{num_runs} for depth={depth}")
            model = CNNDecoderWithActivations(
                input_size=56,
                hidden_size=64,
                n_hidden_layers=depth
            )
            # Train 50 epochs
            train_model(model, train_loader,
                        num_epochs=50, learning_rate=1e-3)

            # Evaluate OOD
            ood_mse = eval_ood_mse(model, ood_loader)
            ood_results.append(ood_mse)
            print(f"    OOD MSE = {ood_mse:.4f}")

            # Keep track of final model from last run
            final_model_for_this_depth = model

        # Summarize runs
        avg_ood = np.mean(ood_results)
        std_ood = np.std(ood_results)
        mean_ood_list.append(avg_ood)
        std_ood_list.append(std_ood)
        print(f"Depth={depth} => Mean OOD={avg_ood:.4f},  Std={std_ood:.4f}")

        # Store the final model from the last run
        final_models[depth] = final_model_for_this_depth

    # 3) Plot OOD MSE results with error bars
    plt.figure()
    plt.errorbar(depths, mean_ood_list, yerr=std_ood_list, fmt='o-', capsize=5)
    plt.xlabel("CNN Depth (n_hidden_layers)")
    plt.ylabel("OOD MSE (mean ± std over 5 runs)")
    plt.title("Variable-Intermediate CNN: OOD MSE vs. Depth")
    out_path = os.path.join(fig_dir, "ood_vs_depth.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print("Experiment complete. OOD plot saved at:", out_path)

    # 4) NEW: Visualize final models on 4 OOD coordinates
    #         We'll produce a 4×(len(depths)) subplot grid
    coords_ood = [(11, 14), (12, 18), (17, 12), (17, 17)]
    fig, axes = plt.subplots(
        nrows=len(coords_ood), ncols=len(depths),
        figsize=(3 * len(depths), 3 * len(coords_ood))
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for col_idx, depth in enumerate(depths):
        model = final_models[depth].eval().to(device)
        for row_idx, (xo, yo) in enumerate(coords_ood):
            # Generate the output image
            xy_56 = build_1d_gaussian_input(xo, yo, grid_size=28, sigma=1.0)
            xy_in = torch.tensor([xy_56], dtype=torch.float32, device=device)
            with torch.no_grad():
                out_2d = model(xy_in)  # shape: (1,1,28,28)
            out_np = out_2d.cpu().numpy()[0, 0]

            ax = axes[row_idx, col_idx]
            ax.imshow(out_np, cmap='gray', origin='lower')
            ax.axis('off')

            # Add bounding box for [lb..ub]^2
            add_ood_box(ax, lb, ub, color='red')

            # Mark the ground-truth coordinate
            ax.scatter(xo, yo, s=40, c='blue', marker='x')

            if col_idx == 0:
                ax.set_ylabel(f"(x={xo}, y={yo})", fontsize=9)
            if row_idx == 0:
                ax.set_title(f"Depth {depth}", fontsize=9)

    plt.tight_layout()
    out_path2 = os.path.join(fig_dir, "final_ood_images_by_depth.png")
    plt.savefig(out_path2, bbox_inches='tight')
    plt.close()
    print(f"Final OOD sample plots saved at: {out_path2}")


##############################################################################
# 6. HELPER FOR BUILDING 1D GAUSSIAN INPUT  (to generate the OOD sample)
##############################################################################

def build_1d_gaussian_input(x, y, grid_size=28, sigma=1.0):
    """
    Build the same 1D Gaussian input of length=56:
        - 28-dim for the x-bump
        - 28-dim for the y-bump
    consistent with how the dataset is created.
    """
    gx = np.exp(-((np.arange(grid_size) - x) ** 2) / (2 * sigma ** 2))
    gy = np.exp(-((np.arange(grid_size) - y) ** 2) / (2 * sigma ** 2))
    return np.concatenate([gx, gy], axis=0).astype(np.float32)


##############################################################################
# 7. MAIN
##############################################################################

if __name__ == "__main__":
    run_experiment_cnn_gaussian_multi()
