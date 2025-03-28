#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

##############################################################################
# 1. DATASET: input => 28×28 image, output => (x, y) coordinate
##############################################################################


class GaussianBumpDatasetCoord(Dataset):
    """
    Each sample:
      - 28×28 input => 2D Gaussian bump (centered at (x,y))
      - 2D target => (x, y) coordinate.

    We can exclude or only sample from [lb..ub]^2 for OOD testing.
    """

    def __init__(
        self,
        num_samples=1000,
        grid_size=28,
        sigma=1.0,
        holdout_center=True,
        only_holdout=False,
        heldout_range=[6, 22],
    ):
        super().__init__()
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.sigma = sigma
        self.holdout_center = holdout_center
        self.only_holdout = only_holdout
        self.lb, self.ub = heldout_range

        # Precompute a meshgrid for generating 2D Gaussians
        self.xv, self.yv = np.meshgrid(
            np.arange(grid_size), np.arange(grid_size), indexing="ij"
        )

        self.data = []
        self.targets = []
        self._generate()

    def _generate(self):
        if self.only_holdout:
            # Only sample from [lb..ub]^2
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
                    (x_b >= self.lb)
                    & (x_b <= self.ub)
                    & (y_b >= self.lb)
                    & (y_b <= self.ub)
                )
                xs.extend(x_b[valid])
                ys.extend(y_b[valid])
            xs = np.array(xs[: self.num_samples])
            ys = np.array(ys[: self.num_samples])
        else:
            # Sample anywhere
            xs = np.random.randint(0, self.grid_size, size=self.num_samples)
            ys = np.random.randint(0, self.grid_size, size=self.num_samples)

        for i in range(self.num_samples):
            x_i, y_i = xs[i], ys[i]

            # 2D Gaussian => shape (28,28)
            gauss_2d = np.exp(
                -(
                    (self.xv - x_i) ** 2 + (self.yv - y_i) ** 2
                )
                / (2 * self.sigma**2)
            )

            self.data.append(gauss_2d.astype(np.float32))
            # Output => (x,y) as float
            self.targets.append(np.array([x_i, y_i], dtype=np.float32))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # input => (1,28,28), output => (2,)
        return (
            torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


##############################################################################
# 2. CNN MODEL: from image (1×28×28) => 2 scalars (x,y)
##############################################################################
class CNNEncoderCoords(nn.Module):
    """
    A CNN encoder that outputs 2 scalars (predicted x,y).
    """

    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2,
                      padding=1),  # => (B,32,14,14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2,
                      padding=1),  # => (B,64,7,7)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,
                      padding=1),  # => (B,128,4,4)
            nn.ReLU(),
        )
        # After conv_stack => (B,128,4,4) => flatten => feed linear
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # outputs (x,y)
        )

    def forward(self, x):
        # x => (B,1,28,28)
        z = self.conv_stack(x)      # => (B,128,4,4)
        z = z.view(z.size(0), -1)   # => (B,128*4*4)
        out = self.fc(z)            # => (B,2)
        return out


##############################################################################
# 3. TRAINING / EVALUATION
##############################################################################
def train_model(model, dataloader, num_epochs=20, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(inputs)  # shape (B,2)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
    return model


def eval_ood_mse(model, dataloader):
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)  # (B,2)
            loss = criterion(preds, targets)
            total_loss += loss.item() * inputs.size(0)
            total_count += inputs.size(0)

    return total_loss / total_count


##############################################################################
# 4. MAIN EXPERIMENT
##############################################################################
def run_experiment_cnn_gaussian_coords():
    fig_dir = "fig_scalar"
    os.makedirs(fig_dir, exist_ok=True)

    # ------------------------------
    # 1) Create train/ood datasets
    # ------------------------------
    train_ds = GaussianBumpDatasetCoord(
        num_samples=2000,
        grid_size=28,
        sigma=1.0,
        holdout_center=True,
        only_holdout=False,
        heldout_range=[6, 22]
    )
    train_loader = DataLoader(train_ds, batch_size=32,
                              shuffle=True, num_workers=2)

    ood_ds = GaussianBumpDatasetCoord(
        num_samples=500,
        grid_size=28,
        sigma=1.0,
        holdout_center=False,
        only_holdout=True,
        heldout_range=[6, 22]
    )
    ood_loader = DataLoader(ood_ds, batch_size=32,
                            shuffle=False, num_workers=2)

    # ------------------------------
    # 2) Create and train the model
    # ------------------------------
    model = CNNEncoderCoords()
    train_model(model, train_loader, num_epochs=50, learning_rate=1e-3)

    # ------------------------------
    # 3) Evaluate OOD
    # ------------------------------
    loss_ood = eval_ood_mse(model, ood_loader)
    print(f"OOD MSE = {loss_ood:.4f}")

    # ------------------------------
    # 4) MSE contour plot across 28×28
    # ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    coords_1d = np.arange(28)
    xv, yv = np.meshgrid(coords_1d, coords_1d, indexing="ij")
    grid_points = np.stack([xv.ravel(), yv.ravel()], axis=-1)  # shape (784,2)

    # Build all 2D Gaussians
    all_imgs = []
    for (xx, yy) in grid_points:
        # 2D Gaussian => shape (28,28)
        rr, cc = np.meshgrid(coords_1d, coords_1d, indexing="ij")
        gauss_2d = np.exp(-((rr-xx)**2 + (cc-yy)**2)/(2*1.0**2))
        all_imgs.append(gauss_2d[None, ...])  # => (1,28,28)

    all_imgs = np.array(all_imgs, dtype=np.float32)  # (784,1,28,28)
    all_imgs_torch = torch.tensor(all_imgs, device=device)

    # Forward pass in batches
    preds_all = []
    batch_size = 64
    for i_start in range(0, len(all_imgs_torch), batch_size):
        i_end = i_start + batch_size
        x_batch = all_imgs_torch[i_start:i_end]
        with torch.no_grad():
            p_batch = model(x_batch)  # => (batch,2)
        preds_all.append(p_batch.cpu().numpy())
    preds_all = np.concatenate(preds_all, axis=0)  # => (784,2)

    # Ground truth coords
    gt_coords = grid_points  # shape (784,2)

    # (A) MSE Map
    mse_per_sample = np.mean((preds_all - gt_coords)**2, axis=1)
    mse_map = mse_per_sample.reshape(28, 28)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(mse_map, origin='lower', cmap='viridis')
    plt.title("MSE across the entire 28×28 grid")
    plt.colorbar(im, orientation='vertical')
    plt.xlabel("y-coordinate")
    plt.ylabel("x-coordinate")

    # bounding box for OOD region [6..22]^2
    rect = patches.Rectangle(
        (6, 6),
        22 - 6 + 1,
        22 - 6 + 1,
        fill=False,
        edgecolor='red',
        linestyle='dashed',
        linewidth=2,
    )
    ax = plt.gca()
    ax.add_patch(rect)

    mse_contour_path = os.path.join(fig_dir, "mse_contour_coords.png")
    plt.savefig(mse_contour_path, bbox_inches='tight')
    plt.close()
    print(f"Saved MSE contour plot to {mse_contour_path}")

    # (B) Accuracy Map
    # We'll define "accurate" = 1 if |x_pred - x_true| <= 0.5 AND |y_pred - y_true| <= 0.5
    # Otherwise 0. Then visualize as a contour [0 or 1].
    acc_per_sample = np.zeros_like(mse_per_sample)
    for i in range(len(mse_per_sample)):
        x_true, y_true = gt_coords[i]
        x_pred, y_pred = preds_all[i]
        if (abs(x_pred - x_true) <= 0.5) and (abs(y_pred - y_true) <= 0.5):
            acc_per_sample[i] = 1.0
        else:
            acc_per_sample[i] = 0.0
    acc_map = acc_per_sample.reshape(28, 28)

    plt.figure(figsize=(6, 5))
    im_acc = plt.imshow(acc_map, origin='lower', cmap='Blues', vmin=0, vmax=1)
    plt.title("Accuracy (±0.5 Tolerance) across the 28×28 grid")
    plt.colorbar(im_acc, orientation='vertical', label="Accuracy (0 or 1)")
    plt.xlabel("y-coordinate")
    plt.ylabel("x-coordinate")

    # bounding box for OOD region [6..22]^2
    rect_acc = patches.Rectangle(
        (6, 6),
        22 - 6 + 1,
        22 - 6 + 1,
        fill=False,
        edgecolor='red',
        linestyle='dashed',
        linewidth=2,
    )
    ax_acc = plt.gca()
    ax_acc.add_patch(rect_acc)

    acc_contour_path = os.path.join(fig_dir, "accuracy_contour_coords.png")
    plt.savefig(acc_contour_path, bbox_inches='tight')
    plt.close()
    print(f"Saved ACCURACY contour plot to {acc_contour_path}")

    # ------------------------------
    # 5) Print 4 OOD predictions
    # ------------------------------
    inputs_batch, targets_batch = next(iter(ood_loader))
    inputs_batch = inputs_batch.to(device)
    targets_batch = targets_batch.to(device)
    with torch.no_grad():
        preds_batch = model(inputs_batch)  # => (B,2)

    preds_np = preds_batch.cpu().numpy()
    targets_np = targets_batch.cpu().numpy()

    idxs = np.random.choice(len(inputs_batch), size=4, replace=False)
    print("---- 4 OOD Samples: predicted vs ground-truth (x,y) ----")
    for idx in idxs:
        x_pred, y_pred = preds_np[idx]
        x_true, y_true = targets_np[idx]
        print(
            f"Sample {idx}: pred=({x_pred:.2f}, {y_pred:.2f}), gt=({x_true:.2f}, {y_true:.2f})")


if __name__ == "__main__":
    run_experiment_cnn_gaussian_coords()
