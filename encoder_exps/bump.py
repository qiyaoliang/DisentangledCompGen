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
# 1. DATASET (Reversed): input => 28×28 image, output => 56-dim vector
##############################################################################
class GaussianBumpDatasetReverse(Dataset):
    """
    Each sample:
      - 28x28 input => 2D Gaussian bump centered at (x,y)
      - 56-dim target => 1D Gaussians for x and y, stacked: [gaussian_x(28), gaussian_y(28)]

    We'll exclude [lb..ub]^2 from training if holdout_center=True
    or only sample from that region if only_holdout=True.
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

            # 2D Gaussian => shape (28×28)
            gauss_2d = np.exp(
                -(
                    (self.xv - x_i) ** 2 + (self.yv - y_i) ** 2
                )
                / (2 * self.sigma**2)
            )

            # 1D Gaussians => 28+28=56-dim output
            gx = np.exp(-((np.arange(self.grid_size) - x_i) ** 2) / (2 * self.sigma**2))
            gy = np.exp(-((np.arange(self.grid_size) - y_i) ** 2) / (2 * self.sigma**2))
            bump_56 = np.concatenate([gx, gy], axis=0)

            self.data.append(gauss_2d.astype(np.float32))
            self.targets.append(bump_56.astype(np.float32))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Input => (1,28,28), Output => (56,)
        return (
            torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


##############################################################################
# 2. CNN ENCODER: from image (1×28×28) => 56-dim vector
##############################################################################
class CNNEncoder(nn.Module):
    """
    A small CNN that takes a 1-channel 28×28 input and outputs a 56-d vector.
    You can adjust hidden sizes, kernel sizes, etc. as needed.
    """

    def __init__(self, output_size=56):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # => 16×16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # => 8×8
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# => 4×4
            nn.ReLU(),
        )
        # After conv_stack => (B, 128, 4, 4) => flatten => feed linear
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        # x => (B,1,28,28)
        z = self.conv_stack(x)  # => (B,128,4,4)
        z = z.view(z.size(0), -1)  # => (B,128*4*4)
        out = self.fc(z)          # => (B,56)
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
            outputs = model(inputs)
            loss = criterion(outputs, targets)
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
    count = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            loss = criterion(preds, targets)
            total_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)
    return total_loss / count


##############################################################################
# 4. HELPER: from a predicted 56-d vector => get (x_pred, y_pred)
##############################################################################
def get_xy_from_56(bumps_56):
    """
    bumps_56: shape (B,56)
      => first 28 are the x-gaussian
      => second 28 are the y-gaussian
    We'll interpret the predicted coordinate as the argmax in each 1D gaussian.
    """
    B = bumps_56.shape[0]
    x_preds = []
    y_preds = []
    for i in range(B):
        # Split
        gx = bumps_56[i, :28]  # (28,)
        gy = bumps_56[i, 28:]  # (28,)
        x_i = torch.argmax(gx).item()
        y_i = torch.argmax(gy).item()
        x_preds.append(x_i)
        y_preds.append(y_i)
    return np.array(x_preds), np.array(y_preds)


##############################################################################
# 5. MAIN EXPERIMENT
##############################################################################
def run_experiment_cnn_gaussian_inverse():
    fig_dir = "fig_gaussian"
    os.makedirs(fig_dir, exist_ok=True)

    # 1) In-distribution dataset => excludes [6..22]^2
    train_ds = GaussianBumpDatasetReverse(
        num_samples=2000,
        grid_size=28,
        sigma=1.0,
        holdout_center=True,
        only_holdout=False,
        heldout_range=[6, 22]
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)

    # 2) OOD dataset => only [6..22]^2
    ood_ds = GaussianBumpDatasetReverse(
        num_samples=500,
        grid_size=28,
        sigma=1.0,
        holdout_center=False,
        only_holdout=True,
        heldout_range=[6, 22]
    )
    ood_loader = DataLoader(ood_ds, batch_size=32, shuffle=False, num_workers=2)

    # 3) Create the CNN model & train
    model = CNNEncoder(output_size=56)
    train_model(model, train_loader, num_epochs=20, learning_rate=1e-3)

    # 4) Evaluate OOD MSE
    loss_ood = eval_ood_mse(model, ood_loader)
    print(f"OOD MSE = {loss_ood:.4f}")

    # 5) MSE contour plot across all x,y in [0..27]
    #    We'll build a 28×28 map of MSE between predicted 56-vector and ground-truth 56-vector
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    coords_1d = np.arange(28)
    xv, yv = np.meshgrid(coords_1d, coords_1d, indexing="ij")
    grid_points = np.stack([xv.ravel(), yv.ravel()], axis=-1)  # shape (784,2)
    # Build (784,1,28,28) images
    all_imgs = []
    all_targets_56 = []
    for (xx, yy) in grid_points:
        # 2D Gaussian image
        # shape => (28,28)
        row_coords, col_coords = np.arange(28), np.arange(28)
        rr, cc = np.meshgrid(row_coords, col_coords, indexing="ij")
        gauss_2d = np.exp(-((rr-xx)**2 + (cc-yy)**2)/(2*1.0**2))

        # 56-d target
        gx = np.exp(-(row_coords - xx)**2 / (2*1.0**2))
        gy = np.exp(-(col_coords - yy)**2 / (2*1.0**2))
        bump_56 = np.concatenate([gx, gy], axis=0)

        all_imgs.append(gauss_2d[None, ...])  # add channel dim => (1,28,28)
        all_targets_56.append(bump_56)

    all_imgs = np.array(all_imgs, dtype=np.float32)        # (784,1,28,28)
    all_targets_56 = np.array(all_targets_56, dtype=np.float32)  # (784,56)

    # Forward pass
    all_imgs_torch = torch.tensor(all_imgs, device=device)
    all_targets_torch = torch.tensor(all_targets_56, device=device)
    batch_size = 64
    preds_all = []
    for i_start in range(0, all_imgs_torch.size(0), batch_size):
        i_end = i_start + batch_size
        x_batch = all_imgs_torch[i_start:i_end]
        with torch.no_grad():
            p_batch = model(x_batch)
        preds_all.append(p_batch.cpu().numpy())
    preds_all = np.concatenate(preds_all, axis=0)  # => (784,56)

    # MSE per sample
    mse_per_sample = np.mean((preds_all - all_targets_56)**2, axis=1)  # (784,)
    mse_map = mse_per_sample.reshape(28, 28)

    # Plot MSE contour
    plt.figure(figsize=(6,5))
    im = plt.imshow(mse_map, origin='lower', cmap='viridis')
    plt.title("MSE across the entire 28×28 grid")
    plt.colorbar(im, orientation='vertical')
    plt.xlabel("y-coordinate")
    plt.ylabel("x-coordinate")

    # bounding box for OOD region [6..22]
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

    out_mse_path = os.path.join(fig_dir, "mse_contour.png")
    plt.savefig(out_mse_path, bbox_inches='tight')
    plt.close()
    print(f"Saved MSE contour plot to {out_mse_path}")

    # 6) Print out 4 different pairs of predicted vs. ground truth coords from OOD
    #    We'll just fetch 1 batch from ood_loader, run model, pick 4 random examples
    inputs_batch, targets_batch = next(iter(ood_loader))
    inputs_batch = inputs_batch.to(device)
    targets_batch = targets_batch.to(device)
    with torch.no_grad():
        preds_batch = model(inputs_batch)
    # Convert to argmax x,y
    x_pred, y_pred = get_xy_from_56(preds_batch)       # numpy arrays
    x_true, y_true = get_xy_from_56(targets_batch)     # numpy arrays

    # Randomly pick 4
    bsize = inputs_batch.shape[0]
    idxs = np.random.choice(bsize, size=4, replace=False)
    print("---- 4 OOD Samples: predicted vs ground-truth (x,y) ----")
    for idx in idxs:
        print(f"Sample {idx}: pred=({x_pred[idx]},{y_pred[idx]}), gt=({x_true[idx]},{y_true[idx]})")


if __name__ == "__main__":
    run_experiment_cnn_gaussian_inverse()
