#!/usr/bin/env python3
# cnn_depth_experiment.py

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

##############################################################################
# 1. DATASET DEFINITIONS
##############################################################################


class GaussianBumpDatasetSquare(Dataset):
    """
    Creates a dataset of 2D Gaussian bump images on a grid, with a 'square' region
    held out if holdout_center=True. If only_holdout=True, it only generates points
    within that region.
    """

    def __init__(
        self,
        num_samples=1000,
        grid_size=28,
        sigma=1.0,
        holdout_center=True,
        only_holdout=False,
        heldout_range=[10, 18]
    ):
        super().__init__()
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.sigma = sigma
        self.holdout_center = holdout_center
        self.only_holdout = only_holdout
        self.lb, self.ub = heldout_range

        # Pre-compute the meshgrid for generating the 2D Gaussian
        self.xv, self.yv = np.meshgrid(
            np.arange(grid_size), np.arange(grid_size)
        )

        self.data = []
        self.targets = []
        self._generate_samples()

    def _generate_samples(self):
        if self.only_holdout:
            # Generate points directly within the holdout range only
            x = np.random.randint(self.lb, self.ub + 1, size=self.num_samples)
            y = np.random.randint(self.lb, self.ub + 1, size=self.num_samples)
        elif self.holdout_center:
            # Generate points outside the holdout region
            x_list = []
            y_list = []
            while len(x_list) < self.num_samples:
                batch_size = min(self.num_samples * 2, 10000)
                x_batch = np.random.randint(0, self.grid_size, size=batch_size)
                y_batch = np.random.randint(0, self.grid_size, size=batch_size)
                valid_mask = ~(
                    (x_batch >= self.lb)
                    & (x_batch <= self.ub)
                    & (y_batch >= self.lb)
                    & (y_batch <= self.ub)
                )
                x_list.extend(x_batch[valid_mask])
                y_list.extend(y_batch[valid_mask])
            x = np.array(x_list[: self.num_samples])
            y = np.array(y_list[: self.num_samples])
        else:
            # Generate points anywhere
            x = np.random.randint(0, self.grid_size, size=self.num_samples)
            y = np.random.randint(0, self.grid_size, size=self.num_samples)

        for i in range(self.num_samples):
            gaussian_image = 1.0 - np.exp(
                -(
                    (self.xv - x[i])**2
                    + (self.yv - y[i])**2
                )
                / (2 * self.sigma**2)
            )
            self.data.append(np.array([x[i], y[i]], dtype=np.float32))
            self.targets.append(gaussian_image.astype(np.float32))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(0),
        )

##############################################################################
# 2. CNN MODEL DEFINITION
##############################################################################


class CNNDecoderWithActivations(nn.Module):
    """
    A CNN-based decoder that takes a 2D input (x,y) and expands it into
    a 28x28 image using transposed convolutions, storing intermediate 
    (post-ReLU) activations in self.activations for analysis (unused here).
    """

    def __init__(self, input_size=2, hidden_size=64, n_hidden_layers=4):
        super().__init__()
        self.activations = []

        layers = []
        # First layer: (B,2,1,1) -> (B, hidden_size, 7,7)
        layers.append(nn.ConvTranspose2d(
            input_size, hidden_size, kernel_size=7, stride=1, padding=0
        ))
        layers.append(nn.ReLU())
        current_size = 7

        upsample_needed = max(0, int(np.log2(28 / current_size)))

        for _ in range(n_hidden_layers):
            if current_size < 28 and upsample_needed > 0:
                # Upsample x2
                layers.append(
                    nn.ConvTranspose2d(
                        hidden_size, hidden_size, kernel_size=4, stride=2, padding=1
                    )
                )
                current_size *= 2
                upsample_needed -= 1
            else:
                # Keep size
                layers.append(
                    nn.ConvTranspose2d(
                        hidden_size, hidden_size, kernel_size=3, stride=1, padding=1
                    )
                )
            layers.append(nn.ReLU())

        # Final layer => (B,1,28,28)
        layers.append(
            nn.ConvTranspose2d(hidden_size, 1, kernel_size=3,
                               stride=1, padding=1)
        )
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        self.activations = []
        x = x.unsqueeze(-1).unsqueeze(-1)  # (B,2)->(B,2,1,1)
        for layer in self.decoder:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                self.activations.append(x.clone().detach())
        return x

##############################################################################
# 3. TRAINING FUNCTION
##############################################################################


def train_model(model, dataloader, num_epochs=50, learning_rate=1e-3):
    """
    Train the model on a given DataLoader (in-distribution).
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Just for label: final layer's in_channels = hidden_size, not "depth"
    hidden_size = model.decoder[-1].in_channels

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
        print(f"Depth={hidden_size}, "
              f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_loss:.4f}")

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
# 5. MAIN EXPERIMENT: Single Run per Depth & Plot OOD Samples
##############################################################################


def run_experiment_cnn_depths():
    # Create folder for figures if needed
    if not os.path.exists("fig_cnn"):
        os.makedirs("fig_cnn")

    # OOD coordinates (held-out region is [6..22])
    coords_ood = [(11, 14), (12, 18), (17, 12), (17, 17)]
    lb, ub = 6, 22

    # Prepare in-distribution dataset => excludes [lb..ub]^2
    train_dataset = GaussianBumpDatasetSquare(
        num_samples=2000,
        grid_size=28,
        holdout_center=True,   # exclude holdout region
        heldout_range=[lb, ub]
    )
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=2
    )

    # We'll run with depths 2..10 (9 different depths)
    depths = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Keep track of a final trained model for each depth
    final_models = {}

    # Train one model per depth
    for depth in depths:
        print("============================================================")
        print(f"Training model with n_hidden_layers={depth}")

        model = CNNDecoderWithActivations(
            input_size=2,
            hidden_size=64,
            n_hidden_layers=depth
        )

        # Train for 50 epochs
        train_model(model, train_loader, num_epochs=50, learning_rate=1e-3)

        # Store the final model
        final_models[depth] = model

    # Now create a single figure with 4 rows x 9 columns
    # (4 OOD coords) x (9 possible depths)
    fig, axes = plt.subplots(nrows=4, ncols=9, figsize=(18, 8))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for col_idx, depth in enumerate(depths):
        model = final_models[depth].eval().to(device)
        for row_idx, (xo, yo) in enumerate(coords_ood):
            # Generate the output image
            xy_in = torch.tensor(
                [[xo, yo]], dtype=torch.float32, device=device)
            with torch.no_grad():
                out_2d = model(xy_in)  # shape: (1,1,28,28)
            out_np = out_2d.cpu().numpy()[0, 0]

            ax = axes[row_idx, col_idx]
            ax.imshow(out_np, cmap='gray', origin='lower')
            ax.axis('off')

            # Draw a red bounding box for the holdout region
            add_ood_box(ax, lb, ub, color='red')

            # Mark the ground-truth coordinate (xo, yo) with a small scatter
            ax.scatter(xo, yo, s=40, c='blue', marker='x')

            # Column 0: label the row with the coordinate
            if col_idx == 0:
                ax.set_ylabel(f"(x={xo}, y={yo})", fontsize=10)

            # Row 0: label the column with depth
            if row_idx == 0:
                ax.set_title(f"Depth {depth}", fontsize=10)

    plt.tight_layout()
    out_plot_path = os.path.join("fig_cnn", "ood_images_by_depth.png")
    plt.savefig(out_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved final figure at: {out_plot_path}")


##############################################################################
# 6. MAIN
##############################################################################

if __name__ == "__main__":
    run_experiment_cnn_depths()
