#!/usr/bin/env python3
# superposition_scalar_2d.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.patches as patches

##############################################################################
# 1. DATASET: 2D scalar input, 28x28 output
##############################################################################


class GaussianBumpDatasetSquare(Dataset):
    """
    Creates a dataset of 2D Gaussian bump images on a grid, 
    with a 'square' region [lb..ub]^2 held out if holdout_center=True 
    (or only that region if only_holdout=True).
    input: (x,y) in [0..grid_size)^2
    output: 28x28 image with Gaussian bump at (x,y).
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

        # For generating 2D Gaussians
        self.xv, self.yv = np.meshgrid(
            np.arange(grid_size), np.arange(grid_size))

        self.data = []
        self.targets = []
        self._generate_samples()

    def _generate_samples(self):
        if self.only_holdout:
            # Directly sample within holdout
            xs = np.random.randint(self.lb, self.ub+1, size=self.num_samples)
            ys = np.random.randint(self.lb, self.ub+1, size=self.num_samples)
        elif self.holdout_center:
            # Exclude [lb..ub]^2
            x_list, y_list = [], []
            while len(x_list) < self.num_samples:
                batch_size = min(self.num_samples*2, 10000)
                x_b = np.random.randint(0, self.grid_size, size=batch_size)
                y_b = np.random.randint(0, self.grid_size, size=batch_size)
                valid_mask = ~(
                    (x_b >= self.lb) & (x_b <= self.ub) &
                    (y_b >= self.lb) & (y_b <= self.ub)
                )
                x_list.extend(x_b[valid_mask])
                y_list.extend(y_b[valid_mask])
            xs = np.array(x_list[:self.num_samples])
            ys = np.array(y_list[:self.num_samples])
        else:
            # Sample anywhere
            xs = np.random.randint(0, self.grid_size, size=self.num_samples)
            ys = np.random.randint(0, self.grid_size, size=self.num_samples)

        for i in range(self.num_samples):
            x_i, y_i = xs[i], ys[i]
            # Build 2D Gaussian
            gaussian_image = 1.0 - np.exp(
                -(
                    (self.xv - x_i)**2 + (self.yv - y_i)**2
                )/(2*self.sigma*self.sigma)
            )
            self.data.append(np.array([x_i, y_i], dtype=np.float32))
            self.targets.append(gaussian_image.astype(np.float32))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_xy = torch.tensor(
            self.data[idx], dtype=torch.float32)   # shape (2,)
        target_2d = torch.tensor(
            self.targets[idx], dtype=torch.float32).unsqueeze(0)  # (1,28,28)
        return input_xy, target_2d


##############################################################################
# 2. CNN MODEL: scalar (2D) => final => 28x28
##############################################################################

class CNNDecoderWithActivations(nn.Module):
    """
    A CNN-based decoder that takes a 2D input (x,y) -> expand to (28,28).
    We'll store post-ReLU activations for optional analysis.
    """

    def __init__(self, input_size=2, hidden_size=64, n_hidden_layers=4):
        super().__init__()
        self.activations = []  # store them in a list or dict if needed

        layers = []
        # First layer: from (2,1,1) -> (hidden_size,7,7)
        layers.append(nn.ConvTranspose2d(input_size, hidden_size,
                      kernel_size=7, stride=1, padding=0))
        layers.append(nn.ReLU())
        current_size = 7

        # figure out how many upsamples to get 28
        upsample_needed = max(0, int(np.log2(28 / current_size)))

        for i in range(n_hidden_layers):
            if current_size < 28 and upsample_needed > 0:
                layers.append(
                    nn.ConvTranspose2d(hidden_size, hidden_size,
                                       kernel_size=4, stride=2, padding=1)
                )
                current_size *= 2
                upsample_needed -= 1
            else:
                layers.append(
                    nn.ConvTranspose2d(hidden_size, hidden_size,
                                       kernel_size=3, stride=1, padding=1)
                )
            layers.append(nn.ReLU())

        # final => (1,28,28)
        layers.append(nn.ConvTranspose2d(hidden_size, 1,
                      kernel_size=3, stride=1, padding=1))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        # x => shape (B,2) => (B,2,1,1)
        self.activations = []
        x = x.unsqueeze(-1).unsqueeze(-1)
        for layer in self.decoder:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                self.activations.append(x.clone().detach())
        return x


##############################################################################
# 3. TRAINING
##############################################################################

def train_model(model, dataloader, num_epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    crit = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        run_loss = 0
        for inp_xy, tgt_2d in dataloader:
            inp_xy, tgt_2d = inp_xy.to(device), tgt_2d.to(device)
            opt.zero_grad()
            out_2d = model(inp_xy)
            loss = crit(out_2d, tgt_2d)
            loss.backward()
            opt.step()
            run_loss += loss.item()
        avg_loss = run_loss/len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss={avg_loss:.4f}")
    return model

##############################################################################
# 4. COLLECT ID OUTPUTS
##############################################################################


def collect_id_outputs(model, dset, lb, ub):
    """
    For each (x,y) in the dataset that is ID (i.e. outside [lb..ub]^2),
    we compute model((x,y)) => shape (1,28,28).
    We'll store them in a dict => id_output_map[(x,y)] = that final image (28,28).
    """
    id_output_map = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    with torch.no_grad():
        for i in range(len(dset)):
            xy, _ = dset[i]
            xval, yval = xy[0].item(), xy[1].item()
            # check if ID
            if not (lb <= xval <= ub and lb <= yval <= ub):
                # pass through model
                xy_in = xy.unsqueeze(0).to(device)  # shape (1,2)
                out_2d = model(xy_in)  # shape (1,1,28,28)
                out_np = out_2d.cpu().numpy()[0, 0]  # shape (28,28)
                id_output_map[(int(xval), int(yval))] = out_np
    return id_output_map

##############################################################################
# 5. OOD EVALUATION: Summation Approach
##############################################################################


def create_sum_id_image(xo, yo, id_output_map):
    """
    Summation of all ID images that share x=xo or y=yo in 'id_output_map'.
    Then return sum_img (28,28) and count how many we added.
    """
    sum_img = np.zeros((28, 28), dtype=np.float32)
    n_added = 0
    # We'll gather images from ID points that share x=xo or y=yo
    # E.g. (xo, any y') or (any x', yo), provided that (x',y') was ID
    for (xID, yID), out_2d in id_output_map.items():
        if xID == xo or yID == yo:
            sum_img += out_2d
            n_added += 1
    return sum_img, n_added


def get_ood_coords(lb, ub, n_samples=4):
    coords = []
    for _ in range(n_samples):
        x_ = np.random.randint(lb, ub+1)
        y_ = np.random.randint(lb, ub+1)
        coords.append((x_, y_))
    return coords

##############################################################################
# 6. VISUALIZATION
##############################################################################


def add_ood_box(ax, lb, ub, color='red'):
    rect = patches.Rectangle(
        (lb, lb), (ub-lb), (ub-lb),
        fill=False, edgecolor=color, linestyle='dashed', linewidth=2
    )
    ax.add_patch(rect)


def visualize_ood_summation(model, coords_list, id_output_map,
                            lb=10, ub=18, out_dir="vis_ood_idsum"):
    """
    For each OOD (xo,yo):
      1) compute model((xo,yo)) => out_ood
      2) create sum_id_image => sum_img
      3) normalize sum_img
      4) show side by side: sum_img_norm, out_ood
    """
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    with torch.no_grad():
        for (xo, yo) in coords_list:
            # model output
            xy_in = torch.tensor(
                [[xo, yo]], dtype=torch.float32, device=device)
            out_2d = model(xy_in).cpu().numpy()[0, 0]  # shape (28,28)

            # sum from id
            sum_img, n_added = create_sum_id_image(xo, yo, id_output_map)
            # if n_added>0, we can do a normalization
            maxval = sum_img.max()
            if maxval > 1e-12:
                sum_norm = sum_img / maxval
            else:
                sum_norm = sum_img

            # side-by-side plot
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            for ax in axes:
                ax.axis('off')

            # Left => sum_norm
            axes[0].imshow(sum_norm, cmap='gray', origin='lower')
            axes[0].set_title(f"Sum of ID outputs (n={n_added})", fontsize=10)

            axes[0].scatter(xo, yo, color='red', s=10)

            add_ood_box(axes[0], lb, ub, color='red')

            # Right => out_ood
            axes[1].imshow(out_2d, cmap='gray', origin='lower')
            axes[1].set_title("model(x,y)", fontsize=10)
            axes[1].scatter(xo, yo, color='red', s=10)

            add_ood_box(axes[1], lb, ub, color='red')

            fig.suptitle(f"Summation vs OOD Output: (x={xo},y={yo})", y=0.93)
            out_fig = os.path.join(out_dir, f"ood_x{xo}_y{yo}.png")
            plt.savefig(out_fig, bbox_inches='tight')
            plt.close()
            print(f"[visualize_ood_summation] => saved {out_fig}")

##############################################################################
# 7. MAIN
##############################################################################


def main():
    lb, ub = 6, 22
    # 1) Create training dataset => excludes [10..18]^2 => ID region
    train_ds = GaussianBumpDatasetSquare(
        num_samples=20000,
        grid_size=28,
        sigma=1.0,
        holdout_center=True,
        only_holdout=False,
        heldout_range=[lb, ub]
    )
    train_loader = DataLoader(train_ds, batch_size=32,
                              shuffle=True, num_workers=2)

    # 2) Build & train model
    model = CNNDecoderWithActivations(
        input_size=2, hidden_size=128, n_hidden_layers=10)
    model = train_model(model, train_loader, num_epochs=30, lr=1e-3)

    # 3) Collect ID outputs => dict{(xID,yID)->(28,28) image}
    id_output_map = collect_id_outputs(model, train_ds, lb, ub)
    print(f"[main] Collected {len(id_output_map)} ID outputs")

    # 4) Pick a few OOD coords in [lb..ub] => visualize
    coords_ood = get_ood_coords(10, 18, n_samples=4)
    print("OOD coords chosen:", coords_ood)

    visualize_ood_summation(model, coords_ood, id_output_map,
                            lb=lb, ub=ub, out_dir="vis_ood_idsum")


if __name__ == "__main__":
    main()
