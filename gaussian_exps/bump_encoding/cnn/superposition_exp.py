#!/usr/bin/env python3
# superposition_xsuper_ysuper_ood_only.py

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
# 1) Increase font sizes globally
##############################################################################
plt.rcParams.update({
    'font.size': 14,         # base font size
    'axes.labelsize': 16,    # x/y labels
    'axes.titlesize': 18,    # subplot title size
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

##############################################################################
# 2. DATASET: 56-dim input => [gaussian_x(28), gaussian_y(28)] => output => (1,28,28)
##############################################################################


class GaussianBumpDatasetSquare(Dataset):
    """
    56-dim input => [gaussian_x(28), gaussian_y(28)] => 28x28 bump.
    We can skip or only sample region [lb..ub]^2 for training/holdout.
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

        self.xv, self.yv = np.meshgrid(
            np.arange(grid_size), np.arange(grid_size)
        )
        self.data = []
        self.targets = []
        self._generate()

    def _generate(self):
        if self.only_holdout:
            xs = np.random.randint(self.lb, self.ub+1, size=self.num_samples)
            ys = np.random.randint(self.lb, self.ub+1, size=self.num_samples)
        elif self.holdout_center:
            xs, ys = [], []
            while len(xs) < self.num_samples:
                batch_size = min(self.num_samples*2, 10000)
                x_b = np.random.randint(0, self.grid_size, size=batch_size)
                y_b = np.random.randint(0, self.grid_size, size=batch_size)
                valid = ~(
                    (x_b >= self.lb) & (x_b <= self.ub) &
                    (y_b >= self.lb) & (y_b <= self.ub)
                )
                xs.extend(x_b[valid])
                ys.extend(y_b[valid])
            xs = np.array(xs[:self.num_samples])
            ys = np.array(ys[:self.num_samples])
        else:
            xs = np.random.randint(0, self.grid_size, size=self.num_samples)
            ys = np.random.randint(0, self.grid_size, size=self.num_samples)

        for i in range(self.num_samples):
            x_i, y_i = xs[i], ys[i]
            gx = np.exp(-((np.arange(self.grid_size)-x_i)**2) /
                        (2*self.sigma**2))
            gy = np.exp(-((np.arange(self.grid_size)-y_i)**2) /
                        (2*self.sigma**2))
            bump_56 = np.concatenate([gx, gy], axis=0).astype(np.float32)

            gauss_2d = 1.0 - np.exp(
                -(((self.xv - x_i)**2 + (self.yv - y_i)**2)/(2*self.sigma**2))
            ).astype(np.float32)

            self.data.append(bump_56)
            self.targets.append(gauss_2d)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # shape => (56,), (1,28,28)
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(0),
        )


##############################################################################
# 3. CNN MODEL
##############################################################################

class CNNDecoderWithActivations(nn.Module):
    def __init__(self, input_size=56, hidden_size=64, n_hidden_layers=4):
        super().__init__()
        self.activations = {}

        layers = []
        layers.append(
            nn.ConvTranspose2d(input_size, hidden_size,
                               kernel_size=7, stride=1, padding=0)
        )
        layers.append(nn.ReLU())
        current_size = 7

        upsample_needed = max(0, int(np.log2(28/current_size)))

        for i in range(n_hidden_layers):
            if current_size < 28 and upsample_needed > 0:
                layers.append(
                    nn.ConvTranspose2d(
                        hidden_size, hidden_size, kernel_size=4, stride=2, padding=1)
                )
                current_size *= 2
                upsample_needed -= 1
            else:
                layers.append(
                    nn.ConvTranspose2d(
                        hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
                )
            layers.append(nn.ReLU())

        layers.append(
            nn.ConvTranspose2d(hidden_size, 1, kernel_size=3,
                               stride=1, padding=1)
        )

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        self.activations = {}
        x = x.unsqueeze(-1).unsqueeze(-1)  # => (B,56,1,1)
        for idx, layer in enumerate(self.decoder):
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                self.activations[f"layer_{idx}"] = x.detach().clone()
        return x


##############################################################################
# 4. TRAINING
##############################################################################

def train_model(model, dataloader, num_epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    crit = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        run_loss = 0
        for inp, tgt in dataloader:
            inp, tgt = inp.to(device), tgt.to(device)
            opt.zero_grad()
            out = model(inp)
            loss = crit(out, tgt)
            loss.backward()
            opt.step()
            run_loss += loss.item()
        avg_loss = run_loss/len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss={avg_loss:.4f}")
    return model

##############################################################################
# 5. AVERAGE X, AVERAGE Y
##############################################################################


def compute_average_xy(dset):
    sum_x = np.zeros(28, dtype=np.float32)
    sum_y = np.zeros(28, dtype=np.float32)
    N = len(dset)
    for i in range(N):
        inp_56, _ = dset[i]
        x_28 = inp_56[:28].numpy()
        y_28 = inp_56[28:].numpy()
        sum_x += x_28
        sum_y += y_28
    avg_x = sum_x / float(N)
    avg_y = sum_y / float(N)
    return avg_x, avg_y

##############################################################################
# 6. PARTIAL UTILS
##############################################################################


def xsuper_56(xval, avg_y, grid_size=28, sigma=1.0):
    gx = np.exp(-((np.arange(grid_size)-xval)**2) /
                (2*sigma*sigma)).astype(np.float32)
    return np.concatenate([gx, avg_y]).astype(np.float32)


def ysuper_56(yval, avg_x, grid_size=28, sigma=1.0):
    gy = np.exp(-((np.arange(grid_size)-yval)**2) /
                (2*sigma*sigma)).astype(np.float32)
    return np.concatenate([avg_x, gy]).astype(np.float32)


def bump_56_xy(xval, yval, grid_size=28, sigma=1.0):
    gx = np.exp(-((np.arange(grid_size)-xval)**2)/(2*sigma*sigma))
    gy = np.exp(-((np.arange(grid_size)-yval)**2)/(2*sigma*sigma))
    return np.concatenate([gx, gy]).astype(np.float32)

##############################################################################
# 7. BOUNDING BOX
##############################################################################


def add_ood_box(ax, lb, ub, color='red'):
    rect = patches.Rectangle(
        (lb, lb), (ub-lb), (ub-lb),
        fill=False, edgecolor=color, linestyle='dashed', linewidth=2
    )
    ax.add_patch(rect)

##############################################################################
# 8. OOD COORDINATES
##############################################################################


def get_ood_coords(lb, ub, n_samples=4):
    """
    Return n_samples random (x,y) from the region [lb..ub].
    """
    coords = []
    for _ in range(n_samples):
        x = np.random.randint(lb, ub+1)
        y = np.random.randint(lb, ub+1)
        coords.append((x, y))
    return coords

##############################################################################
# 9. VISUALIZATION: OOD ONLY
##############################################################################


def visualize_ood_xsuper_ysuper(
    model,
    coords_list,
    avg_x, avg_y,
    lb=10, ub=18,
    grid_size=28, sigma=1.0,
    out_dir="vis_xsuper_ysuper_ood"
):
    """
    For each OOD (x,y) in coords_list:
      - xsuper => [gaussian_x, avg_y]
      - ysuper => [avg_x, gaussian_y]
      - also real xy => [gaussian_x, gaussian_y]
      - we do model(xsuper), model(ysuper), sum_ofOutputs_norm, model(xy)
      - 4 subplots
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    with torch.no_grad():
        for (x_val, y_val) in coords_list:
            # build partial
            xsuper_arr = xsuper_56(x_val, avg_y, grid_size, sigma)
            ysuper_arr = ysuper_56(y_val, avg_x, grid_size, sigma)
            xy_arr = bump_56_xy(x_val, y_val, grid_size, sigma)

            xsuper_t = torch.tensor(xsuper_arr, device=device).unsqueeze(0)
            ysuper_t = torch.tensor(ysuper_arr, device=device).unsqueeze(0)
            xy_t = torch.tensor(xy_arr,     device=device).unsqueeze(0)

            out_xs = model(xsuper_t).cpu().numpy()[0, 0]
            out_ys = model(ysuper_t).cpu().numpy()[0, 0]
            out_xy = model(xy_t).cpu().numpy()[0, 0]

            sum_ofOutputs = out_xs + out_ys
            mxv = sum_ofOutputs.max()
            if mxv > 1e-12:
                sum_norm = sum_ofOutputs / mxv
            else:
                sum_norm = sum_ofOutputs

            fig, axes = plt.subplots(1, 4, figsize=(14, 4))
            # for ax in axes:
            #     ax.axis('off')

            def show_img(ax, data_2d, title, coordinates, lb=6, ub=22, x_toggle=True, y_toggle=True):
                """
                Displays 'data_2d' in grayscale with no axis ticks but keeps
                a visible border around the subplot. Also draws a dashed cross
                and an OOD box.
                """
                # Show the image
                im_ = ax.imshow(data_2d, cmap='gray', origin='lower')

                # Highlight the chosen coordinates with a red dot
                ax.scatter(*coordinates, color='red', s=10)

                # Optional crosshair lines
                if x_toggle:
                    ax.axvline(x=coordinates[0], color='red',
                               linestyle='dashed', linewidth=1)
                if y_toggle:
                    ax.axhline(y=coordinates[1], color='red',
                               linestyle='dashed', linewidth=1)

                # OOD bounding box
                add_ood_box(ax, lb, ub, color='red')

                # Optionally set a title
                # ax.set_title(title, fontsize=12)

                # Remove all tick marks and labels
                ax.set_xticks([])
                ax.set_yticks([])

                # Ensure each spine is visible and set a desired line width/color
                for spine in ax.spines.values():
                    # If spines were previously hidden
                    spine.set_visible(True)
                    spine.set_linewidth(1.0)
                    # or another color if you prefer
                    spine.set_edgecolor('black')

            coordinates = (x_val, y_val)
            show_img(axes[0], out_xs,
                     "model($x^{\\text{super}}$)", coordinates,
                     y_toggle=False)
            show_img(axes[1], out_ys,
                     "model($y^{\\text{super}}$)", coordinates,
                     x_toggle=False)
            show_img(axes[2], sum_norm,
                     "model($x^{\\text{super}}$) + model($y^{\\text{super}}$)",
                     coordinates)
            show_img(axes[3], out_xy,
                     "model($x,y$)", coordinates)

            # fig.suptitle(
            #     f"OOD partial input: (x={x_val}, y={y_val})", fontsize=14)
            out_name = f"ood_x{x_val}_y{y_val}.pdf"
            out_path = os.path.join(out_dir, out_name)
            plt.savefig(out_path, bbox_inches='tight')
            plt.close()
            print(f"[visualize_ood_xsuper_ysuper] => saved {out_path}")

##############################################################################
# 10. MAIN
##############################################################################


def main():
    # 1) Build train dataset => skip region [6..22]
    train_ds = GaussianBumpDatasetSquare(
        num_samples=20000,
        grid_size=28,
        sigma=1.0,
        holdout_center=True,
        only_holdout=False,
        heldout_range=[6, 22]
    )
    train_loader = DataLoader(train_ds, batch_size=32,
                              shuffle=True, num_workers=2)
    model = CNNDecoderWithActivations(
        input_size=56, hidden_size=64, n_hidden_layers=4
    )
    model = train_model(model, train_loader, num_epochs=20, lr=1e-3)

    # 2) compute average_x, average_y
    avg_x, avg_y = compute_average_xy(train_ds)
    print("[main] avg_x shape:", avg_x.shape, "avg_y shape:", avg_y.shape)

    # 3) pick OOD coords from region [10..18]
    lb, ub = 10, 18
    coords_ood = get_ood_coords(lb, ub, n_samples=4)
    print("Coordinates for OOD samples:", coords_ood)

    # 4) visualize partial inputs on OOD coords
    visualize_ood_xsuper_ysuper(
        model, coords_ood, avg_x, avg_y,
        lb=6, ub=22,  # bounding box
        grid_size=28, sigma=1.0,
        out_dir="vis_xsuper_ysuper_ood"
    )


if __name__ == "__main__":
    main()
