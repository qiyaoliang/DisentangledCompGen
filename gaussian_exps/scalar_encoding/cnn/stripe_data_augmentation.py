#!/usr/bin/env python3

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
# 1. DATASET
##############################################################################

class StripedGaussianDataset2D(Dataset):
    """
    Two-scalar input: (x,y) in a 'canvas_size' domain,
    We produce final 28x28 images by cropping from a bigger canvas.
    We hold out [lb..ub]^2 in the final coords if holdout_center=True.
    """

    def __init__(
        self,
        num_samples=2000,
        canvas_size=40,  # bigger than 28
        final_size=28,
        sigma=1.0,
        holdout_center=True,
        only_holdout=False,
        heldout_range=[6, 22],
    ):
        super().__init__()
        self.num_samples = num_samples
        self.canvas_size = canvas_size
        self.final_size = final_size
        self.sigma = sigma
        self.holdout_center = holdout_center
        self.only_holdout = only_holdout
        self.lb, self.ub = heldout_range

        self.data = []  # shape(2,) for each sample
        self.targets = []  # shape(28,28) for each sample
        self._generate()

    def _generate(self):
        top = (self.canvas_size - self.final_size)//2
        bottom = top + self.final_size

        raw_lb = top + self.lb  # e.g. 6+6=12
        raw_ub = top + self.ub  # e.g. 6+22=28

        attempts = 0
        max_attempts = self.num_samples*4
        count = 0
        while count < self.num_samples and attempts < max_attempts:
            xval = np.random.randint(0, self.canvas_size)
            yval = np.random.randint(0, self.canvas_size)
            is_in_ood_box = (xval >= raw_lb and xval <= raw_ub and
                             yval >= raw_lb and yval <= raw_ub)
            if self.only_holdout and not self.holdout_center:
                # OOD => only inside
                if is_in_ood_box:
                    self._append_example(xval, yval)
                    count += 1
            elif self.holdout_center and not self.only_holdout:
                # ID => skip OOD
                if not is_in_ood_box:
                    self._append_example(xval, yval)
                    count += 1
            else:
                # sample anywhere
                self._append_example(xval, yval)
                count += 1
            attempts += 1

        print(
            f"Generated dataset => {len(self.data)} samples. OOD raw coords => [{raw_lb}..{raw_ub}]^2")

    def _append_example(self, xval, yval):
        # big 2D => shape(canvas_size, canvas_size)
        xv, yv = np.meshgrid(np.arange(self.canvas_size),
                             np.arange(self.canvas_size))
        big_2d = 1.0 - np.exp(
            -(((xv - xval)**2 + (yv - yval)**2)/(2*self.sigma**2))
        )
        top = (self.canvas_size - self.final_size)//2
        cropped = big_2d[top: top+self.final_size, top: top+self.final_size]

        self.data.append([xval, yval])
        self.targets.append(cropped.astype(np.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # shape(2,)
        x_2 = torch.tensor(self.data[idx], dtype=torch.float32)
        y_2d = torch.tensor(
            self.targets[idx], dtype=torch.float32).unsqueeze(0)  # (1,28,28)
        return x_2, y_2d


##############################################################################
# 2. MODEL
##############################################################################

class CNNDecoderStripesScalar(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, depth=8):
        super().__init__()
        self.activations = []
        layers = []
        # from (2,1,1) => (hidden_size,7,7)
        layers.append(nn.ConvTranspose2d(input_size, hidden_size,
                      kernel_size=7, stride=1, padding=0))
        layers.append(nn.ReLU())
        current_size = 7

        upsample_needed = max(
            0, int(np.log2(28/current_size)))  # e.g. log2(4)=2

        for i in range(depth):
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
        self.activations = []
        x = x.unsqueeze(-1).unsqueeze(-1)  # shape(B,2)->(B,2,1,1)
        for layer in self.decoder:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                self.activations.append(x.clone().detach())
        return x


##############################################################################
# 3. TRAIN
##############################################################################

def train_model(model, loader, num_epochs=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        run_loss = 0
        for inp_xy, tgt_2d in loader:
            inp_xy, tgt_2d = inp_xy.to(device), tgt_2d.to(device)
            opt.zero_grad()
            out_2d = model(inp_xy)
            loss = crit(out_2d, tgt_2d)
            loss.backward()
            opt.step()
            run_loss += loss.item()
        avg_loss = run_loss/len(loader)
        print(f"Epoch {epoch+1}/{num_epochs}, trainLoss={avg_loss:.4f}")
    return model


##############################################################################
# 4. MSE CONTOUR => (28,28)
##############################################################################

def generate_mse_contour(model, out_dir, canvas_size=40, final_size=28, lb=6, ub=22, sigma=1.0):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    diff_2d = np.zeros((28, 28), dtype=np.float32)
    top = (canvas_size-final_size)//2

    # We'll build a single pass => shape(28*28,2)
    coords_list = []
    for y_ in range(28):
        for x_ in range(28):
            raw_x = x_ + top
            raw_y = y_ + top
            coords_list.append([raw_x, raw_y])
    coords_arr = np.array(coords_list, dtype=np.float32)  # shape(784,2)

    # forward pass
    with torch.no_grad():
        inp_t = torch.tensor(coords_arr, device=device)
        preds = model(inp_t)  # shape(784,1,28,28)

    # compute GT => measure MSE
    xv, yv = np.meshgrid(np.arange(canvas_size), np.arange(canvas_size))
    for idx in range(len(coords_arr)):
        rx, ry = coords_arr[idx]
        x_ = idx % 28
        y_ = idx//28
        big_2d = 1.0 - np.exp(
            -(((xv-rx)**2+(yv-ry)**2)/(2*sigma**2))
        )
        # crop => final(28,28)
        big_crop = big_2d[top:top+final_size, top:top+final_size]
        out_np = preds[idx, 0].cpu().numpy()
        mse_val = np.mean((big_crop - out_np)**2)
        diff_2d[y_, x_] = mse_val

    # plot
    fig, ax = plt.subplots(figsize=(6, 5))
    xx_, yy_ = np.meshgrid(np.arange(28), np.arange(28))
    ctf = ax.contourf(xx_, yy_, diff_2d, levels=30, cmap='viridis')
    fig.colorbar(ctf, ax=ax, label="MSE")
    ax.set_title("MSE contour [0..27]^2")
    rect = patches.Rectangle((lb, lb), (ub-lb), (ub-lb),
                             fill=False, edgecolor='red', linestyle='dashed', linewidth=2
                             )
    ax.add_patch(rect)
    out_fig = os.path.join(out_dir, "mse_contour.png")
    plt.savefig(out_fig, bbox_inches='tight')
    plt.close()
    print(f"[generate_mse_contour] => saved {out_fig}")


##############################################################################
# 5. VISUALIZE FEW OOD SAMPLES
##############################################################################

def visualize_few_ood_samples(model, out_dir, n_samples=9,
                              lb=6, ub=22, canvas_size=40, final_size=28, sigma=1.0):
    """
    We'll pick 'n_samples' random coords in [lb..ub], do a single batched forward pass
    => produce a figure with e.g. sqrt(n_samples) x sqrt(n_samples) subplots
       each with the model's output, bounding box, and a red dot at (x_,y_).
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    coords = []
    top = (canvas_size-final_size)//2
    for _ in range(n_samples):
        x_ = np.random.randint(lb, ub+1)
        y_ = np.random.randint(lb, ub+1)
        # convert to raw coords => raw_x= x_+top
        raw_x = x_ + top
        raw_y = y_ + top
        coords.append([x_, y_, raw_x, raw_y])  # store final coords + raw

    # build batch => shape(n_samples, 2)
    coords_batch = np.array([[c[2], c[3]]
                            for c in coords], dtype=np.float32)  # raw_x, raw_y
    with torch.no_grad():
        inp_t = torch.tensor(coords_batch, device=device)
        out_batch = model(inp_t)  # shape(n_samples,1,28,28)

    # We'll do a sqrt layout => e.g. if n_samples=9 => 3x3
    n_rows = int(np.ceil(np.sqrt(n_samples)))
    n_cols = n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(
        3*n_cols, 3*n_rows), sharex=True, sharey=True)
    axes = axes.flatten() if n_rows > 1 else [axes]

    for i in range(n_samples):
        x_ = coords[i][0]
        y_ = coords[i][1]
        out_np = out_batch[i, 0].cpu().numpy()

        ax = axes[i]
        ax.imshow(out_np, cmap='gray', origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])

        # bounding box
        rect = patches.Rectangle(
            (lb, lb), (ub-lb), (ub-lb),
            fill=False, edgecolor='red', linestyle='dashed', linewidth=2
        )
        ax.add_patch(rect)

        # red dot => (x_, y_)
        ax.plot([x_], [y_], marker='o', markersize=2, color='red')
        ax.set_title(f"({x_},{y_}) OOD")

    # if there's leftover axes, hide them
    for j in range(n_samples, len(axes)):
        axes[j].axis('off')

    out_fig = os.path.join(out_dir, "few_ood_samples.png")
    plt.savefig(out_fig, bbox_inches='tight')
    plt.close()
    print(f"[visualize_few_ood_samples] => saved {out_fig}")


##############################################################################
# 6. MAIN
##############################################################################

def main():
    out_dir = "cnn_stripes_scalar"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Build dataset => exclude region [6..22]^2
    train_ds = StripedGaussianDataset2D(
        num_samples=2000,
        canvas_size=40,
        final_size=28,
        sigma=1.0,
        holdout_center=True,
        only_holdout=False,
        heldout_range=[6, 22]
    )
    train_loader = DataLoader(train_ds, batch_size=32,
                              shuffle=True, num_workers=2)

    # 2) Build model => input_size=2, depth=8
    model = CNNDecoderStripesScalar(input_size=2, hidden_size=128, depth=8)
    # 3) Train
    model = train_model(model, train_loader, num_epochs=30, lr=1e-3)

    # 4) Generate MSE contour => 28×28
    generate_mse_contour(model, out_dir=out_dir,
                         canvas_size=40, final_size=28,
                         lb=6, ub=22, sigma=1.0)

    # 5) Visualize a FEW OOD samples => skip the big 28×28 grid
    visualize_few_ood_samples(model, out_dir=out_dir,
                              n_samples=9,
                              lb=6, ub=22, canvas_size=40, final_size=28, sigma=1.0)


if __name__ == "__main__":
    main()
