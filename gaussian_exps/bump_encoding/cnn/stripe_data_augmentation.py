#!/usr/bin/env python3

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


##############################################################################
# 1. DATASET DEFINITION
##############################################################################

class StripedGaussianDataset(Dataset):
    """
    Here we simulate a 'larger' canvas dimension for x,y, e.g. 28+12=40,
    but produce final images of shape 28x28 for the network output.

    We'll hold out region [6..22]^2 in the final 28x28 space as OOD.
    The coordinate input dimension is 80 => something like [gauss_x(40), gauss_y(40)].
    Then the model must produce a 28x28 image.

    'num_samples' => how many examples to sample
    'canvas_size' => e.g. 40
    'final_size'  => 28
    'heldout_range' => [6,22] is the OOD bounding box in the final 28×28 output space
    """

    def __init__(
        self,
        num_samples=2000,
        canvas_size=40,
        final_size=28,
        sigma=1.0,
        holdout_center=True,
        only_holdout=False,
        heldout_range=[6, 22]
    ):
        super().__init__()
        self.num_samples = num_samples
        self.canvas_size = canvas_size
        self.final_size = final_size  # the final image crop size (28)
        self.sigma = sigma
        self.holdout_center = holdout_center
        self.only_holdout = only_holdout
        self.lb, self.ub = heldout_range

        self.data = []  # each => 80-dim = [gauss_x(40), gauss_y(40)]
        self.targets = []  # each => shape(1,28,28)
        self._generate()

    def _generate(self):
        top = (self.canvas_size - self.final_size)//2  # e.g. 6
        bottom = top + self.final_size                    # e.g. 34

        raw_lb = top + self.lb  # e.g. 6+6=12
        raw_ub = top + self.ub  # e.g. 6+22=28

        attempts = 0
        max_attempts = self.num_samples * 5
        count = 0
        while count < self.num_samples and attempts < max_attempts:
            xval = np.random.randint(0, self.canvas_size)
            yval = np.random.randint(0, self.canvas_size)
            # check if OOD or ID
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
                # fallback => sample anywhere
                self._append_example(xval, yval)
                count += 1
            attempts += 1

        print(
            f"Generated dataset with {len(self.data)} samples. OOD region => raw [{raw_lb}..{raw_ub}]."
        )

    def _append_example(self, xval, yval):
        # build 1D gaussians of length=canvas_size
        gx = np.exp(-((np.arange(self.canvas_size)-xval)**2) /
                    (2*self.sigma*self.sigma))
        gy = np.exp(-((np.arange(self.canvas_size)-yval)**2) /
                    (2*self.sigma*self.sigma))
        # concat => shape(2*canvas_size=80)
        inp_80 = np.concatenate([gx, gy], axis=0).astype(np.float32)

        # build 2D => shape(canvas_size, canvas_size)
        xv, yv = np.meshgrid(np.arange(self.canvas_size),
                             np.arange(self.canvas_size))
        gauss_2d = 1.0 - np.exp(
            -(((xv - xval)**2 + (yv - yval)**2)/(2*self.sigma*self.sigma))
        )
        # Crop => final_size => row col in [top.. top+final_size]
        top = (self.canvas_size - self.final_size)//2
        bottom = top + self.final_size
        cropped_2d = gauss_2d[top:bottom, top:bottom]  # shape(28,28)

        self.data.append(inp_80)
        self.targets.append(cropped_2d.astype(np.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_80 = torch.tensor(self.data[idx], dtype=torch.float32)  # shape (80,)
        y_2d = torch.tensor(
            self.targets[idx], dtype=torch.float32).unsqueeze(0)  # (1,28,28)
        return x_80, y_2d


##############################################################################
# 2. MODEL DEFINITION: input_size=80, final => 28x28
##############################################################################

class CNNDecoderStripes(nn.Module):
    """
    Takes (B,80)->(B,80,1,1). Possibly 1 or more transposed conv => (B,1,28,28).
    We'll set a depth=8, as requested.
    """

    def __init__(self, input_size=80, hidden_size=64, depth=8):
        super().__init__()
        self.activations = []

        layers = []
        # from (80,1,1) => (hidden_size,7,7)
        layers.append(
            nn.ConvTranspose2d(input_size, hidden_size,
                               kernel_size=7, stride=1, padding=0)
        )
        layers.append(nn.ReLU())
        current_size = 7

        # e.g. log2(28/7)=2 if current_size=7
        upsample_needed = max(0, int(np.log2(28 / current_size)))

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

        # final => (1,28,28)
        layers.append(
            nn.ConvTranspose2d(hidden_size, 1, kernel_size=3,
                               stride=1, padding=1)
        )

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        self.activations = []
        # x => shape (B,80) => (B,80,1,1)
        x = x.unsqueeze(-1).unsqueeze(-1)
        for layer in self.decoder:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                self.activations.append(x.clone().detach())
        return x


##############################################################################
# 3. TRAIN & EVAL
##############################################################################

def train_model(model, dataloader, num_epochs=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    crit = nn.MSELoss()

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
        print(f"Epoch {epoch+1}/{num_epochs}, trainLoss={avg_loss:.4f}")
    return model


def generate_mse_contour(model, out_dir, canvas_size=40, final_size=28,
                         lb=6, ub=22):
    """
    We'll do a 28×28 iteration => for each (x,y) in [0..27]^2, build input => shape(80,).
    Then do a forward pass => measure MSE vs. GT => store in diff_2d.
    We'll do a contour plot with a dashed bounding box => [lb..ub]^2.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    diff_2d = np.zeros((28, 28), dtype=np.float32)
    top = (canvas_size - final_size)//2  # 12 if canvas=40, final=28

    # Instead of looping single pass each, let's do it in a single batch for speed:
    coords_list = []
    for yy in range(28):
        for xx in range(28):
            raw_x = xx + top
            raw_y = yy + top
            coords_list.append((xx, yy, raw_x, raw_y))
    # shape => (784,80)
    inputs_list = []
    xv, yv = np.meshgrid(np.arange(canvas_size), np.arange(canvas_size))
    for (xx, yy, rx, ry) in coords_list:
        gx = np.exp(-((np.arange(canvas_size)-rx)**2)/(2*1.0**2))
        gy = np.exp(-((np.arange(canvas_size)-ry)**2)/(2*1.0**2))
        in80 = np.concatenate([gx, gy], axis=0).astype(np.float32)
        inputs_list.append(in80)
    inp_batch = torch.tensor(
        inputs_list, dtype=torch.float32, device=device)  # shape(784,80)

    with torch.no_grad():
        preds = model(inp_batch)  # shape(784,1,28,28)

    # compute GT => measure MSE
    for idx in range(len(coords_list)):
        xx, yy, rx, ry = coords_list[idx]
        # build GT
        big_2d = 1.0 - np.exp(
            -(((xv-rx)**2 + (yv-ry)**2)/(2*1.0**2))
        )
        # crop => shape(28,28)
        big_crop = big_2d[top:top+final_size, top:top+final_size]
        out_np = preds[idx, 0].cpu().numpy()
        mse_val = np.mean((big_crop - out_np)**2)
        diff_2d[yy, xx] = mse_val

    # Plot contour
    fig, ax = plt.subplots(figsize=(6, 5))
    xx_, yy_ = np.meshgrid(np.arange(28), np.arange(28))
    ctf = ax.contourf(xx_, yy_, diff_2d, levels=30, cmap='viridis')
    fig.colorbar(ctf, ax=ax, label="MSE")
    ax.set_title("MSE contour vs. (x,y) in [0..27]^2")

    # bounding box => [lb..ub]^2
    rect = patches.Rectangle(
        (lb, lb), (ub-lb), (ub-lb),
        fill=False, edgecolor='red', linestyle='dashed', linewidth=2
    )
    ax.add_patch(rect)

    ax.set_xlabel("x coord")
    ax.set_ylabel("y coord")
    out_fig = os.path.join(out_dir, "mse_contour.png")
    plt.savefig(out_fig, bbox_inches='tight')
    plt.close()
    print(f"[generate_mse_contour] => saved {out_fig}")


def visualize_few_ood_samples(model, out_dir, n_samples=9,
                              lb=6, ub=22, canvas_size=40, final_size=28):
    """
    We'll pick 'n_samples' random coords in [lb..ub], do a single batched forward pass
    => produce a figure with ~ sqrt(n_samples) x sqrt(n_samples) subplots,
    each with model's output + bounding box + red dot at (x,y).
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    top = (canvas_size - final_size)//2
    coords_ood = []
    for _ in range(n_samples):
        x_ = np.random.randint(lb, ub+1)
        y_ = np.random.randint(lb, ub+1)
        raw_x = x_ + top
        raw_y = y_ + top
        coords_ood.append((x_, y_, raw_x, raw_y))

    # Build batch => shape(n_samples,80)
    input_list = []
    for (x_, y_, rx, ry) in coords_ood:
        gx = np.exp(-((np.arange(canvas_size)-rx)**2)/(2*1.0**2))
        gy = np.exp(-((np.arange(canvas_size)-ry)**2)/(2*1.0**2))
        in80 = np.concatenate([gx, gy], axis=0).astype(np.float32)
        input_list.append(in80)
    inp_t = torch.tensor(input_list, device=device)

    with torch.no_grad():
        out_batch = model(inp_t)  # shape(n_samples,1,28,28)

    # Plot in sqrt(n_samples) x sqrt(n_samples) grid
    n_rows = int(np.ceil(np.sqrt(n_samples)))
    n_cols = n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows),
                             sharex=True, sharey=True)
    axes = axes.flatten() if n_rows > 1 else [axes]

    for i in range(n_samples):
        x_ = coords_ood[i][0]
        y_ = coords_ood[i][1]
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

        # red dot => (x_,y_)
        ax.plot([x_], [y_], marker='o', markersize=2, color='red')
        ax.set_title(f"({x_},{y_}) OOD")

    for j in range(n_samples, len(axes)):
        axes[j].axis('off')

    out_fig = os.path.join(out_dir, "few_ood_samples.png")
    plt.savefig(out_fig, bbox_inches='tight')
    plt.close()
    print(f"[visualize_few_ood_samples] => saved {out_fig}")


##############################################################################
# 4. MAIN
##############################################################################

def main():
    out_dir = "cnn_stripes"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Build dataset => we do ID => skip region [6..22]^2 in the final 28x28
    train_ds = StripedGaussianDataset(
        num_samples=2000,
        canvas_size=40,   # 28+12
        final_size=28,
        sigma=1.0,
        holdout_center=True,
        only_holdout=False,
        heldout_range=[6, 22]
    )
    train_loader = DataLoader(train_ds, batch_size=32,
                              shuffle=True, num_workers=2)

    # 2) Build model => input_size=80, depth=8
    model = CNNDecoderStripes(input_size=80, hidden_size=64, depth=8)

    # 3) Train
    _ = train_model(model, train_loader, num_epochs=20, lr=1e-3)

    # 4) Generate MSE contour => shape(28,28)
    generate_mse_contour(model, out_dir=out_dir, canvas_size=40, final_size=28,
                         lb=6, ub=22)

    # 5) Visualize only a FEW OOD samples in a small grid
    visualize_few_ood_samples(model, out_dir=out_dir,
                              n_samples=9,
                              lb=6, ub=22, canvas_size=40, final_size=28)


if __name__ == "__main__":
    main()
