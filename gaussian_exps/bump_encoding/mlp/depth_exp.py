#!/usr/bin/env python3
# mlp_gaussian_variable_intermediate.py

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


##############################################################################
# 1. DATASET: 56-dim input (two 1D Gaussians), 28×28 output
##############################################################################
class GaussianBumpDatasetSquare(Dataset):
    """
    Each sample:
      - 56-dim input => [gaussian_x(28), gaussian_y(28)]
      - 28×28 target => 2D Gaussian bump at (x,y).

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
                    (x_b >= self.lb) & (x_b <= self.ub)
                    & (y_b >= self.lb) & (y_b <= self.ub)
                )
                xs.extend(x_b[valid])
                ys.extend(y_b[valid])
            xs = np.array(xs[:self.num_samples])
            ys = np.array(ys[:self.num_samples])
        else:
            # Sample anywhere
            xs = np.random.randint(0, self.grid_size, size=self.num_samples)
            ys = np.random.randint(0, self.grid_size, size=self.num_samples)

        for i in range(self.num_samples):
            x_i, y_i = xs[i], ys[i]
            # 1D Gaussians => 28 + 28 = 56
            gx = np.exp(-((np.arange(self.grid_size) - x_i)**2) /
                        (2*self.sigma**2))
            gy = np.exp(-((np.arange(self.grid_size) - y_i)**2) /
                        (2*self.sigma**2))
            bump_56 = np.concatenate([gx, gy], axis=0)

            # 2D Gaussian => shape (28×28)
            gauss_2d = 1.0 - np.exp(
                -(
                    (self.xv - x_i)**2 + (self.yv - y_i)**2
                ) / (2*self.sigma**2)
            )

            self.data.append(bump_56.astype(np.float32))
            self.targets.append(gauss_2d.astype(np.float32))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return (56-d input, 28×28 target) where target is shaped (1,28,28)
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(0),
        )


##############################################################################
# 2. MLP MODEL with variable intermediate layers, storing ReLU activations
##############################################################################
class MLPDecoderWithActivations(nn.Module):
    """
    A variable-depth MLP that:
    - Takes a 56-dim input
    - Has 'n_hidden_layers' fully connected ReLU layers
    - Ends with a linear layer to produce 784 = (28×28) outputs
    - Reshapes final output to (batch,1,28,28)
    - Stores intermediate (post-ReLU) activations in self.activations
      for the same metric-tensor/linear-probe analysis.
    """

    def __init__(self, input_size=56, hidden_size=256, n_hidden_layers=3):
        super().__init__()
        self.activations = []

        layers = []
        in_dim = input_size
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size

        # Final layer => 784 = 28×28
        layers.append(nn.Linear(in_dim, 28*28))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        x shape: (batch, 56)
        We'll manually store ReLU activations in self.activations.
        """
        self.activations = []
        out = x

        for layer in self.model:
            out = layer(out)
            if isinstance(layer, nn.ReLU):
                # shape (batch, hidden_size)
                # Convert to (batch, hidden_size,1,1) to mimic 2D structure
                self.activations.append(
                    out.detach().clone().unsqueeze(-1).unsqueeze(-1))

        # 'out' shape now: (batch, 784)
        out = out.view(out.size(0), 1, 28, 28)  # => (batch,1,28,28)
        return out


##############################################################################
# 3. TRAINING / EVALUATION
##############################################################################
def train_model(model, dataloader, num_epochs=20, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epoch_losses = []
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
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
    return epoch_losses


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
# 4. METRIC TENSOR & Volume Element
##############################################################################
def getg(data_3d):
    """
    data_3d shape: (H, W, D).
    returns g => (H, W, 2, 2).
    """
    H, W, D = data_3d.shape
    g = np.zeros((H, W, 2, 2))

    drow = np.gradient(data_3d, axis=0)  # partial derivative wrt row
    dcol = np.gradient(data_3d, axis=1)  # partial derivative wrt col

    g[:, :, 0, 0] = np.sum(drow**2, axis=-1)
    g[:, :, 1, 1] = np.sum(dcol**2, axis=-1)
    cross = np.sum(drow * dcol, axis=-1)
    g[:, :, 0, 1] = cross
    g[:, :, 1, 0] = cross
    return g


def compute_volume_element(g):
    """
    g shape (H, W, 2, 2).
    dv = sqrt(|det(g)|).
    """
    g_11 = g[:, :, 0, 0]
    g_12 = g[:, :, 0, 1]
    g_21 = g[:, :, 1, 0]
    g_22 = g[:, :, 1, 1]
    detg = g_11 * g_22 - g_12 * g_21
    dv = np.sqrt(np.abs(detg))
    return dv


##############################################################################
# 4b. FACTORIZATION METRIC
##############################################################################
def compute_factorization_metric(data_3d):
    """
    data_3d: shape (28,28,D)
    factorization => 1 - (mean(|g01|) / mean(|g00|+|g11|))
    """
    g_mat = getg(data_3d)
    g00 = g_mat[..., 0, 0]
    g11 = g_mat[..., 1, 1]
    g01 = g_mat[..., 0, 1]

    mean_diag = np.mean(np.abs(g00) + np.abs(g11))
    mean_off = np.mean(np.abs(g01))

    eps = 1e-9
    ratio_off_diag = mean_off / (mean_diag + eps)
    return 1.0 - ratio_off_diag


##############################################################################
# 4c. VOLUME-METRIC PLOT PER LAYER
##############################################################################
def plot_volume_element_layers(activations_list, depth, fig_dir):
    """
    For each layer in `activations_list`, compute the volume element dv, take
    log10(dv), and display it in a subplot. Each layer gets its own colorbar
    with mean±3σ scaling. Only the last colorbar is labeled with '$\\log_{10}(dv)$'.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import os

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Possibly you have these imported globally. If not:
    # from your_code import getg, compute_volume_element

    n_layers = len(activations_list)
    fig, axes = plt.subplots(
        1, n_layers, figsize=(4*n_layers, 4), squeeze=False)

    # If there's only one layer, axes is shape (1,1). Let's unify indexing:
    if n_layers == 1:
        axes = [axes[0, 0]]
    else:
        axes = axes[0]  # shape => (n_layers,)

    for layer_idx, act_tensor in enumerate(activations_list):
        ax = axes[layer_idx]

        # act_tensor shape => (N, C, H, W). E.g., for 28x28 => (784, C, H, W)
        act_cpu = act_tensor.cpu().numpy()
        N_, C_, H_, W_ = act_cpu.shape

        # Flatten each sample => (N, C*H*W)
        act_flat = act_cpu.reshape(N_, C_*H_*W_)

        # Suppose your grid is 28x28
        grid_size = 28  # Adjust if needed
        data_reshape = act_flat.reshape(grid_size, grid_size, -1)

        # compute dv => log10(dv)
        g_mat = getg(data_reshape)
        dv = compute_volume_element(g_mat)
        data_log = np.log10(dv + 1e-12)

        # mean±3σ
        avgg = np.mean(data_log)
        stdd = np.std(data_log)
        vmin_layer = avgg - 3 * stdd
        vmax_layer = avgg + 3 * stdd

        # Show the image
        im = ax.imshow(
            data_log,
            origin='lower',
            cmap='Reds',
            vmin=vmin_layer,
            vmax=vmax_layer
        )
        ax.set_xticks([])
        ax.set_yticks([])

        # OOD bounding box => e.g. (6,6) size 16x16
        rect = patches.Rectangle(
            (6, 6), 16, 16,
            fill=False, edgecolor='white',
            linestyle='--', linewidth=2
        )
        ax.add_patch(rect)

        ax.set_title(f"Layer {layer_idx}", fontsize=16)

        # Create a dedicated colorbar axes for each subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        # Plot color bar
        cb = fig.colorbar(im, cax=cax)
        # Only label the LAST color bar
        if layer_idx < n_layers - 1:
            cb.set_label("")  # remove label on all but last
        else:
            cb.set_label(r"$\log_{10}(dv)$", fontsize=16)

    out_vol_path = os.path.join(fig_dir, f"volume_element_depth_{depth}.pdf")
    plt.tight_layout()
    plt.savefig(out_vol_path, bbox_inches='tight')
    plt.close()
    print(f"Saved volume-element figure => {out_vol_path}")


##############################################################################
# 5. LINEAR PROBE for R^2 wrt x,y
##############################################################################
def linear_probe_r2(activation_tensor, coords_xy):
    """
    activation_tensor: shape (N, C, H, W).
    coords_xy: shape (N, 2).  coords_xy[i] = (x,y).
    Returns (r2_x, r2_y).
    """
    N, C, H, W = activation_tensor.shape
    # Flatten => (N, C*H*W)
    act_flat = activation_tensor.reshape(N, C * H * W)

    y_x = coords_xy[:, 0]
    y_y = coords_xy[:, 1]

    regx = LinearRegression().fit(act_flat, y_x)
    pred_x = regx.predict(act_flat)
    r2_x = r2_score(y_x, pred_x)

    regy = LinearRegression().fit(act_flat, y_y)
    pred_y = regy.predict(act_flat)
    r2_y = r2_score(y_y, pred_y)

    return (r2_x, r2_y)


##############################################################################
# 6. MAIN EXPERIMENT
##############################################################################
def run_experiment_mlp_gaussian():
    # Increase font sizes globally for consistency
    plt.rcParams.update({
        'font.size': 14,         # base font size
        'axes.labelsize': 16,    # x/y labels
        'axes.titlesize': 18,    # subplot title size
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14
    })

    fig_dir = "fig_mlp_gaussian_pdf"
    os.makedirs(fig_dir, exist_ok=True)

    # 1) in-dist dataset => excludes [6..22]^2
    train_ds = GaussianBumpDatasetSquare(
        num_samples=2000,
        grid_size=28,
        sigma=1.0,
        holdout_center=True,
        only_holdout=False,
        heldout_range=[6, 22]
    )
    train_loader = DataLoader(train_ds, batch_size=32,
                              shuffle=True, num_workers=2)

    # 2) OOD dataset => only [6..22]^2
    ood_ds = GaussianBumpDatasetSquare(
        num_samples=500,
        grid_size=28,
        sigma=1.0,
        holdout_center=False,
        only_holdout=True,
        heldout_range=[6, 22]
    )
    ood_loader = DataLoader(ood_ds, batch_size=32,
                            shuffle=False, num_workers=2)

    # We'll try multiple depths
    depths = [3, 4, 5, 6, 7]
    ood_losses = []

    # We'll store factorization & linear-probe results
    factorization_results = {}
    linear_probe_results = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ############################################################################
    # (A*) We'll pick 4 random OOD samples from a single batch (used by all nets)
    ############################################################################
    ood_inputs_full, ood_targets_full = next(iter(ood_loader))
    batch_size = ood_inputs_full.shape[0]
    rand_indices = np.random.choice(batch_size, size=4, replace=False)

    chosen_inputs = ood_inputs_full[rand_indices].clone().to(
        device)   # shape(4,56)
    # shape(4,1,28,28)
    chosen_targets = ood_targets_full[rand_indices].clone()

    # Decode x,y from the 56-d input by argmax
    chosen_coords = []
    for i in range(4):
        bump_56 = chosen_inputs[i].cpu().numpy()
        x_bump = bump_56[:28]
        y_bump = bump_56[28:]
        x_i = np.argmax(x_bump)
        y_i = np.argmax(y_bump)
        chosen_coords.append((x_i, y_i))

    # We'll store predictions from each depth in all_ood_preds
    all_ood_preds = {}

    for depth in depths:
        print("=====================================================")
        print(f"Training MLP with n_hidden_layers = {depth}")
        model = MLPDecoderWithActivations(
            input_size=56,
            hidden_size=256,
            n_hidden_layers=depth
        )

        # Train
        _ = train_model(model, train_loader, num_epochs=20, learning_rate=1e-3)

        # Evaluate OOD
        loss_ood = eval_ood_mse(model, ood_loader)
        print(f"Depth={depth}, OOD MSE={loss_ood:.4f}")
        ood_losses.append(loss_ood)

        # Predictions => same 4 OOD samples
        model.eval().to(device)
        with torch.no_grad():
            preds_4 = model(chosen_inputs)
        all_ood_preds[depth] = preds_4.cpu()

        ######################################################################
        # Factorization & Linear Probe: pass 28×28 grid => model.activations
        ######################################################################
        coords_1d = np.arange(28)
        xv, yv = np.meshgrid(coords_1d, coords_1d)
        grid_points = np.stack([xv.ravel(), yv.ravel()], axis=-1)  # (784,2)

        # Build 56-d input for each (x,y)
        all_bumps = []
        for (xx, yy) in grid_points:
            gx = np.exp(-((np.arange(28) - xx)**2) / (2 * 1.0**2))
            gy = np.exp(-((np.arange(28) - yy)**2) / (2 * 1.0**2))
            bumps_56 = np.concatenate([gx, gy], axis=0)
            all_bumps.append(bumps_56)
        all_bumps = np.array(all_bumps, dtype=np.float32)  # shape (784,56)

        with torch.no_grad():
            inputs_torch = torch.tensor(all_bumps, device=device)
            _ = model(inputs_torch)  # forward pass => fill model.activations

        activations_list = model.activations

        # For each layer, compute factorization metric + linear probe R^2
        layer_factor_scores = []
        layer_x_r2 = []
        layer_y_r2 = []

        for a_idx, act_tensor in enumerate(activations_list):
            # shape => (784, hidden_size,1,1)
            act_cpu = act_tensor.cpu().numpy()
            N, C, _, _ = act_cpu.shape
            # Flatten => (784, C)
            act_flat = act_cpu.reshape(N, C)
            # => (28,28,C)
            data_reshape = act_flat.reshape(28, 28, -1)

            # Factorization
            fact_score = compute_factorization_metric(data_reshape)
            layer_factor_scores.append(fact_score)

            # Linear probe
            r2x, r2y = linear_probe_r2(
                torch.from_numpy(act_cpu),  # shape (N,C,1,1)
                grid_points
            )
            layer_x_r2.append(r2x)
            layer_y_r2.append(r2y)

        factorization_results[depth] = layer_factor_scores
        linear_probe_results[depth] = (layer_x_r2, layer_y_r2)

        # Also plot the volume-element contour for each layer
        plot_volume_element_layers(activations_list, depth, fig_dir)

    ############################################################################
    # (A**) single figure with 1 row of "GT" + (len(depths)) rows of predictions
    # We'll have (len(depths)+1) x 4 subplots.
    ############################################################################
    num_rows = len(depths) + 1
    plt.figure(figsize=(8, 3 * num_rows))

    # Top row => GT
    for col in range(4):
        gt_img = chosen_targets[col, 0].numpy()
        x_i, y_i = chosen_coords[col]
        ax_top = plt.subplot(num_rows, 4, col+1)
        ax_top.imshow(gt_img, cmap='gray', origin='lower')
        rect_gt = patches.Rectangle((6, 6), 16, 16,
                                    fill=False, edgecolor='red',
                                    linestyle='dashed', linewidth=2)
        ax_top.add_patch(rect_gt)
        ax_top.plot(x_i, y_i, color='red', marker='x',
                    markersize=6, markeredgewidth=2)
        if col == 0:
            ax_top.set_ylabel("GT", fontsize=14)
        ax_top.set_title(f"Sample {col+1}")

    # next rows => each depth
    for row_i, depth in enumerate(depths, start=1):
        preds_4 = all_ood_preds[depth]
        for col in range(4):
            pred_img = preds_4[col, 0].numpy()
            x_i, y_i = chosen_coords[col]
            ax_pred = plt.subplot(num_rows, 4, row_i * 4 + (col + 1))
            ax_pred.imshow(pred_img, cmap='gray', origin='lower')
            rect_pd = patches.Rectangle((6, 6), 16, 16,
                                        fill=False, edgecolor='red',
                                        linestyle='dashed', linewidth=2)
            ax_pred.add_patch(rect_pd)
            ax_pred.plot(x_i, y_i, color='red', marker='x',
                         markersize=6, markeredgewidth=2)
            if col == 0:
                ax_pred.set_ylabel(f"Depth={depth}", fontsize=14)

    out_ood_fig = os.path.join(fig_dir, "ood_all_depths.pdf")
    plt.savefig(out_ood_fig, bbox_inches='tight')
    plt.close()

    ##########################################################################
    # (C) OOD MSE vs Depth
    ##########################################################################
    plt.figure()
    plt.plot(depths, ood_losses, marker='o')
    plt.xlabel("MLP Depth (# of hidden layers)")
    plt.ylabel("OOD MSE")
    plt.title("OOD MSE vs MLP Depth")
    out_plot = os.path.join(fig_dir, "ood_vs_mlp_depth.pdf")
    plt.savefig(out_plot, bbox_inches='tight')
    plt.close()

    ##########################################################################
    # (D) FACTORIZATION METRIC vs. LAYER, for each DEPTH
    ##########################################################################
    plt.figure(figsize=(6, 4))
    for depth in depths:
        scores = factorization_results[depth]
        # Insert factor=1 at x=0 => "Input" (just a visual placeholder)
        extended_scores = [1.0] + list(scores)
        x_vals = range(len(extended_scores))
        plt.plot(x_vals, extended_scores, marker='o', label=f"Depth={depth}")

    max_layers = max(len(v) for v in factorization_results.values())
    x_ticks = range(max_layers+1)
    x_labels = ["Input"] + [str(i) for i in range(max_layers)]
    plt.xticks(ticks=x_ticks, labels=x_labels)
    plt.xlabel("Layer Index")
    plt.ylabel("Factorization Metric")
    plt.grid(True)

    plt.legend()
    fact_fig = os.path.join(fig_dir, "factorization_vs_layer.pdf")
    plt.savefig(fact_fig, bbox_inches='tight')
    plt.close()

    ##########################################################################
    # (E) LINEAR PROBE R^2 => separate figure for each depth
    ##########################################################################
    for depth in depths:
        layer_x_r2, layer_y_r2 = linear_probe_results[depth]
        layers = len(layer_x_r2)
        x_vals = range(layers)
        plt.figure(figsize=(6, 3))
        plt.plot(x_vals, layer_x_r2, marker='o', label='$R^2$ wrt x')
        plt.plot(x_vals, layer_y_r2, marker='s', label='$R^2$ wrt y')
        # plt.axhline(y=1, color='gray', linestyle='--', linewidth=1)
        plt.xlabel("Layer Index")
        plt.ylabel("$R^2$ Score")
        plt.grid(True)

        plt.legend()
        # plt.title(f"Linear probe (MLP Depth={depth})")
        out_r2_path = os.path.join(fig_dir, f"probe_r2_depth_{depth}.pdf")
        plt.savefig(out_r2_path, bbox_inches='tight')
        plt.close()

    print("Experiment complete. Figures saved in:", fig_dir)


##############################################################################
# 7. MAIN
##############################################################################
if __name__ == "__main__":
    run_experiment_mlp_gaussian()
