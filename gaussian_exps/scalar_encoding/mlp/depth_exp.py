#!/usr/bin/env python3
# mlp_depth_experiment.py

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
# 1. DATASET DEFINITIONS
##############################################################################


class GaussianBumpDatasetSquare(Dataset):
    """
    Creates a dataset of 2D Gaussian bump images on a grid, with a 'square' region
    held out if holdout_center=True. If only_holdout=True, it only generates points
    within that region.
    Input => shape(2,) = (x,y).
    Output => shape(1,28,28).
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

                # Mask out points inside the holdout region
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
            # Construct a 28x28 image with a Gaussian bump at (x[i], y[i])
            gaussian_image = 1.0 - np.exp(
                -(
                    (self.xv - x[i])**2
                    + (self.yv - y[i])**2
                )
                / (2 * self.sigma**2)
            )

            # Input = (x, y)
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
# 2. MLP MODEL WITH ACTIVATIONS
##############################################################################

class MLPDecoderWithActivations(nn.Module):
    """
    An MLP that:
     - Takes a 2D input (x,y) => shape (B,2).
     - Expands it via fully-connected layers to produce a 28x28 image (784).
     - Has n_hidden_layers ReLU layers + 1 final linear layer.
     - Stores intermediate (post-ReLU) activations in self.activations.
    """

    def __init__(self, input_size=2, hidden_size=128, n_hidden_layers=3):
        super().__init__()
        self.activations = []

        layers = []
        in_dim = input_size
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size

        # Final layer => 784 => reshape to (B,1,28,28)
        layers.append(nn.Linear(in_dim, 28 * 28))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        x shape (B,2). We'll store each ReLU activation in self.activations.
        """
        self.activations = []
        out = x
        for layer in self.model:
            out = layer(out)
            if isinstance(layer, nn.ReLU):
                # shape => (B, hidden_size).
                # We'll artificially add (1,1) dims => (B, hidden_size,1,1).
                self.activations.append(
                    out.detach().clone().unsqueeze(-1).unsqueeze(-1)
                )
        # Now 'out' shape => (B,784).
        out = out.view(out.size(0), 1, 28, 28)  # => (B,1,28,28)
        return out


##############################################################################
# 3. TRAINING / EVALUATION FUNCTIONS
##############################################################################

def train_model(model, dataloader, num_epochs=20, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    avg_losses = []

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
        avg_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_loss:.4f}")
    return avg_losses


def eval_ood_mse(model, dataloader):
    """
    Evaluate the MSE of the model on a given dataloader (intended for OOD set).
    Returns the average MSE.
    """
    criterion = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

    total_loss = 0.0
    num_samples = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)

    return total_loss / num_samples


##############################################################################
# 4. METRIC TENSOR & VOLUME ELEMENT (g-factor)
##############################################################################

def getg(data_3d):
    """
    data_3d: shape (H, W, D)
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
    g shape => (H, W, 2, 2).
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
    data_3d: shape (28,28,D).
    => factorization = 1 - mean(|g01|)/mean(|g00|+|g11|)
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
# 5. LINEAR PROBE: R^2 wrt x,y
##############################################################################


def linear_probe_r2(activation_tensor, coords_xy):
    """
    activation_tensor: shape (N, C, H, W).
    coords_xy: shape (N,2).
    Returns (r2_x, r2_y).
    """
    N, C, H, W = activation_tensor.shape
    act_flat = activation_tensor.reshape(N, C * H * W)

    coords_xy = coords_xy.astype(np.float32)

    reg_x = LinearRegression().fit(act_flat, coords_xy[:, 0])
    pred_x = reg_x.predict(act_flat)
    r2_x = r2_score(coords_xy[:, 0], pred_x)

    reg_y = LinearRegression().fit(act_flat, coords_xy[:, 1])
    pred_y = reg_y.predict(act_flat)
    r2_y = r2_score(coords_xy[:, 1], pred_y)

    return r2_x, r2_y


##############################################################################
# 6. MAIN EXPERIMENT: Vary MLP Depth
##############################################################################

def run_experiment_mlp_depths():
    # (1) Increase global font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14
    })

    fig_dir = "fig_mlp_pdf"
    os.makedirs(fig_dir, exist_ok=True)

    # (2) Prepare training dataset (exclude region [6..22])
    train_dataset = GaussianBumpDatasetSquare(
        num_samples=2000,
        grid_size=28,
        sigma=1.0,
        holdout_center=True,
        only_holdout=False,
        heldout_range=[6, 22]
    )
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=2
    )

    # (3) Prepare OOD dataset (only region [6..22])
    ood_dataset = GaussianBumpDatasetSquare(
        num_samples=500,
        grid_size=28,
        sigma=1.0,
        holdout_center=False,
        only_holdout=True,
        heldout_range=[6, 22]
    )
    ood_loader = DataLoader(ood_dataset, batch_size=32,
                            shuffle=False, num_workers=2)

    # We'll try multiple depths
    depths = [3, 4, 5, 6, 7]
    ood_losses = []

    # We'll store factorization & linear-probe results
    factorization_results = {}
    linear_probe_r2_scores = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ############################################################################
    # (A*) We'll pick 4 random OOD samples from a single batch (used by all nets)
    ############################################################################
    ood_inputs_full, ood_targets_full = next(iter(ood_loader))
    batch_size = ood_inputs_full.shape[0]
    rand_indices = np.random.choice(batch_size, size=4, replace=False)

    chosen_inputs = ood_inputs_full[rand_indices].clone().to(
        device)   # shape(4,2)
    # shape(4,1,28,28)
    chosen_targets = ood_targets_full[rand_indices].clone()

    # decode x,y from the (x,y) pairs
    chosen_coords = []
    for i in range(4):
        x_ = chosen_inputs[i, 0].cpu().item()
        y_ = chosen_inputs[i, 1].cpu().item()
        chosen_coords.append((int(x_), int(y_)))

    # We'll store predictions from each depth in all_ood_preds
    all_ood_preds = {}

    for depth in depths:
        print("============================================================")
        print(f"Training MLP with n_hidden_layers = {depth}")
        model = MLPDecoderWithActivations(
            input_size=2,
            hidden_size=128,
            n_hidden_layers=depth
        )

        # Train on in-distribution
        _ = train_model(model, train_loader, num_epochs=20, learning_rate=1e-3)

        # Evaluate on OOD dataset
        ood_mse = eval_ood_mse(model, ood_loader)
        print(f"Depth={depth}, OOD MSE={ood_mse:.4f}")
        ood_losses.append(ood_mse)

        # predictions => the same 4 OOD samples
        model.eval().to(device)
        with torch.no_grad():
            preds_4 = model(chosen_inputs)  # shape(4,1,28,28)
        all_ood_preds[depth] = preds_4.cpu()

        ######################################################################
        # Factorization & Linear Probe: pass 28×28 grid => model.activations
        ######################################################################
        coords_1d = np.arange(28)
        xv, yv = np.meshgrid(coords_1d, coords_1d)
        grid_points = np.stack([xv.ravel(), yv.ravel()], axis=-1)  # (784,2)

        with torch.no_grad():
            input_grid = torch.tensor(grid_points, dtype=torch.float32,
                                      device=device)
            _ = model(input_grid)  # fill model.activations

        activations_list = model.activations  # list of Tensors

        layer_factor_scores = []
        layer_x_r2 = []
        layer_y_r2 = []

        for act_tensor in activations_list:
            act_cpu = act_tensor.cpu().numpy()  # shape(784, C, 1, 1)
            N_, C_, _, _ = act_cpu.shape
            act_flat = act_cpu.reshape(N_, C_)
            data_reshape = act_flat.reshape(28, 28, -1)

            # Factorization
            fact_score = compute_factorization_metric(data_reshape)
            layer_factor_scores.append(fact_score)

            # Linear probe
            coords_xy = grid_points.astype(np.float32)
            reg_x = LinearRegression().fit(act_flat, coords_xy[:, 0])
            pred_x = reg_x.predict(act_flat)
            r2_x = r2_score(coords_xy[:, 0], pred_x)

            reg_y = LinearRegression().fit(act_flat, coords_xy[:, 1])
            pred_y = reg_y.predict(act_flat)
            r2_y = r2_score(coords_xy[:, 1], pred_y)

            layer_x_r2.append(r2_x)
            layer_y_r2.append(r2_y)

        factorization_results[depth] = layer_factor_scores
        linear_probe_r2_scores[depth] = (layer_x_r2, layer_y_r2)

        # (NEW) Volume-element plot for each layer
        plot_volume_element_layers(activations_list, depth, fig_dir)

    ############################################################################
    # (A**) single figure with (len(depths)+1) rows × 4 columns => OOD images
    #  - top row => GT
    #  - subsequent rows => each depth
    ############################################################################
    num_rows = len(depths) + 1
    plt.figure(figsize=(8, 3 * num_rows))

    # Top row => GT
    for col in range(4):
        gt_img = chosen_targets[col, 0].numpy()
        x_i, y_i = chosen_coords[col]
        ax_top = plt.subplot(num_rows, 4, col+1)
        ax_top.imshow(gt_img, cmap='gray', origin='lower')
        # mark OOD region
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
    # (B) OOD MSE vs Depth
    ##########################################################################
    plt.figure()
    plt.plot(depths, ood_losses, marker='o')
    plt.xlabel("MLP Depth (# hidden layers)")
    plt.ylabel("OOD MSE")
    plt.title("OOD MSE vs. MLP Depth (scalar input)")
    out_plot_path = os.path.join(fig_dir, "ood_vs_mlp_depth.pdf")
    plt.savefig(out_plot_path, bbox_inches='tight')
    plt.close()

    ##########################################################################
    # (C) FACTORIZATION METRIC vs. LAYER, for each DEPTH
    ##########################################################################
    plt.figure(figsize=(6, 4))
    max_layers = max(len(v) for v in factorization_results.values())

    for depth in depths:
        scores = factorization_results[depth]
        # Insert factor=1 at x=0 => "Input"
        extended_scores = [1.0] + list(scores)
        x_vals = range(len(extended_scores))
        plt.plot(x_vals, extended_scores, marker='o', label=f"Depth={depth}")

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
    # (D) LINEAR PROBE R^2 => separate figure for each depth
    ##########################################################################
    for depth in depths:
        layer_x_r2, layer_y_r2 = linear_probe_r2_scores[depth]
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
        # plt.title(f"Linear Probe (MLP Depth={depth})")
        out_r2_plot = os.path.join(fig_dir, f"probe_r2_depth_{depth}.pdf")
        plt.savefig(out_r2_plot, bbox_inches='tight')
        plt.close()

    print(f"Experiment complete. All plots saved in '{fig_dir}' folder.")


##############################################################################
# 7. MAIN
##############################################################################

if __name__ == "__main__":
    run_experiment_mlp_depths()
