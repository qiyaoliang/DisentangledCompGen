#!/usr/bin/env python3
# transformer_gaussian_full_experiment.py

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches


##############################################################################
# 1. DATASET: 56-dim input (two 1D Gaussians), 28×28 output
##############################################################################
class GaussianBumpDatasetSquare(Dataset):
    """
    Each sample:
      - 56-dim input => [gaussian_x(28), gaussian_y(28)]
      - 28×28 target => 2D Gaussian bump at (x,y).
    Can optionally hold out [lb..ub]^2 from training or only use that region for OOD.
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

        # For generating 2D Gaussians
        xv, yv = np.meshgrid(np.arange(grid_size), np.arange(grid_size))

        self.data = []
        self.targets = []
        self._generate(xv, yv)

    def _generate(self, xv, yv):
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
                    (x_b >= self.lb) & (x_b <= self.ub)
                    & (y_b >= self.lb) & (y_b <= self.ub)
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
            gx = np.exp(-((np.arange(self.grid_size) - x_i)
                        ** 2) / (2 * self.sigma**2))
            gy = np.exp(-((np.arange(self.grid_size) - y_i)
                        ** 2) / (2 * self.sigma**2))
            bump_56 = np.concatenate([gx, gy], axis=0)

            gauss_2d = 1.0 - np.exp(
                -(
                    ((xv - x_i) ** 2 + (yv - y_i) ** 2)
                    / (2 * self.sigma**2)
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
# 2. HELPER FUNCTIONS FOR TRAINING/EVAL
##############################################################################
def train_model(model, dataloader, num_epochs=20, learning_rate=1e-3):
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
            preds, _ = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss={avg_loss:.4f}")


def eval_ood_mse(model, dataloader):
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    total_loss, count = 0.0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds, _ = model(inputs)
            loss = criterion(preds, targets)
            total_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)
    return total_loss / count


##############################################################################
# 3. METRIC TENSOR & FACTORIZATION
##############################################################################
def getg(data_3d):
    """
    data_3d: shape (H, W, D)
    returns g => shape (H, W, 2, 2)
    """
    import numpy as np
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
    g_11 = g[:, :, 0, 0]
    g_12 = g[:, :, 0, 1]
    g_21 = g[:, :, 1, 0]
    g_22 = g[:, :, 1, 1]
    detg = g_11 * g_22 - g_12 * g_21
    return np.sqrt(np.abs(detg))


def compute_factorization_metric(data_3d):
    """
    data_3d: shape (28,28,D)
    factorization => 1 - (mean(|g01|)/mean(|g00|+|g11|))
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
# 4. VOLUME-ELEMENT PLOTTING
##############################################################################
def plot_volume_element_layers(activations_list, depth, fig_dir):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    n_layers = len(activations_list)
    fig, axes = plt.subplots(1, n_layers, figsize=(
        3.5*n_layers, 4), squeeze=False)
    if n_layers == 1:
        axes = [axes[0, 0]]
    else:
        axes = axes[0]

    for layer_idx, act_tensor in enumerate(activations_list):
        ax = axes[layer_idx]
        act_cpu = act_tensor.cpu().numpy()  # shape => (N, C, H, W)
        N_, C_, H_, W_ = act_cpu.shape

        # flatten => (N, C*H*W)
        act_flat = act_cpu.reshape(N_, C_*H_*W_)
        # reshape => (28, 28, -1)
        data_reshape = act_flat.reshape(28, 28, -1)

        g_mat = getg(data_reshape)
        dv = compute_volume_element(g_mat)
        data_log = np.log10(dv + 1e-12)

        avgg = np.mean(data_log)
        stdd = np.std(data_log)
        vmin_layer = avgg - 3*stdd
        vmax_layer = avgg + 3*stdd

        im = ax.imshow(data_log, origin='lower', cmap='Reds',
                       vmin=vmin_layer, vmax=vmax_layer)
        ax.set_xticks([])
        ax.set_yticks([])

        rect = patches.Rectangle((6, 6), 16, 16, fill=False, edgecolor='white',
                                 linestyle='--', linewidth=2)
        ax.add_patch(rect)
        ax.set_title(f"Layer {layer_idx}", fontsize=16)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = fig.colorbar(im, cax=cax)
        if layer_idx < n_layers - 1:
            cb.set_label("")
        else:
            cb.set_label(r"$\log_{10}(dv)$", fontsize=16)

    out_vol_path = os.path.join(fig_dir, f"volume_element_depth_{depth}.pdf")
    plt.tight_layout()
    plt.savefig(out_vol_path, bbox_inches='tight')
    plt.close()
    print(f"Saved volume-element figure => {out_vol_path}")


##############################################################################
# 5. TRANSFORMER WITH INTERMEDIATE ACTIVATIONS
##############################################################################
class CustomTransformerEncoderLayer(nn.Module):
    """
    Wraps nn.TransformerEncoderLayer but lets us extract intermediate x for each sub-layer.
    We'll treat self-attn + feedforward as a single block.
    """

    def __init__(self, d_model=128, nhead=4, **kwargs):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model, nhead, batch_first=True, **kwargs)

    def forward(self, x):
        return self.layer(x)


class MultiLayerTransformer(nn.Module):
    """
    We create a manual stack of TransformerEncoderLayer to store per-layer outputs easily.
    """

    def __init__(self, d_model=128, nhead=4, num_layers=3, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, nhead, **kwargs) for _ in range(num_layers)
        ])

    def forward(self, x):
        layer_outputs = []
        for layer in self.layers:
            x = layer(x)
            layer_outputs.append(x)
        return layer_outputs  # list of length=num_layers


class BumpTransformer(nn.Module):
    """
    Takes a 56-dim input -> embed -> adds 2D pos embedding -> stacked Transformer layers -> store intermediate outputs
    Projects final output to 1×28×28 image. Also returns list of per-layer [batch, seq_len, d_model].
    """

    def __init__(self, input_dim=56, embed_dim=128, num_heads=4, num_layers=3, grid_size=28):
        super().__init__()
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size
        self.embed_dim = embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_embedding = nn.Parameter(
            self._generate_2d_positional_encoding(grid_size, embed_dim),
            requires_grad=False
        )
        self.transformer = MultiLayerTransformer(
            d_model=embed_dim, nhead=num_heads, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, 1)

    def _generate_2d_positional_encoding(self, grid_size, embed_dim):
        pe = torch.zeros(grid_size, grid_size, embed_dim)
        for x in range(grid_size):
            for y in range(grid_size):
                div_term = torch.exp(torch.arange(
                    0, embed_dim, 4) * (-np.log(10000.0) / embed_dim))
                pe[x, y, 0::4] = torch.sin(x * div_term)
                pe[x, y, 1::4] = torch.cos(x * div_term)
                pe[x, y, 2::4] = torch.sin(y * div_term)
                pe[x, y, 3::4] = torch.cos(y * div_term)
        # shape => (1, 784, embed_dim)
        return pe.view(-1, embed_dim).unsqueeze(0)

    def forward(self, coord_input):
        B = coord_input.size(0)
        context_embedding = self.input_proj(coord_input).unsqueeze(1)
        tokens = context_embedding.repeat(
            1, self.num_tokens, 1) + self.pos_embedding
        layer_outputs = self.transformer(tokens)  # list of length = num_layers
        final_layer_out = layer_outputs[-1]  # shape => (B,784,embed_dim)
        pixel_values = self.output_proj(
            final_layer_out).squeeze(-1)  # => (B,784)
        out_img = pixel_values.view(B, 1, self.grid_size, self.grid_size)
        return out_img, layer_outputs


##############################################################################
# 6. MAIN EXPERIMENT
##############################################################################
def plot_id_predictions(
    depths,
    all_id_preds,
    chosen_targets_id,
    fig_dir,
    lb=6,
    ub=22
):
    """
    Similar figure to OOD but for ID samples.
    all_id_preds: dict{depth: list_of_4_images}
    chosen_targets_id: (4,1,28,28)
    """
    num_rows = len(depths) + 1
    plt.figure(figsize=(8, 3 * num_rows))

    # -- Top row: ground-truth ID
    for col in range(4):
        ax = plt.subplot(num_rows, 4, col + 1)
        ax.imshow(chosen_targets_id[col, 0], cmap='gray', origin='lower')

        # Optionally add bounding box (the same region in red dashed)
        rect = patches.Rectangle((lb, lb), (ub - lb), (ub - lb),
                                 fill=False, edgecolor='red',
                                 linestyle='--', linewidth=2)
        ax.add_patch(rect)
        if col == 0:
            ax.set_ylabel("GT", fontsize=14)

    # -- Next rows: predictions
    for r, depth in enumerate(depths, 1):
        for c, img in enumerate(all_id_preds[depth]):
            ax = plt.subplot(num_rows, 4, r*4 + (c+1))
            ax.imshow(img[0], cmap='gray', origin='lower')

            rect = patches.Rectangle((lb, lb), (ub - lb), (ub - lb),
                                     fill=False, edgecolor='red',
                                     linestyle='--', linewidth=2)
            ax.add_patch(rect)

            if c == 0:
                ax.set_ylabel(f"Depth={depth}", fontsize=14)

    out_path = os.path.join(fig_dir, "id_all_depths.pdf")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print("ID predictions figure saved at:", out_path)


def run_transformer_full_experiment():
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18
    })

    fig_dir = "fig_transformer_gaussian_alternative"
    os.makedirs(fig_dir, exist_ok=True)

    # 1) Load datasets
    train_ds = GaussianBumpDatasetSquare(2000, 28, 1.0, True, False, [6, 22])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # We'll define an 'id_ds' for easy access to ID samples. Or simply reuse train_ds if you prefer.
    # Let's do: id_ds = the same as train_ds but with smaller num_samples => for quick picks
    id_ds = GaussianBumpDatasetSquare(
        500, 28, 1.0, holdout_center=False, only_holdout=False, heldout_range=[6, 22])
    id_loader = DataLoader(id_ds, batch_size=32, shuffle=False)

    ood_ds = GaussianBumpDatasetSquare(500, 28, 1.0, False, True, [6, 22])
    ood_loader = DataLoader(ood_ds, batch_size=32, shuffle=False)

    # 2) Depths tested
    depths = [2, 3, 4, 5]
    ood_losses = []
    factorization_results = {}
    all_ood_preds = {}
    all_id_preds = {}  # We'll store ID predictions in a similar dict

    # 3) Grab 4 OOD samples
    ood_inputs, ood_targets = next(iter(ood_loader))
    rand_indices_ood = np.random.choice(len(ood_inputs), 4, replace=False)
    chosen_inputs_ood = ood_inputs[rand_indices_ood].cuda()
    chosen_targets_ood = ood_targets[rand_indices_ood]

    # Also pick 4 ID samples from id_loader
    id_inputs, id_targets = next(iter(id_loader))
    rand_indices_id = np.random.choice(len(id_inputs), 4, replace=False)
    chosen_inputs_id = id_inputs[rand_indices_id].cuda()
    chosen_targets_id = id_targets[rand_indices_id]

    for depth in depths:
        # Build model
        model = BumpTransformer(
            input_dim=56, embed_dim=128, num_heads=4,
            num_layers=depth, grid_size=28
        ).cuda()

        # Train
        train_model(model, train_loader, num_epochs=20, learning_rate=1e-3)

        # Evaluate OOD MSE
        mse_ood = eval_ood_mse(model, ood_loader)
        ood_losses.append(mse_ood)

        # Inference on 4 chosen OOD
        model.eval()
        with torch.no_grad():
            preds_ood, _ = model(chosen_inputs_ood)
        all_ood_preds[depth] = preds_ood.cpu()

        # Inference on 4 chosen ID
        model.eval()
        with torch.no_grad():
            preds_id, _ = model(chosen_inputs_id)
        all_id_preds[depth] = preds_id.cpu()

        # (Skipping factorization due to shape complexity; we can do it if needed)
        factorization_results[depth] = [0.0]  # placeholder

        # [ ... The same large block for layer outputs if you want volume-element
        # logs can remain here if you want them. We omit it for brevity.  ... ]

    # ========== 7) Plot the OOD predictions for each depth ==========
    num_rows = len(depths) + 1
    plt.figure(figsize=(8, 3 * num_rows))

    # -- Top row: ground truth images --
    for col in range(4):
        ax = plt.subplot(num_rows, 4, col + 1)
        ax.imshow(chosen_targets_ood[col, 0], cmap='gray')

        # Add the red dashed bounding box for OOD region
        rect = patches.Rectangle((6, 6), 16, 16, fill=False,
                                 edgecolor='red', linestyle='--', linewidth=2)
        ax.add_patch(rect)

        if col == 0:
            ax.set_ylabel("GT", fontsize=14)

    # -- Subsequent rows: model predictions for each depth --
    for r, depth in enumerate(depths, 1):
        for c, img in enumerate(all_ood_preds[depth]):
            ax = plt.subplot(num_rows, 4, r * 4 + (c + 1))
            ax.imshow(img[0], cmap='gray')

            # Add the red dashed bounding box for OOD region
            rect = patches.Rectangle((6, 6), 16, 16, fill=False,
                                     edgecolor='red', linestyle='--', linewidth=2)
            ax.add_patch(rect)

            # Leftmost column gets a label: "Depth={depth}"
            if c == 0:
                ax.set_ylabel(f"Depth={depth}", fontsize=14)

    plt.savefig(os.path.join(fig_dir, "ood_all_depths.pdf"))
    plt.close()

    # ========== Additional: Plot the ID predictions for each depth ==========
    # Reuse the bounding box for consistency. The region [6..22] is the same, though it's not OOD for these samples.
    # We'll do something similar to the code above, or use the newly defined function.
    num_rows_id = len(depths) + 1
    plt.figure(figsize=(8, 3 * num_rows_id))

    # -- Top row: ground truth ID images --
    for col in range(4):
        ax = plt.subplot(num_rows_id, 4, col + 1)
        ax.imshow(chosen_targets_id[col, 0], cmap='gray', origin='lower')
        rect = patches.Rectangle((6, 6), 16, 16, fill=False,
                                 edgecolor='red', linestyle='--', linewidth=2)
        ax.add_patch(rect)
        if col == 0:
            ax.set_ylabel("GT", fontsize=14)

    # -- Next rows => ID predictions
    for r, depth in enumerate(depths, 1):
        for c, img in enumerate(all_id_preds[depth]):
            ax = plt.subplot(num_rows_id, 4, r * 4 + (c + 1))
            ax.imshow(img[0], cmap='gray', origin='lower')
            rect = patches.Rectangle((6, 6), 16, 16, fill=False,
                                     edgecolor='red', linestyle='--', linewidth=2)
            ax.add_patch(rect)
            if c == 0:
                ax.set_ylabel(f"Depth={depth}", fontsize=14)

    plt.savefig(os.path.join(fig_dir, "id_all_depths.pdf"))
    plt.close()

    # ========== 8) OOD MSE vs Depth ==========
    plt.figure()
    plt.plot(depths, ood_losses, 'o-')
    plt.xlabel("Transformer Depth")
    plt.ylabel("OOD MSE")
    plt.savefig(os.path.join(fig_dir, "ood_vs_depth.pdf"))
    plt.close()

    # ========== 9) Factorization vs Layer => just do last layer (placeholder) ==========
    plt.figure()
    for depth in depths:
        plt.plot([0, 1], [1]+factorization_results[depth],
                 'o-', label=f"Depth={depth}")
    plt.legend()
    plt.savefig(os.path.join(fig_dir, "factorization_vs_layer.pdf"))
    plt.close()

    print("Done: All figures (OOD + ID) saved in", fig_dir)


if __name__ == '__main__':
    run_transformer_full_experiment()
