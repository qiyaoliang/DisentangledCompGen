import torch
import torch.nn as nn
import numpy as np

class MultiLayerTransformer(nn.Module):
    """
    We create a manual stack of TransformerEncoderLayer to store per-layer outputs easily.
    """

    def __init__(self, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        mlist = []
        for _ in range(num_layers):
            mlist.append(nn.Linear(d_model, d_model))
            # mlist.append(nn.TransformerEncoderLayer(d_model, nhead, batch_first = True))
            mlist.append(nn.PReLU())
        
        self.layers = nn.ModuleList(mlist)

    def forward(self, x):
        layer_outputs = []
        for layer in self.layers:
            x = layer(x)
            layer_outputs.append(x)
        return layer_outputs  # list of length = num_layers


class BumpTransformer(nn.Module):
    """
    Takes a 56-dim input -> embed -> adds 2D pos embedding -> stacked Transformer layers -> store intermediate outputs
    Projects final output to 1×28×28 image. Also returns list of per-layer [batch, seq_len, d_model].
    """

    def __init__(self, input_dim=56, embed_dim=128, num_heads=4, num_layers=3, grid_size=28):
        super().__init__()
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size # 1 token for each pixel
        self.embed_dim = embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_embedding = nn.Parameter(
            self._generate_2d_positional_encoding(grid_size, embed_dim),
            requires_grad=False
        )
        self.transformer = MultiLayerTransformer(d_model=embed_dim,
                                                 nhead=num_heads,
                                                 num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, 1)

    def _generate_2d_positional_encoding(self, grid_size, embed_dim):
        pe = torch.zeros(grid_size, grid_size, embed_dim)
        for x in range(grid_size):
            for y in range(grid_size):
                div_term = torch.exp(torch.arange(0, embed_dim, 4) * (-np.log(10000.0) / embed_dim))
                pe[x, y, 0::4] = torch.sin(x * div_term)
                pe[x, y, 1::4] = torch.cos(x * div_term)
                pe[x, y, 2::4] = torch.sin(y * div_term)
                pe[x, y, 3::4] = torch.cos(y * div_term)

        return pe.view(-1, embed_dim).unsqueeze(0) # (1, 784, embed_dim)

    def forward(self, coord_input):
        B = coord_input.size(0) # batch size
        context_embedding = self.input_proj(coord_input).unsqueeze(1) # (batch_size, 1, embed_dim)
        tokens = context_embedding.repeat(1, self.num_tokens, 1) # (batch_size, num_tokens, embed_dim)
        tokens += self.pos_embedding
        layer_outputs = self.transformer(tokens)  # list of length = num_layers
        final_layer_out = layer_outputs[-1]  # (batch_size, 784, embed_dim)
        pixel_values = self.output_proj(final_layer_out).squeeze(-1)  # (batch_size, 784)
        out_img = pixel_values.view(B, 1, self.grid_size, self.grid_size)
        return out_img
