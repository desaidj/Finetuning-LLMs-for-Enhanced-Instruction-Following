"""
This file implements various layers used in the Transformer architecture.

## LayerNorm class

- Purpose: Normalizes the inputs, improving training stability.
- Operations: Computes mean and variance, normalizes inputs, applies learnable scale and shift.

## GELU class

- Purpose: Implements the Gaussian Error Linear Unit activation function.
- Operation: Applies a smooth approximation of ReLU, allowing for some gradient even for negative inputs.

## FeedForward class

- Purpose: Applies a position-wise feed-forward network.
- Structure: Two linear transformations with a GELU activation in between.
- Operations: Expands input dimension, applies GELU, then projects back to original dimension.

## TransformerBlock class

- Purpose: Implements a full Transformer block.
- Components:
  1. Multi-Head Attention layer
  2. Feed-Forward layer
  3. Two Layer Normalization layers
  4. Residual connections
- Operations:
  1. Applies attention with a residual connection and layer norm
  2. Applies feed-forward network with a residual connection and layer norm

This file provides the building blocks for constructing the full Transformer model.
"""

import torch
import torch.nn as nn
from attention import MultiHeadAttention

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut

        return x
