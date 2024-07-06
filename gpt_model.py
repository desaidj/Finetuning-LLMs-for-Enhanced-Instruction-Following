"""
This file defines the overall GPT model architecture.

## GPTModel class

- Purpose: Implements the full GPT model.
- Key components:
  1. Token Embedding layer: Converts token IDs to dense vectors
  2. Positional Embedding layer: Adds position information
  3. Dropout layer: For regularization
  4. Transformer Blocks: A sequence of TransformerBlock instances
  5. Final Layer Normalization
  6. Output head: Linear layer for final token prediction

- Forward pass operations:
  1. Convert input token IDs to embeddings
  2. Add positional embeddings
  3. Apply dropout
  4. Pass through the sequence of Transformer blocks
  5. Apply final layer normalization
  6. Project to vocabulary size for token prediction

This class brings together all the components to create the complete GPT model architecture.
"""

import torch
import torch.nn as nn
from layers import TransformerBlock, LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

