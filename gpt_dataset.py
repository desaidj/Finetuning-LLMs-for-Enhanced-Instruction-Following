"""
This file defines the dataset and dataloader for the GPT model.

## GPTDataset class

- Purpose: Creates a dataset from a given text, tokenizing it and preparing input-target pairs.
- Key operations:
  1. Tokenizes the entire input text.
  2. Uses a sliding window to create overlapping sequences of tokens.
  3. Creates input-target pairs where the target is the input shifted by one token.

## create_dataloader function

- Purpose: Creates a PyTorch DataLoader from the dataset.
- Key steps:
  1. Initializes a tokenizer (using tiktoken for GPT-2 tokenization).
  2. Creates a GPTDatasetV1 instance.
  3. Wraps the dataset in a DataLoader, which handles batching, shuffling, and multi-processing.

This file is crucial for efficiently feeding data into the model during training.
"""

import torch
from torch.utils.data import Dataset
import tiktoken

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDataset(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader
