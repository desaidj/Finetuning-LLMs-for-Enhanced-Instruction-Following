"""
This file implements text generation functions for the GPT model.

## generate_text_simple function

- Purpose: Generates text using a simple greedy decoding strategy.
- Operations:
  1. Iteratively predicts the next token
  2. Always chooses the most likely next token
  3. Adds the chosen token to the sequence

## generate function

- Purpose: Generates text with more advanced decoding options.
- Features:
  1. Temperature scaling: Controls randomness of predictions
  2. Top-k sampling: Limits choices to top k most likely tokens
  3. Early stopping: Can stop on encountering an end-of-sequence token

- Operations:
  1. Predicts next token probabilities
  2. Optionally applies top-k filtering
  3. Applies temperature scaling if specified
  4. Samples next token based on probabilities or chooses most likely
  5. Adds chosen token to sequence
  6. Optionally stops on end-of-sequence token

These functions allow for flexible text generation, from deterministic to more random and diverse outputs.
"""
import torch

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx
