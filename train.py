"""
This file implements the training loop and related functions for the GPT model.

## train_model_simple function

- Purpose: Main training loop for the model.
- Operations:
  1. Iterates over epochs and batches
  2. Computes loss and performs backpropagation
  3. Updates model parameters
  4. Periodically evaluates model and generates sample text

## evaluate_model function

- Purpose: Evaluates the model on training and validation data.
- Operations:
  1. Calculates loss on a subset of training and validation data
  2. Returns average losses

## generate_and_print_sample function

- Purpose: Generates and prints a sample text using the current model state.
- Operations:
  1. Encodes a start context
  2. Generates new tokens using the model
  3. Decodes and prints the generated text

## calc_loss_batch and calc_loss_loader functions

- Purpose: Calculate loss for a single batch or multiple batches.
- Operations:
  1. Forward pass through the model
  2. Compute cross-entropy loss

These functions work together to train the model, track its performance, and demonstrate its text generation capabilities throughout the training process.
"""

import torch
from generate import generate_text_simple
"""
text_to_token_ids(text, tokenizer):

This function converts input text to token IDs using the provided tokenizer.
It encodes the text, allowing for a special token '<|endoftext|>'.
The encoded tokens are converted to a PyTorch tensor and a batch dimension is added.
"""
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor

"""
token_ids_to_text(token_ids, tokenizer):

This function converts token IDs back to text using the provided tokenizer.
It removes the batch dimension from the input tensor.
The flattened list of token IDs is then decoded back to text
"""
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())



def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))
    model.train()

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
