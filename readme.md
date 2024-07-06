# LLM Instruction Finetuning Project

This project demonstrates the process of finetuning a GPT-2 medium model for instruction following

## Project Overview

In this project, I have:

- Prepared an instruction dataset for finetuning
- Implemented custom data loading and batching techniques
- Loaded a pretrained GPT-2 medium model (355M parameters)
- Finetuned the model on instruction data

## Key Components

### 1. Data Preparation (InstructionDataset class)

- Created a custom dataset class for instruction data
- Implemented pre-tokenization of texts for efficiency

### 2. Custom Collate Function

- Developed `custom_collate_fn` for efficient batching
- Implemented padding and masking techniques for variable-length sequences

### 3. Model Loading and Finetuning

- Loaded the pretrained GPT-2 medium model (355M parameters)
- Implemented finetuning process using AdamW optimizer
- Trained for 2 epochs with a learning rate of 0.00005 and weight decay of 0.1

### 4. Response Generation and Saving

- Generated responses for the test set using the finetuned model
- Saved the model responses along with original data in a JSON file


## Scripts and Notebooks


## Dataset

Used a custom instruction dataset with entries containing:

- An instruction
- Optional input text
- Expected output

## Model Details

- **Base model**: GPT-2 medium (355M parameters)
- **Finetuning hyperparameters**:
  - Learning rate: 0.00005
  - Weight decay: 0.1
  - Epochs: 2
  - Optimizer: AdamW


## Future Work
- Finetune the model
- Experiment with different model sizes or architectures
- Try longer training durations or different hyperparameters
- Explore more sophisticated evaluation techniques
