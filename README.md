# Sparse MoE Language Model

This repository contains an implementation of a Sparse Mixture of Experts (MoE) Language Model using PyTorch. The model is designed to handle large-scale text generation tasks efficiently by leveraging multiple expert networks and a routing mechanism to dynamically select the most relevant experts for each input.

## Features

- **Sparse Mixture of Experts (MoE)**: Utilizes multiple expert networks with a routing mechanism to select the top-k experts for each input.
- **Transformer Architecture**: Built on a multi-head self-attention mechanism with layer normalization and dropout for stable training.
- **Noisy Top-k Routing**: Implements a noisy routing mechanism to improve the robustness of expert selection.
- **Dynamic Load Balancing**: Includes an auxiliary loss to ensure balanced utilization of experts.
- **Efficient Training**: Supports gradient clipping, learning rate scheduling, and checkpointing for stable and efficient training.
- **Text Generation**: Capable of generating text using top-p sampling with temperature control.

## Requirements

- Python 3.8+
- PyTorch 1.10+
- tiktoken
- mlflow
- tqdm
- matplotlib
- seaborn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sparse-moe-language-model.git
   cd sparse-moe-language-model
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model, run the following command:

```bash
python main.py --train
```

This will start the training process with the default hyperparameters. The model will periodically evaluate on both the training and validation sets, and save checkpoints at regular intervals.

### Text Generation

To generate text using a trained model, run the following command:

```bash
python main.py --generate
```

This will load the latest checkpoint and generate a sample of text. The generated text will be saved to `generated_text.txt`.

## Configuration

The model's hyperparameters can be adjusted in the script. Key parameters include:

- `n_embd`: Embedding dimension.
- `n_head`: Number of attention heads.
- `n_layer`: Number of transformer layers.
- `num_experts`: Number of experts in the MoE layer.
- `top_k`: Number of experts to select for each input.
- `batch_size`: Batch size for training.
- `max_iters`: Maximum number of training iterations.
- `learning_rate`: Initial learning rate.

## Results

The training process will log the following metrics:

- **Training Loss**: The loss on the training set.
- **Validation Loss**: The loss on the validation set.
- **Expert Usage**: The percentage of experts being utilized.
- **Gradient Norm**: The L2 norm of the gradients.

After training, a plot of the training and validation loss will be saved as `loss_curve.png`.

## Checkpoints

Model checkpoints are saved to `model_checkpoint.pth` at regular intervals. You can load a checkpoint to resume training or generate text.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.


## Acknowledgments

- This implementation is inspired by the original Transformer paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) and the Mixture of Experts approach.
- Special thanks to the PyTorch community for their excellent documentation and tutorials.
