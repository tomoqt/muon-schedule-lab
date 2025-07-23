# LLM with Muon Playground

This project is a hackable, minimal implementation of a GPT-style LLM for experimentation, based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy. The main focus of this playground is to explore and experiment with the [Muon](https://github.com/KellerJordan/Muon) optimizer.

## Motivation for Muon

Recent optimizers like [Muon](https://github.com/KellerJordan/Muon) achieve remarkable efficiency by orthogonalizing weights in neural networks. This orthogonalization creates more spread singular value decompositions (SVDs), which appears to improve network training dynamics.

The success of these approaches suggests that standard Euclidean optimization methods might not be optimal for training large neural networks.


## Project Goal

This project aims to provide a simple and flexible playground for training LLMs. It's designed to be easily modified for experiments with different architectures, optimizers, and training techniques. The inclusion of the Muon optimizer serves as a starting point for exploring non-standard optimization methods.

## Implementation

We use the minimal and efficient nanoGPT implementation as our foundation. The code is designed to be as lightweight and readable as possible while enabling meaningful experiments in representation geometry.

## Dataset Preparation

For experimenting with mixed curvature transformers, we use the same dataset preparation approach as the original nanoGPT:

### Shakespeare Dataset (Small Scale Testing)

For quick experimentation, the Shakespeare dataset provides a lightweight option:

```sh
python data/shakespeare_char/prepare.py
```

This creates `train.bin` and `val.bin` files with character-level tokenization.

### OpenWebText Dataset (Full Scale Training)

Fineweb
```sh
python data/fineweb/prepare.py
```

This downloads and tokenizes the Fineweb10B dataset, creating `train.bin` and `val.bin` files with GPT-2 BPE tokenization.

## Getting Started

To get started with training:

### Standard Training (AdamW)
You can run a standard training with the AdamW optimizer on a single GPU:
```sh
python train.py --batch_size=32 --compile=False
```

### Training with Muon
The Muon optimizer implementation requires a distributed process group, so you need to use `torchrun` to launch the training script, even on a single GPU.
To run with Muon on a single GPU:
```sh
torchrun --standalone --nproc_per_node=1 train.py --use_muon=True
```

### Multi-GPU Training
For multi-GPU training (with either AdamW or Muon), you can use `torchrun` as well. For example, on a machine with 4 GPUs:
```sh
torchrun --standalone --nproc_per_node=4 train.py
```
To use Muon in a multi-GPU setup, simply add the flag:
```sh
torchrun --standalone --nproc_per_node=4 train.py --use_muon=True
```
Refer to the top of `train.py` for more advanced distributed training configurations.

## Acknowledgements

- Original code based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- Geometric optimization insights from [Muon](https://github.com/KellerJordan/Muon) by Keller Jordan et al.



