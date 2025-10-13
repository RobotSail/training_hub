# Training Hub Examples

This section provides documentation, tutorials, and examples for using training_hub algorithms.

## Directory Structure

The examples are located in the repository's `examples/` directory:

- **`docs/`** - Usage documentation and guides for supported algorithms
- **`notebooks/`** - Interactive Jupyter notebooks with step-by-step tutorials
- **`scripts/`** - Standalone Python scripts for automation and examples

## Supported Algorithms

### Supervised Fine-Tuning (SFT)

The SFT algorithm supports training language models on supervised datasets with both single-node and multi-node distributed training capabilities.

**Documentation:**
- [SFT Usage Guide](/algorithms/sft) - Comprehensive usage documentation with parameter reference and examples

**Tutorials:**
- LAB Multi-Phase Training Tutorial - Interactive notebook demonstrating LAB multi-phase training workflow
- SFT Comprehensive Tutorial - Interactive notebook covering all SFT parameters with popular model examples

**Scripts:**
- LAB Multi-Phase Training Script - Example script for LAB multi-phase training with full command-line interface
- SFT with Qwen 2.5 7B - Single-node multi-GPU training example
- SFT with Llama 3.1 8B - Single-node multi-GPU training example
- SFT with Phi 4 Mini - Single-node multi-GPU training example
- SFT with GPT-OSS 20B - Single-node multi-GPU training example

**Quick Example:**
```python
from training_hub import sft

result = sft(
    model_path="/path/to/model",
    data_path="/path/to/data",
    ckpt_output_dir="/path/to/checkpoints",
    num_epochs=3,
    effective_batch_size=8,
    learning_rate=2e-5,
    max_seq_len=2048,
    max_tokens_per_gpu=45000
)
```

### Orthogonal Subspace Fine-Tuning (OSFT)

The OSFT algorithm supports continual training of pre-trained or instruction-tuned models without requiring supplementary datasets to maintain the original model distribution. Based on [Nayak et al. (2025)](https://arxiv.org/abs/2504.07097), it enables efficient customization while preventing catastrophic forgetting.

**Documentation:**
- [OSFT Usage Guide](/algorithms/osft) - Comprehensive usage documentation with parameter reference and examples

**Tutorials:**
- OSFT Comprehensive Tutorial - Interactive notebook covering all OSFT parameters with popular model examples
- OSFT Continual Learning - Interactive notebook demonstrating continual learning capabilities
- OSFT Multi-Phase Training Tutorial - Interactive notebook demonstrating OSFT multi-phase training workflow

**Scripts:**
- OSFT Multi-Phase Training Script - Example script for OSFT multi-phase training with full command-line interface
- OSFT with Qwen 2.5 7B - Single-node multi-GPU training example
- OSFT with Llama 3.1 8B - Single-node multi-GPU training example
- OSFT with Phi 4 Mini - Single-node multi-GPU training example
- OSFT with GPT-OSS 20B - Single-node multi-GPU training example
- OSFT Continual Learning Example - Example script demonstrating continual learning without catastrophic forgetting

**Quick Example:**
```python
from training_hub import osft

result = osft(
    model_path="/path/to/model",
    data_path="/path/to/data.jsonl",
    ckpt_output_dir="/path/to/outputs",
    unfreeze_rank_ratio=0.3,
    effective_batch_size=8,
    max_tokens_per_gpu=2048,
    max_seq_len=2048,
    learning_rate=2e-5
)
```

## Getting Started

1. **For detailed parameter documentation**: Check the relevant algorithm guide
2. **For hands-on learning**: Clone the repository and open the interactive notebooks in `examples/notebooks/`
3. **For automation scripts**: Refer to examples in `examples/scripts/`

## Repository

All examples are available in the [Training Hub GitHub repository](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/tree/main/examples).
