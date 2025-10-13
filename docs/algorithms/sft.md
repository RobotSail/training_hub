# Supervised Fine-Tuning (SFT)

> **Conceptual Overview** - For complete API reference, see [sft() Function Reference](/api/functions/sft.md)

## What is SFT?

Supervised Fine-Tuning (SFT) is the standard approach for adapting pre-trained language models to new tasks or domains using labeled training data. The model learns to generate appropriate responses by training on input-output pairs.

In Training Hub, SFT is powered by the battle-tested [InstructLab Training backend](/api/backends/instructlab-training.md), which provides production-grade support for single-GPU, multi-GPU, and multi-node distributed training.

## When to Use SFT

Use SFT when you want to:

- **Adapt a pre-trained model** to a new domain or task (e.g., medical question-answering, coding assistance)
- **Create instruction-following models** from base language models
- **Improve model performance** on specific types of queries with labeled examples
- **Fine-tune openly available models** like Llama, Qwen, or Phi on custom data

SFT works best when:
- You have high-quality labeled training data (input-output pairs)
- You want straightforward, reliable fine-tuning without specialized techniques
- You're not concerned about catastrophic forgetting from previous training

**Note:** If you need to continually train a model without forgetting previous knowledge, consider [OSFT (Orthogonal Subspace Fine-Tuning)](osft.md) instead.

## Quick Start

Here's a minimal example to get started with SFT:

```python
from training_hub import sft

# Run supervised fine-tuning
result = sft(
    model_path="Qwen/Qwen2.5-7B-Instruct",      # Model to fine-tune
    data_path="./training_data.jsonl",          # Your training data
    ckpt_output_dir="./checkpoints",            # Where to save results
    num_epochs=3,                               # Training epochs
    effective_batch_size=8,                     # Batch size across all GPUs
    learning_rate=2e-5,                         # Learning rate
    max_seq_len=2048,                           # Max sequence length
    max_tokens_per_gpu=45000                    # GPU memory limit
)
```

Your training data should be in JSONL format with messages:

```json
{"messages": [{"role": "user", "content": "What is SFT?"}, {"role": "assistant", "content": "SFT is supervised fine-tuning..."}]}
{"messages": [{"role": "user", "content": "How do I use it?"}, {"role": "assistant", "content": "You can use the sft() function..."}]}
```

That's it! The `sft()` function handles all the complexity of distributed training, data processing, and checkpointing automatically.

## Key Concepts

### Training Data

SFT requires training data in **messages format** - conversational exchanges between user and assistant. Each training example teaches the model how to respond to specific inputs.

The backend only trains on assistant responses by default (instruction-tuning mode), but you can use the `unmask` field to include user messages in the training loss as well (system prompts remain masked).

See [Data Formats](/api/data-formats.md) for complete specifications.

### Memory Management

The `max_tokens_per_gpu` parameter is crucial for managing GPU memory. It sets a hard cap on the number of tokens processed per GPU in each mini-batch. The backend automatically calculates gradient accumulation steps to achieve your desired `effective_batch_size` while staying within memory limits.

**If you encounter out-of-memory errors**, reduce `max_tokens_per_gpu`, `effective_batch_size`, or `max_seq_len`.

### Distributed Training

Training Hub automatically handles distributed training across multiple GPUs and nodes. Simply specify:

- `nproc_per_node` - GPUs per machine (auto-detected if not specified)
- `nnodes` - Total number of machines
- `node_rank` - This machine's rank (0 for master)
- `rdzv_endpoint` - Master node address (for multi-node)

The backend uses PyTorch's `torchrun` under the hood for robust distributed execution.

See [Distributed Training Guide](/guides/distributed-training.md) for complete multi-node setup instructions.

## Next Steps

**Learn more about SFT:**
- [sft() Function Reference](/api/functions/sft.md) - Complete parameter documentation and advanced examples
- [SFTAlgorithm Class](/api/classes/SFTAlgorithm.md) - Object-oriented API for advanced use cases
- [InstructLab Training Backend](/api/backends/instructlab-training.md) - Backend implementation details

**Related topics:**
- [OSFT Algorithm](/algorithms/osft.md) - Alternative for continual learning without catastrophic forgetting
- [Data Formats](/api/data-formats.md) - Detailed data format specifications
- [Distributed Training Guide](/guides/distributed-training.md) - Multi-node training setup
- [Data Preparation Guide](/guides/data-preparation.md) - Best practices for preparing training data

**Working examples:**
- Check the [examples directory](/examples/README.md) for Jupyter notebooks and scripts demonstrating SFT in action
