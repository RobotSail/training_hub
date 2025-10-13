# Orthogonal Subspace Fine-Tuning (OSFT)

> **Conceptual Overview** - For complete API reference, see [osft() Function Reference](/api/functions/osft)

## What is OSFT?

Orthogonal Subspace Fine-Tuning (OSFT) is a specialized training algorithm that enables **continual learning without catastrophic forgetting**. Based on research by Nayak et al. (2025) ([arXiv:2504.07097](https://arxiv.org/abs/2504.07097)), OSFT allows you to adapt pre-trained or instruction-tuned models to new tasks while preserving their original capabilities.

The key innovation: OSFT learns in a subspace **orthogonal** to the model's existing knowledge, preventing interference with previously learned information. This eliminates the need for supplementary datasets to maintain the original model's distribution.

In Training Hub, OSFT is powered by the [RHAI Innovation Mini-Trainer backend](/api/backends/mini-trainer), which provides efficient orthogonal subspace computation with support for distributed training.

## When to Use OSFT

Use OSFT when you want to:

- **Continually adapt models** to new domains without forgetting previous training
- **Customize instruction-tuned models** with domain-specific knowledge (e.g., adding medical expertise to a general assistant)
- **Train on small datasets** while preserving the model's general capabilities
- **Avoid catastrophic forgetting** that occurs with standard fine-tuning

OSFT works best when:
- You're adapting an already-trained model (pre-trained or instruction-tuned)
- You want to preserve the model's existing capabilities
- You don't have access to the original training data
- Your new training dataset is relatively small

**Note:** If you're doing initial training or have a large dataset and don't need to preserve previous knowledge, standard [SFT (Supervised Fine-Tuning)](/algorithms/sft) may be simpler and faster.

## Quick Start

Here's a minimal example to get started with OSFT:

```python
from training_hub import osft

# Run orthogonal subspace fine-tuning
result = osft(
    model_path="meta-llama/Llama-3.1-8B-Instruct",  # Model to adapt
    data_path="./medical_qa.jsonl",                 # Your new training data
    ckpt_output_dir="./checkpoints",                # Where to save results
    unfreeze_rank_ratio=0.25,                       # How much to adapt (0.1-0.5)
    effective_batch_size=16,                        # Batch size
    max_tokens_per_gpu=2048,                        # GPU memory limit
    max_seq_len=2048,                               # Max sequence length
    learning_rate=2e-5                              # Learning rate
)
```

Your training data uses the same JSONL messages format as SFT:

```json
{"messages": [{"role": "user", "content": "What is diabetes?"}, {"role": "assistant", "content": "Diabetes is a condition..."}]}
{"messages": [{"role": "user", "content": "How is it treated?"}, {"role": "assistant", "content": "Treatment includes..."}]}
```

The model will learn the new medical domain while retaining its general conversational abilities.

## Key Concepts

### Orthogonal Subspace Learning

OSFT works by identifying the subspace where the model's existing knowledge resides, then learning new information in a direction **orthogonal** (perpendicular) to that subspace. This mathematical property ensures new learning doesn't interfere with old learning.

Think of it like writing on a new sheet of paper instead of erasing and rewriting on the same sheet - both pieces of information coexist without conflict.

### Unfreeze Rank Ratio

The `unfreeze_rank_ratio` parameter (0.0-1.0) controls how much of each weight matrix is adapted during training:

- **0.1-0.3**: Conservative adaptation, minimal changes to the model (recommended for small datasets)
- **0.3-0.5**: Moderate adaptation, balanced preservation and learning
- **>0.5**: Aggressive adaptation (rarely needed, approaches standard fine-tuning)

**Start with 0.25** and adjust based on your needs. Higher values allow more adaptation but slightly increase the risk of forgetting.

### Use Cases

**Example 1: Domain Specialization**
- Start: General instruction-tuned model (e.g., Llama 3.1)
- New data: Medical question-answering pairs
- Result: Model with medical expertise + original general capabilities

**Example 2: Continual Learning**
- Start: Model trained on Task A
- New data: Task B examples
- Result: Model that can handle both Task A and Task B

**Example 3: Low-Resource Adaptation**
- Start: Pre-trained language model
- New data: 500 examples in a new language/domain
- Result: Model with new capabilities without corrupting base knowledge

### Memory Considerations

OSFT has similar memory requirements to standard SFT. If you encounter out-of-memory errors during model loading, use:

```python
result = osft(
    # ... other parameters ...
    osft_memory_efficient_init=True  # Reduces memory during initialization
)
```

For general memory management, adjust `max_tokens_per_gpu`, `effective_batch_size`, or `max_seq_len`.

## Next Steps

**Learn more about OSFT:**
- [osft() Function Reference](/api/functions/osft) - Complete parameter documentation and advanced examples
- [OSFTAlgorithm Class](/api/classes/OSFTAlgorithm) - Object-oriented API for advanced use cases
- [Mini-Trainer Backend](/api/backends/mini-trainer) - Backend implementation details

**Related topics:**
- [SFT Algorithm](/algorithms/sft) - Standard fine-tuning alternative
- [Data Formats](/api/data-formats) - Detailed data format specifications
- [Distributed Training Guide](/guides/distributed-training) - Multi-node training setup

**Research:**
- [Original OSFT Paper](https://arxiv.org/abs/2504.07097) - Nayak et al. (2025) - Mathematical foundations and empirical results

**Working examples:**
- Check the [examples directory](/examples/) for Jupyter notebooks and scripts demonstrating OSFT in action
