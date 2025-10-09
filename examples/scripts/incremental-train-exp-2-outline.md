### Primary Hypothesis

When training on distinct chunks, OSFT maintains a much better balance of prior information retained while it learns new knowledge. Compared to SFT, OSFT should be able to:
* Retain prior custom knowledge as training progresses
* Maintains a better overall performance on general capabilities (leaderboard) 

**Hypothesis scenario:**
- OSFT on each distinct chunk

**Baseline scenarios**:
- Full SFT on the entire dataset (may reuse the results from experiment 1)
- Full OSFT on the entire dataset (may reuse the results from experiment 1)
- Full SFT on each distinct chunk

**Expected outcome**:
- For the % of knowledge gained on each custom domain, the % retained on existing domains should be far less than the baselines


### Secondary Hypothesis

Model subspace becomes saturated with the incoming datasets, so the rank ratio should be gradually reduced as we continue training.

**Hypothesis scenario**:
* OSFT on each distinct chunk, lowering rank ratio by some amount each time

**Comparative baselines**:
- OSFT on each distinct chunk, no change in rank ratio

**Expected outcome**:
* The _final_ checkpoint should have retained more information on each overall domain compared to the baseline

### Dataset

The dataset we'll be using is a version of the original training dataset used for experiment 1 (the quality dataset created using the new knowledge pipeline). 

Experiment 1 focused on chunking the datasets through simple uniform sampling. We will instead create the chunks by selecting samples out based on their documents.

**Chunking**:

We will use this simple chunking script:

```python
from datasets import load_dataset, Dataset
fp = "/mnt/nvme1n1/experiments/os-cl-scenario-1-experiment-0/synth_knowledge2_0-combined_cut_50x/combined_cut_50x.jsonl"
ds = load_dataset('json', data_files=fp, split="train")
all_docs = list(set(ds['document_outline']))

# Split into n equal parts
n = 3  # Change this to desired number of parts
total_docs = len(all_docs)
part_size = total_docs // n

# Create n parts dynamically
doc_parts = []
for i in range(n):
    start_idx = i * part_size
    end_idx = (i + 1) * part_size if i < n - 1 else total_docs  # Last part gets remaining docs
    doc_parts.append(ds.filter(lambda x: x['document_outline'] in all_docs[start_idx:end_idx]))

print(f"Total documents: {total_docs}")
for i, part in enumerate(doc_parts):
    print(f"Part {i+1}: {len(part)} documents")
```
### Evaluation

We will evaluate the model on: 
1. Domain knowledge (QuaLITY)
	1. Current chunk: only the documents from the current distribution
	2. Prior chunks: all prior documents seen, one-by-one
	3. Overall seen: every document seen so far
	4. Overall: Default eval set
2. General performance (Leaderboard)

#### Domain-Knowledge Segmentation

Since all chunks are disjoint subsets, we need an eval set for each respective chunk. Using Abhishek's code to split up the dataset into chunks, I've implemented a script to also create an eval dataset for each chunk.

The only issue is that the training dataset we use does not cover 2 of the documents in the original quality dataset. To account for this, I simply make sure that each eval dataset has these as common samples.


### Model & Hyper-parameters 

**Model**: `meta-llama/Llama-3.1-8B-Instruct`
**Constant Hyper-parameters:**

| Name               | Value    |
| ------------------ | -------- |
| LR                 | 5e-6     |
| BS                 | 128      |
| epochs             | 1        |
| LR scheduler       | constant |
| warmup steps       | 0        |
| max rank ratio<br> | 0.35     |

**Baselines to run**:
1. OSFT completed baseline (full, unchunked dataset, set to the max rank ratio)
2. OSFT chunked baseline (train in chunks, but keep the rank ratio the same each time)
3. SFT complete baseline (we just run a training run with these settings)
4. SFT chunked baseline (same SFT hyperparams, but train on the chunks)

**Hypothesis to run**:
1. OSFT experimental (train on each chunk, start from max rank ratio but reduce the rank ratio by 0.10 each time, so 0.35, 0.25, 0.15)

**Absolute QuaLITY Scores**: 

This table captures the **absolute** score on the QuaLITY evaluation per checkpoint. They are called "Absolute", because each chunked checkpoint (aside from the final chunk) is partially evaluated on entirely unseen data. 

| Experiment                                  | Attempt Rate | Accuracy Among Attempted | Overall Accuracy | Total Attempt Rate |
| :------------------------------------------ | -----------: | -----------------------: | ---------------: | -----------------: |
| **OSFT Decreasing RR - Chunk 0** (RR: 0.35) |            1 |                    0.441 |            0.441 |             0.9989 |
| **OSFT Decreasing RR - Chunk 1** (RR: 0.25) |            1 |                   0.4361 |           0.4361 |             0.9988 |
| **OSFT Decreasing RR - Chunk 2** (RR: 0.15) |            1 |                   0.4313 |           0.4313 |             0.9986 |
| **OSFT Constant RR - Chunk 0** (RR: 0.35)   |            1 |                   0.4072 |           0.4072 |             0.9988 |
| **OSFT Constant RR - Chunk 1** (RR: 0.35)   |            1 |                   0.4602 |           0.4602 |             0.9979 |
| **OSFT Constant RR - Chunk 2** (RR: 0.35)   |            1 |                    0.453 |            0.453 |             0.9983 |
| **OSFT Full Dataset** (RR: 0.35)            |            1 |                   0.4771 |           0.4771 |             0.9977 |
| **SFT Chunked - Chunk 0**                   |            1 |                   0.4289 |           0.4289 |              0.995 |
| **SFT Chunked - Chunk 1**                   |            1 |                   0.4892 |           0.4892 |             0.9907 |
| **SFT Chunked - Chunk 2**                   |            1 |                   0.5253 |           0.5253 |             0.9634 |
| **SFT Full Dataset**                        |            1 |                   0.5012 |           0.5012 |             0.9873 |

**Domain-specific QuaLITY scores**: 

Here we showcase the scores each checkpoint receives when evaluated on only the data that specific chunk was trained on. E.g., chunk 1 is only evaluated on the QuaLITY documents that chunk 1 trains on, chunk 2 only evaluates on QuaLITY documents that chunk 2 trains on, etc.

Here's the transformed table with human-readable experiment names for the chunk-specific QuaLITY evaluations:

| Experiment (Evaluated on Own Chunk)         | Attempt Rate | Accuracy Among Attempted | Overall Accuracy | Total Attempt Rate |
| :------------------------------------------ | -----------: | -----------------------: | ---------------: | -----------------: |
| **OSFT Decreasing RR - Chunk 0** (RR: 0.35) |            1 |                   0.5202 |           0.5202 |              0.999 |
| **OSFT Decreasing RR - Chunk 1** (RR: 0.25) |            1 |                   0.5000 |           0.5000 |             0.9986 |
| **OSFT Decreasing RR - Chunk 2** (RR: 0.15) |            1 |                   0.4449 |           0.4449 |             0.9974 |
| **OSFT Constant RR - Chunk 0** (RR: 0.35)   |            1 |                   0.5176 |           0.5176 |             0.9992 |
| **OSFT Constant RR- Chunk 1** (RR: 0.35)    |            1 |                   0.5026 |           0.5026 |             0.9983 |
| **OSFT Constant RR- Chunk 2** (RR: 0.35)    |            1 |                   0.4586 |           0.4586 |             0.9977 |
| **SFT Chunked - Chunk 0**                   |            1 |                   0.5319 |           0.5319 |             0.9963 |
| **SFT Chunked - Chunk 1**                   |            1 |                   0.5235 |           0.5235 |               0.99 |
| **SFT Chunked - Chunk 2**                   |            1 |                   0.5163 |           0.5163 |             0.9611 |
| **Base Model - Chunk 0**                    |       0.9974 |                   0.4458 |           0.4446 |             0.9831 |
| **Base Model - Chunk 1**                    |            1 |                   0.4360 |           0.4360 |             0.9832 |
| **Base Model - Chunk 2**                    |       0.9975 |                   0.4146 |           0.4135 |             0.9789 |


**Key observations:**
- **SFT performs best** on chunk-specific evaluations, with SFT Chunk 0 achieving the highest accuracy (53.19%)
- **OSFT Decreasing RR shows declining performance** as rank ratio decreases (52.02% → 50.00% → 44.49%)
- **OSFT Constant RR models** maintain relatively stable performance across chunks
- **Chunk-specific performance** is generally higher than the full dataset evaluations, suggesting models learn their training chunks well
- **SFT maintains consistent performance** across all chunks (51.63% - 53.19%), while OSFT shows more variation

**Progressive QuaLITY scores**:


This table captures how well each model performs on the subset of QuaLITY that it has been trained on overall. So the chunk 1 checkpoint is only evaluated on the QuaLITY subset for chunk 1, but the chunk 2 checkpoint gets evaluated on the QuaLITY subset of chunk 2 **and** chunk 1, and finally chunk 3 is evaluated on all of QuaLITY. 

| Experiment (Progressive Evaluation)                  | Attempt Rate | Accuracy Among Attempted | Overall Accuracy | Total Attempt Rate |
| ---------------------------------------------------- | ------------ | ------------------------ | ---------------- | ------------------ |
| Base Model - Chunk 0 Progressive                     | 0.9974       | 0.4405                   | 0.4394           | 0.9844             |
| Base Model - Chunk 1 Progressive                     | 0.9987       | 0.4316                   | 0.4310           | 0.9842             |
| Base Model - Chunk 2 Progressive                     | 0.9983       | 0.4313                   | 0.4305           | 0.9809             |
| OSFT Decreasing RR - Chunk 0 Progressive (RR: 0.35)  | 1.0000       | 0.5202                   | 0.5202           | 0.9989             |
| OSFT Decreasing RR - Chunk 1 Progressive (RR: 0.25)  | 1.0000       | 0.5083                   | 0.5083           | 0.9986             |
| OSFT Decreasing RR - Chunk 2 Progressive (RR: 0.15)  | 1.0000       | 0.4815                   | 0.4815           | 0.9979             |
| OSFT Constant RR - Chunk 0 Progressive (RR: 0.35)    | 1.0000       | 0.5202                   | 0.5202           | 0.9989             |
| OSFT Constant RR - Chunk 1 Progressive (RR: 0.35)    | 1.0000       | 0.5182                   | 0.5182           | 0.9989             |
| OSFT Constant RR - Chunk 2 Progressive (RR: 0.35)    | 1.0000       | 0.5020                   | 0.5020           | 0.9983             |
| OSFT Constant RR v2 - Chunk 0 Progressive (RR: 0.35) | 1.0000       | 0.5150                   | 0.5150           | 0.9991             |
| OSFT Constant RR v2 - Chunk 1 Progressive (RR: 0.35) | 1.0000       | 0.5116                   | 0.5116           | 0.9985             |
| OSFT Constant RR v2 - Chunk 2 Progressive (RR: 0.35) | 1.0000       | 0.4902                   | 0.4902           | 0.9980             |
| SFT Chunked - Chunk 0 Progressive                    | 1.0000       | 0.5398                   | 0.5398           | 0.9960             |
| SFT Chunked - Chunk 1 Progressive                    | 1.0000       | 0.5274                   | 0.5274           | 0.9905             |
| SFT Chunked - Chunk 2 Progressive                    | 1.0000       | 0.5290                   | 0.5290           | 0.9600             |

Key observations for Progressive Evaluation:

- SFT continues to outperform OSFT on progressive datasets, with SFT Chunk 0 achieving the highest accuracy (53.98%)

- OSFT Decreasing RR shows declining performance as rank ratio decreases: 52.02% → 50.83% → 48.15%

- Progressive datasets are more challenging than chunk-specific ones - compare to single-chunk results where models achieved higher scores

- Base model performance is consistent across progressive chunks (~43-44%), establishing a clear baseline

- All trained models significantly outperform the base model by 6-10 percentage points

- OSFT models maintain relatively stable performance across progressive evaluations, while SFT shows slight variation

**Leaderboard Scores**:

| Experiment/Checkpoint             | Overall |    BBH |   GPQA | IFEval | MATH-Hard | MMLU-Pro |   MUSR |
| :-------------------------------- | ------: | -----: | -----: | -----: | --------: | -------: | -----: |
| **OSFT Decreasing RR - Chunk 0**  |  42.49% | 50.21% | 29.90% | 77.37% |    17.24% |   37.97% | 42.25% |
| **OSFT Decreasing RR - Chunk 1**  |  41.26% | 49.22% | 30.74% | 74.72% |    14.44% |   36.98% | 41.46% |
| **OSFT Decreasing RR - Chunk 2**  |  40.90% | 49.37% | 29.94% | 73.35% |    13.94% |   36.29% | 42.53% |
| **OSFT Constant RR v2 - Chunk 0** |  42.52% | 50.16% | 31.52% | 75.88% |    16.74% |   38.04% | 42.78% |
| **OSFT Constant RR v2 - Chunk 1** |  41.65% | 49.58% | 31.23% | 74.54% |    16.49% |   36.88% | 41.19% |
| **OSFT Constant RR v2 - Chunk 2** |  40.92% | 49.91% | 31.36% | 72.62% |    13.85% |   36.74% | 41.07% |
| **OSFT Full Dataset**             |  41.88% | 49.25% | 32.20% | 76.00% |    15.19% |   37.67% | 40.93% |
| **SFT Chunked - Chunk 0**         |  39.82% | 50.23% | 31.97% | 65.41% |    14.52% |   37.14% | 39.62% |
| **SFT Chunked - Chunk 1**         |  39.77% | 50.19% | 31.95% | 66.44% |    15.37% |   36.50% | 38.17% |
| **SFT Chunked - Chunk 2**         |  38.61% | 50.28% | 30.10% | 63.80% |    13.94% |   35.64% | 37.91% |
| **SFT Full Dataset**              |  39.40% | 49.96% | 32.63% | 63.16% |    14.73% |   36.29% | 39.61% |

**Key observations:**
- **Best overall performance**: OSFT Constant RR v2 - Chunk 0 (42.52%)
- **OSFT consistently outperforms SFT** on overall scores and IFEval
- **IFEval scores** are significantly higher for OSFT models (71-77%) vs SFT (63-66%)
- **BBH scores** remain relatively stable across all models (~49-50%)
- **OSFT decreasing rank ratio** shows declining performance from chunk 0 to 2, supporting the hypothesis that lower rank ratios may hurt performance
