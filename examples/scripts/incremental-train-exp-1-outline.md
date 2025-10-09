## Experiment 1 - multiple chunks from single data-generating distribution

 Incremental Training Experiment

**Results on QuaLITY** (EntityGraph)
	
| Experiment                                                       | Attempt Rate | Accuracy Among Attempted | Overall Accuracy | Total Attempt Rate |
| :--------------------------------------------------------------- | -----------: | -----------------------: | ---------------: | -----------------: |
| Llama-3.1-8b-instruct-baseline                                   |       0.9976 |                   0.4179 |           0.4169 |              0.973 |
| First Chunk (BS 128, LR 5e-6, 1 epoch, cosine LR, RR=0.5)        |            1 |                   0.4361 |           0.4361 |             0.9976 |
| Second Chunk (BS 128, LR 5e-6, 1 epoch, cosine LR, rr=0.5)       |            1 |                   0.4819 |           0.4819 |             0.9979 |
| Third Chunk (BS 128, LR 5e-6, 1 epoch, cosine LR, rr=0.5)        |            1 |                    0.506 |            0.506 |             0.9982 |
| OSFT, Full Dataset (BS 128, LR 5e-6, 1 epoch, cosine LR, rr=0.5) |            1 |                    0.494 |            0.494 |             0.9986 |
| SFT baseline, full dataset (BS 128, LR 5e-6, 1 epoch, cosine LR) |            1 |                   0.5181 |           0.5181 |              0.998 |
| SFT baseline, chunk 1 (BS 128, LR 5e-6, 1 epoch, cosine LR)      |            1 |                   0.4506 |           0.4506 |             0.9914 |
| SFT baseline, chunk 2 (BS 128, LR 5e-6, 1 epoch, cosine LR)      |            1 |                   0.5229 |           0.5229 |             0.9926 |
| SFT baseline, chunk 3 (BS 128, LR 5e-6, 1 epoch, cosine LR)      |            1 |                   0.5325 |           0.5325 |             0.9363 |


Now we assess each checkpoint's leaderboard score. For this evaluation, I used the evaluation script located in https://github.com/instructlab/eval/blob/main/scripts/evaluate_best_checkpoint.py, using only default settings.

| Experiment                                                       | Overall   | BBH       | GPQA      | MMLU-Pro  | MUSR      | IFEval    | MATH-Hard |
| ---------------------------------------------------------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| meta-llama/Llama-3.1-8b-instruct baseline                        | 41.72     | **51.01** | 27.25     | 37.78     | 38.57     | **77.55** | **18.13** |
| First Chunk (BS 128, LR 5e-6, 1 epoch, cosine LR, RR=0.5)        | **42.61** | 50.01     | **31.91** | **37.87** | 43.06     | 76.61     | 16.19     |
| Second Chunk (BS 128, LR 5e-6, 1 epoch, cosine LR, rr=0.5)       | 41.28     | 49.18     | 30.06     | **37.87** | **44.51** | 71.35     | 14.71     |
| Third Chunk (BS 128, LR 5e-6, 1 epoch, cosine LR, rr=0.5)        | 40.70     | 49.51     | 31.77     | 37.06     | 41.19     | 70.40     | 14.29     |
| Full dataset (BS 128, LR 5e-6, 1 epoch, cosine LR, rr=0.5)       | 40.93     | 49.00     | 31.90     | 37.39     | 39.61     | 73.32     | 14.35     |
| SFT baseline, full dataset (BS 128, LR 5e-6, 1 epoch, cosine LR) | 39.50     | 49.52     | 31.22     | 37.15     | 39.60     | 65.05     | 14.48     |
| SFT baseline, chunk 1 (BS 128, LR 5e-6, 1 epoch, cosine LR)      | 40.23     | 49.81     | 31.68     | 38.09     | 41.l74    | 66.06     | 14.02     |
| SFT baseline, chunk 2 (BS 128, LR 5e-6, 1 epoch, cosine LR)      | 38.95     | 48.95     | 31.81     | 37.71     | 41.35     | 59.67     | 14.21     |
| SFT baseline, chunk 3 (BS 128, LR 5e-6, 1 epoch, cosine LR)      | 36.14     | 49.84     | 29.13     | 36.86     | 36.17     | 49.32     | 15.52     |

**Incremental training experiment, 2 epochs** 

	
| Experiment                                                       | Attempt Rate | Accuracy Among Attempted | Overall Accuracy | Total Attempt Rate |
| :--------------------------------------------------------------- | -----------: | -----------------------: | ---------------: | -----------------: |
| Llama-3.1-8b-instruct-baseline                                   |       0.9976 |                   0.4179 |           0.4169 |              0.973 |
| First Chunk (BS 128, LR 5e-6, 2 epoch, cosine LR, RR=0.5)        |            1 |                   0.4554 |           0.4554 |             0.9988 |
| Second Chunk (BS 128, LR 5e-6, 2 epoch, cosine LR, rr=0.5)       |            1 |                   0.4747 |           0.4747 |             0.9976 |
| Third Chunk (BS 128, LR 5e-6, 2 epoch, cosine LR, rr=0.5)        |            1 |                   0.5133 |           0.5133 |              0.998 |
| OSFT, Full Dataset (BS 128, LR 5e-6, 2 epoch, cosine LR, rr=0.5) |            1 |                   0.4795 |           0.4795 |             0.9965 |
| SFT baseline, full dataset (BS 128, LR 5e-6, 2 epoch, cosine LR) |            1 |                   0.5205 |           0.5205 |             0.9944 |
| SFT baseline, chunk 1 (BS 128, LR 5e-6, 2 epoch, cosine LR)      |            1 |                   0.4482 |           0.4482 |             0.9907 |
| SFT baseline, chunk 2 (BS 128, LR 5e-6, 2 epoch, cosine LR)      |            1 |                   0.5422 |           0.5422 |             0.9833 |
| SFT baseline, chunk 3 (BS 128, LR 5e-6, 2 epoch, cosine LR)      |            1 |                   0.5349 |           0.5349 |             0.9801 |



| Experiment                                                       | Overall   | BBH       | GPQA      | MMLU-Pro  | MUSR      | IFEval    | MATH-Hard |
| ---------------------------------------------------------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| meta-llama/Llama-3.1-8b-instruct baseline                        | 41.72     | **51.01** | 27.25     | 37.78     | 38.57     | **77.55** | **18.13** |
| First Chunk (BS 128, LR 5e-6, 1 epoch, cosine LR, RR=0.5)        | **42.88** | 49.37     | **34.07** | **37.99** | **43.86** | 76.95     | 15.07     |
| Second Chunk (BS 128, LR 5e-6, 1 epoch, cosine LR, rr=0.5)       | 40.84     | 47.88     | 30.76     | 36.88     | 44.37     | 71.65     | 13.52     |
| Third Chunk (BS 128, LR 5e-6, 1 epoch, cosine LR, rr=0.5)        | 40.39     | 48.98     | 32.33     | 36.63     | 39.19     | 70.70     | 14.54     |
| Full dataset (BS 128, LR 5e-6, 1 epoch, cosine LR, rr=0.5)       | 40.58     | 49.26     | 31.00     | 37.07     | 40.15     | 70.40     | 15.61     |
| SFT baseline, full dataset (BS 128, LR 5e-6, 1 epoch, cosine LR) | 39.79     | 49.14     | 31.90     | 36.95     | 38.30     | 68.06     | 14.40     |
| SFT baseline, chunk 1 (BS 128, LR 5e-6, 1 epoch, cosine LR)      | 40.26     | 49.10     | 32.12     | 37.65     | 43.33     | 65.65     | 13.71     |
| SFT baseline, chunk 2 (BS 128, LR 5e-6, 1 epoch, cosine LR)      | 39.50     | 48.07     | 32.42     | 37.11     | 41.73     | 64.29     | 13.40     |
| SFT baseline, chunk 3 (BS 128, LR 5e-6, 1 epoch, cosine LR)      | 37.54     | 49.05     | 30.67     | 36.08     | 37.23     | 57.69     | 14.52     |
