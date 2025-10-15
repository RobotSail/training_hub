"""
Process datasets with the extended Moonlight tokenizer for training.

This script tokenizes JSONL datasets using the Moonlight Extended Tokenizer
which supports adding new special tokens dynamically.

Note: Due to pickling constraints with custom tokenizer classes, this uses
num_cpu_procs=1. For large datasets, consider running in the background.
"""
from instructlab.training.data_process import process_messages_into_input_ids

train_dataset = '/mnt/vde/workspace/osilkin/stashed-data/muon-quality/train.jsonl'
val_dataset = '/mnt/vde/workspace/osilkin/stashed-data/muon-quality/val.jsonl'
test_dataset = '/mnt/vde/workspace/osilkin/stashed-data/muon-quality/test.jsonl'

tokenizer_path = '/mnt/vde/workspace/osilkin/experiment-repos/moonlight-extended-tokenizer'

for ds in [train_dataset, val_dataset, test_dataset]:
    version = ds.split('/')[-1].split('.')[0]
    process_messages_into_input_ids(
        data_path=ds,
        data_output_path=ds.replace('.jsonl', f'_moonlight_{version}_input_ids.jsonl'),
        max_seq_len=8196,
        model_path=tokenizer_path,
        num_cpu_procs=1,  # Must use 1 due to pickling issues with custom tokenizer (cannot use multiprocessing)
        trust_remote_code=True,
    )
