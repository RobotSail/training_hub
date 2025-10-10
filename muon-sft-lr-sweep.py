from training_hub import osft, create_algorithm
from itertools import product
import os
import time


EXP_SUFFIX = 'v1-muon-adamw'
PROJECT_NAME = 'muon-sft-lr-sweep'

osft_algo = create_algorithm('osft', 'mini-trainer')

def sft_lr_sweep():
    
    if not os.environ.get('WANDB_API_KEY'):
        raise ValueError('WANDB_API_KEY is not set')

    # Base configuration
    base_model = 'meta-llama/Llama-3.1-8B-Instruct'
    
    # New data paths
    train_data_path = '/mnt/vde/workspace/osilkin/stashed-data/muon-quality/train.jsonl'
    validation_data_path = '/mnt/vde/workspace/osilkin/stashed-data/muon-quality/val.jsonl'
    
    # Muon-specific parameters
    muon_adamw_lr = 5e-6
    muon_momentum = 0.95
    
    # Training configuration
    effective_batch_size = 128
    tokens_per_gpu = 40_000
    max_seq_len = 8196
    num_gpus = 8
    delay_between_runs = 30

    # LR sweep configuration
    # muon_lrs = [5e-6, 1e-5, 2e-5, 5e-5, 1e-4]  # Note: 5e-6 appears twice in your list, I included it once
    # adamw_lrs = [1e-6, 2e-6, 5e-6, 1e-5, 2e-5]
    # muon_lrs = [5e-6, 1e-5, 2e-5, 5e-5, 1e-4]  # Note: 5e-6 appears twice in your list, I included it once
    adamw_lrs = [
        5e-5, 1e-4,  # AdamW should complete Muon's existing LRs
        3e-5, 4e-5, 6e-5 # Next, run a sweep over the area we haven't fully explored
    ]
    muon_lrs = [
        3e-5, 4e-5, 6e-5 # Next, run a sweep over the area we haven't fully explored
    ]
    
    # Create optimizer-lr pairs
    experiments = [
        ('muon', lr) for lr in muon_lrs
    ] + [
        ('adamw', lr) for lr in adamw_lrs
    ]

    for optimizer, lr in experiments:
        run_name = f'sft-{optimizer}-lr-{lr}-{EXP_SUFFIX}'
        print(f"Running experiment: {run_name}")

        extra_kwargs = {
            'learning_rate': lr,
            'optimizer': optimizer,
            # Disable OSFT to run standard SFT
            'osft': False,
            'unfreeze_rank_ratio': 0.0,
        }
        
        if optimizer == 'muon':
            extra_kwargs.update({
                'muon_momentum': muon_momentum, 
                'adamw_learning_rate': muon_adamw_lr,
            })

        retry_count = 0
        success = False
        while not success and retry_count < 3:
            try:
                osft(
                    model_path=base_model,
                    
                    # Training configuration
                    effective_batch_size=effective_batch_size,
                    max_tokens_per_gpu=tokens_per_gpu,
                    max_seq_len=max_seq_len,

                    # Data paths
                    data_path=train_data_path,
                    validation_data_path=validation_data_path,
                    use_processed_dataset=False,

                    # Output directories
                    ckpt_output_dir=f'/mnt/vde/ckpts/sft-{run_name}',
                    data_output_dir='/dev/shm',
                    
                    # Distributed training settings
                    seed=67,
                    nproc_per_node=num_gpus,
                    nnodes=1,
                    node_rank=0,
                    rdzv_endpoint='localhost:1738',
                    rdzv_id=67,
                    
                    # Training schedule
                    save_final_checkpoint=True,
                    checkpoint_at_epoch=False,
                    num_epochs=2,
                    lr_scheduler='cosine',
                    use_liger=True,

                    # Wandb configuration
                    wandb_project=PROJECT_NAME,
                    wandb_run_name=run_name,
    
                    # Validation settings
                    validation_frequency=100,
                    save_best_val_loss=True,
                    val_loss_improvement_threshold=0.01,  # 1% 

                    # Experiment-specific settings
                    **extra_kwargs,
                )
                success = True
                print(f"Successfully completed experiment: {run_name}")
                
            except KeyboardInterrupt:
                print(f"Keyboard interrupt, exiting...")
                break
            except Exception as e:
                print(f"Error running experiment: {e}")
                print(f"Run {run_name} failed, continuing...")
                retry_count += 1
                if retry_count >= 3:
                    print(f"Run {run_name} failed after 3 retries, skipping...")
                    break
            finally:
                if success or retry_count >= 3:
                    print(f"Sleeping for {delay_between_runs} seconds before proceeding to next experiment")
                    time.sleep(delay_between_runs)


if __name__ == '__main__':
    sft_lr_sweep()
