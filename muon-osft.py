from training_hub import osft, create_algorithm
from itertools import product
# from mini_trainer import run_training, TorchrunArgs, TrainingArgs
import os
import time


EXP_SUFFIX = 'v4-adamuon_ema'
OSFT_DISABLED = True  # for our baselines
PROJECT_NAME = 'sft-muon-quality-lr-sweep'

osft_algo = create_algorithm('osft', 'mini-trainer')

def sft_lr_sweep():
    
    if not os.environ.get('WANDB_API_KEY'):
        raise ValueError('WANDB_API_KEY is not set')


    # LR that Muon will use for the AdamW parameters
    base_model = 'meta-llama/Llama-3.1-8B-Instruct'
    
    # we have a cached dataset already
    # base_dataset = '/dev/shm/data.jsonl'
    # use_processed_dataset = True
    use_processed_dataset = False
    base_dataset='/mnt/nvme0n1/datasets/quality-knowledge2.0/combined_cut_50x.jsonl/combined_cut_50x.jsonl'
    
    muon_adamw_lr = 5e-6
    base_lr = 1e-6
    muon_momentum = 0.95
    effective_batch_size = 128
    tokens_per_gpu = 15000
    max_seq_len = 8196
    num_gpus = 8
    osft_unfreeze_rank_ratio = 0.5
    validation_split = 0.2
    delay_between_runs = 30

    optimizers = [
        'muon',
        'adamw',
    ]
    learning_rates = [
        # base_lr,  # 1e-6
        # base_lr * 5,   # 5e-6
        # base_lr * 20,  # 2e-5 --> use known lr
        2e-5,
        # 1e-4,  # 2e-5 --> use known lr
        # 2.5e-5,  # this one worked well for Muon, so let's try for Adamw
        # 5e-5,
        # base_lr * 100, # 1e-4
    ]
    # Create all combinations of optimizers and learning rates

    for optimizer, lr in product(optimizers, learning_rates):
        run_name = f'osft-{optimizer}-lr-{lr}-{EXP_SUFFIX}'
        print(f"Running experiment: {run_name}")

        extra_kwargs = {
            'learning_rate': lr,
            'optimizer': optimizer,
            # 'optimizer': 'adamuon',
        }
        if optimizer == 'muon':
            extra_kwargs.update({
                'muon_momentum': muon_momentum, 
                'adamw_learning_rate': muon_adamw_lr,
                # 'adamw_beta1': 0.8,
                # 'adamw_beta2': 0.98,
                
                # try doing grad norm clip 2.0 here, since muon stabilizes the gradients
                'grad_norm_clip': 1.0,
            })
        # if optimizer == 'adamw':
        #     extra_kwargs.update({
        #         'adamw_beta1': 0.9,
        #         'adamw_beta2': 0.98,
        #         'grad_norm_clip': 1.0,
        #     })

        retry_count = 0
        success = False
        while not success and retry_count < 3:
            try:
                osft(
                    model_path=base_model,
                    # data_path = '/dev/shm/data.jsonl',
                    # data_path='/mnt/nvm0n1/outputs/os-test-cpu-offload/_internal_data_processing/data.jsonl',
                    effective_batch_size=effective_batch_size,
                    max_tokens_per_gpu=tokens_per_gpu,
                    max_seq_len=max_seq_len,  # Required parameter not in original call
                    # adamw learning rate
                    # learning_rate=5e-6,

                    # data
                    data_path=base_dataset,
                    use_processed_dataset=use_processed_dataset,

                    ckpt_output_dir=f'/mnt/vde/ckpts/os-{run_name}',
                    data_output_dir='/dev/shm',
                    unfreeze_rank_ratio=osft_unfreeze_rank_ratio,
                    seed=67,
                    nproc_per_node=num_gpus,
                    nnodes=1,
                    node_rank=0,
                    rdzv_endpoint='localhost:1738',
                    rdzv_id=67,
                    save_final_checkpoint=True,
                    checkpoint_at_epoch=False,
                    num_epochs=2,
                    lr_scheduler='cosine',
                    use_liger=True,

                    # wandb
                    wandb_project=project_name,
                    wandb_run_name=run_name,
                    validation_split=validation_split,
    
                    # validation settings
                    validation_frequency=100,
                    save_best_val_loss=True,
                    val_loss_improvement_threshold=0.01,

                    # our settings go here
                    **extra_kwargs,
                )
                success = True
            except KeyboardInterrupt:
                print(f"Keyboard interrupt, exiting...")
                break
            except Exception as e:
                print(f"Error running experiment: {e}")
                print(f"run {run_name} failed, continuing...")
                retry_count += 1
                if retry_count >= 3:
                    print(f"run {run_name} failed after 3 retries, skipping...")
                    break
            finally:
                print(f"Sleeping for {delay_between_runs} seconds before proceeding")
                time.sleep(delay_between_runs)



def lr_sweep():
    
    if not os.environ.get('WANDB_API_KEY'):
        raise ValueError('WANDB_API_KEY is not set')


    # LR that Muon will use for the AdamW parameters
    base_model = 'meta-llama/Llama-3.1-8B-Instruct'
    
    # we have a cached dataset already
    # base_dataset = '/dev/shm/data.jsonl'
    # use_processed_dataset = True
    use_processed_dataset = False
    base_dataset='/mnt/nvme0n1/datasets/quality-knowledge2.0/combined_cut_50x.jsonl/combined_cut_50x.jsonl'
    
    muon_adamw_lr = 5e-6
    base_lr = 1e-6
    muon_momentum = 0.95
    effective_batch_size = 128
    tokens_per_gpu = 15000
    max_seq_len = 8196
    num_gpus = 8
    osft_unfreeze_rank_ratio = 0.5
    validation_split = 0.2
    delay_between_runs = 30

    project_name = 'osft-muon-quality-lr-sweep'

    # use 4 LRs
    # optimizers = ['muon', 'adamw']
    # run only muon
    optimizers = [
        'adamuon',
        # 'muon',
        # 'adamw',
    ]
    learning_rates = [
        # base_lr,  # 1e-6
        # base_lr * 5,   # 5e-6
        # base_lr * 20,  # 2e-5 --> use known lr
        2e-5,
        # 1e-4,  # 2e-5 --> use known lr
        # 2.5e-5,  # this one worked well for Muon, so let's try for Adamw
        # 5e-5,
        # base_lr * 100, # 1e-4
    ]
    # Create all combinations of optimizers and learning rates

    for optimizer, lr in product(optimizers, learning_rates):
        run_name = f'osft-{optimizer}-lr-{lr}-{EXP_SUFFIX}'
        print(f"Running experiment: {run_name}")

        extra_kwargs = {
            'learning_rate': lr,
            'optimizer': optimizer,
            # 'optimizer': 'adamuon',
        }
        if optimizer == 'muon':
            extra_kwargs.update({
                'muon_momentum': muon_momentum, 
                'adamw_learning_rate': muon_adamw_lr,
                # 'adamw_beta1': 0.8,
                # 'adamw_beta2': 0.98,
                
                # try doing grad norm clip 2.0 here, since muon stabilizes the gradients
                'grad_norm_clip': 1.0,
            })
        # if optimizer == 'adamw':
        #     extra_kwargs.update({
        #         'adamw_beta1': 0.9,
        #         'adamw_beta2': 0.98,
        #         'grad_norm_clip': 1.0,
        #     })

        retry_count = 0
        success = False
        while not success and retry_count < 3:
            try:
                osft(
                    model_path=base_model,
                    # data_path = '/dev/shm/data.jsonl',
                    # data_path='/mnt/nvm0n1/outputs/os-test-cpu-offload/_internal_data_processing/data.jsonl',
                    effective_batch_size=effective_batch_size,
                    max_tokens_per_gpu=tokens_per_gpu,
                    max_seq_len=max_seq_len,  # Required parameter not in original call
                    # adamw learning rate
                    # learning_rate=5e-6,

                    # data
                    data_path=base_dataset,
                    use_processed_dataset=use_processed_dataset,

                    ckpt_output_dir=f'/mnt/vde/ckpts/os-{run_name}',
                    data_output_dir='/dev/shm',
                    unfreeze_rank_ratio=osft_unfreeze_rank_ratio,
                    seed=67,
                    nproc_per_node=num_gpus,
                    nnodes=1,
                    node_rank=0,
                    rdzv_endpoint='localhost:1738',
                    rdzv_id=67,
                    save_final_checkpoint=True,
                    checkpoint_at_epoch=False,
                    num_epochs=2,
                    lr_scheduler='cosine',
                    use_liger=True,

                    # wandb
                    wandb_project=project_name,
                    wandb_run_name=run_name,
                    validation_split=validation_split,
    
                    # validation settings
                    validation_frequency=100,
                    save_best_val_loss=True,
                    val_loss_improvement_threshold=0.01,

                    # our settings go here
                    **extra_kwargs,
                )
                success = True
            except KeyboardInterrupt:
                print(f"Keyboard interrupt, exiting...")
                break
            except Exception as e:
                print(f"Error running experiment: {e}")
                print(f"run {run_name} failed, continuing...")
                retry_count += 1
                if retry_count >= 3:
                    print(f"run {run_name} failed after 3 retries, skipping...")
                    break
            finally:
                print(f"Sleeping for {delay_between_runs} seconds before proceeding")
                time.sleep(delay_between_runs)


if __name__ == '__main__':
    lr_sweep()
