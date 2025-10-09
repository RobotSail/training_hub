#!/usr/bin/env python3
"""
Multi-experiment training script for continual learning scenario.

This script implements several experimental conditions:
1. OSFT completed baseline: Full dataset training with OSFT
2. OSFT chunked baseline: Sequential chunk training with constant rank ratio
3. SFT complete baseline: Full dataset training with standard SFT
4. SFT chunked baseline: Sequential chunk training with SFT
5. OSFT experimental: Sequential chunk training with decreasing rank ratio

Each experiment tests different approaches to continual learning on disjoint
data distributions.
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import typer
from glob import glob
from enum import Enum

from training_hub import osft

app = typer.Typer(help="Run continual learning experiments with OSFT and SFT")

# ===== CENTRALIZED HYPERPARAMETERS =====
# Common hyperparameters for all experiments
class HyperParams:
    # Model
    BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Dataset
    DEFAULT_FULL_DATASET_PATH = "/mnt/nvme1n1/experiments/os-cl-scenario-1-experiment-0/synth_knowledge2_0-combined_cut_50x/combined_cut_50x.jsonl"
    
    # OSFT specific
    MAX_UNFREEZE_RANK_RATIO = 0.5  # Starting/max rank ratio
    RANK_RATIO_DECAY = 0.10  # Amount to decrease rank ratio per chunk
    TARGET_PATTERNS = None  # Use default patterns
    
    # Training parameters
    EFFECTIVE_BATCH_SIZE = 128
    MAX_TOKENS_PER_GPU = 15_000
    MAX_SEQ_LEN = 8192
    OSFT_LEARNING_RATE = 5e-6  # learning rate for OSFT experiments
    SFT_LEARNING_RATE = 5e-6   # learning rate for SFT experiments
    NUM_EPOCHS = 2
    
    # Learning rate scheduler
    LR_SCHEDULER = "cosine"
    WARMUP_STEPS = 0
    
    # Hardware/optimization settings
    USE_LIGER = True
    SEED = 42
    
    # Data processing
    USE_PROCESSED_DATASET = False
    UNMASK_MESSAGES = True
    
    # Checkpointing
    CHECKPOINT_AT_EPOCH = False
    SAVE_FINAL_CHECKPOINT = True
    
    # Distributed training
    NPROC_PER_NODE = 8
    NNODES = 1
    NODE_RANK = 0
    RDZV_ID = 123
    RDZV_ENDPOINT = "localhost:29500"
    
    # W&B defaults
    WANDB_PROJECT = "osft-disjoint-distributions"

# =======================================

class ExperimentType(str, Enum):
    """Available experiment types."""
    OSFT_FULL = "osft-full"
    OSFT_CHUNKED = "osft-chunked"
    OSFT_CHUNKED_DECREASING = "osft-chunked-decreasing"
    SFT_FULL = "sft-full"
    SFT_CHUNKED = "sft-chunked"


def find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Find the most recent checkpoint in the hf_format subdirectory."""
    hf_format_dir = ckpt_dir / "hf_format"
    if not hf_format_dir.exists():
        return None
    
    # get all directories under hf_format
    checkpoint_dirs = [d for d in hf_format_dir.iterdir() if d.is_dir()]
    
    # if no directories, check if hf_format itself contains the model
    if not checkpoint_dirs:
        # check if hf_format directory contains model files
        if (hf_format_dir / "config.json").exists():
            return hf_format_dir
        return None
    
    # return the most recently created directory
    return max(checkpoint_dirs, key=lambda x: x.stat().st_ctime)


def load_chunk_metadata(chunk_data_dir: Path) -> dict:
    """Load metadata about data chunks."""
    metadata_file = chunk_data_dir / "metadata.json"
    if not metadata_file.exists():
        typer.echo(f"Error: metadata.json not found in {chunk_data_dir}", err=True)
        typer.echo("Make sure this directory was created by split_training_data.py", err=True)
        raise typer.Exit(1)
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    # validate expected structure
    if "n_chunks" not in metadata:
        typer.echo("Error: Invalid metadata.json - missing n_chunks field", err=True)
        raise typer.Exit(1)
    
    return metadata


def get_full_dataset_path(chunk_data_dir: Path, full_dataset_path: Optional[str] = None) -> Path:
    """Get the path to the full dataset."""
    # use provided path first
    if full_dataset_path:
        full_data_path = Path(full_dataset_path)
    else:
        # try default path
        full_data_path = Path(HyperParams.DEFAULT_FULL_DATASET_PATH)
    
    if not full_data_path.exists():
        # try looking in metadata for original dataset path as fallback
        metadata = load_chunk_metadata(chunk_data_dir)
        if "original_dataset" in metadata:
            full_data_path = Path(metadata["original_dataset"])
    
    if not full_data_path.exists():
        # last resort: try parent directory
        full_data_path = chunk_data_dir.parent / "combined_cut_50x.jsonl"
    
    if not full_data_path.exists():
        typer.echo(f"Error: Cannot find full dataset. Tried:", err=True)
        typer.echo(f"  - Provided path: {full_dataset_path}", err=True)
        typer.echo(f"  - Default path: {HyperParams.DEFAULT_FULL_DATASET_PATH}", err=True)
        typer.echo(f"  - Metadata path (if exists)", err=True)
        typer.echo(f"  - Parent directory: {chunk_data_dir.parent / 'combined_cut_50x.jsonl'}", err=True)
        raise typer.Exit(1)
    
    return full_data_path


def run_training(
    model_path: str,
    data_path: str,
    output_dir: Path,
    data_output_dir: Path,
    wandb_run_name: Optional[str],
    wandb_project: Optional[str],
    use_osft: bool = True,
    unfreeze_rank_ratio: Optional[float] = None,
) -> None:
    """Run a single training session with the specified parameters."""
    # use default rank ratio if not specified
    if unfreeze_rank_ratio is None:
        unfreeze_rank_ratio = HyperParams.MAX_UNFREEZE_RANK_RATIO
    
    # select appropriate learning rate based on training method
    learning_rate = HyperParams.OSFT_LEARNING_RATE if use_osft else HyperParams.SFT_LEARNING_RATE
    
    osft(
        model_path=model_path,
        data_path=data_path,
        ckpt_output_dir=str(output_dir),
        data_output_dir=str(data_output_dir),
        
        # OSFT parameters
        osft=use_osft,
        unfreeze_rank_ratio=unfreeze_rank_ratio,  # always pass a valid value
        
        # training parameters
        effective_batch_size=HyperParams.EFFECTIVE_BATCH_SIZE,
        max_tokens_per_gpu=HyperParams.MAX_TOKENS_PER_GPU,
        max_seq_len=HyperParams.MAX_SEQ_LEN,
        learning_rate=learning_rate,
        num_epochs=HyperParams.NUM_EPOCHS,
        
        # scheduler
        lr_scheduler=HyperParams.LR_SCHEDULER,
        warmup_steps=HyperParams.WARMUP_STEPS,
        
        # optimization
        use_liger=HyperParams.USE_LIGER,
        seed=HyperParams.SEED,
        
        # data processing
        use_processed_dataset=HyperParams.USE_PROCESSED_DATASET,
        unmask_messages=HyperParams.UNMASK_MESSAGES,
        
        # checkpointing
        checkpoint_at_epoch=HyperParams.CHECKPOINT_AT_EPOCH,
        save_final_checkpoint=HyperParams.SAVE_FINAL_CHECKPOINT,
        
        # distributed training
        nproc_per_node=HyperParams.NPROC_PER_NODE,
        nnodes=HyperParams.NNODES,
        node_rank=HyperParams.NODE_RANK,
        rdzv_id=HyperParams.RDZV_ID,
        rdzv_endpoint=HyperParams.RDZV_ENDPOINT,
        
        # wandb
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )


def train_full_osft(
    data_path: Path,
    output_dir: Path,
    data_output_dir: Path,
    wandb_project: Optional[str],
    wandb_run_prefix: str,
) -> None:
    """Baseline 1: OSFT on full dataset."""
    typer.echo("\n" + "="*60)
    typer.echo("Running OSFT Full Dataset Baseline")
    typer.echo("="*60 + "\n")
    
    output_path = output_dir / "osft_full"
    wandb_run_name = f"{wandb_run_prefix}_osft_full" if wandb_project else None
    
    typer.echo(f"Model: {HyperParams.BASE_MODEL}")
    typer.echo(f"Training data: {data_path}")
    typer.echo(f"Output directory: {output_path}")
    typer.echo(f"Rank ratio: {HyperParams.MAX_UNFREEZE_RANK_RATIO}")
    typer.echo(f"Learning rate: {HyperParams.OSFT_LEARNING_RATE}")
    if wandb_run_name:
        typer.echo(f"W&B run name: {wandb_run_name}")
    
    run_training(
        model_path=HyperParams.BASE_MODEL,
        data_path=str(data_path),
        output_dir=output_path,
        data_output_dir=data_output_dir / "osft_full",
        wandb_run_name=wandb_run_name,
        wandb_project=wandb_project,
        use_osft=True,
        unfreeze_rank_ratio=HyperParams.MAX_UNFREEZE_RANK_RATIO,
    )
    
    typer.echo("\n✓ Completed OSFT full dataset baseline")


def train_full_sft(
    data_path: Path,
    output_dir: Path,
    data_output_dir: Path,
    wandb_project: Optional[str],
    wandb_run_prefix: str,
) -> None:
    """Baseline 3: SFT on full dataset."""
    typer.echo("\n" + "="*60)
    typer.echo("Running SFT Full Dataset Baseline")
    typer.echo("="*60 + "\n")
    
    output_path = output_dir / "sft_full"
    wandb_run_name = f"{wandb_run_prefix}_sft_full" if wandb_project else None
    
    typer.echo(f"Model: {HyperParams.BASE_MODEL}")
    typer.echo(f"Training data: {data_path}")
    typer.echo(f"Output directory: {output_path}")
    typer.echo(f"Learning rate: {HyperParams.SFT_LEARNING_RATE}")
    if wandb_run_name:
        typer.echo(f"W&B run name: {wandb_run_name}")
    
    run_training(
        model_path=HyperParams.BASE_MODEL,
        data_path=str(data_path),
        output_dir=output_path,
        data_output_dir=data_output_dir / "sft_full",
        wandb_run_name=wandb_run_name,
        wandb_project=wandb_project,
        use_osft=False,
    )
    
    typer.echo("\n✓ Completed SFT full dataset baseline")


def train_chunked_osft(
    chunk_data_dir: Path,
    output_dir: Path,
    data_output_dir: Path,
    wandb_project: Optional[str],
    wandb_run_prefix: str,
    start_from_chunk: int = 0,
) -> None:
    """Baseline 2: OSFT on chunks with constant rank ratio."""
    typer.echo("\n" + "="*60)
    typer.echo("Running OSFT Chunked Baseline (constant rank ratio)")
    typer.echo("="*60 + "\n")
    
    metadata = load_chunk_metadata(chunk_data_dir)
    n_chunks = metadata["n_chunks"]
    
    typer.echo(f"Found {n_chunks} chunks in {chunk_data_dir}")
    typer.echo(f"Starting from chunk {start_from_chunk + 1}")
    typer.echo(f"Using constant rank ratio: {HyperParams.MAX_UNFREEZE_RANK_RATIO}")
    
    output_base = output_dir / "osft_chunked"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # determine starting model
    if start_from_chunk == 0:
        current_model = HyperParams.BASE_MODEL
    else:
        prev_chunk_dir = output_base / f"chunk_{start_from_chunk - 1}"
        prev_checkpoint = find_latest_checkpoint(prev_chunk_dir)
        if prev_checkpoint is None:
            typer.echo(f"Error: No checkpoint found for chunk {start_from_chunk}", err=True)
            raise typer.Exit(1)
        current_model = str(prev_checkpoint)
    
    # train on each chunk
    for chunk_idx in range(start_from_chunk, n_chunks):
        typer.echo(f"\nTraining on chunk {chunk_idx + 1} of {n_chunks}")
        
        chunk_dir = chunk_data_dir / f"chunk_{chunk_idx}"
        training_data = chunk_dir / "training.jsonl"
        chunk_output_dir = output_base / f"chunk_{chunk_idx}"
        
        wandb_run_name = f"{wandb_run_prefix}_osft_chunked_{chunk_idx + 1}" if wandb_project else None
        
        run_training(
            model_path=current_model,
            data_path=str(training_data),
            output_dir=chunk_output_dir,
            data_output_dir=data_output_dir / "osft_chunked" / f"chunk_{chunk_idx}",
            wandb_run_name=wandb_run_name,
            wandb_project=wandb_project,
            use_osft=True,
            unfreeze_rank_ratio=HyperParams.MAX_UNFREEZE_RANK_RATIO,
        )
        
        # update model for next chunk
        if chunk_idx < n_chunks - 1:
            next_checkpoint = find_latest_checkpoint(chunk_output_dir)
            if next_checkpoint is None:
                typer.echo(f"Error: No checkpoint found after chunk {chunk_idx + 1}", err=True)
                raise typer.Exit(1)
            current_model = str(next_checkpoint)
    
    typer.echo("\n✓ Completed OSFT chunked baseline")


def train_chunked_osft_decreasing_rank(
    chunk_data_dir: Path,
    output_dir: Path,
    data_output_dir: Path,
    wandb_project: Optional[str],
    wandb_run_prefix: str,
    start_from_chunk: int = 0,
) -> None:
    """Hypothesis experiment: OSFT on chunks with decreasing rank ratio."""
    typer.echo("\n" + "="*60)
    typer.echo("Running OSFT Chunked Experiment (decreasing rank ratio)")
    typer.echo("="*60 + "\n")
    
    metadata = load_chunk_metadata(chunk_data_dir)
    n_chunks = metadata["n_chunks"]
    
    typer.echo(f"Found {n_chunks} chunks in {chunk_data_dir}")
    typer.echo(f"Starting from chunk {start_from_chunk + 1}")
    typer.echo(f"Rank ratio schedule: {HyperParams.MAX_UNFREEZE_RANK_RATIO} → "
               f"{HyperParams.MAX_UNFREEZE_RANK_RATIO - HyperParams.RANK_RATIO_DECAY} → "
               f"{HyperParams.MAX_UNFREEZE_RANK_RATIO - 2 * HyperParams.RANK_RATIO_DECAY}...")
    
    output_base = output_dir / "osft_chunked_decreasing"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # determine starting model
    if start_from_chunk == 0:
        current_model = HyperParams.BASE_MODEL
    else:
        prev_chunk_dir = output_base / f"chunk_{start_from_chunk - 1}"
        prev_checkpoint = find_latest_checkpoint(prev_chunk_dir)
        if prev_checkpoint is None:
            typer.echo(f"Error: No checkpoint found for chunk {start_from_chunk}", err=True)
            raise typer.Exit(1)
        current_model = str(prev_checkpoint)
    
    # train on each chunk with decreasing rank ratio
    for chunk_idx in range(start_from_chunk, n_chunks):
        # calculate rank ratio for this chunk
        rank_ratio = HyperParams.MAX_UNFREEZE_RANK_RATIO - (chunk_idx * HyperParams.RANK_RATIO_DECAY)
        rank_ratio = max(rank_ratio, 0.05)  # ensure we don't go below 0.05
        
        typer.echo(f"\nTraining on chunk {chunk_idx + 1} of {n_chunks}")
        typer.echo(f"Rank ratio: {rank_ratio}")
        
        chunk_dir = chunk_data_dir / f"chunk_{chunk_idx}"
        training_data = chunk_dir / "training.jsonl"
        chunk_output_dir = output_base / f"chunk_{chunk_idx}"
        
        wandb_run_name = f"{wandb_run_prefix}_osft_decreasing_{chunk_idx + 1}" if wandb_project else None
        
        run_training(
            model_path=current_model,
            data_path=str(training_data),
            output_dir=chunk_output_dir,
            data_output_dir=data_output_dir / "osft_chunked_decreasing" / f"chunk_{chunk_idx}",
            wandb_run_name=wandb_run_name,
            wandb_project=wandb_project,
            use_osft=True,
            unfreeze_rank_ratio=rank_ratio,
        )
        
        # update model for next chunk
        if chunk_idx < n_chunks - 1:
            next_checkpoint = find_latest_checkpoint(chunk_output_dir)
            if next_checkpoint is None:
                typer.echo(f"Error: No checkpoint found after chunk {chunk_idx + 1}", err=True)
                raise typer.Exit(1)
            current_model = str(next_checkpoint)
    
    typer.echo("\n✓ Completed OSFT chunked experiment with decreasing rank ratio")


def train_chunked_sft(
    chunk_data_dir: Path,
    output_dir: Path,
    data_output_dir: Path,
    wandb_project: Optional[str],
    wandb_run_prefix: str,
    start_from_chunk: int = 0,
) -> None:
    """Baseline 4: SFT on chunks."""
    typer.echo("\n" + "="*60)
    typer.echo("Running SFT Chunked Baseline")
    typer.echo("="*60 + "\n")
    
    metadata = load_chunk_metadata(chunk_data_dir)
    n_chunks = metadata["n_chunks"]
    
    typer.echo(f"Found {n_chunks} chunks in {chunk_data_dir}")
    typer.echo(f"Starting from chunk {start_from_chunk + 1}")
    
    output_base = output_dir / "sft_chunked"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # determine starting model
    if start_from_chunk == 0:
        current_model = HyperParams.BASE_MODEL
    else:
        prev_chunk_dir = output_base / f"chunk_{start_from_chunk - 1}"
        prev_checkpoint = find_latest_checkpoint(prev_chunk_dir)
        if prev_checkpoint is None:
            typer.echo(f"Error: No checkpoint found for chunk {start_from_chunk}", err=True)
            raise typer.Exit(1)
        current_model = str(prev_checkpoint)
    
    # train on each chunk
    for chunk_idx in range(start_from_chunk, n_chunks):
        typer.echo(f"\nTraining on chunk {chunk_idx + 1} of {n_chunks}")
        
        chunk_dir = chunk_data_dir / f"chunk_{chunk_idx}"
        training_data = chunk_dir / "training.jsonl"
        chunk_output_dir = output_base / f"chunk_{chunk_idx}"
        
        wandb_run_name = f"{wandb_run_prefix}_sft_chunked_{chunk_idx + 1}" if wandb_project else None
        
        run_training(
            model_path=current_model,
            data_path=str(training_data),
            output_dir=chunk_output_dir,
            data_output_dir=data_output_dir / "sft_chunked" / f"chunk_{chunk_idx}",
            wandb_run_name=wandb_run_name,
            wandb_project=wandb_project,
            use_osft=False,
        )
        
        # update model for next chunk
        if chunk_idx < n_chunks - 1:
            next_checkpoint = find_latest_checkpoint(chunk_output_dir)
            if next_checkpoint is None:
                typer.echo(f"Error: No checkpoint found after chunk {chunk_idx + 1}", err=True)
                raise typer.Exit(1)
            current_model = str(next_checkpoint)
    
    typer.echo("\n✓ Completed SFT chunked baseline")


@app.command()
def train(
    experiment: ExperimentType = typer.Argument(
        ...,
        help="Type of experiment to run"
    ),
    chunk_data_dir: Path = typer.Argument(
        ...,
        help="Directory containing chunk_0, chunk_1, etc. (for chunked experiments) or path to full dataset",
        exists=True
    ),
    output_base_dir: Path = typer.Option(
        "./experiment_outputs",
        "--output-dir", "-o",
        help="Base directory for all experiment outputs"
    ),
    wandb_project: Optional[str] = typer.Option(
        HyperParams.WANDB_PROJECT,
        "--wandb-project", "-wp",
        help="Weights & Biases project name"
    ),
    wandb_run_prefix: str = typer.Option(
        ...,
        "--wandb-run-prefix", "-wr",
        help="Prefix for W&B run names"
    ),
    start_from_chunk: int = typer.Option(
        0,
        "--start-from-chunk", "-s",
        help="Which chunk to start from (for chunked experiments)"
    ),
    data_output_dir: Optional[Path] = typer.Option(
        None,
        "--data-output-dir", "-d",
        help="Directory for data processing outputs"
    ),
    full_dataset_path: Optional[str] = typer.Option(
        None,
        "--full-dataset-path", "-f",
        help=f"Path to full dataset for full training experiments (default: {HyperParams.DEFAULT_FULL_DATASET_PATH})"
    ),
):
    """Run continual learning experiments with different training strategies."""
    
    # create output directories
    output_base_dir.mkdir(parents=True, exist_ok=True)
    if data_output_dir is None:
        data_output_dir = output_base_dir / "_data_processing"
    data_output_dir.mkdir(parents=True, exist_ok=True)
    
    # run the appropriate experiment
    if experiment == ExperimentType.OSFT_FULL:
        # for full dataset experiments, chunk_data_dir should point to the dataset file
        if chunk_data_dir.is_dir():
            # if it's a directory, try to find the full dataset
            data_path = get_full_dataset_path(chunk_data_dir, full_dataset_path)
        else:
            data_path = chunk_data_dir
        
        train_full_osft(
            data_path=data_path,
            output_dir=output_base_dir,
            data_output_dir=data_output_dir,
            wandb_project=wandb_project,
            wandb_run_prefix=wandb_run_prefix,
        )
        
    elif experiment == ExperimentType.SFT_FULL:
        # for full dataset experiments, chunk_data_dir should point to the dataset file
        if chunk_data_dir.is_dir():
            # if it's a directory, try to find the full dataset
            data_path = get_full_dataset_path(chunk_data_dir, full_dataset_path)
        else:
            data_path = chunk_data_dir
        
        train_full_sft(
            data_path=data_path,
            output_dir=output_base_dir,
            data_output_dir=data_output_dir,
            wandb_project=wandb_project,
            wandb_run_prefix=wandb_run_prefix,
        )
        
    elif experiment == ExperimentType.OSFT_CHUNKED:
        if not chunk_data_dir.is_dir():
            typer.echo("Error: For chunked experiments, provide the directory containing chunks", err=True)
            raise typer.Exit(1)
        
        train_chunked_osft(
            chunk_data_dir=chunk_data_dir,
            output_dir=output_base_dir,
            data_output_dir=data_output_dir,
            wandb_project=wandb_project,
            wandb_run_prefix=wandb_run_prefix,
            start_from_chunk=start_from_chunk,
        )
        
    elif experiment == ExperimentType.OSFT_CHUNKED_DECREASING:
        if not chunk_data_dir.is_dir():
            typer.echo("Error: For chunked experiments, provide the directory containing chunks", err=True)
            raise typer.Exit(1)
        
        train_chunked_osft_decreasing_rank(
            chunk_data_dir=chunk_data_dir,
            output_dir=output_base_dir,
            data_output_dir=data_output_dir,
            wandb_project=wandb_project,
            wandb_run_prefix=wandb_run_prefix,
            start_from_chunk=start_from_chunk,
        )
        
    elif experiment == ExperimentType.SFT_CHUNKED:
        if not chunk_data_dir.is_dir():
            typer.echo("Error: For chunked experiments, provide the directory containing chunks", err=True)
            raise typer.Exit(1)
        
        train_chunked_sft(
            chunk_data_dir=chunk_data_dir,
            output_dir=output_base_dir,
            data_output_dir=data_output_dir,
            wandb_project=wandb_project,
            wandb_run_prefix=wandb_run_prefix,
            start_from_chunk=start_from_chunk,
        )
    
    typer.echo(f"\n{'='*60}")
    typer.echo(f"✓ Experiment '{experiment}' completed successfully!")
    typer.echo(f"Results saved to: {output_base_dir}")
    typer.echo(f"{'='*60}")


@app.command()
def list_experiments():
    """List all available experiment types."""
    typer.echo("\nAvailable experiments:")
    typer.echo("-" * 40)
    for exp in ExperimentType:
        typer.echo(f"  {exp.value}")
        if exp == ExperimentType.OSFT_FULL:
            typer.echo("    → Baseline 1: OSFT on full dataset")
        elif exp == ExperimentType.OSFT_CHUNKED:
            typer.echo("    → Baseline 2: OSFT on chunks (constant rank ratio)")
        elif exp == ExperimentType.SFT_FULL:
            typer.echo("    → Baseline 3: SFT on full dataset")
        elif exp == ExperimentType.SFT_CHUNKED:
            typer.echo("    → Baseline 4: SFT on chunks")
        elif exp == ExperimentType.OSFT_CHUNKED_DECREASING:
            typer.echo("    → Hypothesis: OSFT on chunks (decreasing rank ratio)")
    typer.echo()


if __name__ == "__main__":
    app()
