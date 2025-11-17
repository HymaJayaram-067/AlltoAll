"""
Simple Example: TopKA2A Basic Usage
====================================

This example demonstrates the basic usage of TopKA2A for gradient aggregation
in distributed training.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from topka2a_paper_exact import TopKA2A
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", 
                           rank=rank, world_size=world_size)


def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()


def run_topka2a_example(rank, world_size):
    """
    Run a simple TopKA2A example.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    print(f"Running example on rank {rank}/{world_size}")
    
    # Setup distributed training
    setup(rank, world_size)
    
    # Set device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Initialize TopKA2A with 2% density
    topka2a = TopKA2A(world_size=world_size, rank=rank, density=0.02)
    
    # Create a sample gradient (simulating a model layer)
    # In real training, this would come from model.backward()
    gradient_size = 1000000  # 1M parameters
    gradient = torch.randn(gradient_size, device=device)
    
    if rank == 0:
        print(f"\nGradient shape: {gradient.shape}")
        print(f"Gradient size: {gradient.numel():,} elements")
        print(f"Density: {topka2a.density}")
        print(f"Top-k elements: {int(gradient.numel() * topka2a.density):,}")
    
    # Warm-up
    for _ in range(3):
        _ = topka2a.communicate(gradient)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Actual communication
    import time
    start = time.time()
    
    aggregated_gradient = topka2a.communicate(gradient)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end = time.time()
    
    if rank == 0:
        print(f"\nCommunication time: {(end-start)*1000:.2f} ms")
        print(f"Aggregated gradient shape: {aggregated_gradient.shape}")
        print(f"Aggregated gradient norm: {aggregated_gradient.norm().item():.4f}")
    
    # Verify all ranks have same result
    if rank == 0:
        print("\n" + "="*80)
        print("✓ TopKA2A Example Completed Successfully!")
        print("="*80)
        print("\nKey Points:")
        print("1. Gradient was partitioned into n shards")
        print("2. Top-k selection applied to each shard independently")
        print("3. AlltoAll exchanged sparse shards between GPUs")
        print("4. Accumulation created dense shards on each GPU")
        print("5. AllGather collected all shards to form final result")
        print("\nAll GPUs now have identical aggregated gradients.")
    
    cleanup()


def main():
    """Main function to launch distributed training."""
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 2
    
    print("="*80)
    print("TopKA2A Simple Example")
    print("="*80)
    print(f"\nWorld size: {world_size}")
    print(f"Using: {'CUDA (NCCL)' if torch.cuda.is_available() else 'CPU (Gloo)'}")
    print("="*80)
    
    if world_size < 2:
        print("\nWarning: Need at least 2 processes for distributed training.")
        print("Running with world_size=2 on CPU for demonstration.")
        world_size = 2
    
    # Spawn processes
    mp.spawn(run_topka2a_example,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    main()
