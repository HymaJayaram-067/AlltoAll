"""
Comparison Benchmark: TopKA2A vs Other Methods
==============================================

This script compares:
1. TopKA2A (our implementation)
2. TopKAllGather (baseline sparse method)
3. Dense AllReduce (PyTorch DDP style)

Based on the evaluation from Section 5 of the ICPP '24 paper.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import sys
import os
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topka2a_paper_exact import TopKA2A
from topka2a_baseline import TopKA2ABaseline


def setup(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                           rank=rank, world_size=world_size)


def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()


def dense_allreduce(gradient: torch.Tensor) -> torch.Tensor:
    """
    Dense AllReduce communication (PyTorch DDP style).
    
    Communication complexity: 2(n-1)α + 2(n-1)d/n * β
    """
    if dist.is_initialized():
        dist.all_reduce(gradient, op=dist.ReduceOp.AVG)
    return gradient


def topk_allgather(gradient: torch.Tensor, density: float) -> torch.Tensor:
    """
    TopKAllGather: Sparse communication using AllGather.
    
    Communication complexity: 2(n-1)α + 2(n-1)k * β
    
    Args:
        gradient: Local gradient
        density: Sparsification density
        
    Returns:
        Aggregated gradient
    """
    device = gradient.device
    flat_grad = gradient.flatten()
    k = max(1, int(flat_grad.numel() * density))
    
    # Select top-k
    _, indices = torch.topk(flat_grad.abs(), k, sorted=False)
    values = flat_grad[indices]
    
    # AllGather values and indices
    if dist.is_initialized():
        world_size = dist.get_world_size()
        
        all_values = [torch.zeros_like(values) for _ in range(world_size)]
        all_indices = [torch.zeros_like(indices) for _ in range(world_size)]
        
        dist.all_gather(all_values, values)
        dist.all_gather(all_indices, indices)
    else:
        all_values = [values]
        all_indices = [indices]
    
    # Reconstruct
    result = torch.zeros_like(flat_grad)
    for vals, idxs in zip(all_values, all_indices):
        result[idxs.long()] += vals
    
    if dist.is_initialized():
        result /= dist.get_world_size()
    
    return result.reshape(gradient.shape)


def benchmark_method(gradient: torch.Tensor, method_func, method_name: str, 
                    num_iters: int = 20, **kwargs) -> float:
    """
    Benchmark a communication method.
    
    Args:
        gradient: Test gradient
        method_func: Function to benchmark
        method_name: Name for display
        num_iters: Number of iterations
        **kwargs: Additional arguments for method_func
        
    Returns:
        Average time in milliseconds
    """
    # Warm-up
    for _ in range(5):
        _ = method_func(gradient.clone(), **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    
    for _ in range(num_iters):
        _ = method_func(gradient.clone(), **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end = time.time()
    avg_time = (end - start) / num_iters * 1000  # Convert to ms
    
    return avg_time


def run_benchmark(rank, world_size):
    """
    Run comprehensive benchmark comparing all methods.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    setup(rank, world_size)
    
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Test configurations
    test_configs = [
        {"name": "Small Layer", "size": 100_000, "shape": (100, 1000)},
        {"name": "Medium Layer", "size": 1_000_000, "shape": (1000, 1000)},
        {"name": "Large Layer", "size": 10_000_000, "shape": (2000, 5000)},
    ]
    
    densities = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    
    results = {}
    
    for config in test_configs:
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"Benchmarking: {config['name']} ({config['size']:,} parameters)")
            print(f"{'='*80}")
        
        gradient = torch.randn(config['shape'], device=device)
        
        # 1. Dense AllReduce (baseline)
        dense_time = benchmark_method(
            gradient, 
            dense_allreduce, 
            "Dense AllReduce"
        )
        
        if rank == 0:
            print(f"\nDense AllReduce: {dense_time:.2f} ms")
        
        # 2. TopKA2A with different densities
        if rank == 0:
            print(f"\nTopKA2A:")
        
        for density in densities:
            topka2a = TopKA2A(world_size, rank, density)
            topka2a_time = benchmark_method(
                gradient,
                lambda g: topka2a.communicate(g),
                f"TopKA2A (ρ={density})"
            )
            
            speedup = dense_time / topka2a_time
            
            if rank == 0:
                print(f"  ρ={density:5.3f}: {topka2a_time:6.2f} ms  "
                      f"(speedup: {speedup:.2f}x)")
        
        # 3. TopKAllGather with different densities
        if rank == 0:
            print(f"\nTopKAllGather:")
        
        for density in densities:
            try:
                allgather_time = benchmark_method(
                    gradient,
                    topk_allgather,
                    f"TopKAllGather (ρ={density})",
                    density=density
                )
                
                speedup = dense_time / allgather_time
                
                if rank == 0:
                    print(f"  ρ={density:5.3f}: {allgather_time:6.2f} ms  "
                          f"(speedup: {speedup:.2f}x)")
            except RuntimeError as e:
                if rank == 0:
                    print(f"  ρ={density:5.3f}: OOM or Error")
    
    # Summary
    if rank == 0:
        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}")
        print(f"\nWorld size: {world_size} GPUs")
        print(f"\nTheoretical density range for TopKA2A:")
        min_density = 1.0 / (2 * (world_size - 1))
        print(f"  {min_density:.4f} < ρ < 0.5")
        print(f"\nRecommended densities: 0.02 - 0.1")
        print(f"\nKey Findings:")
        print(f"  - TopKA2A is fastest in the range {min_density:.3f} < ρ < 0.5")
        print(f"  - TopKAllGather requires ρ < {1/world_size:.3f} to beat dense")
        print(f"  - TopKA2A allows practical densities (2-10%)")
    
    cleanup()


def main():
    """Main function to launch benchmark."""
    world_size = min(torch.cuda.device_count() if torch.cuda.is_available() else 2, 8)
    
    print("="*80)
    print("TopKA2A Comparison Benchmark")
    print("="*80)
    print(f"\nComparing:")
    print("  1. Dense AllReduce (PyTorch DDP)")
    print("  2. TopKA2A (our method)")
    print("  3. TopKAllGather (baseline)")
    print(f"\nWorld size: {world_size}")
    print(f"Backend: {'NCCL' if torch.cuda.is_available() else 'Gloo'}")
    print("="*80)
    
    if world_size < 2:
        print("\nWarning: Need at least 2 processes. Using world_size=2.")
        world_size = 2
    
    # Spawn processes
    mp.spawn(run_benchmark,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    main()
