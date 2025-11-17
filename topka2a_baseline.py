"""
TopKA2A Baseline Implementation
================================
Distributed training communication using Top-K gradient sparsification + AlltoAll + AllGather.

This is the baseline implementation before optimizations.
"""

import torch
import torch.distributed as dist
from typing import Tuple, List, Optional
import time


class TopKA2ABaseline:
    """
    Baseline TopKA2A implementation:
    1. Top-K sparsification on local gradients
    2. AlltoAll to exchange sparse gradients
    3. AllGather to collect all sparse gradients
    """
    
    def __init__(self, world_size: int, rank: int, k_ratio: float = 0.1):
        """
        Args:
            world_size: Total number of processes
            rank: Current process rank
            k_ratio: Ratio of gradients to keep (0.1 = top 10%)
        """
        self.world_size = world_size
        self.rank = rank
        self.k_ratio = k_ratio
        
    def topk_sparsify(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform Top-K sparsification on gradient tensor.
        
        Args:
            tensor: Input gradient tensor
            
        Returns:
            values: Top-K values
            indices: Indices of top-K values
        """
        # Flatten tensor for top-k selection
        flat_tensor = tensor.flatten()
        k = max(1, int(flat_tensor.numel() * self.k_ratio))
        
        # Get top-k by absolute value
        _, top_indices = torch.topk(flat_tensor.abs(), k)
        top_values = flat_tensor[top_indices]
        
        return top_values, top_indices
    
    def all_to_all_sparse(
        self, 
        values: torch.Tensor, 
        indices: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Perform AlltoAll exchange of sparse gradients.
        Each rank sends its sparse gradients to all other ranks.
        
        Args:
            values: Top-K gradient values
            indices: Top-K gradient indices
            
        Returns:
            all_values: List of value tensors from all ranks
            all_indices: List of index tensors from all ranks
        """
        # Prepare output buffers
        all_values = [torch.zeros_like(values) for _ in range(self.world_size)]
        all_indices = [torch.zeros_like(indices) for _ in range(self.world_size)]
        
        # Perform all-to-all
        dist.all_gather(all_values, values)
        dist.all_gather(all_indices, indices)
        
        return all_values, all_indices
    
    def reconstruct_gradient(
        self,
        all_values: List[torch.Tensor],
        all_indices: List[torch.Tensor],
        original_shape: torch.Size
    ) -> torch.Tensor:
        """
        Reconstruct full gradient from sparse components.
        
        Args:
            all_values: Value tensors from all ranks
            all_indices: Index tensors from all ranks
            original_shape: Original gradient tensor shape
            
        Returns:
            Reconstructed gradient tensor
        """
        reconstructed = torch.zeros(original_shape).flatten()
        
        # Aggregate sparse gradients from all ranks
        for values, indices in zip(all_values, all_indices):
            reconstructed[indices] += values
        
        # Average by world size
        reconstructed /= self.world_size
        
        return reconstructed.reshape(original_shape)
    
    def communicate(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Full TopKA2A communication pattern.
        
        Args:
            gradient: Local gradient tensor
            
        Returns:
            Aggregated gradient tensor
        """
        original_shape = gradient.shape
        
        # Step 1: Top-K sparsification
        values, indices = self.topk_sparsify(gradient)
        
        # Step 2: AlltoAll exchange
        all_values, all_indices = self.all_to_all_sparse(values, indices)
        
        # Step 3: Reconstruct
        aggregated = self.reconstruct_gradient(all_values, all_indices, original_shape)
        
        return aggregated


def benchmark_baseline(world_size: int, rank: int, tensor_size: int = 1000000):
    """Benchmark baseline TopKA2A implementation."""
    
    # Initialize
    topk = TopKA2ABaseline(world_size, rank, k_ratio=0.1)
    gradient = torch.randn(tensor_size, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Warmup
    for _ in range(5):
        _ = topk.communicate(gradient)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    num_iters = 20
    start = time.time()
    
    for _ in range(num_iters):
        result = topk.communicate(gradient)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end = time.time()
    avg_time = (end - start) / num_iters
    
    if rank == 0:
        print(f"Baseline TopKA2A - Avg time: {avg_time*1000:.2f}ms")
    
    return avg_time
