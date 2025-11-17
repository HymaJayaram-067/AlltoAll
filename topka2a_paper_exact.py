"""
TopKA2A Algorithm - Exact Implementation from ICPP '24 Paper
============================================================

Paper: "Sparse Gradient Communication with AlltoAll for Accelerating 
Distributed Deep Learning" by Jing Peng et al., ICPP 2024

This implementation follows Algorithm 1 from the paper exactly.

ALGORITHM OVERVIEW:
==================

The TopKA2A algorithm consists of 4 main steps:

Step 1: Gradient Shard Sparsification
--------------------------------------
- Partition the local gradient tensor into n parts (n = number of GPUs)
- Each GPU has n shards: g_j_i where j is the GPU rank, i is the shard index
- Apply MSTopK independently on each shard to select k/n largest elements
- This produces: values (κ_j_i) and indices (ι_j_i) for each shard

Step 2: AlltoAll Aggregation
-----------------------------
- Exchange sparse shards between GPUs using AlltoAll collective
- GPU j sends shard i to GPU i and receives shard j from GPU i
- This is like a matrix transpose operation
- Both values and indices are exchanged (2k/n elements per shard)

Step 3: Accumulation
-------------------
- Each GPU accumulates received sparse shards onto its designated shard
- GPU j accumulates all received shards onto shard j
- Uses scatter_add based on indices to handle sparse additions
- Result: Each GPU has one dense shard of size d/n

Step 4: AllGather
-----------------
- Collect all dense shards from all GPUs
- Each GPU sends its accumulated shard to all others
- Final result: All GPUs have identical full gradient

COMMUNICATION COMPLEXITY:
========================

Time Cost: t_TopKA2A = 3(n-1)α + (2k + d)(n-1)/n * β

Where:
- n = number of GPUs
- d = total gradient dimension
- k = ρ × d (number of top-k elements, ρ is density)
- α = startup latency
- β = per-byte transfer time

Key Insight: The bandwidth term (2k + d)(n-1)/n is nearly independent of n,
making TopKA2A more scalable than TopKAllGather which has 2(n-1)kβ.

PERFORMANCE CONDITIONS:
======================

TopKA2A outperforms TopKAllGather when: ρ > 1/(2(n-1))
TopKA2A outperforms Dense AllReduce when: ρ < 0.5

Example: For 32 GPUs, TopKA2A is best when 0.016 < ρ < 0.5
This allows practical densities like 0.02-0.10 while maintaining speedup.
"""

import torch
import torch.distributed as dist
from typing import Tuple, List, Optional
import time


class TopKA2A:
    """
    TopKA2A: Sparse gradient communication using AlltoAll.
    
    Based on Algorithm 1 from the ICPP '24 paper.
    """
    
    def __init__(self, world_size: int, rank: int, density: float = 0.02):
        """
        Initialize TopKA2A algorithm.
        
        Args:
            world_size: Total number of GPUs/processes (n)
            rank: Current process rank (j)
            density: Sparsification density ρ (e.g., 0.02 = 2%)
        """
        self.world_size = world_size  # n
        self.rank = rank              # j
        self.density = density        # ρ
        
        # Validate density is in practical range
        min_density = 1.0 / (2 * (world_size - 1))
        if density <= min_density:
            print(f"Warning: density {density} <= {min_density:.4f} may not outperform TopKAllGather")
        if density >= 0.5:
            print(f"Warning: density {density} >= 0.5 may not outperform dense AllReduce")
    
    def mstopk(self, tensor: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MSTopK: Efficient approximate top-k selection.
        
        From the paper: "we use the MSTopK algorithm proposed in [25] to 
        approximately select top-k elements such that the overhead of top-k 
        selection is negligible."
        
        For simplicity, we use exact top-k here. In production, use the 
        optimized MSTopK implementation.
        
        Args:
            tensor: Input gradient shard (1D)
            k: Number of elements to select
            
        Returns:
            values: Top-k values (κ)
            indices: Top-k indices (ι)
        """
        # Get top-k by absolute value, sorted=False for better performance
        _, top_indices = torch.topk(tensor.abs(), k, sorted=False)
        top_values = tensor[top_indices]
        
        return top_values, top_indices
    
    def communicate(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Execute the full TopKA2A algorithm (Algorithm 1 from paper).
        
        Args:
            gradient: Local gradient tensor g_j (can be any shape)
            
        Returns:
            Aggregated gradient g̃ (same shape as input)
        """
        original_shape = gradient.shape
        device = gradient.device
        n = self.world_size
        j = self.rank
        
        # Flatten gradient for processing
        g_flat = gradient.flatten()  # Total size: d
        d = g_flat.numel()
        k = max(1, int(self.density * d))  # k = ρ × d
        
        # Ensure d is divisible by n for equal sharding
        shard_size = d // n
        if d % n != 0:
            # Pad to make divisible
            pad_size = n * shard_size - d
            g_flat = torch.cat([g_flat, torch.zeros(pad_size, device=device)])
            d = g_flat.numel()
            shard_size = d // n
        
        # ================================================================
        # STEP 1: Gradient Shard Sparsification
        # ================================================================
        # Split gradient into n shards: g_j_1, g_j_2, ..., g_j_n
        # Each shard has size d/n
        
        g_shards = g_flat.view(n, shard_size)  # Shape: [n, d/n]
        
        # Apply MSTopK to each shard independently
        # Select k/n largest elements from each shard
        k_per_shard = max(1, k // n)
        
        sparse_values_list = []  # Will hold κ_j_i for all i
        sparse_indices_list = []  # Will hold ι_j_i for all i
        
        for i in range(n):
            values, indices = self.mstopk(g_shards[i], k_per_shard)
            sparse_values_list.append(values)
            sparse_indices_list.append(indices)
        
        # Stack for AlltoAll: shape [n, k/n] for both values and indices
        sparse_values = torch.stack(sparse_values_list)  # [n, k/n]
        sparse_indices = torch.stack(sparse_indices_list)  # [n, k/n]
        
        # ================================================================
        # STEP 2: AlltoAll Aggregation  
        # ================================================================
        # Exchange sparse shards between GPUs
        # GPU j sends shard i to GPU i, receives shard j from GPU i
        # This is like transposing the [n, k/n] matrix across GPUs
        
        # Prepare output tensors for AlltoAll
        received_values = torch.zeros_like(sparse_values)
        received_indices = torch.zeros_like(sparse_indices)
        
        # Create tensor lists for all_to_all_single
        # We need to send sparse_values[i] to rank i and receive from rank i
        send_values_list = [sparse_values[i].contiguous() for i in range(n)]
        recv_values_list = [torch.zeros(k_per_shard, dtype=sparse_values.dtype, 
                                       device=device) for _ in range(n)]
        
        send_indices_list = [sparse_indices[i].contiguous() for i in range(n)]
        recv_indices_list = [torch.zeros(k_per_shard, dtype=sparse_indices.dtype,
                                        device=device) for _ in range(n)]
        
        # Execute AlltoAll for values
        if dist.is_initialized():
            dist.all_to_all(recv_values_list, send_values_list)
            dist.all_to_all(recv_indices_list, send_indices_list)
        else:
            # For single GPU testing
            recv_values_list = send_values_list
            recv_indices_list = send_indices_list
        
        # Stack received data
        received_values = torch.stack(recv_values_list)  # [n, k/n]
        received_indices = torch.stack(recv_indices_list)  # [n, k/n]
        
        # ================================================================
        # STEP 3: Accumulation
        # ================================================================
        # Each GPU j accumulates all received sparse shards onto shard j
        # This creates a dense shard g̃_j_j of size d/n
        
        accumulated_shard = torch.zeros(shard_size, dtype=gradient.dtype, device=device)
        
        # Accumulate from all n received sparse shards
        for i in range(n):
            # Get sparse values and indices from rank i
            values = received_values[i]  # κ̃_j_i: [k/n]
            indices = received_indices[i]  # ι̃_j_i: [k/n]
            
            # Accumulate: g̃_j_j[ι] += κ
            accumulated_shard.scatter_add_(0, indices.long(), values)
        
        # ================================================================
        # STEP 4: AllGather
        # ================================================================
        # Collect accumulated shards from all GPUs
        # Each GPU broadcasts its accumulated_shard to all others
        
        # Prepare output tensor for AllGather
        gathered_shards = [torch.zeros_like(accumulated_shard) for _ in range(n)]
        
        if dist.is_initialized():
            dist.all_gather(gathered_shards, accumulated_shard)
        else:
            # For single GPU testing
            gathered_shards = [accumulated_shard]
        
        # Concatenate all shards to form final gradient
        g_tilde = torch.cat(gathered_shards)  # Size: d
        
        # Remove padding if any was added
        if d != original_shape.numel():
            g_tilde = g_tilde[:original_shape.numel()]
        
        # Reshape to original shape
        g_tilde = g_tilde.reshape(original_shape)
        
        return g_tilde


def explain_topka2a_with_example():
    """
    Detailed explanation of TopKA2A with a 4-GPU example.
    
    This matches Figure 1 from the paper.
    """
    
    print("="*80)
    print("TopKA2A Algorithm Explanation - 4 GPU Example")
    print("="*80)
    
    print("\nInitial Setup:")
    print("-" * 80)
    print("- 4 GPUs (n=4)")
    print("- Each GPU has a gradient vector of size d=16")
    print("- Density ρ=0.25, so k=4 elements selected in total")
    print("- Each shard gets k/n = 4/4 = 1 element")
    
    print("\n" + "="*80)
    print("STEP 1: Gradient Shard Sparsification")
    print("="*80)
    
    print("\nGPU 0 has gradient: [8.0, -3.0, 2.0, -9.0, | 4.0, 1.0, -7.0, 3.0, | ...]")
    print("                     └─── Shard 0 ───┘  └─── Shard 1 ───┘")
    print("\nPartition into 4 shards (each size d/n=4):")
    print("  Shard 0: [8.0, -3.0, 2.0, -9.0]")
    print("  Shard 1: [4.0, 1.0, -7.0, 3.0]")
    print("  Shard 2: [6.0, -2.0, 5.0, -1.0]")
    print("  Shard 3: [-4.0, 7.0, 2.0, 3.0]")
    
    print("\nApply MSTopK(k/n=1) to each shard:")
    print("  Shard 0: Select largest absolute value: -9.0 at index 3")
    print("           → κ_0_0 = [-9.0], ι_0_0 = [3]")
    print("  Shard 1: Select largest absolute value: -7.0 at index 2")
    print("           → κ_0_1 = [-7.0], ι_0_1 = [2]")
    print("  Shard 2: Select largest absolute value: 6.0 at index 0")
    print("           → κ_0_2 = [6.0], ι_0_2 = [0]")
    print("  Shard 3: Select largest absolute value: 7.0 at index 1")
    print("           → κ_0_3 = [7.0], ι_0_3 = [1]")
    
    print("\n(Similarly for GPU 1, 2, 3...)")
    
    print("\n" + "="*80)
    print("STEP 2: AlltoAll Exchange")
    print("="*80)
    
    print("\nBefore AlltoAll (GPU perspective):")
    print("  GPU 0 has: [κ_0_0, κ_0_1, κ_0_2, κ_0_3]  - Its 4 sparse shards")
    print("  GPU 1 has: [κ_1_0, κ_1_1, κ_1_2, κ_1_3]")
    print("  GPU 2 has: [κ_2_0, κ_2_1, κ_2_2, κ_2_3]")
    print("  GPU 3 has: [κ_3_0, κ_3_1, κ_3_2, κ_3_3]")
    
    print("\nAlltoAll operation (like matrix transpose):")
    print("  GPU 0 sends κ_0_1 to GPU 1, κ_0_2 to GPU 2, κ_0_3 to GPU 3")
    print("  GPU 0 receives κ_1_0 from GPU 1, κ_2_0 from GPU 2, κ_3_0 from GPU 3")
    
    print("\nAfter AlltoAll:")
    print("  GPU 0 has: [κ_0_0, κ_1_0, κ_2_0, κ_3_0]  - All sparse for shard 0")
    print("  GPU 1 has: [κ_0_1, κ_1_1, κ_2_1, κ_3_1]  - All sparse for shard 1")
    print("  GPU 2 has: [κ_0_2, κ_1_2, κ_2_2, κ_3_2]  - All sparse for shard 2")
    print("  GPU 3 has: [κ_0_3, κ_1_3, κ_2_3, κ_3_3]  - All sparse for shard 3")
    
    print("\n" + "="*80)
    print("STEP 3: Accumulation")
    print("="*80)
    
    print("\nGPU 0 accumulates onto shard 0 (size 4, initially all zeros):")
    print("  From κ_0_0 at index 3: g̃_0_0[3] += -9.0  → [0, 0, 0, -9.0]")
    print("  From κ_1_0 at index 1: g̃_0_0[1] += -6.5  → [0, -6.5, 0, -9.0]")
    print("  From κ_2_0 at index 2: g̃_0_0[2] += 8.2   → [0, -6.5, 8.2, -9.0]")
    print("  From κ_3_0 at index 0: g̃_0_0[0] += 4.1   → [4.1, -6.5, 8.2, -9.0]")
    
    print("\n  Final accumulated shard on GPU 0: [4.1, -6.5, 8.2, -9.0]")
    print("\n(Similarly, GPU 1, 2, 3 accumulate their respective shards)")
    
    print("\n" + "="*80)
    print("STEP 4: AllGather")
    print("="*80)
    
    print("\nEach GPU broadcasts its accumulated shard:")
    print("  GPU 0 broadcasts: [4.1, -6.5, 8.2, -9.0]   (shard 0)")
    print("  GPU 1 broadcasts: [2.3, -1.1, -7.0, 3.8]   (shard 1)")
    print("  GPU 2 broadcasts: [6.0, -2.7, 5.5, -1.2]   (shard 2)")
    print("  GPU 3 broadcasts: [-4.4, 7.0, 2.9, 3.3]    (shard 3)")
    
    print("\nAll GPUs receive all shards and concatenate:")
    print("  Final g̃ = [4.1, -6.5, 8.2, -9.0,  2.3, -1.1, -7.0, 3.8,")
    print("             6.0, -2.7, 5.5, -1.2,  -4.4, 7.0, 2.9, 3.3]")
    
    print("\n" + "="*80)
    print("KEY BENEFITS")
    print("="*80)
    
    print("\n1. Scalability:")
    print("   - Communication: (2k + d)(n-1)/n")
    print("   - Almost independent of n for large d")
    print("   - Compare to TopKAllGather: 2(n-1)k (grows with n)")
    
    print("\n2. Practical Density Range:")
    print("   - Works well with ρ = 0.01 to 0.1 (1% to 10%)")
    print("   - TopKAllGather requires ρ < 0.01 on 32 GPUs")
    
    print("\n3. Memory Efficiency:")
    print("   - Each GPU only stores one dense shard (d/n)")
    print("   - Sparse communication reduces bandwidth")
    
    print("\n" + "="*80)


# Example usage and benchmarking
def benchmark_topka2a():
    """
    Benchmark TopKA2A and compare with theoretical predictions.
    """
    
    # Setup
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0
        print("Warning: Running in single-GPU mode for testing")
    
    # Test parameters
    d = 10_000_000  # 10M parameters (typical for a layer)
    density = 0.02   # 2% density
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize TopKA2A
    topka2a = TopKA2A(world_size, rank, density)
    
    # Create random gradient
    gradient = torch.randn(d, device=device)
    
    # Warmup
    for _ in range(3):
        _ = topka2a.communicate(gradient)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    num_iters = 10
    start = time.time()
    
    for _ in range(num_iters):
        result = topka2a.communicate(gradient)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end = time.time()
    avg_time = (end - start) / num_iters
    
    if rank == 0:
        print(f"\nTopKA2A Benchmark Results:")
        print(f"  World size: {world_size}")
        print(f"  Gradient size: {d:,}")
        print(f"  Density: {density}")
        print(f"  Avg iteration time: {avg_time*1000:.2f} ms")
        print(f"  Theoretical speedup range: {1/(2*(world_size-1)):.4f} < ρ < 0.5")
        print(f"  Current density in range: {1/(2*(world_size-1)) < density < 0.5}")


if __name__ == "__main__":
    # Print detailed explanation
    explain_topka2a_with_example()
    
    # Run benchmark if distributed is initialized
    if dist.is_available() and dist.is_initialized():
        benchmark_topka2a()
