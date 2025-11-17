# User Guide: Getting Started with TopKA2A

## Overview

This guide will help you understand and use the TopKA2A implementation for distributed deep learning.

## What You Need to Know

### 1. The Problem

In distributed training, multiple GPUs need to aggregate gradients after each backward pass:
- **Traditional approach**: Send all gradients (100% of data)
- **Problem**: Network becomes bottleneck as models grow larger

### 2. TopKA2A Solution

TopKA2A reduces communication by:
- **Sending only top-k% of gradients** (e.g., 2% instead of 100%)
- **Using efficient AlltoAll communication** instead of AllGather
- **Result**: Up to 73% faster training on 32 GPUs!

### 3. When to Use TopKA2A

✅ **Good for:**
- Multi-GPU training (8+ GPUs)
- Large models (BERT, GPT, ResNet, ViT)
- Network-bound scenarios
- Multi-node training

❌ **Not needed for:**
- Single GPU training
- Very small models
- When network is not the bottleneck

## Quick Start

### Step 1: Installation

```bash
# Install PyTorch with CUDA
pip install torch torchvision

# Clone repository
git clone https://github.com/HymaJayaram-067/AlltoAll.git
cd AlltoAll

# Verify installation
python validate.py
```

### Step 2: Choose Your Version

We provide 4 implementations:

1. **`topka2a_paper_exact.py`** ⭐ **Recommended for learning**
   - Exact algorithm from the paper
   - Heavily commented with explanations
   - Best for understanding how it works

2. **`topka2a_optimized_hpc.py`** ⭐ **Recommended for production**
   - All HPC optimizations included
   - Async communication, tensor fusion, etc.
   - Best performance

3. **`topka2a_baseline.py`**
   - TopKAllGather for comparison
   - Shows traditional sparse approach

4. **`topka2a_optimized.py`**
   - Original optimized version
   - Alternative implementation

### Step 3: Basic Usage

```python
import torch
import torch.distributed as dist
from topka2a_paper_exact import TopKA2A

# Initialize distributed training
dist.init_process_group(backend='nccl')
world_size = dist.get_world_size()
rank = dist.get_rank()

# Create TopKA2A instance
# density=0.02 means keep 2% of gradients
topka2a = TopKA2A(world_size, rank, density=0.02)

# In your training loop
for batch in dataloader:
    # Forward pass
    outputs = model(batch)
    loss = criterion(outputs, labels)
    
    # Backward pass
    loss.backward()
    
    # Aggregate gradients with TopKA2A
    for param in model.parameters():
        if param.grad is not None:
            param.grad = topka2a.communicate(param.grad)
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
```

### Step 4: With HPC Optimizations

```python
from topka2a_optimized_hpc import TopKA2AWithOptimizations

# Create optimized instance
topka2a = TopKA2AWithOptimizations(
    world_size=world_size,
    rank=rank,
    density=0.02,                # 2% sparsity
    enable_async=True,           # Use async streams
    enable_tensor_fusion=True,   # Batch small tensors
    fusion_buffer_mb=25,         # 25MB buffer
    use_hierarchical=True,       # Multi-node optimization
    gpus_per_node=8             # 8 GPUs per node
)

# Use same as basic version
aggregated_gradient = topka2a.communicate(gradient)
```

## Choosing the Right Density

The density parameter (ρ) controls how many gradients to keep.

### Recommended Densities

| # GPUs | Min ρ | Max ρ | **Recommended** |
|--------|-------|-------|-----------------|
| 8      | 0.07  | 0.5   | **0.10-0.20**  |
| 16     | 0.033 | 0.5   | **0.05-0.10**  |
| 32     | 0.016 | 0.5   | **0.02-0.10**  |
| 64     | 0.008 | 0.5   | **0.01-0.05**  |

### Guidelines

- **Higher density** (0.1-0.2):
  - Better accuracy
  - Faster convergence
  - Use for: Critical models, early experimentation

- **Medium density** (0.02-0.05):
  - Good balance
  - ~2-5% of gradients
  - Use for: Most production scenarios

- **Lower density** (0.01-0.02):
  - Maximum speedup
  - May affect accuracy slightly
  - Use for: Very large scale, when speed critical

### Testing Different Densities

```python
# Try multiple densities to find best for your model
densities = [0.01, 0.02, 0.05, 0.10]

for density in densities:
    topka2a = TopKA2A(world_size, rank, density)
    # Train and measure accuracy
    train_model(topka2a)
```

## Running Examples

### 1. Simple Example

```bash
# Single node, multiple GPUs
python examples/simple_example.py
```

This shows:
- Basic TopKA2A usage
- Gradient aggregation
- Timing information

### 2. Performance Comparison

```bash
# Compare different methods
python benchmarks/compare_methods.py
```

This compares:
- Dense AllReduce (baseline)
- TopKAllGather
- TopKA2A
- Different densities

## Multi-Node Setup

### Launch Script

```bash
#!/bin/bash
# launch_distributed.sh

# Node 0 (master)
export MASTER_ADDR=node0.cluster.com
export MASTER_PORT=29500
export WORLD_SIZE=32
export RANK=0

python train.py \
    --use-topka2a \
    --density 0.02 \
    --enable-async \
    --gpus-per-node 8

# Node 1
export RANK=8
python train.py ...

# Node 2
export RANK=16
python train.py ...

# Node 3
export RANK=24
python train.py ...
```

### NCCL Configuration

For best performance, set these environment variables:

```bash
# For large messages (after tensor fusion)
export NCCL_PROTO=Simple
export NCCL_ALGO=Tree

# For small sparse messages
export NCCL_PROTO=LL128
export NCCL_ALGO=Ring

# Enable GPUDirect RDMA (if InfiniBand)
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_HCA=mlx5_0,mlx5_1

# Buffer optimization
export NCCL_BUFFSIZE=8388608

# For NVLink systems (DGX A100/H100)
export NCCL_P2P_LEVEL=NVL
export NCCL_NTHREADS=512
```

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution 1**: Reduce batch size
```python
batch_size = 32  # Instead of 64
```

**Solution 2**: Lower density
```python
density = 0.01  # Instead of 0.05
```

**Solution 3**: Disable tensor fusion
```python
enable_tensor_fusion=False
```

### Issue: Slower than expected

**Check 1**: Verify NCCL environment variables are set

**Check 2**: Ensure density is in optimal range
```python
min_density = 1.0 / (2 * (world_size - 1))
assert density > min_density
```

**Check 3**: Enable optimizations
```python
enable_async=True
enable_tensor_fusion=True
use_hierarchical=True  # For multi-node
```

### Issue: Accuracy degradation

**Solution 1**: Increase density
```python
density = 0.05  # Instead of 0.01
```

**Solution 2**: Use error feedback (future feature)

**Solution 3**: Adjust learning rate
```python
lr = lr * math.sqrt(density)  # Scale with sparsity
```

## Performance Tuning

### 1. Profile Your Training

```python
import time

# Time communication
start = time.time()
aggregated = topka2a.communicate(gradient)
torch.cuda.synchronize()
comm_time = time.time() - start

print(f"Communication time: {comm_time*1000:.2f} ms")
```

### 2. Optimize Buffer Size

The implementation includes binary search for automatic tuning:

```python
from topka2a_optimized_hpc import TensorFusionController

controller = TensorFusionController(
    model_size_mb=420,  # Your model size
    min_buffer_mb=1,
    max_iterations=10
)

# It will automatically find optimal buffer size
```

### 3. Monitor Network Utilization

```bash
# On Linux
iftop -i eth0  # Monitor network interface

# Check NCCL debug info
export NCCL_DEBUG=INFO
python train.py
```

## Understanding the Output

When you run TopKA2A, you'll see:

```
TopKA2A Benchmark Results:
  World size: 32
  Gradient size: 10,000,000
  Density: 0.02
  Avg iteration time: 42.50 ms
  Theoretical speedup range: 0.0161 < ρ < 0.5
  Current density in range: True
```

**Interpreting:**
- **Avg iteration time**: Time for one gradient aggregation
- **Theoretical speedup range**: Valid density range for your setup
- **Current density in range**: Whether your choice is optimal

## Best Practices

### 1. Start Conservative

```python
# Start with higher density
density = 0.05

# Monitor accuracy
# If good, try lower density for more speedup
density = 0.02
```

### 2. Use Hierarchical for Multi-Node

```python
# Always enable for multi-node
if num_nodes > 1:
    use_hierarchical = True
```

### 3. Enable All Optimizations

```python
topka2a = TopKA2AWithOptimizations(
    world_size, rank, density=0.02,
    enable_async=True,           # ✓
    enable_tensor_fusion=True,   # ✓
    use_hierarchical=True        # ✓ (multi-node)
)
```

### 4. Validate Convergence

```python
# Compare with dense baseline first
# Train for a few epochs with dense
# Then switch to TopKA2A
# Ensure validation accuracy is similar
```

## Next Steps

1. **Read the documentation**:
   - `HOW_IT_WORKS.md` - Understand the algorithm
   - `OPTIMIZATIONS.md` - Learn about optimizations
   - `DIAGRAMS.py` - Visual explanations

2. **Run the examples**:
   ```bash
   python examples/simple_example.py
   python benchmarks/compare_methods.py
   ```

3. **Integrate with your code**:
   - Replace gradient aggregation with TopKA2A
   - Tune density for your model
   - Enable optimizations

4. **Monitor and optimize**:
   - Profile communication time
   - Tune NCCL parameters
   - Adjust density based on accuracy

## Getting Help

- **Issues**: Open an issue on GitHub
- **Questions**: Check HOW_IT_WORKS.md
- **Paper**: Read the original ICPP '24 paper
- **Examples**: See examples/ directory

## Summary

TopKA2A provides:
- ✅ Up to 73% faster training
- ✅ Works with practical densities (2-10%)
- ✅ Scales efficiently to 64+ GPUs
- ✅ Production-ready with HPC optimizations
- ✅ Easy to integrate (3 lines of code)

**Start simple, optimize gradually, monitor results!**
