# TopKA2A: Sparse Gradient Communication for Distributed Deep Learning

Implementation of the **TopKA2A** algorithm from the ICPP '24 paper:

**"Sparse Gradient Communication with AlltoAll for Accelerating Distributed Deep Learning"**  
by Jing Peng, Zihan Li, Shaohuai Shi, and Bo Li

## What is TopKA2A?

TopKA2A is a novel sparse gradient communication algorithm that combines:
- **Top-K sparsification** to reduce communication volume
- **AlltoAll collective** for efficient sparse gradient exchange
- **Scalable design** that works with practical sparsity levels (1-10%)

### Key Innovation

Unlike TopKAllGather which requires very low densities (< 1%) to outperform dense communication, TopKA2A maintains efficiency with **practical densities of 2-10%**, making it suitable for production use without sacrificing model accuracy.

## Algorithm Overview

TopKA2A consists of 4 steps:

```
1. Gradient Shard Sparsification
   - Partition gradient into n parts
   - Apply Top-K to each part independently
   
2. AlltoAll Exchange  
   - Exchange sparse shards between GPUs
   - Like a matrix transpose operation
   
3. Accumulation
   - Each GPU accumulates its designated shard
   - Handles sparse additions efficiently
   
4. AllGather
   - Collect all dense shards
   - All GPUs get identical result
```

## Performance

**Communication Complexity:**  
- TopKA2A: `3(n-1)α + (2k + d)(n-1)/n * β`
- TopKAllGather: `2(n-1)α + 2(n-1)k * β`
- Dense AllReduce: `2(n-1)α + 2(n-1)d/n * β`

**Key Advantage:** TopKA2A's bandwidth term is nearly independent of n (number of GPUs), making it highly scalable.

**Experimental Results (from paper):**
- Up to **73% faster** than PyTorch DDP on 32 GPUs
- Up to **41% faster** on BERT-Large
- Up to **72% faster** on GPT-2
- Scales efficiently from 8 to 32+ GPUs

## Repository Structure

```
AlltoAll/
├── topka2a_baseline.py          # Baseline implementation for comparison
├── topka2a_paper_exact.py       # Exact algorithm from ICPP '24 paper
├── topka2a_optimized_hpc.py     # With HPC optimizations
├── OPTIMIZATIONS.md             # Detailed optimization guide
├── examples/                    # Usage examples
│   ├── simple_example.py
│   ├── bert_training.py
│   └── multi_node_setup.py
└── benchmarks/                  # Benchmarking scripts
    └── compare_methods.py
```

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.8+ with CUDA support (recommended)
- NCCL 2.7+ (for multi-GPU training)
- (Optional) InfiniBand or high-speed network for multi-node

### Install Dependencies

```bash
# Install PyTorch (choose appropriate version for your CUDA)
pip install torch torchvision torchaudio

# For distributed training
pip install mpi4py  # Optional, for MPI backend
```

### Clone Repository

```bash
git clone https://github.com/HymaJayaram-067/AlltoAll.git
cd AlltoAll
```

### Verify Installation

```bash
python validate.py
```

## Quick Start

### Basic Usage

```python
from topka2a_paper_exact import TopKA2A
import torch
import torch.distributed as dist

# Initialize distributed training
dist.init_process_group(backend='nccl')
world_size = dist.get_world_size()
rank = dist.get_rank()

# Create TopKA2A instance
topka2a = TopKA2A(world_size, rank, density=0.02)  # 2% density

# In your training loop:
gradient = model.get_gradient()  # Your gradient tensor
aggregated = topka2a.communicate(gradient)
model.apply_gradient(aggregated)
```

### With HPC Optimizations

```python
from topka2a_optimized_hpc import TopKA2AWithOptimizations

# Create optimized instance
topka2a = TopKA2AWithOptimizations(
    world_size=world_size,
    rank=rank,
    density=0.02,
    enable_async=True,           # Async communication
    enable_tensor_fusion=True,   # Tensor fusion
    fusion_buffer_mb=25,         # 25MB fusion buffer
    use_hierarchical=True,       # Multi-node optimization
    gpus_per_node=8             # 8 GPUs per node
)

aggregated = topka2a.communicate(gradient)
```

## When to Use TopKA2A?

**Best suited for:**
- ✅ Multi-GPU training (8+ GPUs)
- ✅ Network-bound scenarios (slow interconnect)
- ✅ Large models (BERT, GPT, ViT)
- ✅ When you need practical sparsity (2-10%)
- ✅ Multi-node training

**Performance Conditions:**

For n GPUs, TopKA2A is optimal when density ρ satisfies:
```
1/(2(n-1)) < ρ < 0.5
```

**Examples:**
- 8 GPUs: 0.07 < ρ < 0.5  ✓ (use 0.1-0.2)
- 16 GPUs: 0.033 < ρ < 0.5  ✓ (use 0.05-0.1)
- 32 GPUs: 0.016 < ρ < 0.5  ✓ (use 0.02-0.1)
- 64 GPUs: 0.008 < ρ < 0.5  ✓ (use 0.01-0.05)

## HPC Optimizations Included

1. **Async Communication** - Overlap communication with computation
2. **Tensor Fusion** - Batch small tensors to reduce overhead
3. **Binary Search Tuning** - Automatically find optimal buffer size
4. **Hierarchical Communication** - Fast intra-node, slower inter-node
5. **NCCL Tuning** - Optimized collective parameters
6. **Memory Pooling** - Reduce allocation overhead

See [OPTIMIZATIONS.md](OPTIMIZATIONS.md) for detailed explanations.

## Hardware Requirements

**Recommended:**
- GPUs: NVIDIA A100 or H100 (NVLink support)
- Network: InfiniBand HDR (200 Gb/s) or RoCE
- Topology: 8 GPUs per node
- NCCL: Version 2.10+

**Minimum:**
- GPUs: Any CUDA-capable GPU
- Network: 10 GbE or better
- NCCL: Version 2.7+

## NCCL Configuration

For optimal performance, set these environment variables:

```bash
# For large messages
export NCCL_PROTO=Simple
export NCCL_ALGO=Tree

# For small sparse messages
export NCCL_PROTO=LL128
export NCCL_ALGO=Ring

# Enable GPUDirect RDMA (InfiniBand)
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_HCA=mlx5_0,mlx5_1

# Buffer optimization
export NCCL_BUFFSIZE=8388608

# For NVLink systems
export NCCL_P2P_LEVEL=NVL
export NCCL_NTHREADS=512
```

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{peng2024topka2a,
  title={Sparse Gradient Communication with AlltoAll for Accelerating Distributed Deep Learning},
  author={Peng, Jing and Li, Zihan and Shi, Shaohuai and Li, Bo},
  booktitle={The 53rd International Conference on Parallel Processing (ICPP)},
  year={2024},
  organization={ACM}
}
```

## License

This implementation is provided for research and educational purposes.

## Related Work

- **TopKAllGather**: Requires very low density (< 1%)
- **PowerSGD/ACP-SGD**: Low-rank compression methods
- **DeAR**: Optimized AllReduce pipelining
- **PyTorch DDP**: Standard dense communication

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## Contact

For questions or issues, please open a GitHub issue. 
