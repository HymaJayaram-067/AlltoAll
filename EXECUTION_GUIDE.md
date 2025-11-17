# Step-by-Step Execution Guide for TopKA2A

This guide provides detailed step-by-step instructions to execute TopKA2A on your system.

## Prerequisites Check

Before starting, ensure you have:
- ✅ NVIDIA GPU(s) with CUDA support
- ✅ Python 3.7 or higher
- ✅ Git installed
- ✅ Internet connection for downloading dependencies

---

## Step 1: Install PyTorch

### Option A: With CUDA Support (Recommended for GPU training)

```bash
# For CUDA 11.8 (most common)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only (testing purposes)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Verify PyTorch Installation

```bash
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
```

Expected output:
```
PyTorch version: 2.x.x
CUDA available: True
CUDA version: 11.8 (or your version)
```

---

## Step 2: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/HymaJayaram-067/AlltoAll.git

# Navigate to the directory
cd AlltoAll

# Verify you're on the correct branch
git branch
```

---

## Step 3: Validate Installation

Run the validation script to ensure all files are present and correct:

```bash
python3 validate.py
```

Expected output:
```
================================================================================
TopKA2A Implementation Validation
================================================================================

1. Checking file structure...
   ✓ All required files present

2. Checking Python syntax...
   ✓ topka2a_baseline.py
   ✓ topka2a_paper_exact.py
   ✓ topka2a_optimized.py
   ✓ topka2a_optimized_hpc.py
   ✓ DIAGRAMS.py

3. Checking documentation...
   ✓ Documentation complete

4. Checking examples...
   ✓ examples/simple_example.py
   ✓ benchmarks/compare_methods.py

================================================================================
✓ ALL VALIDATIONS PASSED
================================================================================
```

---

## Step 4: Understand the Algorithm (Optional but Recommended)

Before running, understand what TopKA2A does:

```bash
# Read the detailed explanation
cat HOW_IT_WORKS.md | less

# View visual diagrams
python3 DIAGRAMS.py | less

# Quick overview
cat IMPLEMENTATION_SUMMARY.md | less
```

---

## Step 5: Run Simple Example (Single Machine)

### For Single GPU or CPU Testing:

```bash
# This will run a simple gradient aggregation example
python3 examples/simple_example.py
```

Expected output:
```
================================================================================
TopKA2A Simple Example
================================================================================

World size: 2
Using: CUDA (NCCL) or CPU (Gloo)
================================================================================

Running example on rank 0/2
Running example on rank 1/2

Gradient shape: torch.Size([1000000])
Gradient size: 1,000,000 elements
Density: 0.02
Top-k elements: 20,000

Communication time: XX.XX ms
Aggregated gradient shape: torch.Size([1000000])
Aggregated gradient norm: X.XXXX

================================================================================
✓ TopKA2A Example Completed Successfully!
================================================================================
```

---

## Step 6: Run Performance Benchmarks

Compare TopKA2A with other methods:

```bash
python3 benchmarks/compare_methods.py
```

This will compare:
- Dense AllReduce (PyTorch DDP baseline)
- TopKAllGather (sparse baseline)
- TopKA2A (our method)

Expected output:
```
================================================================================
Benchmarking: Small Layer (100,000 parameters)
================================================================================

Dense AllReduce: XX.XX ms

TopKA2A:
  ρ=0.001: XX.XX ms  (speedup: X.XXx)
  ρ=0.005: XX.XX ms  (speedup: X.XXx)
  ρ=0.010: XX.XX ms  (speedup: X.XXx)
  ρ=0.020: XX.XX ms  (speedup: X.XXx)
  ...
```

---

## Step 7: Integrate into Your Training Code

### Basic Integration:

```python
import torch
import torch.distributed as dist
from topka2a_paper_exact import TopKA2A

# Initialize distributed training
dist.init_process_group(backend='nccl')  # or 'gloo' for CPU
world_size = dist.get_world_size()
rank = dist.get_rank()

# Create TopKA2A instance
topka2a = TopKA2A(world_size, rank, density=0.02)

# In your training loop
for epoch in range(num_epochs):
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

### With HPC Optimizations:

```python
from topka2a_optimized_hpc import TopKA2AWithOptimizations

# Create optimized instance
topka2a = TopKA2AWithOptimizations(
    world_size=world_size,
    rank=rank,
    density=0.02,                # 2% sparsity
    enable_async=True,           # Async communication
    enable_tensor_fusion=True,   # Tensor fusion
    fusion_buffer_mb=25,         # 25MB buffer
    use_hierarchical=True,       # Multi-node optimization
    gpus_per_node=8             # 8 GPUs per node
)

# Use same as basic version
for param in model.parameters():
    if param.grad is not None:
        param.grad = topka2a.communicate(param.grad)
```

---

## Step 8: Multi-GPU Training (Single Node)

### Using torchrun (Recommended):

```bash
# For 4 GPUs on a single node
torchrun --nproc_per_node=4 your_training_script.py
```

### Using python -m torch.distributed.launch:

```bash
# For 4 GPUs
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    your_training_script.py
```

### Example with the provided simple example:

```bash
# This should work if you have multiple GPUs
torchrun --nproc_per_node=2 examples/simple_example.py
```

---

## Step 9: Multi-Node Training (Advanced)

### Setup for 2 nodes with 8 GPUs each:

**On Node 0 (Master):**
```bash
export MASTER_ADDR=192.168.1.100  # IP of master node
export MASTER_PORT=29500
export WORLD_SIZE=16              # Total GPUs (2 nodes × 8 GPUs)
export RANK=0                     # Master node rank

torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    your_training_script.py
```

**On Node 1 (Worker):**
```bash
export MASTER_ADDR=192.168.1.100  # Same IP as master
export MASTER_PORT=29500
export WORLD_SIZE=16
export RANK=8                     # Worker node rank (8 for node 1)

torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    your_training_script.py
```

---

## Step 10: NCCL Configuration (For Optimal Performance)

Set these environment variables before running:

```bash
# For large messages (after tensor fusion)
export NCCL_PROTO=Simple
export NCCL_ALGO=Tree

# For small sparse messages
export NCCL_PROTO=LL128
export NCCL_ALGO=Ring

# Enable GPUDirect RDMA (if InfiniBand available)
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_HCA=mlx5_0,mlx5_1

# Buffer optimization
export NCCL_BUFFSIZE=8388608

# For NVLink systems (DGX A100/H100)
export NCCL_P2P_LEVEL=NVL
export NCCL_NTHREADS=512

# For debugging (optional)
export NCCL_DEBUG=INFO
```

Then run your training:
```bash
torchrun --nproc_per_node=4 your_training_script.py
```

---

## Troubleshooting

### Issue 1: "No module named 'torch'"

**Solution:**
```bash
pip install torch torchvision torchaudio
```

### Issue 2: "CUDA out of memory"

**Solution 1:** Reduce batch size
```python
batch_size = 16  # Instead of 32
```

**Solution 2:** Lower density
```python
topka2a = TopKA2A(world_size, rank, density=0.01)  # Instead of 0.05
```

### Issue 3: "Process group not initialized"

**Solution:** Ensure you initialize distributed training:
```python
import torch.distributed as dist
dist.init_process_group(backend='nccl')  # or 'gloo' for CPU
```

### Issue 4: Slower than expected

**Check 1:** Verify density is in optimal range
```python
min_density = 1.0 / (2 * (world_size - 1))
print(f"Minimum density for {world_size} GPUs: {min_density:.4f}")
print(f"Your density: {density}")
assert density > min_density, "Density too low!"
```

**Check 2:** Enable all optimizations
```python
topka2a = TopKA2AWithOptimizations(
    world_size, rank, density=0.02,
    enable_async=True,
    enable_tensor_fusion=True,
    use_hierarchical=True  # For multi-node
)
```

---

## Quick Reference Commands

```bash
# 1. Install PyTorch
pip install torch torchvision torchaudio

# 2. Clone repository
git clone https://github.com/HymaJayaram-067/AlltoAll.git
cd AlltoAll

# 3. Validate
python3 validate.py

# 4. Run simple example
python3 examples/simple_example.py

# 5. Run benchmarks
python3 benchmarks/compare_methods.py

# 6. Multi-GPU training (4 GPUs)
torchrun --nproc_per_node=4 your_training_script.py
```

---

## Choosing the Right Density

| # GPUs | Recommended Density | Notes |
|--------|-------------------|-------|
| 8      | 0.10-0.20        | Higher density for small scale |
| 16     | 0.05-0.10        | Balanced |
| 32     | 0.02-0.10        | ⭐ Most common scenario |
| 64+    | 0.01-0.05        | Lower density for large scale |

**Rule of thumb:** Start with 0.02 (2%) and adjust based on:
- **Increase** if accuracy drops
- **Decrease** if you need more speedup

---

## Next Steps

1. ✅ Read `USER_GUIDE.md` for detailed usage instructions
2. ✅ Read `HOW_IT_WORKS.md` to understand the algorithm
3. ✅ Read `OPTIMIZATIONS.md` for performance tuning
4. ✅ Check `DIAGRAMS.py` for visual explanations

---

## Getting Help

- **Documentation**: See `USER_GUIDE.md`, `HOW_IT_WORKS.md`
- **Issues**: Open an issue on GitHub
- **Paper**: Read the original ICPP '24 paper for theoretical details

---

## Summary

**Minimum steps to get started:**

```bash
# 1. Install
pip install torch

# 2. Clone
git clone https://github.com/HymaJayaram-067/AlltoAll.git
cd AlltoAll

# 3. Test
python3 validate.py
python3 examples/simple_example.py

# 4. Use in your code
from topka2a_paper_exact import TopKA2A
topka2a = TopKA2A(world_size, rank, density=0.02)
aggregated = topka2a.communicate(gradient)
```

**That's it! You're ready to use TopKA2A for faster distributed training!** 🚀
