# Project2: GPU MLP Executor

 Folder: `gpu-mlp-executor/`

---

##  Objective

Train multilayer perceptrons (MLPs) using custom CUDA GPU kernels, and compare the performance with NumPy-based implementations.

---

##  Technologies Used

- **Python**
- **CUDA** (custom GPU kernel programming)
- **ctypes** (C–Python bridge for kernel integration)

---

##  Key Components

| File             | Description                                      |
|------------------|--------------------------------------------------|
| `gpu_op.cu`      | CUDA kernels for `MatMul`, `ReLU`, `Softmax`     |
| `autodiff.py`    | Symbolic computation graph and executor          |

---

##  Benchmarking Tools

- `nvprof` or `nsys` for memory profiling and kernel timing
- Evaluate CUDA memory allocation, kernel execution, and memory reuse

---

##  Expected Performance

- MLP training on GPU should be **significantly faster** than the NumPy baseline
- Target **~97% test accuracy** on MNIST using provided training configuration
-  Efficient memory reuse should minimize `cudaMalloc` calls across epochs
