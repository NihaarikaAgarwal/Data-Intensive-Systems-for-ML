#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

__global__ void array_set_kernel(index_t n, float *data, float value) {
  index_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n)
    data[idx] = value;
}

int DLGpuArraySet(DLArrayHandle arr, float value) { /* TODO: Your code here */
  index_t n = 1;
  for (int i = 0; i < arr->ndim; i++)
    n *= arr->shape[i];

  float *data = (float *) arr->data;
  int thread_per_block = 512;
  int n_blocks = (n + thread_per_block - 1) / thread_per_block;
  array_set_kernel<<<n_blocks, thread_per_block>>>(n, data, value);
  return 0;
}

__global__ void broadcast_to_kernel(const float *input_data, 
  float *output_data,
  index_t input_n,
  index_t output_n) {
index_t idx = blockDim.x * blockIdx.x + threadIdx.x;
if (idx < output_n) {
output_data[idx] = input_data[idx % input_n];
}
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  index_t input_n = 1;
  for (int i = 0; i < input->ndim; i++)
    input_n *= input->shape[i];

  index_t output_n = 1;
  for (int i = 0; i < output->ndim; i++)
    output_n *= output->shape[i];

  const float *input_data = (const float *) input->data;
  float *output_data = (float *) output->data;

  int thread_per_block = 512;
  int n_blocks = (output_n + thread_per_block - 1) / thread_per_block;
  broadcast_to_kernel<<<n_blocks, thread_per_block>>>(input_data, output_data,
                                                      input_n, output_n);
  return 0;
}

__global__ void reduce_sum_axis_zero_kernel(const float *input_data, 
  float *output_data,
  index_t input_n,
  index_t output_n) {
index_t idx = blockDim.x * blockIdx.x + threadIdx.x;
if (idx < output_n) {
output_data[idx] = 0.0; 
for (index_t i = 0; i < input_n / output_n; i++)
output_data[idx] += input_data[i * output_n + idx];
}
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  index_t input_n = 1;
  for (int i = 0; i < input->ndim; i++)
    input_n *= input->shape[i];

  index_t output_n = 1;
  for (int i = 0; i < output->ndim; i++)
    output_n *= output->shape[i];

  const float *input_data = (const float *) input->data;
  float *output_data = (float *) output->data;

  int thread_per_block = 512;
  int n_blocks = (output_n + thread_per_block - 1) / thread_per_block;
  reduce_sum_axis_zero_kernel<<<n_blocks, thread_per_block>>>(input_data, 
                                                              output_data,
                                                              input_n, 
                                                              output_n);
  return 0;
}

__global__ void matrix_elementwise_add_kernel(const float *matA_data,
  const float *matB_data,
  float *output_data, 
  index_t n) {
index_t idx = blockDim.x * blockIdx.x + threadIdx.x;
if (idx < n) 
output_data[idx] = matA_data[idx] + matB_data[idx];
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  /* TODO: Your code here */
  index_t n = 1;
  for (int i = 0; i < output->ndim; i++)
    n *= output->shape[i];

  const float *matA_data = (const float *) matA->data;
  const float *matB_data = (const float *) matB->data;
  float *output_data = (float *) output->data;
  int thread_per_block = 512;
  int n_blocks = (n + thread_per_block - 1) / thread_per_block;
  matrix_elementwise_add_kernel<<<n_blocks, thread_per_block>>>(matA_data, 
                                                                matB_data, 
                                                                output_data,
                                                                n); 
  return 0;
}

__global__ void matrix_elementwise_add_by_const_kernel(const float *input_data,
  float *output_data,
  float val,
  index_t n) {
index_t idx = blockDim.x * blockIdx.x + threadIdx.x;
if (idx < n) 
output_data[idx] = input_data[idx] + val;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  /* TODO: Your code here */
  index_t n = 1;
  for (int i = 0; i < output->ndim; i++)
    n *= output->shape[i];

  const float *input_data = (const float *) input->data;
  float *output_data = (float *) output->data;
  int thread_per_block = 512;
  int n_blocks = (n + thread_per_block - 1) / thread_per_block;
  matrix_elementwise_add_by_const_kernel<<<n_blocks, thread_per_block>>>(input_data, 
                                                                         output_data,
                                                                         val,
                                                                         n); 
  return 0;
}

__global__ void matrix_elementwise_multiply_kernel(const float *matA_data,
  const float *matB_data,
  float *output_data, 
  index_t n) {
index_t idx = blockDim.x * blockIdx.x + threadIdx.x;
if (idx < n) 
output_data[idx] = matA_data[idx] * matB_data[idx];
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  /* TODO: Your code here */
  index_t n = 1;
  for (int i = 0; i < output->ndim; i++)
    n *= output->shape[i];

  const float *matA_data = (const float *) matA->data;
  const float *matB_data = (const float *) matB->data;
  float *output_data = (float *) output->data;
  int thread_per_block = 512;
  int n_blocks = (n + thread_per_block - 1) / thread_per_block;
  matrix_elementwise_multiply_kernel<<<n_blocks, thread_per_block>>>(matA_data, 
                                                                     matB_data, 
                                                                     output_data,
                                                                     n);
  return 0;
}

__global__ void matrix_elementwise_multiply_by_const_kernel(const float *input_data,
  float *output_data,
  float val,
  index_t n) {
index_t idx = blockDim.x * blockIdx.x + threadIdx.x;
if (idx < n) 
output_data[idx] = input_data[idx] * val;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  /* TODO: Your code here */
  index_t n = 1;
  for (int i = 0; i < output->ndim; i++)
    n *= output->shape[i];

  const float *input_data = (const float *) input->data;
  float *output_data = (float *) output->data;
  int thread_per_block = 512;
  int n_blocks = (n + thread_per_block - 1) / thread_per_block;
  matrix_elementwise_multiply_by_const_kernel<<<n_blocks, thread_per_block>>>(input_data, 
                                                                              output_data,
                                                                              val,
                                                                              n); 
  return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  /* TODO: Your code here */
  // Hint: use cublas
  // cublas assume matrix is column major
  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) 
    printf("CUBLAS initialization failed\n");

  const float *matA_data = (const float *) matA->data;
  const float *matB_data = (const float *) matB->data;
  float *matC_data = (float *) matC->data;

  cublasOperation_t transa = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;

  int m = transposeB ? matB->shape[0] : matB->shape[1];
  int n = transposeA ? matA->shape[1] : matA->shape[0];
  int k = transposeA ? matA->shape[0] : matA->shape[1];

  float alpha = 1.0f;
  float beta = 0.0f;
  stat = cublasSgemm(handle, transb, transa,
                     m, n, k,
                     &alpha, matB_data, matB->shape[1],
                     matA_data, matA->shape[1], 
                     &beta, matC_data, m);
  if (stat != CUBLAS_STATUS_SUCCESS) 
    printf("CUBLAS kernel execution error.\n");
                 
  stat = cublasDestroy(handle);
    if (stat != CUBLAS_STATUS_SUCCESS) 
      printf("CUBLAS shutdown error\n");
                  
  return 0;
}

__global__ void relu_kernel(const float *input_data, float *output_data, 
  index_t n) {
index_t idx = blockDim.x * blockIdx.x + threadIdx.x;
if (idx < n) 
output_data[idx] = input_data[idx] > 0 ? input_data[idx] : 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  index_t n = 1;
  for (int i = 0; i < output->ndim; i++)
    n *= output->shape[i];

  const float *input_data = (const float *) input->data;
  float *output_data = (float *) output->data;
  int thread_per_block = 512;
  int n_blocks = (n + thread_per_block - 1) / thread_per_block;
  relu_kernel<<<n_blocks, thread_per_block>>>(input_data, output_data, n);
  return 0;
}

__global__ void relu_gradient_kernel(const float *input_data, 
  const float *in_grad_data, 
  float *output_data,
  index_t n) {
index_t idx = blockDim.x * blockIdx.x + threadIdx.x;
if (idx < n) 
output_data[idx] = input_data[idx] > 0 ? in_grad_data[idx] : 0;
}


int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  /* TODO: Your code here */
  index_t n = 1;
  for (int i = 0; i < input->ndim; i++)
    n *= input->shape[i];

  const float *input_data = (const float *) input->data;
  const float *in_grad_data = (const float *) in_grad->data;
  float *output_data = (float *) output->data;

  int thread_per_block = 512;
  int n_blocks = (n + thread_per_block - 1) / thread_per_block;
  relu_gradient_kernel<<<n_blocks, thread_per_block>>>(input_data, 
                                                       in_grad_data,
                                                       output_data,
                                                       n);
  return 0;
}

__global__ void softmax_kernel(const float *input_data, float *output_data,
  index_t row, index_t col) {
index_t idx = blockDim.x * blockIdx.x + threadIdx.x;
// Operate on each row
if (idx < row) {
// Find max
float max_val = input_data[idx*col];
for (int i = idx*col; i < (idx+1) * col; i++) 
max_val = max(max_val, input_data[i]);

// Subtract max and exp
float sum = 0.0;
for (int i = idx*col; i < (idx+1) * col; i++) {
output_data[i] = exp(input_data[i] - max_val);
sum += output_data[i];
}

// Divide sum of this row
for (int i = idx*col; i < (idx+1) * col; i++) 
output_data[i] /= sum;
}

}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  const float *input_data = (const float *) input->data;
  float *output_data = (float *) output->data;

  int n_blocks = output->shape[0];
  softmax_kernel<<<n_blocks, 1>>>(input_data, output_data, output->shape[0],
                                  output->shape[1]);
  return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
