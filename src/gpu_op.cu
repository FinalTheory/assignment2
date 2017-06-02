#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

static const size_t THREADS_PER_BLK = 512;

/* all GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

__global__ void array_set(float *arr, size_t len, const float val) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) { arr[i] = val; }
}

__global__ void array_broadcast(const float *src, size_t src_len, float *dst, size_t dst_len) {
  auto i = threadIdx.x;
  auto from = src;
  auto to = dst + src_len * blockIdx.x;
  auto skip = blockDim.x;
  while (i < src_len) {
    to[i] = from[i];
    i += skip;
  }
}

__global__ void reduce_sum_axis_zero(const float *src, size_t src_len, float *dst, size_t dst_len) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < dst_len) {
    auto idx = i;
    float sum = 0.f;
    while (idx < src_len) {
      sum += src[idx];
      idx += dst_len;
    }
    dst[i] = sum;
  }
}

__global__ void matrix_add(const float *matA, const float *matB, float *out, size_t len) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) { out[i] = matA[i] + matB[i]; }
}

__global__ void matrix_add_by_const(const float *src, float *dst, size_t len, const float val) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) { dst[i] = src[i] + val; }
}

__global__ void matrix_dotmul(const float *matA, const float *matB, float *out, size_t len) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) { out[i] = matA[i] * matB[i]; }
}

__global__ void matrix_dotmul_by_const(const float *src, float *dst, size_t len, const float val) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) { dst[i] = src[i] * val; }
}

__global__ void array_relu(const float *src, float *dst, size_t len) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) { dst[i] = src[i] > 0.f ? src[i] : 0.f; }
}

__global__ void array_relu_grad(const float *input, const float *src, float *dst, size_t len) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) { dst[i] = input[i] >= 0.f ? src[i] : 0.f; }
}

__global__ void matrix_softmax(const float *src, float *dst, size_t rows, size_t cols) {
  __shared__ float temp[THREADS_PER_BLK];
  // locates to row
  auto from = src + blockIdx.x * cols;
  auto to  = dst + blockIdx.x * cols;
  const auto tid = threadIdx.x;
  const auto skip = blockDim.x;
  float _max = from[0];

  auto idx = tid;
  while (idx < cols) {
    _max = max(_max, from[idx]);
    idx += skip;
  }
  temp[tid] = _max;
  __syncthreads();
  // reduction find max
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      temp[tid] = max(temp[tid], temp[tid + s]);
    }
    __syncthreads();
  }
  // this is row max
  _max = temp[0];
  float _sum = 0.f;
  idx = tid;
  while (idx < cols) {
    auto v = __expf(from[idx] - _max);
    _sum += v;
    to[idx] = v;
    idx += skip;
  }
  temp[tid] = _sum;
  __syncthreads();
  // reduction sum
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      temp[tid] += temp[tid + s];
    }
    __syncthreads();
  }
  // this is row sum
  _sum = temp[0];
  // divide all elementes by sum
  idx = tid;
  while (idx < cols) {
    to[idx] /= _sum;
    idx += skip;
  }
}

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

static inline size_t array_len(const DLArrayHandle arr) {
  size_t len = 1;
  for (int i = 0; i < arr->ndim; i++) { len *= arr->shape[i]; }
  return len;
}

int DLGpuArraySet(DLArrayHandle arr, float value) {
  size_t len = array_len(arr);
  auto nblocks = (len + THREADS_PER_BLK - 1) / THREADS_PER_BLK;
  array_set<<<nblocks, THREADS_PER_BLK>>>((float *)arr->data, len, value);
  if (cudaGetLastError() != cudaSuccess) return -1;
  return 0;
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  auto src_len = array_len(input);
  auto dst_len = array_len(output);
  if (dst_len % src_len != 0) { return -1; }
  array_broadcast<<<dst_len / src_len, THREADS_PER_BLK>>>((float *)input->data, src_len, (float *)output->data, dst_len);
  if (cudaGetLastError() != cudaSuccess) return -1;
  return 0;
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  auto src_len = array_len(input);
  auto dst_len = array_len(output);
  if (src_len % dst_len != 0) { return -1; }
  auto nblocks = (dst_len + THREADS_PER_BLK - 1) / THREADS_PER_BLK;
  reduce_sum_axis_zero<<<nblocks, THREADS_PER_BLK>>>((float *)input->data, src_len, (float *)output->data, dst_len);
  if (cudaGetLastError() != cudaSuccess) return -1;
  return 0;
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  auto lenA = array_len(matA);
  auto lenB = array_len(matB);
  auto len = array_len(output);
  if (len != lenA || len != lenB) { return -1; }
  auto nblocks = (len + THREADS_PER_BLK - 1) / THREADS_PER_BLK;
  matrix_add<<<nblocks, THREADS_PER_BLK>>>((float *)matA->data, (float *)matB->data, (float *)output->data, len);
  if (cudaGetLastError() != cudaSuccess) return -1;
  return 0;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  auto src_len = array_len(input);
  auto dst_len = array_len(output);
  if (src_len != dst_len) { return -1; }
  auto nblocks = (src_len + THREADS_PER_BLK - 1) / THREADS_PER_BLK;
  matrix_add_by_const<<<nblocks, THREADS_PER_BLK>>>((float *)input->data, (float *)output->data, src_len, val);
  if (cudaGetLastError() != cudaSuccess) return -1;
  return 0;
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  auto lenA = array_len(matA);
  auto lenB = array_len(matB);
  auto len = array_len(output);
  if (len != lenA || len != lenB) { return -1; }
  auto nblocks = (len + THREADS_PER_BLK - 1) / THREADS_PER_BLK;
  matrix_dotmul<<<nblocks, THREADS_PER_BLK>>>((float *)matA->data, (float *)matB->data, (float *)output->data, len);
  if (cudaGetLastError() != cudaSuccess) return -1;
  return 0;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  auto src_len = array_len(input);
  auto dst_len = array_len(output);
  if (src_len != dst_len) { return -1; }
  auto nblocks = (src_len + THREADS_PER_BLK - 1) / THREADS_PER_BLK;
  matrix_dotmul_by_const<<<nblocks, THREADS_PER_BLK>>>((float *)input->data, (float *)output->data, src_len, val);
  if (cudaGetLastError() != cudaSuccess) return -1;
  return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  static cublasHandle_t handle;
  cublasCreate(&handle);
  auto m = transposeA ? matA->shape[1] : matA->shape[0];
  auto k = transposeA ? matA->shape[0] : matA->shape[1];
  auto n = transposeB ? matB->shape[0] : matB->shape[1];
  auto A = (float *)matA->data;
  auto B = (float *)matB->data;
  auto C = (float *)matC->data;
  float alpha = 1.f;
  float beta = 0.f;
  // cublas assume matrix is column major
  // [n, k] x [k, m] => [n, m]
  return (cublasSgemm(handle, transposeB ? CUBLAS_OP_T : CUBLAS_OP_N, transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
                      n, m, k, &alpha, B, transposeB ? k : n, A, transposeA ? m : k, &beta, C, n) == CUBLAS_STATUS_SUCCESS) ? 0 : -1;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  auto src_len = array_len(input);
  auto dst_len = array_len(output);
  if (src_len != dst_len) { return -1; }
  auto nblocks = (src_len + THREADS_PER_BLK - 1) / THREADS_PER_BLK;
  array_relu<<<nblocks, THREADS_PER_BLK>>>((float *)input->data, (float *)output->data, src_len);
  if (cudaGetLastError() != cudaSuccess) return -1;
  return 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  auto src_len = array_len(input);
  auto dst_len = array_len(output);
  auto grad_len = array_len(output);
  if (src_len != dst_len || src_len != grad_len) { return -1; }
  auto nblocks = (src_len + THREADS_PER_BLK - 1) / THREADS_PER_BLK;
  array_relu_grad<<<nblocks, THREADS_PER_BLK>>>((float *)input->data, (float *)in_grad->data, (float *)output->data, src_len);
  if (cudaGetLastError() != cudaSuccess) return -1;
  return 0;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  assert(input->ndim == output->ndim && input->ndim == 2);
  for (int i = 0; i < input->ndim; i++) assert(input->shape[i] == output->shape[i]);
  auto rows = input->shape[0];
  auto cols = input->shape[1];
  matrix_softmax<<<rows, THREADS_PER_BLK>>>((float *)input->data, (float *)output->data,
                                            rows, cols);
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
