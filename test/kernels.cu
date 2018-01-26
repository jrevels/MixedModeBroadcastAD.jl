#include <cuda.h>
#include <iostream>

#define cudaErrCheck(stat)                                                     \
  { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
  if (stat != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(stat) << " " << file
              << " " << line << std::endl;
  }
}

//
// Fused
//

__forceinline__ __device__ float sigmoidf(float in) {
  return 1.f / (1.f + expf(-in));
}

__global__ void lstm_update_c_kernel(int numElements, float *out, const float *c,
                                    const float *Wx_f, const float *Wx_i,
                                    const float *Wx_c, const float *Rh_f,
                                    const float *Rh_i, const float *Rh_c,
                                    const float *b_f, const float *b_i,
                                    const float *b_c) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numElements)
    return;

  out[index] = sigmoidf(Wx_f[index] + Rh_f[index] + b_f[index]) * c[index] +
               sigmoidf(Wx_i[index] + Rh_i[index] + b_i[index]) *
                   tanh(Wx_c[index] + Rh_c[index] + b_c[index]);
}

extern "C" void lstm_update_c(int numElements, float *out, const float *c, const float *Wx_f,
                        const float *Wx_i, const float *Wx_c, const float *Rh_f,
                        const float *Rh_i, const float *Rh_c, const float *b_f,
                        const float *b_i, const float *b_c) {
    dim3 blockDim;
    dim3 gridDim;

    blockDim.x = 256;
    gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;

    lstm_update_c_kernel<<<gridDim, blockDim>>>(
        numElements, out, c, Wx_f, Wx_i, Wx_c, Rh_f, Rh_i, Rh_c, b_f, b_i, b_c);

}


//
// Unfused
//

__global__ void pw_tanh(float *y, const float *a, int numElements) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numElements)
    y[i] = tanh(a[i]);
}

__global__ void pw_sigmoid(float *y, const float *a, int numElements) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numElements)
    y[i] = sigmoidf(a[i]);
}

__global__ void pw_vecAdd(float *y, const float *a, const float *b,
                          int numElements) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numElements)
    y[i] = a[i] + b[i];
}

__global__ void pw_vecMul(float *y, const float *a, const float *b,
                          int numElements) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numElements)
    y[i] = a[i] * b[i];
}

extern "C" void unfused_lstm_update_c(int numElements, float *out, float *tmp1,
                                      float *tmp2, const float *c, const float *Wx_f,
                                      const float *Wx_i, const float *Wx_c,
                                      const float *Rh_f, const float *Rh_i,
                                      const float *Rh_c, const float *b_f,
                                      const float *b_i, const float *b_c) {
  dim3 blockDim;
  dim3 gridDim;

  blockDim.x = 256;
  gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;

  // sigmoid(Wx_f + Rh_f + b_f) * c
  pw_vecAdd<<<gridDim, blockDim>>>(tmp1, Wx_f, Rh_f, numElements);
  pw_vecAdd<<<gridDim, blockDim>>>(tmp1, tmp1, b_f, numElements);
  pw_sigmoid<<<gridDim, blockDim>>>(tmp1, tmp1, numElements);
  pw_vecMul<<<gridDim, blockDim>>>(tmp1, tmp1, c, numElements);

  // sigmoid(Wx_i + Rh_i + b_i)
  pw_vecAdd<<<gridDim, blockDim>>>(tmp2, Wx_i, Rh_i, numElements);
  pw_vecAdd<<<gridDim, blockDim>>>(tmp2, tmp2, b_i, numElements);
  pw_sigmoid<<<gridDim, blockDim>>>(tmp2, tmp2, numElements);

  // tanh(Wx_c + Rh_c + b_c)
  pw_vecAdd<<<gridDim, blockDim>>>(out, Wx_c, Rh_c, numElements);
  pw_vecAdd<<<gridDim, blockDim>>>(out, out, b_c, numElements);
  pw_tanh<<<gridDim, blockDim>>>(out, out, numElements);

  // sigmoid(...) * tanh(...)
  pw_vecMul<<<gridDim, blockDim>>>(out, out, tmp2, numElements);

  // sigmoid(...) + sigmoid(...) * tanh(...)
  pw_vecAdd<<<gridDim, blockDim>>>(out, out, tmp1, numElements);
}
