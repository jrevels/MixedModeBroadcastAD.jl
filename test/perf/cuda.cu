#include <cstdlib>
#include <iostream>
#include <chrono>

#include <cuda.h>

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      std::cerr << "CUDA Error: " << cudaGetErrorString(stat) << " " << file << " " << line << std::endl;
   }
}


//
// Fused
//

__forceinline__ __device__ float sigmoidf(float in) {
     return 1.f / (1.f + expf(-in));  
}

__global__ void fused_lstm_update_c(int numElements, float* out,
                                    const float* c,
                                    const float* Wx_f, const float* Wx_i, const float* Wx_c,
                                    const float* Rh_f, const float* Rh_i, const float* Rh_c,
                                    const float* b_f,  const float* b_i,  const float* b_c) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numElements) return;

    out[index] = sigmoidf(Wx_f[index] + Rh_f[index] + b_f[index]) * c[index] +
                 sigmoidf(Wx_i[index] + Rh_i[index] + b_i[index]) *
                 tanh(Wx_c[index] + Rh_c[index] + b_c[index]);
}


//
// Unfused
//

__global__ void pw_tanh(float *y, const float *a, int numElements) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < numElements) y[i] = tanh(a[i]);
}

__global__ void pw_sigmoid(float *y, const float *a, int numElements) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < numElements) y[i] = sigmoidf(a[i]);
}

__global__ void pw_vecAdd(float *y, const float *a,  const float *b, int numElements) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < numElements) y[i] = a[i] + b[i];
}

__global__ void pw_vecMul(float *y, const float *a,  const float *b, int numElements) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < numElements) y[i] = a[i] * b[i];
}

void unfused_lstm_update_c(int numElements, float* out,
                           const float* c,
                           const float* Wx_f, const float* Wx_i, const float* Wx_c,
                           const float* Rh_f, const float* Rh_i, const float* Rh_c,
                           const float* b_f,  const float* b_i,  const float* b_c) {

    dim3 blockDim;
    dim3 gridDim;

    blockDim.x = 256;
    gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;

    float *tmp1, *tmp2;
    cudaErrCheck(cudaMalloc((void**)&tmp1, numElements * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&tmp2, numElements * sizeof(float)));

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

    cudaErrCheck(cudaFree(tmp1));
    cudaErrCheck(cudaFree(tmp2));
}


//
// Entry-points
//

extern "C" void execute(int numElements, int fused, float* out,
                        const float* c,
                        const float* Wx_f, const float* Wx_i, const float* Wx_c,
                        const float* Rh_f, const float* Rh_i, const float* Rh_c,
                        const float* b_f,  const float* b_i,  const float* b_c) {
    if (fused) {
        dim3 blockDim;
        dim3 gridDim;

        blockDim.x = 256;
        gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;

        fused_lstm_update_c<<<gridDim, blockDim>>>(numElements, out, c, Wx_f, Wx_i, Wx_c, Rh_f, Rh_i, Rh_c, b_f, b_i, b_c);
    } else {
        unfused_lstm_update_c(numElements, out, c, Wx_f, Wx_i, Wx_c, Rh_f, Rh_i, Rh_c, b_f, b_i, b_c);
    }
}

extern "C" float benchmark(int n, int fused) {
    int numElements = n*n;

    float *out, *c, *Wx_f, *Wx_i, *Wx_c, *Rh_f, *Rh_i, *Rh_c, *b_f, *b_i, *b_c;
    cudaErrCheck(cudaMalloc((void**)&c, numElements * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&Wx_f, numElements * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&Rh_f, numElements * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&b_f, numElements * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&Wx_i, numElements * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&Rh_i, numElements * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&b_i, numElements * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&Wx_c, numElements * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&Rh_c, numElements * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&b_c, numElements * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&out, numElements * sizeof(float)));

    dim3 blockDim;
    dim3 gridDim;

    blockDim.x = 256;
    gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;

    auto start = std::chrono::system_clock::now();
    execute(numElements, fused, out, c, Wx_f, Wx_i, Wx_c, Rh_f, Rh_i, Rh_c, b_f, b_i, b_c);
    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<float> elapsed = end - start;
    return elapsed.count();
}


//
// Main
//

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " N" << std::endl;
        return EXIT_FAILURE;
    }
    int n = atoi(argv[1]);
    std::cout << "Fused: " << benchmark(n, 1) << std::endl;
    std::cout << "Unfused: " << benchmark(n, 0) << std::endl;
    return EXIT_SUCCESS;
}
