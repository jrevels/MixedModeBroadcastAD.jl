#include <cstdlib>
#include <iostream>
#include <chrono>

#include <cuda.h>

#define N 2048

__forceinline__ __device__ float sigmoidf(float in) {
     return 1.f / (1.f + expf(-in));  
}

__global__ void cuda_lstm_update_c(int n, float* out,
                                   const float* c,
                                   const float* Wx_f, const float* Wx_i, const float* Wx_c,
                                   const float* Rh_f, const float* Rh_i, const float* Rh_c,
                                   const float* b_f,  const float* b_i,  const float* b_c) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    out[index] = sigmoidf(Wx_f[index] + Rh_f[index] + b_f[index]) * c[index] +
                 sigmoidf(Wx_i[index] + Rh_i[index] + b_i[index]) *
                 tanh(Wx_c[index] + Rh_c[index] + b_c[index]);
}

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      std::cerr << "CUDA Error: " << cudaGetErrorString(stat) << " " << file << " " << line << std::endl;
   }
}

extern "C" float benchmark(int n) {
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
    cuda_lstm_update_c<<<gridDim, blockDim>>>(numElements, out, c, Wx_f, Wx_i, Wx_c, Rh_f, Rh_i, Rh_c, b_f, b_i, b_c);
    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<float> elapsed = end - start;
    return elapsed.count();
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " N" << std::endl;
        return EXIT_FAILURE;
    }
    int n = atoi(argv[1]);
    std::cout << benchmark(n) << std::endl;
    return EXIT_SUCCESS;
}
