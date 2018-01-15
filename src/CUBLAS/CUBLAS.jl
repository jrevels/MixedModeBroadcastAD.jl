module CUBLAS

import Base: one, zero
using CUDAapi, CUDAdrv
using ..MixedModeBroadcastAD: CuArray, CuVector, CuMatrix, CuVecOrMat

const toolkit = CUDAapi.find_toolkit()
const libcublas = CUDAapi.find_cuda_library("cublas", toolkit)
@assert libcublas != nothing

include("libcublas_types.jl")
include("error.jl")
const cudaStream_t = Ptr{Void}
include("libcublas.jl")

const libcublas_handle = Ref{cublasHandle_t}()
function __init__()
    cublasCreate_v2(libcublas_handle)
    atexit(()->cublasDestroy_v2(libcublas_handle[]))
end

include("wrap.jl")

end
