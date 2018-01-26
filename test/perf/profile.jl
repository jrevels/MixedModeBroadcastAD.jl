using MixedModeBroadcastAD: record, forward!, backward!, CuArray
using CUDAnative
import CUDAdrv
import NVTX
include("../kernels.jl")

# NOTE: use with `--profile-from-start off`
NVTX.stop()

const N     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2048
const fused = length(ARGS) >= 2 ? parse(Bool, ARGS[2]) : true

inputs = Tuple(CuArray{Float32}((N,N)) for i in 1:10)

const cuda_fun       = fused ? cuda_lstm_update_c       : unfused_cuda_lstm_update_c
const cudanative_fun = fused ? cudanative_lstm_update_c : unfused_cudanative_lstm_update_c

# warm-up
cuda_fun(inputs...)
cudanative_fun(inputs...)
gc()

NVTX.@activate CUDAdrv.@profile begin
    ccall(:jl_dump_compiles, Void, (Ptr{Nothing},), STDERR.handle)
    NVTX.@range "CUDA" cuda_fun(inputs...)
    NVTX.@range "CuArray" cudanative_fun(inputs...)
    ccall(:jl_dump_compiles, Void, (Ptr{Void},), C_NULL)
end
