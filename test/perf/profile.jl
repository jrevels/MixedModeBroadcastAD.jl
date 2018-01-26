using MixedModeBroadcastAD: record, forward!, backward!, CuArray
using CUDAnative
import CUDAdrv
import NVTX
include("../kernels.jl")

# NOTE: use with `--profile-from-start off`
NVTX.stop()

const n = 2048

inputs = Tuple(CuArray{Float32}((n,n)) for i in 1:10)

# warm-up
cuda_lstm_update_c(inputs...)
cudanative_lstm_update_c(inputs...)
gc()

NVTX.@activate CUDAdrv.@profile begin
    ccall(:jl_dump_compiles, Void, (Ptr{Nothing},), STDERR.handle)
    NVTX.@range "CUDA" cuda_lstm_update_c(inputs...)
    NVTX.@range "CuArray" cudanative_lstm_update_c(inputs...)
    ccall(:jl_dump_compiles, Void, (Ptr{Void},), C_NULL)
end
