using MixedModeBroadcastAD: forward!, backward!
using CUDAnative
import CUDAdrv
import NVTX

include("kernels.jl")

# NOTE: use with `--profile-from-start off`
NVTX.stop()

const DIMS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2048
const TFSTYLE = length(ARGS) >= 2 ? parse(Bool, ARGS[2]) : false
const KERNEL!, INPUTS, DERIVS = get_kernel(:gpu, DIMS, TFSTYLE)

function benchmark(kernel!, inputs, derivs)
    NVTX.@range "kernel" (kernel!(inputs, derivs); CUDAdrv.synchronize())
end

benchmark(KERNEL!, INPUTS, DERIVS) # warmup

NVTX.@activate CUDAdrv.@profile begin
    ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), STDERR.handle)
    benchmark(KERNEL!, INPUTS, DERIVS)
    ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), C_NULL)
end
