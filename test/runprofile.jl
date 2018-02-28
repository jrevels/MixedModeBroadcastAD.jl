import CUDAdrv
import NVTX

include("kernels.jl")

# NOTE: use with `--profile-from-start off`
NVTX.stop()

const TFSTYLE = length(ARGS) >= 1 ? parse(Bool, ARGS[1]) : false
const DIMS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2048
const KERNEL!, INPUTS, DERIVS, BUFFERS = get_hmlstm_kernel(TFSTYLE, true, DIMS)

function benchmark(kernel!, inputs, derivs, buffers)
    NVTX.@range "kernel" (kernel!(inputs, derivs, buffers); CUDAdrv.synchronize())
end

benchmark(KERNEL!, INPUTS, DERIVS, BUFFERS) # warmup

NVTX.@activate CUDAdrv.@profile begin
    ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), STDERR.handle)
    benchmark(KERNEL!, INPUTS, DERIVS, BUFFERS)
    ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), C_NULL)
end
