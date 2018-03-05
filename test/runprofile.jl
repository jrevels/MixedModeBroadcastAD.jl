import CUDAdrv
import NVTX

include("kernels.jl")

# NOTE: use with `--profile-from-start off`
NVTX.stop()

const TFSTYLE = length(ARGS) >= 1 ? parse(Bool, ARGS[1]) : false
const DIMS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2048
const ITERATIONS = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
const ARITY = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 0

ARITY > 0 && @assert !TFSTYLE
const KERNEL!, INPUTS, DERIVS, BUFFERS = if ARITY == 0
    get_hmlstm_kernel(TFSTYLE, true, DIMS)
else
    get_arity_scaling_kernel(true, DIMS, ARITY)
end

function benchmark(kernel!, inputs, derivs, buffers)
    NVTX.@range "kernel" (kernel!(inputs, derivs, buffers); CUDAdrv.synchronize())
end

benchmark(KERNEL!, INPUTS, DERIVS, BUFFERS) # warmup

NVTX.@activate CUDAdrv.@profile begin
    ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), STDERR.handle)
    for i in 1:ITERATIONS
        benchmark(KERNEL!, INPUTS, DERIVS, BUFFERS)
    end
    ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), C_NULL)
end
