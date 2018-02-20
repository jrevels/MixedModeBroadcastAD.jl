using MixedModeBroadcastAD: forward!, backward!
using CUDAnative
import CUDAdrv
import NVTX

include("../kernels.jl")

# NOTE: use with `--profile-from-start off`
NVTX.stop()

const DIMS = length(ARGS) >= 1 ? parse(Int,  ARGS[1]) : 2048
const KERNEL, OUTPUT, INPUTS, DERIVS = getkernel(:gpu, DIMS)

function benchmark(kernel, output, inputs, derivs)
    NVTX.@range "dual_broadcast" (dual_broadcast!(kernel, output, inputs, derivs), CUDAdrv.synchronize())
end

benchmark(KERNEL, OUTPUT, INPUTS, DERIVS) # warmup

NVTX.@activate CUDAdrv.@profile begin
    ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), STDERR.handle)
    benchmark(KERNEL, OUTPUT, INPUTS, DERIVS)
    ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), C_NULL)
end
