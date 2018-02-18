using MixedModeBroadcastAD: forward!, backward!
using CUDAnative
import CUDAdrv
import NVTX

include("../kernels.jl")

# NOTE: use with `--profile-from-start off`
NVTX.stop()

const DIMS = length(ARGS) >= 1 ? parse(Int,  ARGS[1]) : 2048
const KERNEL, INPUT_VALUES, INPUT_DERIVS, OUTPUT_VALUE = getkernel(:gpu, DIMS)

function benchmark(kernel, input_values, input_derivs, output_value)
    NVTX.@range "autodiff_broadcast" (autodiff_broadcast!(kernel, input_values, input_derivs, output_value),  CUDAdrv.synchronize())
end

benchmark(KERNEL, INPUT_VALUES, INPUT_DERIVS, OUTPUT_VALUE) # warmup

NVTX.@activate CUDAdrv.@profile begin
    ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), STDERR.handle)
    benchmark(KERNEL, INPUT_VALUES, INPUT_DERIVS, OUTPUT_VALUE)
    ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), C_NULL)
end
