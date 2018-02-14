using MixedModeBroadcastAD: forward!, backward!
using CUDAnative
import CUDAdrv
import NVTX

include("../kernels.jl")

# NOTE: use with `--profile-from-start off`
NVTX.stop()

const DIMS = length(ARGS) >= 1 ? parse(Int,  ARGS[1]) : 2048
const SOA  = length(ARGS) >= 2 ? parse(Bool, ARGS[2]) : true
const TAPE = gettape(:gpu, SOA, DIMS)

function benchmark(tape)
    NVTX.@range "forward pass"  (forward!(tape),  CUDAdrv.synchronize())
    NVTX.@range "backward pass" (backward!(tape), CUDAdrv.synchronize())
end

benchmark(TAPE) # warm-up

NVTX.@activate CUDAdrv.@profile begin
    ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), STDERR.handle)
    benchmark(TAPE)
    ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), C_NULL)
end
