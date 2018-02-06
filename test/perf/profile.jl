using MixedModeBroadcastAD: forward!, backward!
using CUDAnative
import CUDAdrv
import NVTX

include("../kernels.jl")

# NOTE: use with `--profile-from-start off`
NVTX.stop()

const PRECOMPUTED = length(ARGS) >= 1 ? parse(Bool, ARGS[1]) : false
const DIMS        = length(ARGS) >= 2 ? parse(Int,  ARGS[2]) : 2048
const TAPE        = gettape(:gpu, PRECOMPUTED, DIMS)

function benchmark(tape)
    NVTX.@range "forward pass"  (forward!(tape), CUDAdrv.synchronize())
    NVTX.@range "backward pass" (backward!(tape), CUDAdrv.synchronize())
end

benchmark(tape) # warm-up

NVTX.@activate CUDAdrv.@profile begin
    ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), STDERR.handle)
    benchmark(tape)
    ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), C_NULL)
end
