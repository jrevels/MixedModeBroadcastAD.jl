using MixedModeBroadcastAD: forward!, backward!
using CUDAnative
import CUDAdrv
import NVTX

include("../kernels.jl")

# NOTE: use with `--profile-from-start off`
NVTX.stop()

const DIMS        = length(ARGS) >= 1 ? parse(Int, ARGS[1])  : 2048
const PRECOMPUTED = length(ARGS) >= 2 ? parse(Bool, ARGS[2]) : false
const TAPE        = gettape(:gpu, precomputed, dims)

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
