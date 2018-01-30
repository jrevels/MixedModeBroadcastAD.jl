using MixedModeBroadcastAD: forward!, backward!
using CUDAnative
import CUDAdrv
import NVTX
include("util.jl")

# NOTE: use with `--profile-from-start off`
NVTX.stop()

const N            = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2048
const fusion_level = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
const tape = gettape(:cudanative, fusion_level, N)

function benchmark(tape)
    NVTX.@range "forward pass"  (forward!(tape), CUDAdrv.synchronize())
    NVTX.@range "backward pass" (backward!(tape), CUDAdrv.synchronize())
end

# warm-up
benchmark(tape)
benchmark(tape) # additional compilation by re-using the tape

# profile
NVTX.@activate CUDAdrv.@profile begin
    ccall(:jl_dump_compiles, Void, (Ptr{Nothing},), STDERR.handle)
    benchmark(tape)
    ccall(:jl_dump_compiles, Void, (Ptr{Void},), C_NULL)
end
