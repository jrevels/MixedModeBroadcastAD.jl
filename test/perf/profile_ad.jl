import MixedModeBroadcastAD
using MixedModeBroadcastAD: record, forward!, backward!, CuArray
using CUDAnative
import CUDAdrv
import NVTX
include("../kernels.jl")

# NOTE: use with `--profile-from-start off`
NVTX.stop()

const N     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2048
const fused = length(ARGS) >= 2 ? parse(Bool, ARGS[2]) : true

function prepare()
    input = Tuple(CuArray{Float32}((N,N)) for i in 1:10)
    tape, _, _ = record(fused ? cudanative_lstm_update_c : unfused_cudanative_lstm_update_c,
                        input...)
    tape
end

function benchmark(tape)
    NVTX.@range "forward pass"  (forward!(tape), CUDAdrv.synchronize())
    NVTX.@range "backward pass" (backward!(tape), CUDAdrv.synchronize())
end

# warm-up
tape = prepare()
benchmark(tape)
benchmark(tape) # re-run on existing tape triggers additional compilation
MixedModeBroadcastAD.setcacheconfig(MixedModeBroadcastAD.CU_FUNC_CACHE_PREFER_L1)

NVTX.@activate CUDAdrv.@profile begin
    ccall(:jl_dump_compiles, Void, (Ptr{Nothing},), STDERR.handle)
    benchmark(tape)
    ccall(:jl_dump_compiles, Void, (Ptr{Void},), C_NULL)
end
