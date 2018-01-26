using MixedModeBroadcastAD: record, forward!, backward!, CuArray
using CUDAnative
import CUDAdrv
import NVTX
include("../kernels.jl")

# NOTE: use with `--profile-from-start off`
NVTX.stop()

const N = 2048

function prepare()
    input = Tuple(CuArray{Float32}((N,N)) for i in 1:10)
    tape, _, _ = record(cudanative_lstm_update_c, input...)
    tape
end

function benchmark(tape)
    NVTX.@range "forward pass"      (forward!(tape), CUDAdrv.synchronize())
    NVTX.@range "backward pass"     (backward!(tape), CUDAdrv.synchronize())
end

tape = prepare()
benchmark(tape)
NVTX.@activate CUDAdrv.@profile begin
    ccall(:jl_dump_compiles, Void, (Ptr{Nothing},), STDERR.handle)
    benchmark(tape)
    ccall(:jl_dump_compiles, Void, (Ptr{Void},), C_NULL)
end
