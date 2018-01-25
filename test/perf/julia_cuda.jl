# limited version of julia.jl with extra annotations, for ease of profiling & analysis

using MixedModeBroadcastAD: record, forward!, backward!, CuArray
import CUDAdrv
import NVTX
NVTX.stop()

include("../kernels.jl")

# NOTE: use with `--profile-from-start off`

const N = 2048

function prepare()
    input = Tuple(CuArray{Float32}((N,N)) for i in 1:10)
    tape, _, _ = record(cuda_lstm_update_c, input...)
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
