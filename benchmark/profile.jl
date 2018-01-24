using MixedModeBroadcastAD: record, forward!, backward!, CuArray, cuda_lstm_update_c
import CUDAdrv
import NVTX
NVTX.stop()

# NOTE: use with `--profile-from-start off`

const N = 2048

function benchmark()
    NVTX.@range "input generation"  input = Tuple(CuArray{Float32}((N,N)) for i in 1:10)
    NVTX.@range "tape construction" tape, _, _ = record(cuda_lstm_update_c, input...)
    NVTX.@range "forward pass"      (forward!(tape), CUDAdrv.synchronize())
    NVTX.@range "backward pass"     (backward!(tape), CUDAdrv.synchronize())
end

info("Warming up...")
benchmark.()
gc()

info("Benchmarking...")
NVTX.@activate CUDAdrv.@profile begin
    ccall(:jl_dump_compiles, Void, (Ptr{Nothing},), STDERR.handle)
    benchmark()
    ccall(:jl_dump_compiles, Void, (Ptr{Void},), C_NULL)
end
