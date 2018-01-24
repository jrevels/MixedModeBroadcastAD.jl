using MixedModeBroadcastAD: record, forward!, backward!, CuArray, lstm_update_c, cuda_lstm_update_c
import CUDAdrv
#=
Kernels benchmarked here are implemented in `src/kernels` and tested for correctness
in `test/runtests.jl`.

Since our technique is agnostic to a framework's method of tape construction, the
benchmarking code below pre-records the tape in order to avoid timing irrelevant
overhead due to tape construction.
=#

function benchmark(::Type{Array}, n::Int)
    input = Tuple(rand(Float32, n, n) for i in 1:10)
    tape, _, _ = record(lstm_update_c, input...)
    @elapsed(forward!(tape)), @elapsed(backward!(tape))
end

function benchmark(::Type{CuArray}, n::Int)
    input = Tuple(rand(Float32, n, n) for i in 1:10)
    tape, _, _ = record(cuda_lstm_update_c, CuArray.(input)...)
    CUDAdrv.@elapsed(forward!(tape)), CUDAdrv.@elapsed(backward!(tape))
end

rows = Any[["size", "type", "forwards", "backwards"]]
for n in (2^i for i in 9:11), T in [Array, CuArray]
    benchmark(T, n) # warm-up
    fwd, bwd = benchmark(T, n)
    push!(rows, [T, "$(n)x$(n)", "$fwd s", "$bwd s"])
end
show(STDOUT, Markdown.MD(Markdown.Table(rows, [:r, :c, :c, :c])))
