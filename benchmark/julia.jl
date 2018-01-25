using MixedModeBroadcastAD: record, forward!, backward!, CuArray, lstm_update_c, cuda_lstm_update_c
import CUDAdrv
#=
Kernels benchmarked here are implemented in `src/kernels` and tested for correctness
in `test/runtests.jl`.

Since our technique is agnostic to a framework's method of tape construction, the
benchmarking code below pre-records the tape in order to avoid timing irrelevant
overhead due to tape construction.
=#

const TEST_TYPES = [Array, CuArray]

kernel(::Type{Array}) = lstm_update_c
kernel(::Type{CuArray}) = cuda_lstm_update_c

function prepare(::Type{T}, n::Int) where {T<:AbstractArray}
    input = Tuple(T{Float32}((n,n)) for i in 1:10)
    tape, _, _ = record(kernel(T), input...)
    tape
end

function benchmark(::Type{Array}, tape)
    @elapsed(forward!(tape)), @elapsed(backward!(tape))
end

function benchmark(::Type{CuArray}, tape)
    @elapsed((forward!(tape), CUDAdrv.synchronize())),
    @elapsed((backward!(tape), CUDAdrv.synchronize()))
end

rows = Any[["size", "type", "forwards", "backwards"]]
for n in (2^i for i in 9:11), T in TEST_TYPES
    tape = prepare(T, n)
    benchmark(T, tape) # warm-up
    fwd, bwd = benchmark(T, tape)
    push!(rows, [T, "$(n)x$(n)", "$fwd s", "$bwd s"])
end
show(STDOUT, Markdown.MD(Markdown.Table(rows, [:r, :c, :c, :c])))
