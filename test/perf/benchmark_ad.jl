using MixedModeBroadcastAD: record, forward!, backward!, CuArray
using CUDAnative
import CUDAdrv
include("util.jl")
include("../kernels.jl")

kernel(::Type{Array}, fused) = fused ? lstm_update_c : unfused_lstm_update_c
kernel(::Type{CuArray}, fused) = fused ? cuda_lstm_update_c : unfused_cuda_lstm_update_c

function prepare(::Type{T}, n::Int, fused) where {T<:AbstractArray}
    input = Tuple(T{Float32}((n,n)) for i in 1:10)
    output = T{Float32}((n,n))
    tape, _, _ = if fused
        record(kernel(T, fused), input...)
    else
        record(kernel(T, fused), output, input...)
    end
    tape
end

function benchmark(::Type{Array}, tape)
    @elapsed(forward!(tape)), @elapsed(backward!(tape))
end

function benchmark(::Type{CuArray}, tape)
    @elapsed((forward!(tape), CUDAdrv.synchronize())),
    @elapsed((backward!(tape), CUDAdrv.synchronize()))
end

rows = Any[["environment", "size", "fused", "forwards", "backwards"]]
for fused in [true, false], n in (2^i for i in 9:11), T in [Array, CuArray]
    tape = prepare(T, n, fused)
    benchmark(T, tape) # warm-up
    fwd, bwd = benchmark(T, tape)
    push!(rows, ["Julia $T", "$(n)x$(n)", fused, timedelta(fwd), timedelta(bwd)])
end
println(Markdown.MD(Markdown.Table(rows, [:r, :c, :c, :c, :c])))
