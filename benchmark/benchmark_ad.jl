using MixedModeBroadcastAD: record, forward!, backward!, CuArray, lstm_update_c, cuda_lstm_update_c
import CUDAdrv
include("util.jl")

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

rows = Any[["environment", "size", "forwards", "backwards"]]
for n in (2^i for i in 9:11), T in [Array, CuArray]
    tape = prepare(T, n)
    benchmark(T, tape) # warm-up
    fwd, bwd = benchmark(T, tape)
    push!(rows, ["Julia $T", "$(n)x$(n)", timedelta(fwd), timedelta(bwd)])
end
println(Markdown.MD(Markdown.Table(rows, [:r, :c, :c, :c])))
