using MixedModeBroadcastAD: record, forward!, backward!, CuArray
using BenchmarkTools
using CUDAnative
import CUDAdrv
include("util.jl")
include("../kernels.jl")

# speed it up a little
BenchmarkTools.DEFAULT_PARAMETERS.samples = 1
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1
# make sure we collect CuArrays from previous iterations
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true

function benchmark(::Type{Array}, n, fused)
    inputs = Tuple(Array{Float32}((n,n)) for i in 1:10)
    if fused
        @belapsed lstm_update_c($inputs...)
    else
        temps = Tuple(Array{Float32}((n,n)) for i in 1:2)
        @belapsed unfused_lstm_update_c($temps..., $inputs...)
    end
end

function benchmark(::Type{CuArray}, n, fused)
    inputs = Tuple(CuArray{Float32}((n,n)) for i in 1:10)
    if fused
        @belapsed cudanative_lstm_update_c($inputs...)
    else
        temps = Tuple(CuArray{Float32}((n,n)) for i in 1:2)
        @belapsed unfused_cudanative_lstm_update_c($temps..., $inputs...)
    end
end

function benchmark_cuda(n, fused)
    inputs = Tuple(CuArray{Float32}((n,n)) for i in 1:10)
    if fused
        @belapsed cuda_lstm_update_c($inputs...)
    else
        temps = Tuple(CuArray{Float32}((n,n)) for i in 1:2)
        @belapsed unfused_cuda_lstm_update_c($temps..., $inputs...)
    end
end

rows = Any[["environment", "size", "fused", "elapsed"]]
for fused in [true, false], n in (2^i for i in 9:11)
    info("benchmarking fused=$fused size=$n")

    # Julia arrays
    for T in [Array, CuArray]
        benchmark(T, n, fused) # warm-up
        elapsed = benchmark(T, n, fused)
        push!(rows, ["Julia $T", "$(n)x$(n)", fused, timedelta(elapsed)])
    end

    # CUDA
    push!(rows, ["CUDA", "$(n)x$(n)", fused, timedelta(benchmark_cuda(n, fused))])
end
println(Markdown.MD(Markdown.Table(rows, [:r, :c, :c, :c])))
