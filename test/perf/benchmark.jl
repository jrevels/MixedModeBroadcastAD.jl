using MixedModeBroadcastAD: record, forward!, backward!, CuArray
using CUDAnative
import CUDAdrv
include("util.jl")
include("../kernels.jl")

function benchmark(::Type{Array}, n, fused)
    inputs = Tuple(Array{Float32}((n,n)) for i in 1:10)
    output = Array{Float32}((n,n))
    temps = Tuple(Array{Float32}((n,n)) for i in 1:2)
    @elapsed if fused
        lstm_update_c(inputs...)
    else
        unfused_lstm_update_c(output, temps..., inputs...)
    end
end

function benchmark(::Type{CuArray}, n, fused)
    inputs = Tuple(CuArray{Float32}((n,n)) for i in 1:10)
    output = CuArray{Float32}((n,n))
    temps = Tuple(CuArray{Float32}((n,n)) for i in 1:2)
    @elapsed begin
        if fused
            cudanative_lstm_update_c(inputs...)
        else
            unfused_cudanative_lstm_update_c(output, temps..., inputs...)
        end
        CUDAdrv.synchronize()
    end
end

function benchmark_cuda(n, fused)
    inputs = Tuple(CuArray{Float32}((n,n)) for i in 1:10)
    output = CuArray{Float32}((n,n))
    temps = Tuple(CuArray{Float32}((n,n)) for i in 1:2)
    @elapsed begin
        if fused
            cuda_lstm_update_c(output, inputs...)
        else
            unfused_cuda_lstm_update_c(output, temps..., inputs...)
        end
        CUDAdrv.synchronize()
    end
end

rows = Any[["environment", "size", "fused", "elapsed"]]
for fused in [true, false], n in (2^i for i in 9:11)
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
