using MixedModeBroadcastAD: record, forward!, backward!, CuArray, lstm_update_c, cuda_lstm_update_c
import CUDAdrv
include("util.jl")

function benchmark(::Type{Array}, n)
    inputs = Tuple(Array{Float32}((n,n)) for i in 1:10)
    @elapsed(lstm_update_c.(inputs...))
end

function benchmark(::Type{CuArray}, n)
    inputs = Tuple(CuArray{Float32}((n,n)) for i in 1:10)
    @elapsed((cuda_lstm_update_c.(inputs...), CUDAdrv.synchronize()))
end

function benchmark_cuda(n, fused)
    lib = Libdl.dlopen("./cuda.so")
    fun = Libdl.dlsym(lib, "benchmark")
    ccall(fun, Cfloat, (Cint, Cint), n, fused)
end

rows = Any[["environment", "size", "fused", "elapsed"]]
for n in (2^i for i in 9:11)
    # Julia arrays
    for T in [Array, CuArray]
        benchmark(T, n) # warm-up
        elapsed = benchmark(T, n)
        push!(rows, ["Julia $T", "$(n)x$(n)", true, timedelta(elapsed)])
    end

    # CUDA
    for fused in [true, false]
        push!(rows, ["CUDA", "$(n)x$(n)", fused, timedelta(benchmark_cuda(n, fused))])
    end
end
println(Markdown.MD(Markdown.Table(rows, [:r, :c, :c, :c])))
