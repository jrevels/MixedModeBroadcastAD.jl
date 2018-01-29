using MixedModeBroadcastAD: forward!, backward!
using BenchmarkTools
using CUDAnative
import CUDAdrv
include("util.jl")

# make sure we collect CuArrays from previous iterations
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true

function benchmark(kind::Symbol, args...)
    tape = gettape(kind, args...)
    if kind == :cpu
        return (@belapsed(forward!($tape)),
                @belapsed(backward!($tape)))
    elseif kind == :cudanative
        return (@belapsed((forward!($tape), CUDAdrv.synchronize())),
                @belapsed((backward!($tape), CUDAdrv.synchronize())))
    end
end

rows = Any[["environment", "size", "fusion_level", "forwards", "backwards"]]
for fusion_level in [0, 1, 2], n in (2^i for i in 9:11)
    info("benchmarking fusion_level=$fusion_level size=$n")
    for kind in [:cpu, :cudanative]
        fwd, bwd = benchmark(kind, fusion_level, n)
        push!(rows, ["$kind", "$(n)x$(n)", fusion_level, timedelta(fwd), timedelta(bwd)])
    end
end
println(Markdown.MD(Markdown.Table(rows, [:r, :c, :c, :c, :c])))
