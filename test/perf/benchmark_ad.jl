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

rows = Any[["environment", "size", "fusion_level", "soa", "forwards", "backwards"]]
for fusion_level in [0, 1, 2], n in (2^i for i in 9:11), soa in [true, false]
    info("benchmarking fusion_level=$fusion_level size=$n, soa=$soa")
    for kind in [:cpu, :cudanative]
        fwd, bwd = benchmark(kind, fusion_level, n, soa)
        push!(rows, [kind, n, fusion_level, soa, fwd, bwd])
    end
end

# raw output
using DelimitedFiles
writedlm(joinpath(@__DIR__, "benchmark_ad.csv"), rows, ',')

# table output
for row in rows[2:end]
    row[2] = "$(row[2])x$(row[2])"
    row[5] = timedelta(row[5])
    row[6] = timedelta(row[6])
end
println(Markdown.MD(Markdown.Table(rows, [:r, :c, :c, :c, :c, :c])))
