using BenchmarkTools
using CUDAnative
import CUDAdrv
include("util.jl")

# make sure we collect CuArrays from previous iterations
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true

function benchmark(args...)
    kernel, inputs = getkernel(args...)
    @belapsed $(kernel)($inputs...)
end

rows = Any[["environment", "size", "fusion_level", "elapsed"]]
for fusion_level in [0, 1, 2], n in (2^i for i in 9:11)
    info("benchmarking fusion_level=$fusion_level size=$n")
    for kind in [:cpu, :cudanative, :cudaraw]
        kind == :cudaraw && fusion_level == 1 && continue
        elapsed = benchmark(kind, fusion_level, n)
        push!(rows, [kind, n, fusion_level, elapsed])
    end
end

# raw output
using DelimitedFiles
writedlm(joinpath(@__DIR__, "benchmark_no_ad.csv"), rows, ',')

# table output
for row in rows[2:end]
    row[2] = "$(row[2])x$(row[2])"
    row[4] = timedelta(row[4])
end
println(Markdown.MD(Markdown.Table(rows, [:r, :c, :c, :c])))
