using MixedModeBroadcastAD: forward!, backward!
using BenchmarkTools
using DelimitedFiles
using Printf
using CUDAnative
import CUDAdrv

include("../kernels.jl")

#########
# setup #
#########

# make sure we collect CuArrays from previous iterations
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true

#############
# execution #
#############

rows = Any[["environment", "precomputed layers?", "size", "forward time", "backward time"]]
for kind in (:cpu, :gpu),
    for precomputed in (false, true)
        for dims in (2^i for i in 9:11)
            tape = gettape(kind, precomputed, dims)
            fwdtime = @belapsed (forward!($tape), CUDAdrv.synchronize()) evals=1
            bwdtime = @belapsed (backward!($tape), CUDAdrv.synchronize()) evals=1
            push!(rows, [kind, precomputed, dims, fwdtime, bwdtime])
        end
    end
end

##################
# writing output #
##################

# raw output
writedlm(joinpath(@__DIR__, "timings.csv"), rows, ',')

# table output
for row in rows[2:end]
    row[2] = "$(row[2])x$(row[2])"
    row[4] = BenchmarkTools.prettytime(row[4])
    row[5] = BenchmarkTools.prettytime(row[5])
end
println(Markdown.MD(Markdown.Table(rows, [:r, :c, :c, :c, :c])))
