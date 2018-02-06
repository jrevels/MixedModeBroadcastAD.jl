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

rows = Any[["environment", "precomputed layers?", "size", "SoA enabled?", "forward time", "backward time"]]
for kind in (:cpu, :gpu)
    for precomputed in (false, true)
        for dims in (2^i for i in 9:11)
            for soa in (false, true)
                println("benchmarking kind=:", kind, "; precomputed=", precomputed, "; dims=", dims, "; soa=", soa)
                tape = gettape(kind, precomputed, dims, soa)
                fwdtime = @belapsed (forward!($tape), CUDAdrv.synchronize())  evals=1
                bwdtime = @belapsed (backward!($tape), CUDAdrv.synchronize()) evals=1
                push!(rows, Any[kind, precomputed, dims, soa, fwdtime, bwdtime])
            end
        end
    end
end

##################
# writing output #
##################

function pretty_print_time(s)
    (unit, factor) = if s < 1e-6
        ("n", 1e9)
    elseif s < 1e-3
        ("u", 1e6)
    elseif s < 1
        ("m", 1e3)
    else
        ("", 1)
    end
    @sprintf("%.2f %ss", factor*s, unit)
end

# raw output
writedlm(joinpath(@__DIR__, "timings.csv"), rows, ',')

# table output
for row in rows[2:end]
    row[5] = pretty_print_time(row[5])
    row[6] = pretty_print_time(row[6])
end
println(Markdown.MD(Markdown.Table(rows, [:r, :c, :c, :c, :c, :c])))
