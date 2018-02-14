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

const RUNALL = length(ARGS) >= 1 ? parse(Bool, ARGS[1]) : false

# make sure we collect CuArrays from previous iterations
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true

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

#############
# execution #
#############

kind_opts = RUNALL ? (:cpu, :gpu) : (:gpu,)
soa_opts = RUNALL ? (false, true) : (true,)
rows = Any[["environment", "SoA enabled?", "size", "forward time", "backward time"]]
for kind in kind_opts
    for soa in soa_opts
        for dims in (2^i for i in 9:11)
            println("benchmarking kind=:", kind, "; soa=", soa, "; dims=", dims)
            tape = gettape(kind, soa, dims)
            fwdtime = @belapsed (forward!($tape), CUDAdrv.synchronize())  evals=1
            bwdtime = @belapsed (backward!($tape), CUDAdrv.synchronize()) evals=1
            push!(rows, Any[kind, soa, dims, fwdtime, bwdtime])
            println("\tforward time:  ", pretty_print_time(fwdtime))
            println("\tbackward time: ", pretty_print_time(bwdtime))
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
    row[4] = pretty_print_time(row[4])
    row[5] = pretty_print_time(row[5])
end
println(Markdown.MD(Markdown.Table(rows, [:r, :c, :c, :c, :c])))
