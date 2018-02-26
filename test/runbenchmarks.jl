using BenchmarkTools
using DelimitedFiles
using Printf
using CUDAnative
import CUDAdrv

include("kernels.jl")

#########
# setup #
#########

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

rows = Any[["environment", "size", "TF-style?", "time"]]
for kind in (:cpu, :gpu)
    for dims in (2^i for i in 9:11)
        for tfstyle in (false, true)
            println("benchmarking kind=:", kind, "; dims=", dims, "; tfstyle=", tfstyle)
            kernel!, inputs, derivs, buffers = get_kernel(kind, dims, tfstyle)
            time = @belapsed ($kernel!($inputs, $derivs, $buffers); CUDAdrv.synchronize()) evals=1
            push!(rows, Any[kind, dims, tfstyle, time])
            println("\ttime:  ", pretty_print_time(time))
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
    row[end] = pretty_print_time(row[end])
end
println(Markdown.MD(Markdown.Table(rows, [:r, :c, :c, :c])))
