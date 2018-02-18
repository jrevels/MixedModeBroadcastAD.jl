using MixedModeBroadcastAD: autodiff_broadcast!
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

rows = Any[["environment", "size", "time"]]
for kind in (:cpu, :gpu)
    for dims in (2^i for i in 9:11)
        println("benchmarking kind=:", kind, "; dims=", dims)
        kernel, input_values, input_derivs, output_value = getkernel(kind, dims)
        time = @belapsed (autodiff_broadcast!($kernel, $input_values, $input_derivs, $output_value), CUDAdrv.synchronize()) evals=1
        push!(rows, Any[kind, dims, time])
        println("\ttime:  ", pretty_print_time(time))
    end
end

##################
# writing output #
##################

# raw output
writedlm(joinpath(@__DIR__, "timings.csv"), rows, ',')

# table output
for row in rows[2:end]
    row[3] = pretty_print_time(row[3])
end
println(Markdown.MD(Markdown.Table(rows, [:r, :c, :c])))
