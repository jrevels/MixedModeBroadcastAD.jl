using BenchmarkTools
using DelimitedFiles
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

println("### Benchmarking HMLSTM Kernels")
hmlstm_rows = Any[["TF-style?", "on GPU?", "size", "time"]]
for tfstyle in (false, true)
    for usegpu in (false, true)
        for dims in (2^i for i in 9:11)
            println("benchmarking tfstyle=", tfstyle, "; usegpu=", usegpu, "; dims=", dims)
            kernel!, inputs, derivs, buffers = get_hmlstm_kernel(tfstyle, usegpu, dims)
            time = @belapsed ($kernel!($inputs, $derivs, $buffers); CUDAdrv.synchronize()) evals=1
            push!(hmlstm_rows, Any[tfstyle, usegpu, dims, time])
            println("\ttime:  ", pretty_print_time(time))
        end
    end
end

println("### Benchmarking Arity-Scaling Kernels")
arity_scaling_rows = Any[["on GPU?", "size", "arity", "time"]]
for usegpu in (false, true)
    for dims in (2^i for i in 9:11)
        for arity in 1:10
            println("benchmarking usegpu=", usegpu, "; dims=", dims, "; arity=", arity)
            kernel!, inputs, derivs, buffers = get_arity_scaling_kernel(usegpu, dims, arity)
            time = @belapsed ($kernel!($inputs, $derivs, $buffers); CUDAdrv.synchronize()) evals=1
            push!(arity_scaling_rows, Any[usegpu, dims, arity, time])
            println("\ttime:  ", pretty_print_time(time))
        end
    end
end

##################
# writing output #
##################

function generate_output!(filename, rows, ncols)
    writedlm(joinpath(@__DIR__, filename), rows, ',')
    for row in rows[2:end]
        row[end] = pretty_print_time(row[end])
    end
    format = [:c for i in 1:ncols]
    format[1] = :r
    return Markdown.MD(Markdown.Table(rows, format))
end

println(generate_output!("hmlstm_timings.csv", hmlstm_rows, 4))

println(generate_output!("arity_scaling_timings.csv", arity_scaling_rows, 4))
