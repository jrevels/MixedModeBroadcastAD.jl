#!/usr/bin/env julia

using DataFrames
using Measurements
using DataStructures

mutable struct Kernel{T<:AbstractFloat}
    name::String
    registers::Int
    runtime::T
end
runtime(kernel::Kernel) = kernel.runtime
registers(kernel::Kernel) = kernel.registers

mutable struct Iteration{T<:AbstractFloat}
    runtime::T

    # kernels
    kernels::Vector{Kernel{T}}

    # memory transfers
    transfer_size::Float64
    transfer_count::Int
    transfer_runtime::T

    # API calls
    api_count::Int
    api_runtime::T

    Iteration{T}() where {T} = new{T}(zero(T), Kernel{T}[], 0.0, 0, zero(T), 0, zero(T))
end

function Base.show(io::IO, it::Iteration)
    println(io, "Iteration taking $(round(it.runtime,2)) us:")
    println(io, " - $(length(it.kernels)) kernel launches: $(round(sum(runtime, it.kernels), 2)) us")
    println(io, " - $(it.transfer_count) memory transfers: $(round(it.transfer_size, 2)) MB in $(round(it.transfer_runtime, 2)) us")
    print(io, " - $(it.api_count) API calls: $(round(it.api_runtime, 2)) us")
end

function group_trace(table)
    its = Iteration[]
    it = nothing
    it_start = 0.
    for row in eachrow(table)
        # handle start/stop of a new iteration
        if contains(row[:Name], "[Range start]")
            @assert it == nothing
            it = Iteration{Float64}()
            it_start = row[:Start]
            continue
        end
        if contains(row[:Name], "[Marker]")
            continue
        end
        @assert it != nothing
        if contains(row[:Name], "[Range end]")
            it_stop = row[:Start]
            it.runtime = it_stop - it_start
            push!(its, it)
            it = nothing
            continue
        end

        # other entries are either API traces, memory transfers or executed kernels
        if startswith(row[:Name], "cu")
            it.api_count += 1
            # TODO: only count API calls that contribute to the total execution (ie. no
            # async calls, event queries, etc)
            if row[:Name] != "cuCtxSynchronize"
                it.api_runtime += row[:Duration]
            end
        elseif contains(row[:Name], "[CUDA memcpy")
            it.transfer_size += row[:Size]
            it.transfer_count += 1
            it.transfer_runtime += row[:Duration]
        else
            push!(it.kernels, Kernel{Float64}(row[:Name], row[:Registers_Per_Thread], row[:Duration]))
        end
    end

    return its
end

function average_trace(its)
    # NOTE: better just chuck everything in the DataFrame,
    #       and group+average when displaying

    popfirst!(its)

    avg_it = Iteration{Measurement{Float64}}()

    # kernels
    @assert all(it->length(its[1].kernels)==length(it.kernels), its)
    for i in 1:length(its[1].kernels)
        @assert all(it->its[1].kernels[i].name==it.kernels[i].name, its)
        @assert all(it->its[1].kernels[i].registers==it.kernels[i].registers, its)
        push!(avg_it.kernels,
            Kernel{Measurement{Float64}}(
                its[1].kernels[i].name,
                its[1].kernels[i].registers,
                mean([it.kernels[i].runtime for it in its]) ±
                 std([it.kernels[i].runtime for it in its]),
            )
        )
    end

    # other
    ## counters
    for field in (:transfer_size, :transfer_count, :api_count)
        @assert all(it->getfield(its[1], field) == getfield(it, field), its)
        setfield!(avg_it, field, getfield(its[1], field))
    end
    ## timings
    for field in (:runtime, :transfer_runtime, :api_runtime)
        val = mean([getfield(it, field) for it in its]) ± std([getfield(it, field) for it in its])
        setfield!(avg_it, field, val)
    end

    return avg_it
end

function read_trace(tag)
    df = open("$(tag)_trace.jls", "r") do io
        deserialize(io)
    end

    its = group_trace(df)
    return average_trace(its)
end

function read_metrics(tag)
    df = open("$(tag)_metrics.jls", "r") do io
        deserialize(io)
    end

    return df
end

function read(tag)
    return read_metrics(tag), read_trace(tag)
end

function get_metric(df, kernel, metric)
    mask = (df[:Kernel] .== kernel) .& (df[:Metric_Name] .== metric)
    return first(eachrow(df[mask, [:Invocations, :Min, :Max, :Avg]]))
end


# basic timings

function add_row!(df, trace, metrics; implementation, arity=missing, problem_size)
    push!(df, [implementation, arity, problem_size,
               trace.runtime,
               length(trace.kernels), maximum(registers.(trace.kernels)),
               sum(runtime, trace.kernels),
               mean(kernel->get_metric(metrics, kernel.name, "achieved_occupancy")[:Avg],
                    trace.kernels),
               trace.transfer_size, trace.transfer_count, trace.transfer_runtime,
               trace.api_count, trace.api_runtime])
end

function process(dir)
    df = DataFrame(implementation = Symbol[], arity=Union{Missing,Int}[],
                   problem_size=Int[],
                   runtime = AbstractFloat[],
                   kernel_count = Int[], kernel_registers = Int[],
                   kernel_runtime = AbstractFloat[],  kernel_occupancy = AbstractFloat[],
                   transfer_size = Float64[], transfer_count = Int[], transfer_runtime = AbstractFloat[],
                   api_count = Int[], api_runtime = AbstractFloat[])

    cd(dir) do
        for problem_size in [512,1024,2048]
            let
                metrics, trace = read("python_$(problem_size)")
                add_row!(df, trace, metrics; problem_size=problem_size, implementation=:python)
            end

            for tfstyle in [true, false]
                metrics, trace = read("julia_$(tfstyle ? "tf_" : "")$(problem_size)")
                add_row!(df, trace, metrics; problem_size=problem_size, implementation=(tfstyle ? :julia_tfstyle : :julia))
            end

            for arity in 1:10
                metrics, trace = read("julia_arity$(arity)_$(problem_size)")
                add_row!(df, trace, metrics; problem_size=problem_size, implementation=:julia, arity=arity)
            end
        end
    end

    function prepare_properties!(df)
        df[:runtime_err] = Measurements.uncertainty.(df[:runtime])
        df[:runtime] = Measurements.value.(df[:runtime])

        df[:kernel_runtime_err] = Measurements.uncertainty.(df[:kernel_runtime])
        df[:kernel_runtime] = Measurements.value.(df[:kernel_runtime])

        df[:transfer_runtime_err] = Measurements.uncertainty.(df[:transfer_runtime])
        df[:transfer_runtime] = Measurements.value.(df[:transfer_runtime])

        df[:api_runtime_err] = Measurements.uncertainty.(df[:api_runtime])
        df[:api_runtime] = Measurements.value.(df[:api_runtime])
    end

    let df = df[ismissing.(df[:arity]),:]
        delete!(df, :arity)
        prepare_properties!(df)

        writetable(joinpath(dirname(@__DIR__), "img", "hmlstm.csv"), df)
    end

    let df = df[.!ismissing.(df[:arity]),:]
        delete!(df, :implementation)
        prepare_properties!(df)

        writetable(joinpath(dirname(@__DIR__), "img", "arity.csv"), df)
    end

    df
end

dir = if length(ARGS) >= 1
    ARGS[1]
else
    @__DIR__
end

println(process(dir))
