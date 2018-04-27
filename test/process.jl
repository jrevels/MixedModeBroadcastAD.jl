#!/usr/bin/env julia

using DataFrames
using Measurements
using DataStructures

mutable struct Kernel{T<:AbstractFloat}
    name::String
    registers::Int
    duration::T
end
duration(kernel::Kernel) = kernel.duration
registers(kernel::Kernel) = kernel.registers

mutable struct Iteration{T<:AbstractFloat}
    duration::T

    # kernels
    kernels::Vector{Kernel{T}}

    # memory operations
    memcpy_size::Float64
    memcpy_count::Int
    memcpy_duration::T

    # API calls
    api_count::Int
    api_duration::T

    Iteration{T}() where {T} = new{T}(zero(T), Kernel{T}[], 0.0, 0, zero(T), 0, zero(T))
end

function Base.show(io::IO, it::Iteration)
    println(io, "Iteration taking $(round(it.duration,2)) us:")
    println(io, " - $(length(it.kernels)) kernel launches: $(round(sum(duration, it.kernels), 2)) us")
    println(io, " - $(it.memcpy_count) memory copies: $(round(it.memcpy_size, 2)) MB in $(round(it.memcpy_duration, 2)) us")
    print(io, " - $(it.api_count) API calls: $(round(it.api_duration, 2)) us")
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
        @assert it != nothing
        if contains(row[:Name], "[Range end]")
            it_stop = row[:Start]
            it.duration = it_stop - it_start
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
                it.api_duration += row[:Duration]
            end
        elseif contains(row[:Name], "[CUDA memcpy")
            it.memcpy_size += row[:Size]
            it.memcpy_count += 1
            it.memcpy_duration += row[:Duration]
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
                mean([it.kernels[i].duration for it in its]) ±
                 std([it.kernels[i].duration for it in its]),
            )
        )
    end

    # other
    ## counters
    for field in (:memcpy_size, :memcpy_count, :api_count)
        @assert all(it->getfield(its[1], field) == getfield(it, field), its)
        setfield!(avg_it, field, getfield(its[1], field))
    end
    ## timings
    for field in (:duration, :memcpy_duration, :api_duration)
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

function add_row!(df, trace, metrics; system, TFstyle=missing, arity=missing, dims)
    push!(df, [system, TFstyle, arity, dims,
               trace.duration,
               length(trace.kernels), maximum(registers.(trace.kernels)),
               sum(duration, trace.kernels),
               mean(kernel->get_metric(metrics, kernel.name, "achieved_occupancy")[:Avg],
                    trace.kernels),
               trace.memcpy_size, trace.memcpy_count, trace.memcpy_duration,
               trace.api_count, trace.api_duration])
end

function process(dir)
    df = DataFrame(system = Symbol[], TFstyle=Union{Missing,Bool}[], arity=Union{Missing,Int}[],
                   dims=Int[],
                   duration = AbstractFloat[],
                   kernel_count = Int[], kernel_registers = Int[],
                   kernel_duration = AbstractFloat[],  kernel_occupancy = AbstractFloat[],
                   memcpy_size = Float64[], memcpy_count = Int[], memcpy_duration = AbstractFloat[],
                   api_count = Int[], api_duration = AbstractFloat[])

    cd(dir) do
        for dims in [512,1024,2048]
            let
                metrics, trace = read("python_$(dims)")
                add_row!(df, trace, metrics; dims=dims, system=:python)
            end

            for tfstyle in [true, false]
                metrics, trace = read("julia_$(tfstyle ? "tf_" : "")$(dims)")
                add_row!(df, trace, metrics; dims=dims, system=:julia, TFstyle=tfstyle)
            end

            for arity in 1:10
                metrics, trace = read("julia_arity$(arity)_$(dims)")
                add_row!(df, trace, metrics; dims=dims, system=:julia, arity=arity)
            end
        end
    end

    let df = df
        df = df[.!ismissing.(df[:arity]),:]
        df = df[[:arity, :dims, :kernel_registers, :kernel_duration, :kernel_occupancy]]

        df[:duration_val] = Measurements.value.(df[:kernel_duration])
        df[:duration_err] = Measurements.uncertainty.(df[:kernel_duration])
        delete!(df, :kernel_duration)

        rename!(df, :kernel_registers, :registers)

        rename!(df, :kernel_occupancy, :occupancy)

        writetable(joinpath(dirname(@__DIR__), "img", "arity.csv"), df)
    end

    let df = df
        df = df[ismissing.(df[:arity]),:]
        df = df[[:system, :TFstyle, :dims, :kernel_duration, :memcpy_duration]]

        df[(df[:system] .== :julia) .& (df[:TFstyle] .== true), :system] = :julia_tfstyle
        delete!(df, :TFstyle)

        df[:compute_val] = Measurements.value.(df[:kernel_duration])
        df[:compute_err] = Measurements.uncertainty.(df[:kernel_duration])
        delete!(df, :kernel_duration)

        df[:memory_val] = Measurements.value.(df[:memcpy_duration])
        df[:memory_err] = Measurements.uncertainty.(df[:memcpy_duration])
        delete!(df, :memcpy_duration)

        writetable(joinpath(dirname(@__DIR__), "img", "compute.csv"), df)
    end

    df
end

dir = if length(ARGS) >= 1
    ARGS[1]
else
    @__DIR__
end

println(process(dir))
