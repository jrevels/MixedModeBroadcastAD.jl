#!/usr/bin/env julia

using DataFrames
using Measurements

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
    memcpy_count::Int
    memcpy_duration::T

    # API calls
    api_count::Int
    api_duration::T

    Iteration{T}() where {T} = new{T}(zero(T), Kernel{T}[], 0, zero(T), 0, zero(T))
end

function Base.show(io::IO, it::Iteration)
    println(io, "Iteration taking $(round(it.duration,2)) us:")
    println(io, " - $(length(it.kernels)) kernel launches: $(round(sum(duration, it.kernels), 2)) us")
    println(io, " - $(it.memcpy_count) memory copies: $(round(it.memcpy_duration, 2)) us")
    print(io, " - $(it.api_count) API calls: $(round(it.api_duration, 2)) us")
end

info("#26278 workaround") # yeah no kiddin' printing something makes it work. fml
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
    for field in (:memcpy_count, :api_count)
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

function read_trace(path)
    df = open(path, "r") do io
        deserialize(io)
    end

    its = group_trace(df)
    return average_trace(its)
end

function add_row!(df, it; system, TFstyle=missing, arity=missing, dims)
    push!(df, [system, TFstyle, arity, dims,
               it.duration,
               length(it.kernels), sum(registers, it.kernels), sum(duration, it.kernels),
               it.memcpy_count, it.memcpy_duration,
               it.api_count, it.api_duration])
end

df = DataFrame(system = Symbol[], TFstyle=Union{Missing,Bool}[], arity=Union{Missing,Int}[],
               dims=Int[],
               duration = AbstractFloat[],
               kernel_count = Int[], kernel_registers = Int[], kernel_duration = AbstractFloat[],
               memcpy_count = Int[], memcpy_duration = AbstractFloat[],
               api_count = Int[], api_duration = AbstractFloat[])

cd(@__DIR__) do
    for dims in [512,1024,2048]
        let
            it = read_trace("python_$(dims)_trace.jls")
            add_row!(df, it; dims=dims, system=:python)
        end

        for tfstyle in [true, false]
            it = read_trace("julia_$(tfstyle ? "tf_" : "")$(dims)_trace.jls")
            add_row!(df, it; dims=dims, system=:julia, TFstyle=tfstyle)
        end

        for arity in 1:13
            it = read_trace("julia_arity$(arity)_$(dims)_trace.jls")
            add_row!(df, it; dims=dims, system=:julia, arity=arity)
        end
    end
end

println(df)
