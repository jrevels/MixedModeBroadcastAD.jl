#!/usr/bin/env julia

using DataFrames
using Measurements

mutable struct Iteration{T<:AbstractFloat}
    total_time::T

    # kernels
    kernel_count::Int
    kernel_time::T

    # memory operations
    memcpy_count::Int
    memcpy_time::T

    # API calls
    api_count::Int
    api_time::T

    Iteration{T}() where {T} = new{T}(zero(T), 0, zero(T), 0, zero(T), 0, zero(T))
end

function Base.show(io::IO, it::Iteration)
    println(io, "Iteration taking $(round(it.total_time,2)) us:")
    println(io, " - $(it.kernel_count) kernel launches: $(round(it.kernel_time, 2)) us")
    println(io, " - $(it.memcpy_count) memory copies: $(round(it.memcpy_time, 2)) us")
    print(io, " - $(it.api_count) API calls: $(round(it.api_time, 2)) us")
end

info("#26278 workaround") # yeah no kiddin' printing something makes it work. fml
function process_data(table)
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
            it.total_time = it_stop - it_start
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
                it.api_time += row[:Duration]
            end
        elseif contains(row[:Name], "[CUDA memcpy")
            it.memcpy_count += 1
            it.memcpy_time += row[:Duration]
        else
            it.kernel_count += 1
            it.kernel_time += row[:Duration]
        end
    end

    return its
end

function average_data(its)
    # NOTE: better just chuck everything in the DataFrame,
    #       and group+average when displaying

    popfirst!(its)

    avg_it = Iteration{Measurement{Float64}}()

    # counters should be identical
    for field in (:kernel_count, :memcpy_count, :api_count)
        @assert all(it->getfield(its[1], field) == getfield(it, field), its)
        setfield!(avg_it, field, getfield(its[1], field))
    end

    # average timings
    for field in (:total_time, :kernel_time, :memcpy_time, :api_time)
        val = mean([getfield(it, field) for it in its]) ± std([getfield(it, field) for it in its])
        setfield!(avg_it, field, val)
    end

    return avg_it
end

function process(path)
    df = open(path, "r") do io
        deserialize(io)
    end

    its = process_data(df)
    return average_data(its)
end

function add_row!(df, it; system, TFstyle=missing, arity=missing, dims)
    push!(df, [system, TFstyle, arity, dims,
               it.total_time,
               it.kernel_count, it.kernel_time,
               it.memcpy_count, it.memcpy_time,
               it.api_count, it.api_time])
end

df = DataFrame(system = Symbol[], TFstyle=Union{Missing,Bool}[], arity=Union{Missing,Int}[],
               dims=Int[],
               total_time = AbstractFloat[],
               kernel_count = Int[], kernel_time = AbstractFloat[],
               memcpy_count = Int[], memcpy_time = AbstractFloat[],
               api_count = Int[], api_time = AbstractFloat[])

cd(@__DIR__) do
    for dims in [512,1024,2048]
        let
            it = process("python_$(dims).jls")
            add_row!(df, it; dims=dims, system=:python)
        end

        for tfstyle in [true, false]
            it = process("julia_$(tfstyle ? "tf_" : "")$(dims).jls")
            add_row!(df, it; dims=dims, system=:julia, TFstyle=tfstyle)
        end

        for arity in 1:10
            it = process("julia_arity$(arity)_$(dims).jls")
            add_row!(df, it; dims=dims, system=:julia, arity=arity)
        end
    end
end

println(df)
