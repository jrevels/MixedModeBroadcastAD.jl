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
function process_data(path)
    table = open(path, "r") do io
        deserialize(io)
    end

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

@assert length(ARGS) >= 1
const INPUT = ARGS[1]

its = process_data(INPUT)
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

println(avg_it)
