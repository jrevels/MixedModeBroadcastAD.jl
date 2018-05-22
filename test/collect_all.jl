#!/usr/bin/env julia

using CUDAdrv

const gpus = Dict{String,Int}()
for (i,device) in enumerate(devices())
    id = name(device)
    get!(gpus, id, i-1)
end

const ITERATIONS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128

cd(@__DIR__) do
    for (gpu,i) in gpus
        @info "Collecting data for $gpu (device $i)"
        withenv("CUDA_VISIBLE_DEVICES" => i) do
            run(
                ignorestatus(
                    `julia --depwarn=no --compiled-modules=no collect.jl $ITERATIONS $gpu`
                )
            )
        end
    end
end
