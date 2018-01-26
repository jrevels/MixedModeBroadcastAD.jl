module MixedModeBroadcastAD

#=
Many parts of the reverse-mode implementation in this package
are derived from the prototypical AD package Capstan
(https://github.com/JuliaDiff/Capstan.jl).
=#

using ForwardDiff
using DiffResults
using StaticArrays
using FastSplat

include("utilities.jl")
include("tape.jl")
include("variables.jl")
include("primitives.jl")
include("gpu.jl")
include("unfused.jl")

function record(f, input...)
    tape = Tape()
    recorded_input = map(x -> Record(tape, Variable(x)), input)
    recorded_output = f(recorded_input...)
    return tape, recorded_output, recorded_input
end

function autograd(f, input...)
    tape, recorded_output, recorded_input = record(f, input...)
    forward!(tape)
    seed!(recorded_output)
    backward!(tape)
    return (value(recorded_output), map(deriv, recorded_input))
end

end # module
