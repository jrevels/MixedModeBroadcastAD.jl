module MixedModeBroadcastAD

#=
Many parts of the reverse-mode implementation in this package
are derived from the prototypical AD package Capstan
(https://github.com/JuliaDiff/Capstan.jl).
=#

using ForwardDiff
using DiffRules
using StaticArrays
using FastSplat

# GPU code
include("cuarray.jl")
include("soa.jl")
include("gpuarray.jl")

# AD code
include("utilities.jl")
include("tape.jl")
include("variables.jl")
include("primitives.jl")

DiffRules.@define_diffrule CUDAnative.exp(x) = :(CUDAnative.exp($x))
DiffRules.@define_diffrule CUDAnative.tanh(x) = :(1 - CUDAnative.tanh($x)^2)

@eval $(ForwardDiff.unary_dual_definition(:CUDAnative, :exp))
@eval $(ForwardDiff.unary_dual_definition(:CUDAnative, :tanh))

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
    return (value(recorded_output), deriv.(recorded_input))
end

end # module
