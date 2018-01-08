module MixedModeBroadcastAD

#=
Many parts of the reverse-mode implementation in this package
are derived from the prototypical AD package Capstan
(https://github.com/JuliaDiff/Capstan.jl).
=#

using ForwardDiff

include("utilities.jl")
include("tape.jl")
include("variables.jl")
include("primitives.jl")

end # module
