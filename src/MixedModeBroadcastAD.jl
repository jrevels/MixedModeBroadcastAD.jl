module MixedModeBroadcastAD

using ForwardDiff
using DiffRules
using StaticArrays
using FastSplat

############
# GPU code #
############

include("gpu/CuArrays.jl")
include("gpu/StructOfArrays.jl")
include("gpu/GPUArrays.jl")

###########
# AD code #
###########

include("ad/utilities.jl")
include("ad/tape.jl")
include("ad/variables.jl")
include("ad/primitives.jl")

####################
# misc definitions #
####################

Broadcast.BroadcastStyle(::CuArrayStyle, s::SoAStyle) = s
Broadcast.BroadcastStyle(::CuArrayStyle, s::RecordOtherStyle) = s
Broadcast.BroadcastStyle(::CuArrayStyle, s::RecordArrayStyle) = s
Broadcast.BroadcastStyle(::SoAStyle,     s::RecordOtherStyle) = s
Broadcast.BroadcastStyle(::SoAStyle,     s::RecordArrayStyle) = s

DiffRules.@define_diffrule CUDAnative.exp(x) = :(CUDAnative.exp($x))
DiffRules.@define_diffrule CUDAnative.tanh(x) = :(1 - CUDAnative.tanh($x)^2)

@eval $(ForwardDiff.unary_dual_definition(:CUDAnative, :exp))
@eval $(ForwardDiff.unary_dual_definition(:CUDAnative, :tanh))

end # module
