module MixedModeBroadcastAD

using ForwardDiff
using ForwardDiff: Dual, value, dualize, partials
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

include("ad/broadcast.jl")

####################
# misc definitions #
####################

Broadcast.BroadcastStyle(::Broadcast.ArrayStyle{CuArray}, s::Broadcast.ArrayStyle{<:StructOfArrays}) = s
Broadcast.BroadcastStyle(::Broadcast.ArrayStyle{CuArray}, s::RecordOtherStyle) = s
Broadcast.BroadcastStyle(::Broadcast.ArrayStyle{CuArray}, s::RecordArrayStyle) = s
Broadcast.BroadcastStyle(::Broadcast.ArrayStyle{<:StructOfArrays}, s::RecordOtherStyle) = s
Broadcast.BroadcastStyle(::Broadcast.ArrayStyle{<:StructOfArrays}, s::RecordArrayStyle) = s

DiffRules.@define_diffrule CUDAnative.exp(x) = :(CUDAnative.exp($x))
DiffRules.@define_diffrule CUDAnative.tanh(x) = :(1 - CUDAnative.tanh($x)^2)

@eval $(ForwardDiff.unary_dual_definition(:CUDAnative, :exp))
@eval $(ForwardDiff.unary_dual_definition(:CUDAnative, :tanh))

end # module
