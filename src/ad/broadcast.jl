###################
# BroadcastResult #
###################

#=
This is just a very poor version of the StructOfArrays-style approach we already have in
`gpu/StructOfArrays.jl` (I just ended up rolling my own while messing around with this
approach).
=#

struct BroadcastResult{T,N,V,D} <: AbstractArray{T,N}
    output_value::V
    input_derivs::D
    function BroadcastResult(output_value::V,
                             input_derivs::D) where {T,N,V<:AbstractArray{T,N},D<:Tuple}
        # TODO: a proper StructOfArrays-style version shouldn't need these asserts,
        # which are here to enforce that this code only really works for `map`-equivalent
        # `broadcast`s.
        @assert isa(IndexStyle(output_value), IndexLinear)
        @assert all(IndexStyle(output_value) === IndexStyle(input_deriv) for input_deriv in input_derivs)
        @assert all(size(output_value) === size(input_deriv) for input_deriv in input_derivs)
        return new{T,N,V,D}(output_value, input_derivs)
    end
end

@inline Base.size(result::BroadcastResult) = size(result.output_value)

@inline Base.IndexStyle(::Type{<:BroadcastResult}) where {T,N,V} = IndexLinear()

@generated function Base.setindex!(result::BroadcastResult{<:Any,<:Any,<:Any,D},
                                   dual::Dual{<:Any,<:Any,M},
                                   i::Int) where {M,D<:NTuple{M,AbstractArray}}
    body = Expr(:block)
    push!(body.args, Expr(:meta, :inline))
    push!(body.args, :(result.output_value[i] = value(dual)))
    for j in 1:M
        # Note that, in real-world code, the graph compiler would have to
        # prove that it's safe to do `=` instead of `+=` here by determining
        # that each input variable only has a single reverse-dependency. For
        # our benchmark, we can assume that the hypothetical graph compiler
        # has already determined that the broadcast output is our input
        # variables' only reverse-dependency.
        push!(body.args, :(result.input_derivs[$j][i] = partials(dual, $j)))
    end
    return body
end

#######################
# autodiff_broadcast! #
#######################

# a fused forwards/backwards pass to compute value and gradients of broadcast(kernel, input_values...)
function autodiff_broadcast!(kernel::K,
                             input_values::NTuple{N,AbstractArray},
                             input_derivs::NTuple{N,AbstractArray},
                             output_value::AbstractArray,
                             output_deriv::Union{Real,AbstractArray} = one(eltype(output_value))) where {K,N}
    dual_kernel = (xs...) -> dualcall(kernel, xs...)
    bcast_result = BroadcastResult(output_value, input_derivs)
    broadcast!(dual_kernel, bcast_result, output_deriv, input_values...)
    return nothing
end

@inline function dualcall(kernel::K, output_deriv, input_values...) where {K}
    dual_inputs = dualize(Nothing, StaticArrays.SVector(input_values))
    dual_output = kernel(dual_inputs...)
    return Dual(value(dual_output), output_deriv * partials(dual_output))
end
