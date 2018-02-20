##################
# BroadcastDuals #
##################

struct BroadcastDuals{V<:AbstractArray,D<:Tuple,T,N,M} <: AbstractArray{Dual{Nothing,T,N},M}
    value::V
    derivs::D
    function BroadcastDuals(value::V,
                            derivs::D) where {T,N,M,
                                              V<:AbstractArray{T,M},
                                              D<:NTuple{N,AbstractArray{T}}}
        return new{V,D,T,N,M}(value, derivs)
    end
end

@inline Base.size(x::BroadcastDuals) = size(x.value)

@inline Base.IndexStyle(::Type{<:BroadcastDuals}) = IndexCartesian()

@inline function bound(x::AbstractArray, ::Val{N}) where {N}
    if @generated
        shape = Expr(:tuple, (:(size(x, $i)) for i in 1:N)...)
        return quote
            $(Expr(:meta, :inline))
            CartesianIndex{N}($shape)
        end
    else
        return CartesianIndex{N}(NTuple{N}(size(x, i) for i in 1:N))
    end
end

@generated function Base.setindex!(x::BroadcastDuals{O,I,T,N,M},
                                   dual::Dual{Nothing,T,N},
                                   i::CartesianIndex) where {O,I,T,N,M}
    body = Expr(:block)
    push!(body.args, Expr(:meta, :inline))
    push!(body.args, :(x.value[i] = value(dual)))
    for j in 1:N
        push!(body.args, quote
            deriv = x.derivs[$j]
            deriv[min(bound(deriv, Val(M)), i)] = partials(dual, $j)
        end)
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
    bcast_result = BroadcastDuals(output_value, input_derivs)
    broadcast!(dual_kernel, bcast_result, output_deriv, input_values...)
    return nothing
end

@inline function dualcall(kernel::K, output_deriv, input_values...) where {K}
    dual_inputs = dualize(Nothing, StaticArrays.SVector(input_values))
    dual_output = kernel(dual_inputs...)
    return Dual(value(dual_output), output_deriv * partials(dual_output))
end
