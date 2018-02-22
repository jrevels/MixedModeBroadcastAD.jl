module MixedModeBroadcastAD

using ForwardDiff
using DiffRules
using StaticArrays
using FastSplat
using Base.Broadcast
using Base.Cartesian
using CUDAdrv
using CUDAnative

##################
# CUDA utilities #
##################

function cuda_dimensions(n::Integer)
    threads = min(n, 256)
    return ceil(Int, n / threads), threads
end

macro cuda_linear_index(n)
    return esc(quote
        i = (blockIdx().x - UInt32(1)) * blockDim().x + threadIdx().x
        i > $n && return
        i
    end)
end

macro cuda_index(shape)
    return esc(quote
        i = @cuda_linear_index(prod(length, $shape))
        @inbounds CartesianIndices($shape)[i]
    end)
end

#######################################
# broadcast!(f, ::CuArray, inputs...) #
#######################################

Broadcast.broadcast_indices(::Type{<:CuArray}, A::Ref) = ()
Broadcast.broadcast_indices(::Type{<:CuArray}, A) = indices(A)

@inline function Broadcast.broadcast!(kernel, output::CuArray, ::Nothing, inputs...)
    shape = Broadcast.broadcast_indices(output)
    @boundscheck Broadcast.check_broadcast_indices(shape, inputs...)
    keep_bools, default_indices = Broadcast.map_newindexer(shape, first(inputs), Base.tail(inputs))
    blocks, threads = cuda_dimensions(prod(length, shape))
    @cuda(blocks=blocks,
          threads=threads,
          _cuda_broadcast_kernel!(kernel, output, inputs, keep_bools, default_indices, shape))
    return dest
end

@generated function _cuda_broadcast_kernel!(kernel::K,
                                            output::CuDeviceArray,
                                            inputs::NTuple{N,Any},
                                            keep_bools,
                                            default_indices,
                                            shape) where {K,N}
    quote
        $(Expr(:meta, :inline))
        @nexprs $N i -> (input_i = inputs[i])
        @nexprs $N i -> (keep_bools_i = keep_bools[i])
        @nexprs $N i -> (default_indices_i = default_indices[i])
        let idx = @cuda_index(shape)
            @nexprs $N i -> (idx_i = Broadcast.newindex(idx, keep_bools_i, default_indices_i))
            @nexprs $N i -> (@inbounds element_i = Broadcast._broadcast_getindex(input_i, idx_i))
            output[idx] = @ncall $N kernel element
        end
        return nothing
    end
end

########################
# broadcast_gradients! #
########################

@inline function dual_call(kernel::K, inputs...) where {K}
    dual_inputs = ForwardDiff.dualize(Nothing, StaticArrays.SVector(inputs))
    return @fastsplat(kernel(dual_inputs...))
end

# a fused forwards/backwards pass to compute value and gradients of broadcast(kernel, inputs...)
function broadcast_gradients!(kernel::K,
                              inputs::NTuple{N,AbstractArray},
                              derivs::NTuple{N,AbstractArray}) where {K,N}
    @inline dual_kernel(elements...) = @fastsplat(dual_call(kernel, elements...))
    @assert all(indices(derivs[i]) === indices(inputs[i]) for i in 1:N)
    shape = Broadcast.combine_indices(inputs...)
    @boundscheck Broadcast.check_broadcast_indices(shape, inputs...)
    keep_bools, default_indices = Broadcast.map_newindexer(shape, first(inputs), Base.tail(inputs))
    _dual_broadcast_kernel!(dual_kernel, inputs, derivs, keep_bools, default_indices, shape)
    return nothing
end

#=== Generic/CPU ===#

@generated function _dual_broadcast_kernel!(dual_kernel::K,
                                            inputs::NTuple{N,AbstractArray{T}},
                                            derivs::NTuple{N,AbstractArray{T}},
                                            keep_bools,
                                            default_indices,
                                            shape) where {K,T,N}
    quote
        $(Expr(:meta, :inline))
        @nexprs $N i -> (input_i = inputs[i])
        @nexprs $N i -> (deriv_i = derivs[i])
        @nexprs $N i -> (keep_bools_i = keep_bools[i])
        @nexprs $N i -> (default_indices_i = default_indices[i])
        @simd for idx in CartesianIndices(shape)
            @nexprs $N i -> (idx_i = Broadcast.newindex(idx, keep_bools_i, default_indices_i))
            @nexprs $N i -> (@inbounds element_i = input_i[idx_i])
            dual::ForwardDiff.Dual{Nothing,$T,$N} = @ncall $N dual_kernel element
            @nexprs $N i -> (deriv_i[idx_i] = ForwardDiff.partials(dual, i))
        end
        return nothing
    end
end

#=== GPU ===#

function _dual_broadcast_kernel!(dual_kernel::K,
                                 inputs::NTuple{N,CuArray{T}},
                                 derivs::NTuple{N,CuArray{T}},
                                 keep_bools,
                                 default_indices,
                                 shape) where {K,T,N}
    blocks, threads = cuda_dimensions(prod(length, shape))
    @cuda(blocks=blocks,
          threads=threads,
          _cuda_dual_broadcast_kernel!(dual_kernel, inputs, derivs,
                                       keep_bools, default_indices,
                                       shape))
    return nothing
end

@generated function _cuda_dual_broadcast_kernel!(dual_kernel::K,
                                                 inputs::NTuple{N,CuDeviceArray{T}},
                                                 derivs::NTuple{N,CuDeviceArray{T}},
                                                 keep_bools,
                                                 default_indices,
                                                 shape) where {K,T,N}
    quote
        $(Expr(:meta, :inline))
        @nexprs $N i -> (input_i = inputs[i])
        @nexprs $N i -> (deriv_i = derivs[i])
        @nexprs $N i -> (keep_bools_i = keep_bools[i])
        @nexprs $N i -> (default_indices_i = default_indices[i])
        let idx = @cuda_index(shape)
            @nexprs $N i -> (idx_i = Broadcast.newindex(idx, keep_bools_i, default_indices_i))
            @nexprs $N i -> (@inbounds element_i = input_i[idx_i])
            dual::ForwardDiff.Dual{Nothing,$T,$N} = @ncall $N dual_kernel element
            @nexprs $N i -> (@inbounds deriv_i[idx_i] = ForwardDiff.partials(dual, i))
        end
        return nothing
    end
end

#######################################
# DiffRules for CUDAnative primitives #
#######################################

DiffRules.@define_diffrule CUDAnative.exp(x) = :(CUDAnative.exp($x))
DiffRules.@define_diffrule CUDAnative.tanh(x) = :(1 - CUDAnative.tanh($x)^2)

@eval $(ForwardDiff.unary_dual_definition(:CUDAnative, :exp))
@eval $(ForwardDiff.unary_dual_definition(:CUDAnative, :tanh))

end # module
