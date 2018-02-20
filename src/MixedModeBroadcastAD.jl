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

cuda_dimensions(a::AbstractArray) = cuda_dimensions(length(a))

function cuda_dimensions(n::Integer)
    threads = min(n, 256)
    ceil(Int, n / threads), threads
end

macro cuda_linear_index(A)
    esc(quote
        i = (blockIdx().x-UInt32(1)) * blockDim().x + threadIdx().x
        i > length($A) && return
        i
    end)
end

macro cuda_index(A)
    esc(quote
        i = @cuda_linear_index($A)
        @inbounds CartesianIndices($A)[i]
    end)
end

###################
# dual_broadcast! #
###################

@inline function dualcall(kernel::K, inputs...) where {K}
    dual_inputs = ForwardDiff.dualize(Nothing, StaticArrays.SVector(inputs))
    return @fastsplat(kernel(dual_inputs...))
end

# a fused forwards/backwards pass to compute value and gradients of broadcast(kernel, inputs...)
function dual_broadcast!(kernel::K,
                         output::AbstractArray,
                         inputs::NTuple{N,AbstractArray},
                         derivs::NTuple{N,AbstractArray}) where {K,N}
    @inline dual_kernel(elements...) = @fastsplat(dualcall(kernel, elements...))
    shape = Broadcast.broadcast_indices(output)
    @assert all(indices(derivs[i]) === indices(inputs[i]) for i in 1:N)
    @boundscheck Broadcast.check_broadcast_indices(shape, inputs...)
    keep_bools, default_indices = Broadcast.map_newindexer(shape, first(inputs), Base.tail(inputs))
    _dual_broadcast_kernel!(dual_kernel, output, inputs, derivs,
                            keep_bools, default_indices,
                            shape)
    return nothing
end

#=== Generic/CPU ===#

@generated function _dual_broadcast_kernel!(dual_kernel::K,
                                            output::AbstractArray{T},
                                            inputs::NTuple{N,AbstractArray},
                                            derivs::NTuple{N,AbstractArray},
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
            @inbounds output[idx] = ForwardDiff.value(dual)
            @nexprs $N i -> (deriv_i[idx_i] = ForwardDiff.partials(dual, i))
        end
        return nothing
    end
end

#=== GPU ===#

function _dual_broadcast_kernel!(dual_kernel::K,
                                 output::CuArray{T},
                                 inputs::NTuple{N,CuArray},
                                 derivs::NTuple{N,CuArray},
                                 keep_bools,
                                 default_indices,
                                 shape) where {K,T,N}
    blocks, threads = cuda_dimensions(output)
    @cuda(blocks=blocks,
          threads=threads,
          _cuda_dual_broadcast_kernel!(dual_kernel, output, inputs, derivs,
                                       keep_bools, default_indices))
    return nothing
end

@generated function _cuda_dual_broadcast_kernel!(dual_kernel::K,
                                                 output::CuDeviceArray{T},
                                                 inputs::NTuple{N,CuDeviceArray},
                                                 derivs::NTuple{N,CuDeviceArray},
                                                 keep_bools,
                                                 default_indices) where {K,T,N}
    quote
        $(Expr(:meta, :inline))
        @nexprs $N i -> (input_i = inputs[i])
        @nexprs $N i -> (deriv_i = derivs[i])
        @nexprs $N i -> (keep_bools_i = keep_bools[i])
        @nexprs $N i -> (default_indices_i = default_indices[i])
        let idx = @cuda_index(output)
            @nexprs $N i -> (idx_i = Broadcast.newindex(idx, keep_bools_i, default_indices_i))
            @nexprs $N i -> (@inbounds element_i = input_i[idx_i])
            dual::ForwardDiff.Dual{Nothing,$T,$N} = @ncall $N dual_kernel element
            @inbounds output[idx] = ForwardDiff.value(dual)
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
