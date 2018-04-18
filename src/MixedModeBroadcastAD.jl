module MixedModeBroadcastAD

using ForwardDiff
using DiffRules
using StaticArrays
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

# this is just `prod(length, x)`, but works on the GPU
@generated function lengthproduct(x::NTuple{N,Any}) where {N}
    if N == 0
        body = :(Int32(0))
    elseif N == 1
        body = :(length(x[1]))
    else
        body = Expr(:call, :*, [:(length(x[$i])) for i in 1:N]...)
    end
    return quote
        $(Expr(:meta, :inline))
        $body
    end
end

macro cuda_index(shape)
    return esc(quote
        i = @cuda_linear_index(lengthproduct($shape))
        @inbounds CartesianIndices($shape)[i]
    end)
end

#######################################
# broadcast!(f, ::CuArray, inputs...) #
#######################################

Broadcast.broadcast_indices(::Type{<:CuArray}, A::Ref) = ()
Broadcast.broadcast_indices(::Type{<:CuArray}, A) = axes(A)

@inline function Broadcast.broadcast!(kernel, output::CuArray, ::Nothing, inputs::Vararg{Any,N}) where {N}
    shape = Broadcast.broadcast_indices(output)
    @boundscheck Broadcast.check_broadcast_indices(shape, inputs...)
    keep_bools, default_indices = Broadcast.map_newindexer(shape, first(inputs), Base.tail(inputs))
    blocks, threads = cuda_dimensions(prod(length, shape))
    @cuda(blocks=blocks,
          threads=threads,
          _cuda_broadcast_kernel!(kernel, output, inputs, keep_bools, default_indices, shape))
    return output
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

struct Wrt{X}
    value::X
end

unwrt(x) = x
unwrt(x::Wrt) = x.value

# `I` here is a tuple of indices of the inputs that are marked for differentiation, computed
# from the input type tuple `T`. For example, if `T` specifies that the first and third
# inputs are marked `Wrt`, then `I === (1, 3)`.
struct DualKernel{K,I}
    kernel::K
    DualKernel(kernel::K, ::Type{T}) where {K,T} = new{K,wrtidx(T)}(kernel)
end

@generated function wrtidx(::Type{T}) where {T<:Tuple}
    body = Expr(:tuple)
    for i in 1:fieldcount(T)
        if T.parameters[i] <: Wrt
            push!(body.args, i)
        end
    end
    return quote
        $(Expr(:meta, :inline))
        $body
    end
end

@generated function (dk::DualKernel{K,I})(inputs...) where {K,I}
    wrts = Expr(:tuple)
    duals_and_fixed = Expr[]
    dual_i = 1
    for i in 1:fieldcount(typeof(inputs))
        if in(i, I)
            push!(wrts.args, :(inputs[$i]))
            push!(duals_and_fixed, :(duals[$dual_i]))
            dual_i += 1
        else
            push!(duals_and_fixed, :(inputs[$i]))
        end
    end
    return quote
        $(Expr(:meta, :inline))
        duals = ForwardDiff.dualize(Nothing, StaticArrays.SVector($wrts))
        return dk.kernel($(duals_and_fixed...))
    end
end

@generated function is_valid_input_and_derivs(inputs::NTuple{N,Union{AbstractArray,Wrt}},
                                              derivs::NTuple{D,AbstractArray}) where {N,D}
    predicates = Expr(:tuple)
    i_deriv = 0
    for i in 1:N
        if inputs.parameters[i] <: Wrt
            i_deriv += 1
            push!(predicates.args, :(axes(unwrt(inputs[$i])) === axes(derivs[$i_deriv])))
        end
    end
    if i_deriv == D
        return :(all($predicates))
    else
        return :(false)
    end
end

# a fused forwards pass to compute value and gradients of broadcast(kernel, inputs...)
function broadcast_gradients!(kernel::K,
                              wrtinputs::NTuple{N,Union{AbstractArray,Wrt}},
                              derivs::NTuple{D,AbstractArray}) where {K,N,D}
    @assert is_valid_input_and_derivs(wrtinputs, derivs)
    dual_kernel = DualKernel(kernel, typeof(wrtinputs))
    inputs = unwrt.(wrtinputs)
    shape = Broadcast.combine_indices(inputs...)
    @boundscheck Broadcast.check_broadcast_indices(shape, inputs...)
    keep_bools, default_indices = Broadcast.map_newindexer(shape, first(inputs), Base.tail(inputs))
    _dual_broadcast_kernel!(dual_kernel, inputs, derivs, keep_bools, default_indices, shape)
    return nothing
end

#=== Generic/CPU ===#

@generated function _dual_broadcast_kernel!(dual_kernel::DualKernel{K,I},
                                            inputs::NTuple{N,AbstractArray{T}},
                                            derivs::NTuple{D,AbstractArray{T}},
                                            keep_bools,
                                            default_indices,
                                            shape) where {K,I,T,N,D}
    deriv_loads = Expr[]
    for i in 1:D
        idx_sym = Symbol("idx_", I[i])
        deriv_sym = Symbol("deriv_", i)
        push!(deriv_loads, :(@inbounds $deriv_sym[$idx_sym] = ForwardDiff.partials(dual, $i)))
    end
    quote
        $(Expr(:meta, :inline))
        @nexprs $N i -> (input_i = inputs[i])
        @nexprs $D i -> (deriv_i = derivs[i])
        @nexprs $N i -> (keep_bools_i = keep_bools[i])
        @nexprs $N i -> (default_indices_i = default_indices[i])
        @simd for idx in CartesianIndices(shape)
            @nexprs $N i -> (idx_i = Broadcast.newindex(idx, keep_bools_i, default_indices_i))
            @nexprs $N i -> (@inbounds element_i = input_i[idx_i])
            dual::ForwardDiff.Dual{Nothing,$T,$D} = @ncall $N dual_kernel element
            $(deriv_loads...)
        end
        return nothing
    end
end

#=== GPU ===#

function _dual_broadcast_kernel!(dual_kernel::DualKernel{K,I},
                                 inputs::NTuple{N,CuArray{T}},
                                 derivs::NTuple{D,CuArray{T}},
                                 keep_bools,
                                 default_indices,
                                 shape) where {K,I,T,N,D}
    blocks, threads = cuda_dimensions(prod(length, shape))
    @cuda(blocks=blocks,
          threads=threads,
          _cuda_dual_broadcast_kernel!(dual_kernel, inputs, derivs,
                                       keep_bools, default_indices,
                                       shape))
    return nothing
end

@generated function _cuda_dual_broadcast_kernel!(dual_kernel::DualKernel{K,I},
                                                 inputs::NTuple{N,CuDeviceArray{T}},
                                                 derivs::NTuple{D,CuDeviceArray{T}},
                                                 keep_bools,
                                                 default_indices,
                                                 shape) where {K,I,T,N,D}
    deriv_loads = Expr[]
    for i in 1:D
        idx_sym = Symbol("idx_", I[i])
        deriv_sym = Symbol("deriv_", i)
        push!(deriv_loads, :(@inbounds $deriv_sym[$idx_sym] = ForwardDiff.partials(dual, $i)))
    end
    quote
        $(Expr(:meta, :inline))
        @nexprs $N i -> (input_i = inputs[i])
        @nexprs $D i -> (deriv_i = derivs[i])
        @nexprs $N i -> (keep_bools_i = keep_bools[i])
        @nexprs $N i -> (default_indices_i = default_indices[i])
        let idx = @cuda_index(shape)
            @nexprs $N i -> (idx_i = Broadcast.newindex(idx, keep_bools_i, default_indices_i))
            @nexprs $N i -> (@inbounds element_i = input_i[idx_i])
            dual::ForwardDiff.Dual{Nothing,$T,$D} = @ncall $N dual_kernel element
            $(deriv_loads...)
        end
        return nothing
    end
end

#######################################
# DiffRules for CUDAnative primitives #
#######################################

DiffRules.@define_diffrule CUDAnative.exp_fast(x) = :(CUDAnative.exp_fast($x))
DiffRules.@define_diffrule CUDAnative.tanh(x) = :(1 - CUDAnative.tanh($x)^2)

@eval $(ForwardDiff.unary_dual_definition(:CUDAnative, :exp_fast))
@eval $(ForwardDiff.unary_dual_definition(:CUDAnative, :tanh))

end # module
