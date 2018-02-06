## broadcast
const GPUSoA = StructOfArrays{T,N,CuArray{T,N},U} where {T,N,U}
const GPUDeviceSoA = StructOfArrays{T,N,CuDeviceArray{T,N,AS},U} where {T,N,AS,U}
const GPUArrays = Union{<:CuArray{T,N}, <:GPUSoA{T,N}} where {T,N}
const GPUDeviceArrays = Union{<:CuDeviceArray{T,N,AS}, <:GPUDeviceSoA{T,N,AS}} where {T,N,AS}

function CUDAnative.cudaconvert(A::GPUSoA{T, N}) where {T, N}
    arrays = map(CUDAnative.cudaconvert, A.arrays)
    tt = typeof(arrays)
    StructOfArrays{T, N, CuDeviceArray{T,N,AS.Global}, tt}(arrays)
end

### base interface

@inline function Base.broadcast!(f, dest::GPUArrays, ::Nothing, As::Vararg{Any, N}) where N
    _broadcast!(f, dest, As...)
    return dest
end

Base.Broadcast.broadcast_indices(::Type{<:GPUArrays}, A::Ref) = ()
Base.Broadcast.broadcast_indices(::Type{<:GPUArrays}, A) = indices(A)

@generated function unsafe_getindex(A::GPUSoA{T}, i::Integer...) where {T}
    exprs = Any[Expr(:meta, :inline), Expr(:meta, :propagate_inbounds)]
    if isempty(T.types)
        push!(exprs, :(return unsafe_getindex(A.arrays[1], i...)))
    else
        strct, _ = generate_getindex(T, unsafe_getindex, 1)
        push!(exprs, strct)
    end
    quote
        $(exprs...)
    end
end

### internal implementation (mostly copied from Base)

using Base.Broadcast: broadcast_indices, check_broadcast_indices, map_newindexer

# This indirection allows size-dependent implementations.
@inline function _broadcast!(f, C::GPUArrays, A, Bs::Vararg{Any,N}) where N
    shape = broadcast_indices(C)
    @boundscheck check_broadcast_indices(shape, A, Bs...)
    keeps, Idefaults = map_newindexer(shape, A, Bs)
    blk, thr = cuda_dimensions(C)
    @cuda blocks=blk threads=thr _broadcast!(f, C, keeps, Idefaults, A, Bs, Val(N))
    return C
end

using Base.Broadcast: newindex, _broadcast_getindex
using Base.Cartesian: @nexprs, @ncall

# nargs encodes the number of As arguments (which matches the number
# of keeps). The first two type parameters are to ensure specialization.
@generated function _broadcast!(f, B::GPUDeviceArrays, keeps::K, Idefaults::ID,
                                A::AT, Bs::BT, ::Val{N}) where {K,ID,AT,BT,N}
    nargs = N + 1
    quote
        # destructure the keeps and As tuples
        A_1 = A
        @nexprs $N i->(A_{i+1} = Bs[i])
        @nexprs $nargs i->(keep_i = keeps[i])
        @nexprs $nargs i->(Idefault_i = Idefaults[i])
        let I = @cuda_index(B)
            # reverse-broadcast the indices
            @nexprs $nargs i->(I_i = newindex(I, keep_i, Idefault_i))
            # extract array values
            @nexprs $nargs i->(@inbounds val_i = _broadcast_getindex(A_i, I_i))
            # call the function and store the result
            result = @ncall $nargs f val
            @inbounds B[I] = result
        end
        return
    end
end


## high-level operations

function Base.fill!(xs::GPUArrays, x)
    function _fill!(xs::GPUDeviceArrays, x)
        I = @cuda_index xs
        @inbounds xs[I] = x
        return
    end
    blk, thr = cuda_dimensions(xs)
    @cuda blocks=blk threads=thr _fill!(xs, convert(eltype(xs), x))
    return xs
end

Base.map(f, y::GPUArrays, xs::GPUArrays...) = f.(y, xs...)

Base.map!(f, y::GPUArrays, xs::GPUArrays...) = y .= f.(xs...)
Base.map!(f, y::GPUArrays) =
  invoke(map!, Tuple{Any,GPUArrays,Vararg{GPUArrays}}, f, y)
Base.map!(f, y::GPUArrays, x::GPUArrays) =
  invoke(map!, Tuple{Any,GPUArrays,Vararg{GPUArrays}}, f, y, x)
Base.map!(f, y::GPUArrays, x1::GPUArrays, x2::GPUArrays) =
  invoke(map!, Tuple{Any,CuArray,Vararg{CuArray}}, f, y, x1, x2)

### reductions
Base.sum(xs::GPUArrays) = reduce(+, 0, xs)

function Base.reduce(f, v0::T, xs::GPUArrays{T}) where T
  dim = cuda_reduce_dimensions(length(xs))
  scratch = similar(xs, dim[2])
  _reduce(f, v0, xs, scratch, dim)
  return unsafe_getindex(scratch, 1)
end
Base.reduce(f, v0, xs::GPUArrays) = reduce(f, convert(eltype(xs), v0), xs)

function reduce_grid(op, v0::T, input::GPUDeviceArrays{T}, output::GPUDeviceArrays{T},
                     len::Integer) where {T}
    val = v0
    i = (blockIdx().x-UInt32(1)) * blockDim().x + threadIdx().x
    step = blockDim().x * gridDim().x
    while i <= len
        @inbounds val = op(val, input[i])
        i += step
    end
    val = reduce_block(op, v0, val)
    if threadIdx().x == UInt32(1)
        @inbounds output[blockIdx().x] = val
    end
    return
end
