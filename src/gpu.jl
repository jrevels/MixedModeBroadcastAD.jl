##############
# CUDA Array #
##############

using CUDAnative
using CUDAdrv

mutable struct CuArray{T,N} <: AbstractArray{T,N}
    buf::Mem.Buffer
    shape::NTuple{N,Int}

    function CuArray{T,N}(shape::NTuple{N,Int}) where {T,N}
        len = prod(shape)
        buf = Mem.alloc(len*sizeof(T))

        obj = new{T,N}(buf, shape)
        finalizer(unsafe_free!, obj)
        return obj
    end
end

CuArray{T}(shape::NTuple{N,Integer}) where {T,N} = CuArray{T,N}(shape)

function unsafe_free!(a::CuArray)
    CUDAdrv.isvalid(a.buf.ctx) && Mem.free(a.buf)
end


## array interface

Base.size(a::CuArray) = a.shape

Base.print_array(::IO, ::CuArray) = nothing


## memory copy operations

function CuArray{T,N}(src::Array{T,N}) where {T,N}
    dst = CuArray{T,N}(size(src))
    Mem.upload!(dst.buf, pointer(src), length(src) * sizeof(T))
    return dst
end
CuArray(src::Array{T,N}) where {T,N} = CuArray{T,N}(src)

function Base.Array{T,N}(src::CuArray{T,N}) where {T,N}
    dst = Array{T,N}(uninitialized, src.shape)
    Mem.download!(pointer(dst), src.buf, prod(src.shape) * sizeof(T))
    return dst
end
Array(src::CuArray{T,N}) where {T,N} = Array{T,N}(src)


## conversions

function CUDAnative.cudaconvert(a::CuArray{T,N}) where {T,N}
    ptr = Base.unsafe_convert(Ptr{T}, a.buf)
    devptr = CUDAnative.DevicePtr{T,AS.Global}(ptr)
    CuDeviceArray{T,N,AS.Global}(a.shape, devptr)
end


## broadcast

### base interface

Base.BroadcastStyle(::Type{<:CuArray}) = Broadcast.ArrayStyle{CuArray}()

function Base.broadcast_similar(f, ::Broadcast.ArrayStyle{CuArray}, ::Type{T}, inds, As...) where T
    @assert isleaftype(T)
    similar(CuArray{T}, inds)
end

@inline function Base.broadcast!(f, dest::CuArray, ::Nothing, As::Vararg{Any, N}) where N
    _broadcast!(f, dest, As...)
    return dest
end

Base.Broadcast.broadcast_indices(::Type{CuArray}, A::Ref) = ()
Base.Broadcast.broadcast_indices(::Type{CuArray}, A) = indices(A)

### internal implementation (mostly copied from Base)

using Base.Broadcast: broadcast_indices, check_broadcast_indices, map_newindexer

# This indirection allows size-dependent implementations.
@inline function _broadcast!(f, C::CuArray, A, Bs::Vararg{Any,N}) where N
    shape = broadcast_indices(C)
    @boundscheck check_broadcast_indices(shape, A, Bs...)
    keeps, Idefaults = map_newindexer(shape, A, Bs)
    blk, thr = cuda_dimensions(C)
    @cuda (blk, thr) _broadcast!(f, C, keeps, Idefaults, A, Bs, Val(N))
    return C
end

using Base.Broadcast: newindex, _broadcast_getindex
using Base.Cartesian: @nexprs, @ncall

# nargs encodes the number of As arguments (which matches the number
# of keeps). The first two type parameters are to ensure specialization.
@generated function _broadcast!(f, B::CuDeviceArray, keeps::K, Idefaults::ID,
                                A::AT, Bs::BT, ::Val{N}) where {K,ID,AT,BT,N}
    nargs = N + 1
    quote
        # destructure the keeps and As tuples
        A_1 = A
        @nexprs $N i->(A_{i+1} = Bs[i])
        @nexprs $nargs i->(keep_i = keeps[i])
        @nexprs $nargs i->(Idefault_i = Idefaults[i])
        # FIXME: @inbounds because 0.7's exceptions are incompatible with CUDAnative
        @inbounds let I = @cuda_index(B)
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

### auxiliary functionality

cuda_dimensions(a::AbstractArray) = cuda_dimensions(length(a))
function cuda_dimensions(n::Integer)
    threads = 256
    ceil(Int, n / threads), threads
end

macro cuda_index(A)
    esc(quote
        i = (blockIdx().x-UInt32(1)) * blockDim().x + threadIdx().x
        i > length($A) && return
        CartesianIndices($A)[i]
    end)
end
