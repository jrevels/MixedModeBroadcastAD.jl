##############
# CUDA Array #
##############

using CUDAdrv, CUDAnative

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
CuArray{T,N}(::Uninitialized, shape::NTuple{N,Integer}) where {T,N} = CuArray{T,N}(shape)

function unsafe_free!(a::CuArray)
      CUDAdrv.isvalid(a.buf.ctx) && Mem.free(a.buf)
end

CuVector{T} = CuArray{T,1}
CuMatrix{T} = CuArray{T,2}
CuVecOrMat{T} = Union{CuVector{T},CuMatrix{T}}


## array interface

Base.size(a::CuArray) = a.shape

Base.show(io::IO, a::CuArray{T,N}) where {T,N} = print(io, "$(length(a))-element CuArray{$T,$N}")

Base.print_array(::IO, ::CuArray) = nothing

Base.similar(a::CuArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} =  CuArray{T,N}(dims)

Base.BroadcastStyle(::Type{<:CuArray}) = Broadcast.ArrayStyle{CuArray}()

function Base.broadcast_similar(f, ::Broadcast.ArrayStyle{CuArray}, ::Type{T}, inds, As...) where T
    @assert isconcretetype(T) "$T is not a leaf type"
    similar(CuArray{T}, inds)
end


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

function unsafe_getindex(xs::CuArray{T}, i::Integer) where T
  buf = Mem.view(xs.buf, (i-1)*sizeof(T))
  return Mem.download(T, buf)[1]
end


## conversions

Base.cconvert(::Type{Ptr{T}}, x::CuArray{T}) where T = x.buf
Base.cconvert(::Type{Ptr{Cvoid}}, x::CuArray) = x.buf

function CUDAnative.cudaconvert(a::CuArray{T,N}) where {T,N}
    ptr = Base.unsafe_convert(Ptr{T}, a.buf)
    devptr = CUDAnative.DevicePtr{T,AS.Global}(ptr)
    CuDeviceArray{T,N,AS.Global}(a.shape, devptr)
end

function CUDAnative.cudaconvert(A::SubArray)
    SubArray(CUDAnative.cudaconvert(A.parent), A.indices)
end

### auxiliary functionality

cuda_dimensions(a::AbstractArray) = cuda_dimensions(length(a))
function cuda_dimensions(n::Integer)
    threads = min(n, 256)
    ceil(Int, n / threads), threads
end

macro cuda_index(A)
    esc(quote
        i = (blockIdx().x-UInt32(1)) * blockDim().x + threadIdx().x
        i > length($A) && return
        i
    end)
end
