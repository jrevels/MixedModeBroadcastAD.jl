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
      Core.println("Freeing $(pointer_from_objref(a)): $(a.buf.ptr) -> $(a.buf.ptr+a.buf.bytesize)")
end

CuVector{T} = CuArray{T,1}
CuMatrix{T} = CuArray{T,2}
CuVecOrMat{T} = Union{CuVector{T},CuMatrix{T}}


## array interface

Base.size(a::CuArray) = a.shape

Base.show(io::IO, a::CuArray{T,N}) where {T,N} = print(io, "$(length(a))-element CuArray{$T,$N}")

Base.print_array(::IO, ::CuArray) = nothing

Base.similar(a::CuArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} =  CuArray{T,N}(dims)

Base.BroadcastStyle(::Type{T}) where T<:CuArray = Broadcast.ArrayStyle{T}()

function Base.broadcast_similar(f, ::Broadcast.ArrayStyle{<:CuArray}, ::Type{T}, inds, As...) where T
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
    threads = 256
    ceil(Int, n / threads), threads
end

macro cuda_index(A)
    esc(quote
        i = (blockIdx().x-UInt32(1)) * blockDim().x + threadIdx().x
        i > length($A) && return
        @inbounds CartesianIndices($A)[i]
    end)
end

### matrix operations

using LinearAlgebra

include("CUBLAS/CUBLAS.jl")

function cublas_gemm!(C::CuVecOrMat{T}, tA::Char, tB::Char,
                      A::CuVecOrMat{T},
                      B::CuVecOrMat{T},
                      alpha = one(T),
                      beta = zero(T)) where T <: CUBLAS.CublasFloat
    CUBLAS.gemm!(tA, tB, alpha, A, B, beta, C)
end

LinearAlgebra.mul!(C::CuMatrix{T}, A::CuMatrix{T}, B::CuMatrix{T}) where T<:CUBLAS.CublasFloat =
    cublas_gemm!(C, 'N', 'N', A, B)
LinearAlgebra.mul!(C::CuMatrix, A::CuMatrix, adjB::Adjoint{<:Any,<:CuMatrix}) =
    cublas_gemm!(C, 'N', 'C', A, adjB.parent)
LinearAlgebra.mul!(C::CuMatrix, adjA::Adjoint{<:Any,<:CuMatrix}, B::CuMatrix) =
    cublas_gemm!(C, 'C', 'N', adjA.parent, B)

# dimension calculation specific to the reduction algorithm below
function cuda_reduce_dimensions(n)
  threads = 512
  blocks = min((n + threads - 1) ÷ threads, 1024)
  return threads, blocks
end

function _reduce(op, v0, input, output, dim = reduce_cudim(length(input)))
  threads, blocks = dim
  if length(output) < blocks
    throw(ArgumentError("output array too small, should be at least $blocks elements"))
  end
  @cuda blocks=blocks threads=threads reduce_grid(op, v0, input, output, Int32(length(input)))
  @cuda threads=1024 reduce_grid(op, v0, output, output, Int32(blocks))
  return
end

@inline function reduce_warp(op, val::T)::T where {T}
  offset = CUDAnative.warpsize() ÷ UInt32(2)
  while offset > 0
    val = op(val, shfl_down(val, offset))
    offset ÷= UInt32(2)
  end
  return val
end

@inline function reduce_block(op, v0::T, val::T)::T where {T}
  shared = @cuStaticSharedMem(T, 32)
  wid  = div(threadIdx().x-UInt32(1), CUDAnative.warpsize()) + UInt32(1)
  lane = rem(threadIdx().x-UInt32(1), CUDAnative.warpsize()) + UInt32(1)
  val = reduce_warp(op, val)
  if lane == 1
    @inbounds shared[wid] = val
  end
  sync_threads()
  @inbounds val = (threadIdx().x <= fld(blockDim().x, CUDAnative.warpsize())) ? shared[lane] : v0
  if wid == 1
    val = reduce_warp(op, val)
  end
  return val
end
