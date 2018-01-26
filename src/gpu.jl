##############
# CUDA Array #
##############

using CUDAdrv, CUDAnative
using LinearAlgebra

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

CuVector{T} = CuArray{T,1}
CuMatrix{T} = CuArray{T,2}
CuVecOrMat{T} = Union{CuVector{T},CuMatrix{T}}


## array interface

Base.size(a::CuArray) = a.shape

Base.show(io::IO, a::CuArray{T,N}) where {T,N} = print(io, "$(length(a))-element CuArray{$T,$N}")

Base.print_array(::IO, ::CuArray) = nothing

Base.similar(a::CuArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} =  CuArray{T,N}(dims)


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

import Base.Broadcast
# Pssh we are flattening the broadcast here, don't tell anyone
function CUDAnative.cudaconvert(bc::Broadcast.Broadcasted{Style,ElType}) where {Style,ElType}
    # concatenate the nested arguments into {a, b, c, d}
    args = Broadcast.mapTupleLL(CUDAnative.cudaconvert, Broadcast.cat_nested(x->x.args, bc))

    # build a function `makeargs` that takes a "flat" argument list and
    # and creates the appropriate input arguments for `f`, e.g.,
    #          makeargs = (w, x, y, z) -> (w, g(x, y), z)
    #
    # `makeargs` is built recursively and looks a bit like this:
    #     makeargs(w, x, y, z) = (w, makeargs1(x, y, z)...)
    #                          = (w, g(x, y), makeargs2(z)...)
    #                          = (w, g(x, y), z)
    let makeargs = Broadcast.make_makeargs(bc)
        newf = @inline function(args::Vararg{Any,N}) where N
            bc.f(makeargs(args...)...)
        end
        return Broadcast.Broadcasted{Style,ElType}(newf, args)
    end
end

function CUDAnative.cudaconvert(bc::Broadcast.BroadcastedF{Style,ElType}) where {Style,ElType}
    # Since bc is instantiated, let's preserve the instatiation in the result
    args = Broadcast.mapTupleLL(CUDAnative.cudaconvert, Broadcast.cat_nested(x->x.args, bc))
    indexing = Broadcast.cat_nested(x->x.indexing, bc)
    let makeargs = Broadcast.make_makeargs(bc)
        newf = @inline function(args::Vararg{Any,N}) where N
            bc.f(makeargs(args...)...)
        end
        return Broadcast.Broadcasted{Style,ElType}(newf, args, axes(bc), indexing)
    end
end

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


## broadcast
### base interface

Base.BroadcastStyle(::Type{<:CuArray}) = Broadcast.ArrayStyle{CuArray}()

function Base.broadcast_similar(::Broadcast.ArrayStyle{CuArray}, ::Type{T}, inds, As...) where T
    @assert isleaftype(T) "$T is not a leaf type"
    similar(CuArray{T}, inds)
end

@inline function Base.copyto!(dest::CuArray, bc::Broadcast.Broadcasted{Nothing})
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    blk, thr = cuda_dimensions(dest)
    @cuda (blk, thr) _copyto!(dest, bc)
    return dest
end

function _copyto!(dest::CuDeviceArray, bc::Broadcast.Broadcasted{Nothing})
    let I = @cuda_index(dest)
        @inbounds dest[I] = Broadcast._broadcast_getindex(bc, I)
    end
    return
end


## high-level operations

function Base.fill!(xs::CuArray, x)
    function _fill!(xs::CuDeviceArray, x)
        I = @cuda_index xs
        @inbounds xs[I] = x
        return
    end
    blk, thr = cuda_dimensions(xs)
    @cuda (blk, thr) _fill!(xs, convert(eltype(xs), x))
    return xs
end

Base.map(f, y::CuArray, xs::CuArray...) = f.(y, xs...)

Base.map!(f, y::CuArray, xs::CuArray...) = y .= f.(xs...)
Base.map!(f, y::CuArray) =
  invoke(map!, Tuple{Any,CuArray,Vararg{CuArray}}, f, y)
Base.map!(f, y::CuArray, x::CuArray) =
  invoke(map!, Tuple{Any,CuArray,Vararg{CuArray}}, f, y, x)
Base.map!(f, y::CuArray, x1::CuArray, x2::CuArray) =
  invoke(map!, Tuple{Any,CuArray,Vararg{CuArray}}, f, y, x1, x2)

### matrix operations

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

### reductions

Base.sum(xs::CuArray) = reduce(+, 0, xs)

function Base.reduce(f, v0::T, xs::CuArray{T}) where T
  dim = cuda_reduce_dimensions(length(xs))
  scratch = similar(xs, dim[2])
  _reduce(f, v0, xs, scratch, dim)
  return unsafe_getindex(scratch, 1)
end
Base.reduce(f, v0, xs::CuArray) = reduce(f, convert(eltype(xs), v0), xs)

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
  @cuda (blocks,threads) reduce_grid(op, v0, input, output, Int32(length(input)))
  @cuda (1,1024) reduce_grid(op, v0, output, output, Int32(blocks))
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

function reduce_grid(op, v0::T, input::CuDeviceArray{T}, output::CuDeviceArray{T},
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
