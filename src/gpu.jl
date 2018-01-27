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

function unsafe_free!(a::CuArray)
    if !isnull(a.buf)
        CUDAdrv.isvalid(a.buf.ctx) && Mem.free(a.buf)
        a.buf = Mem.Buffer(C_NULL, 0, CuContext(C_NULL))
    end
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

function CUDAnative.cudaconvert(A::SubArray)
    SubArray(CUDAnative.cudaconvert(A.parent), A.indices)
end

## broadcast

### base interface

Base.BroadcastStyle(::Type{<:CuArray}) = Broadcast.ArrayStyle{CuArray}()

function Base.broadcast_similar(f, ::Broadcast.ArrayStyle{CuArray}, ::Type{T}, inds, As...) where T
    @assert isleaftype(T) "$T is not a leaf type"
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

@enum(CUfunc_cache, CU_FUNC_CACHE_PREFER_NONE   = 0x00,
                    CU_FUNC_CACHE_PREFER_SHARED = 0x01,
                    CU_FUNC_CACHE_PREFER_L1     = 0x02,
                    CU_FUNC_CACHE_PREFER_EQUAL  = 0x03)

function setcacheconfig(config::CUfunc_cache)
    CUDAdrv.@apicall(:cuCtxSetCacheConfig, (CUfunc_cache,), config)
end

### 
@noinline function dual_eval_broadcast!(output_value::CuMatrix, input_derivs::CuArray{<:Any, 3},
                                        kernel, input_values::NTuple{N,<:CuMatrix}) where N
    @assert all(size(iv) === size(output_value) for iv in input_values)
    blk, thr = cuda_dimensions(output_value)
    @cuda (blk, thr) _dual_eval_broadcast!(output_value, input_derivs, kernel, input_values, Val(N))
end

@generated function _dual_eval_broadcast!(output_value::CuDeviceArray, input_derivs, kernel, input_values, ::Val{N}) where N
    quote
        let I = @cuda_index(output_value)
            @nexprs $N i->(iv_{i} = input_values[i][I])
            ivs = @ncall $N SVector iv 
            ij_result = dual_eval_kernel(kernel, ivs)
            @inbounds output_value[i] = ForwardDiff.value(ij_result)
            iderivs = view(input_derivs, I, :)
            for k in 1:$N
                @inbounds iderivs[k] = ForwardDiff.partials(ij_result, k)
            end
        end
        return
    end
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

Base.LinAlg.mul!(C::CuMatrix{T}, A::CuMatrix{T}, B::CuMatrix{T}) where T<:CUBLAS.CublasFloat =
    cublas_gemm!(C, 'N', 'N', A, B)
Base.LinAlg.mul!(C::CuMatrix, A::CuMatrix, adjB::Adjoint{<:Any,<:CuMatrix}) =
    cublas_gemm!(C, 'N', 'C', A, adjB.parent)
Base.LinAlg.mul!(C::CuMatrix, adjA::Adjoint{<:Any,<:CuMatrix}, B::CuMatrix) =
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
