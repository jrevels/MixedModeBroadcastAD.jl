# Based upon https://github.com/simonster/StructsOfArrays.jl v0.3
# MIT License

###
# Changes:
# - Accepts primitive types (e.g. types with no fields).
#   This allows broadcast to produce SoA arrays starting
#   from something like `SoA{Float32, 2}`.
# - Supports multiple storage array types. Most notably
#   `CuArrays` and `CuDeviceArray`.
###

struct StructOfArrays{T,N,A<:AbstractArray{T,N},U<:Tuple} <: AbstractArray{T,N}
    arrays::U
end

_types(T) = isbits(T) && isempty(T.types) ? Core.svec(T) : T.types

# Storage types of StructOfArrays need to implement this
type_with_eltype(::Type{<:Array}, T, N) = Array{T, N}
type_with_eltype(::Type{<:CuArray}, T, N) = CuArray{T, N}

@generated function StructOfArrays(::Type{T}, ::Type{ArrayT}, dims::Integer...) where {T, ArrayT<:AbstractArray} 
    if !isconcretetype(T) || T.mutable
        return :(throw(ArgumentError("can only create an StructOfArrays of concrete types")))
    end
    if !isbits(T) && isempty(T.types)
        return :(throw(ArgumentError("cannot create an StructOfArrays of an empty type")))
    end
    N = length(dims) 
    pArrayT = type_with_eltype(ArrayT, T, N)
    typvec = _types(T)
    arrtuple = Tuple{map(t->type_with_eltype(ArrayT, t, N), typvec)...}
    :(StructOfArrays{T,$N,$(pArrayT),$arrtuple}(($([:($(pArrayT)(uninitialized,dims)) for i = 1:length(typvec)]...),)))
end
StructOfArrays(T::Type, AT::Type, dims::Tuple{Vararg{Integer}}) = StructOfArrays(T, AT, dims...)

Base.IndexStyle(::Type{<:StructOfArrays{T,N,A}}) where {T,N,A<:AbstractArray} =  Base.IndexStyle(A)
Base.BroadcastStyle(::Type{<:StructOfArrays{T,N,A}}) where {T,N,A<:AbstractArray} = Broadcast.ArrayStyle{StructOfArrays{T, N, A}}()

function Base.similar(A::StructOfArrays{T1,N,AT}, ::Type{T}, dims::Dims) where {T1,N,AT,T}
    StructOfArrays(T, AT, dims)
end

function Base.broadcast_similar(f, ::Broadcast.ArrayStyle{StructOfArrays{T1, N, A}}, ::Type{T}, inds, As...) where {T1, N, A, T}
    StructOfArrays(T, A, Base.to_shape(inds))
end

@inline function Base.broadcast!(f, dest::StructOfArrays{T, N1, <:CuArray, U}, ::Nothing,  As::Vararg{Any, N2}) where {T, N1, U, N2}
    gpu_broadcast!(f, dest, As...)
    return dest
end

Base.convert(::Type{StructOfArrays{T,N}}, A::AbstractArray) where {T,N} =
    copyto!(StructOfArrays(T,typeof(A),size(A)), A)
Base.convert(::Type{StructOfArrays{T}}, A::AbstractArray{S,N}) where {T,S,N} =
    convert(StructOfArrays{T,N}, A)
Base.convert(::Type{StructOfArrays}, A::AbstractArray{T,N}) where {T,N} =
    convert(StructOfArrays{T,N}, A)

function CUDAnative.cudaconvert(A::StructOfArrays{T, N, AT, U}) where {T, AT<:CuArray, N, U}
    tt = Tuple{map(CUDAnative.cudaconvert, U.parameters)...}
    nAT = CUDAnative.cudaconvert(AT)
    arrays = map(CUDAnative.cudaconvert, A.arrays)
    StructOfArrays{T, N, nAT, tt}(arrays)
end

_gpu(::Type{<:AbstractArray{T, N}}) where {T, N} = CuArray{T, N}
function gpu(A::StructOfArrays{T, N, <:AbstractArray{T, N}, U}) where {T, N, U}
    tt = Tuple{map(_gpu, U.parameters)...}
    arrays = map(CuArray, A.arrays)
    StructOfArrays{T, N, CuArray{T,N}, tt}(arrays)
end

_cpu(::Type{<:CuArray{T, N}}) where {T, N} = Array{T, N}
function cpu(A::StructOfArrays{T, N, <:CuArray{T, N}, U}) where {T, N, U}
    tt = Tuple{map(cpu, U.parameters)...}
    arrays = map(Array, A.arrays)
    StructOfArrays{T, N, Array{T,N}, tt}(arrays)
end

Base.size(A::StructOfArrays) = size(A.arrays[1])
Base.size(A::StructOfArrays, d) = size(A.arrays[1], d)

@generated function Base.getindex(A::StructOfArrays{T}, i::Integer...) where {T}
    typvec = _types(T)
    exprs = Any[Expr(:meta, :inline), Expr(:meta, :propagate_inbounds)]
    if length(typvec) == 1
        push!(exprs, :(return A.arrays[1][i...]))
    else
        push!(exprs, Expr(:new, T, [:(A.arrays[$j][i...]) for j in 1:length(typvec)]...))
    end
    quote
        $(exprs...)
    end
end
@generated function Base.setindex!(A::StructOfArrays{T}, x, i::Integer...) where {T}
    typvec = _types(T)
    exprs = Any[Expr(:meta, :inline), Expr(:meta, :propagate_inbounds)]
    push!(exprs, :(v = convert(T, x)))
    if length(typvec) == 1
        push!(exprs, :(A.arrays[1][i...] = v))
    else
        for j = 1:length(typvec)
            push!(exprs, :(A.arrays[$j][i...] = getfield(v, $j)))
        end
    end
    push!(exprs, :(return x))
    quote
        $(exprs...)
    end
end

function _fill!(xs::StructOfArrays{T,N, <:CuDeviceArray{T,N}}, x) where {T,N}
    I = @cuda_index xs
    @inbounds xs[I] = x
    return
end
function Base.fill!(xs::StructOfArrays{T, N, <:CuArray{T, N}}, x) where {T, N}
    blk, thr = cuda_dimensions(xs)
    @cuda blocks=blk threads=thr _fill!(xs, convert(eltype(xs), x))
    return xs
end
