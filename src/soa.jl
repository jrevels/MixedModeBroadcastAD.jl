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
    # TODO: Verify U
end

_types(T) = isbits(T) && isempty(T.types) ? Core.svec(T) : T.types

# Storage types of StructOfArrays need to implement this
_type_with_eltype(::Type{<:Array}, T, N) = Array{T, N}
_type_with_eltype(::Type{<:CuArray}, T, N) = CuArray{T, N}
_type_with_eltype(::Type{CuDeviceArray{_T,_N,AS}}, T, N) where{_T,_N,AS} = CuDeviceArray(T,N,AS)
_type(::Type{<:Array}) = Array
_type(::Type{<:CuArray}) = CuArray
_type(::Type{<:CuDeviceArray}) = CuDeviceArray

@generated function StructOfArrays(::Type{T}, ::Type{ArrayT}, dims::Integer...) where {T, ArrayT<:AbstractArray} 
    if !isconcretetype(T) || T.mutable
        return :(throw(ArgumentError("can only create an StructOfArrays of concrete types")))
    end
    if !isbits(T) && isempty(T.types)
        return :(throw(ArgumentError("cannot create an StructOfArrays of an empty type")))
    end
    N         = length(dims)
    pArrayT   = _type_with_eltype(ArrayT, T, N)
    typvec    = _types(T)
    arrtypvec = map(t->_type_with_eltype(ArrayT, t, N), typvec)
    arrtuple  = Tuple{arrtypvec...}
    :(StructOfArrays{T,$N,$(pArrayT),$arrtuple}(($([:($(arrtypvec[i])(uninitialized,dims)) for i = 1:length(typvec)]...),)))
end
StructOfArrays(T::Type, AT::Type, dims::Tuple{Vararg{Integer}}) = StructOfArrays(T, AT, dims...)

Base.size(A::StructOfArrays) = size(@inbounds(A.arrays[1]))

Base.show(io::IO, a::StructOfArrays{T,N,A}) where {T,N,A} = print(io, "$(length(a))-element SoA{$T,$N,$A}")

Base.print_array(::IO, ::StructOfArrays) = nothing

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

Base.IndexStyle(::Type{<:StructOfArrays{T,N,A}}) where {T,N,A<:AbstractArray} =  Base.IndexStyle(A)
Base.BroadcastStyle(::Type{<:StructOfArrays{T,N,A}}) where {T,N,A<:AbstractArray} = Broadcast.ArrayStyle{StructOfArrays{T,N,A}}()

function Base.similar(A::StructOfArrays{T1,N,AT}, ::Type{T}, dims::Dims) where {T1,N,AT,T}
    StructOfArrays(T, AT, dims)
end

function Base.broadcast_similar(f, ::Broadcast.ArrayStyle{StructOfArrays{T1, N, A}}, ::Type{T}, inds, As...) where {T1,N,A,T}
    StructOfArrays(T, A, Base.to_shape(inds))
end

function Base.convert(::Type{<:StructOfArrays{T,N,AT}}, A::StructOfArrays{T, N}) where {T,N,AT<:AbstractArray{T,N}}
    if AT <: StructOfArrays
        error("Can't embed a SoA array in a SoA array")
    end
    arrays = map(a->convert(_type(AT), a), A.arrays)
    tt = typeof(arrays)
    StructOfArrays{T, N, AT, tt}(arrays)
end

function Base.convert(::Type{<:StructOfArrays{T,N,AT}}, A::StructOfArrays{S,N,BT}) where {T,N,AT,S,BT}
    BT != AT && AT<:CuArray && error("Can't convert from $BT to $AT with different eltypes")
    copyto!(StructOfArrays(T, _type_with_eltype(AT, T, N), size(A)), A)
end

function Base.convert(::Type{<:StructOfArrays{T,N}}, A::AbstractArray) where {T,N}
    @assert !(A isa StructOfArrays)
    copyto!(StructOfArrays(T, _type_with_eltype(typeof(A), T, N), size(A)), A)
end

Base.convert(::Type{<:StructOfArrays{T}}, A::AbstractArray{S,N}) where {T,S,N} =
    convert(StructOfArrays{T,N}, A)

Base.convert(::Type{<:StructOfArrays}, A::AbstractArray{T,N}) where {T,N} =
    convert(StructOfArrays{T,N}, A)
