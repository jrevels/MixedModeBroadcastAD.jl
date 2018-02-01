# Based upon https://github.com/simonster/StructsOfArrays.jl
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

# Storage types of StructOfArrays need to implement this
_type_with_eltype(::Type{<:Array}, T, N) = Array{T, N}
_type_with_eltype(::Type{<:CuArray}, T, N) = CuArray{T, N}
_type_with_eltype(::Type{CuDeviceArray{_T,_N,AS}}, T, N) where{_T,_N,AS} = CuDeviceArray(T,N,AS)
_type(::Type{<:Array}) = Array
_type(::Type{<:CuArray}) = CuArray
_type(::Type{<:CuDeviceArray}) = CuDeviceArray

function gather_eltypes(T, visited = Set{Type}())
    (!isconcretetype(T) || T.mutable) && throw(ArgumentError("can only create an StructOfArrays of leaf type immutables"))
    if isempty(T.types)
        return Type[T]
    end
    types = Type[]
    push!(visited, T)
    for S in T.types
        sizeof(S) == 0 && continue
        (S in visited) && throw(ArgumentError("Recursive types are not allowed for SoA conversion"))
        if isempty(S.types)
            push!(types, S)
        else
            append!(types, gather_eltypes(S, copy(visited)))
        end
    end
    types
end

@generated function StructOfArrays(::Type{T}, ::Type{ArrayT}, dims::Integer...) where {T, ArrayT<:AbstractArray}
    N         = length(dims)
    pArrayT   = _type_with_eltype(ArrayT, T, N)
    types     = gather_eltypes(T)
    arrtypes  = map(t->_type_with_eltype(ArrayT, t, N), types)
    arrtuple  = Tuple{arrtypes...}
    :(StructOfArrays{T,$N,$(pArrayT),$arrtuple}(($([:($(arrtypes[i])(uninitialized,dims)) for i = 1:length(types)]...),)))
end
StructOfArrays(T::Type, AT::Type, dims::Tuple{Vararg{Integer}}) = StructOfArrays(T, AT, dims...)

Base.size(A::StructOfArrays) = size(@inbounds(A.arrays[1]))

Base.show(io::IO, a::StructOfArrays{T,N,A}) where {T,N,A} = print(io, "$(length(a))-element SoA{$T,$N,$A}")

Base.print_array(::IO, ::StructOfArrays) = nothing

function generate_getindex(T, arraynum)
    members = Expr[]
    for S in T.types
        sizeof(S) == 0 && push!(members, :($(S())))
        if isempty(S.types)
            push!(members, :(A.arrays[$arraynum][i...]))
            arraynum += 1
        else
            member, arraynum = generate_getindex(S, arraynum)
            push!(members, member)
        end
    end
    Expr(:new, T, members...), arraynum
end

@generated function Base.getindex(A::StructOfArrays{T}, i::Integer...) where T
    exprs = Any[Expr(:meta, :inline), Expr(:meta, :propagate_inbounds)]
    if isempty(T.types)
        push!(exprs, :(return A.arrays[1][i...]))
    else
        strct, _ = generate_getindex(T, 1)
        push!(exprs, strct)
    end
    quote
        $(exprs...)
    end
end

function generate_setindex(T, x, arraynum)
    s = gensym()
    exprs = Expr[:($s = $x)]
    for (el,S) in enumerate(T.types)
        sizeof(S) == 0 && push!(exprs, :($(S())))
        if isempty(S.types)
            push!(exprs, :(A.arrays[$arraynum][i...] = getfield($s, $el)))
            arraynum += 1
        else
            nexprs, arraynum = generate_setindex(S, :(getfield($s, $el)), arraynum)
            append!(exprs, nexprs)
        end
    end
    exprs, arraynum
end

@generated function Base.setindex!(A::StructOfArrays{T}, x, i::Integer...) where {T}
    exprs = Any[Expr(:meta, :inline), Expr(:meta, :propagate_inbounds)]
    push!(exprs, :(v = convert(T, x)))
    if isempty(T.types)
        push!(exprs, :(A.arrays[1][i...] = v))
    else
        append!(exprs, generate_setindex(T, :v, 1)[1])
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
