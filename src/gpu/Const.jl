## Const Arg
struct Const{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    Const{T,N,A}(data::A) where {T,N,A<:AbstractArray{T,N}} = new{T,N,A}(data)
end
Const{T,N,A}(::Uninitialized, shape::NTuple{N,Integer}) where {T,N,A} = Const{T,N,A}(A(uninitialized, shape))

Base.size(A::Const) = size(A.data)
Base.size(A::Const, i) = size(A.data, i)
Base.show(io::IO, a::Const{T,N,A}) where {T,N,A} = print(io, "$(length(a))-element Const{$T,$N,$A}")
Base.print_array(::IO, ::Const) = nothing

Base.getindex(A::Const, I...) = (Base.@_propagate_inbounds_meta; A.data[I...])
@inline function Base.getindex(A::Const{T,N,<:CuDeviceArray{T,N,AS}}, index::Integer) where {T,N,AS}
    @boundscheck checkbounds(A, index)
    align = CUDAnative.datatype_align(T)
    CUDAnative.unsafe_cached_load(pointer(A.data), index, Val(align))::T
end

Base.BroadcastStyle(::Type{<:Const{T,N,A}}) where {T,N,A<:AbstractArray} = Broadcast.ArrayStyle{Const{T,N,A}}()

function Base.similar(A::Const{T1,N,AT}, ::Type{T}, dims::Dims) where {T1,N,AT,T}
    similar(AT, T, dims)
end

function Base.broadcast_similar(f, ::Broadcast.ArrayStyle{Const{T1, N, A}}, ::Type{T}, inds, As...) where {T1,N,A,T}
    Base.broadcast_similar(f, Base.BroadcastStyle(A), T, inds, As...)
end

Base.setindex!(::Const, x, I...) = error("setindex! is not allowed for Const array")

Base.IndexStyle(::Type{<:Const{T,N,A}}) where {T,N,A<:AbstractArray} =  Base.IndexStyle(A)

readonly(a::A) where{T,N,A<:AbstractArray{T,N}} = Const{T,N,A}(a)
