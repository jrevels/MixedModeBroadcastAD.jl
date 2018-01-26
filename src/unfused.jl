import Base.Broadcast

struct Unfused{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    data::A
    function Unfused(data::A) where {T, N, A <: AbstractArray{T, N}}
        new{T, N, A}(data)
    end
end

Base.getindex(u::Unfused, I...) = u.data[I...]
Base.setindex!(u::Unfused, val, I...) = u.data[I...] = val
Base.size(u::Unfused) = size(u.data)
Base.similar(u::Unfused, T::Type=eltype(u),dims=size(u)) = Unfused(similar(u.data, T, dims))

struct UnfusedStyle{N} <: Broadcast.AbstractArrayStyle{N} end
UnfusedStyle(::Val{N}) where N = UnfusedStyle{N}()
UnfusedStyle{M}(::Val{N}) where {M,N} = UnfusedStyle{max(M, N)}()
Broadcast.BroadcastStyle(::Type{<:Unfused{T, N}}) where {T, N} = UnfusedStyle{N}()
Broadcast.broadcast_similar(::UnfusedStyle{N}, ::Type{ElType}, inds::Base.Indices{N}, bc) where {N,ElType} =
      Unfused(similar(Array{ElType}, inds))
Broadcast.is_broadcast_incremental(::Broadcast.Broadcasted{<:UnfusedStyle}) = true

import Base.Broadcast: broadcast
# implement specific functions needed for test/kernels
broadcast(::typeof(+), u::Unfused, v::Unfused) = (@info("Unfused +"); Unfused(u.data .+ v.data))
broadcast(::typeof(*), u::Unfused, v::Unfused) = (@info("Unfused *"); Unfused(u.data .* v.data))
broadcast(f::F, u::Unfused) where F <: Function = (@info("Unfused $f"); Unfused(f.(u.data)))