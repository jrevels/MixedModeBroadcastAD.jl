tuplize(x) = tuple(x)
tuplize(x::Tuple) = x


tuplemap(f, x) = f(x)
tuplemap(f, x::Tuple) = map(f, x)

# optimized splat operator for single-argument functions (eg. `f(x...)`)

macro fastsplat(ex)
    @assert ex.head == :call
    f = ex.args[1]
    @assert length(ex.args) == 2
    splat = ex.args[2]
    @assert splat.head == :(...)
    arg = splat.args[1]
    esc(quote
        _fastsplat($f, $arg, Val{length($arg)})
    end)
end

@generated function _fastsplat(f, args, ::Type{Val{N}}) where {N}
    call = Expr(:call, :(f))
    for i in 1:N
        push!(call.args, :(args[$i]))
    end
    call
end
