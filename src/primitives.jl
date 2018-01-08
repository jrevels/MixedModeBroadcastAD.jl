################
# Forward Pass #
################

const FORWARD_METHODS = [(Base.sum, 1), (Base.:+, 2), (Base.:*, 3)]

for (f, arity) in FORWARD_METHODS
    if arity == 1
        @eval begin
            $(f)(x::Record) = Record($f, x)
        end
    elseif arity == 2
        @eval begin
            $(f)(x::Record{T}, y::Record{T}) where {T} = Record($f, x, y)
            $(f)(x::Record{T}, y) where {T}            = Record($f, x, y)
            $(f)(x, y::Record{T}) where {T}            = Record($f, x, y)
        end
    else
        error("unsupported arity $arity for method $f")
    end
end

##################
# Backwards Pass #
##################

backward!(::typeof(sum), x, y) = @propagate!(x, deriv(y))

function backward!(::typeof(+), x, y, z)
    @propagate!(x, deriv(z) .* value(y))
    @propagate!(y, deriv(z) .* value(x))
end

function backward!(::typeof(*), x, y, z)
    @propagate!(x, A_mul_Bc(deriv(z), value(y)))
    @propagate!(y, Ac_mul_B(value(x), deriv(z)))
end
