mutable struct Variable{V}
    value::V
    deriv::V
    Variable(value::V) where {V} = new{V,V}(value, initderiv(value))
end

struct Record{T,V,D}
    tape::Tape{T}
    variable::Variable{V,D}
end

function Record(tape::Tape, f, input...)
    output = tuplemap(Variable, f(map(value, input...)...))
    push!(tape, Instruction(f, input, output))
    return Record(tape, output)
end

macro propagate!(x, Δ)
    return esc(quote
        (typeof($x) <: $Variable) && $incrderiv!($x, $Δ)
    end)
end

initderiv(x::AbstractArray) = fill!(similar(x), 0)
initderiv(x) = zero(x)

incrderiv!(v::Variable, x) = (v.deriv .+= x; nothing)

value(x) = x
value(v::Variable) = v.value
value(r::Record) = value(r.variable)

deriv(v::Variable) = v.deriv
