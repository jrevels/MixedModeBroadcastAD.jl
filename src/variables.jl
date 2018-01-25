mutable struct Variable{V}
    value::V
    deriv::V
    Variable(value::V) where {V} = new{V}(value, initderiv(value))
end

struct Record{tag,V}
    tape::Tape{tag}
    variable::Variable{V}
end

function Record(tape::Tape, f, input...)
    input_variables = map(i -> isa(i, Record) ? i.variable : i, input)
    output = tuplemap(Variable, f(map(value, input_variables)...))
    push!(tape, Instruction(f, input_variables, output))
    return tuplemap(x -> Record(tape, x), output)
end

macro propagate!(x, Δ)
    return esc(quote
        isa($x, $Variable) && $incrderiv!($x, $Δ)
    end)
end

initderiv(x::AbstractArray) = fill!(similar(x), 0)
initderiv(x) = zero(x)

incrderiv!(v::Variable, x) = (v.deriv .+= x; nothing)

seed!(v::Variable) = (v.deriv = one(v.deriv); nothing)
seed!(r::Record) = seed!(r.variable)

value(x) = x
value(v::Variable) = v.value
value(r::Record) = value(r.variable)

deriv(v::Variable) = v.deriv
deriv(r::Record) = deriv(r.variable)
