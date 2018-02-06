mutable struct Variable{V}
    value::V
    deriv::V
    Variable(value::V) where {V} = new{V}(value, initderiv(value))
end

struct Record{V}
    tape::Tape
    variable::Variable{V}
end

function Record(tape::Tape, f, input...)
    input_variables = map(i -> isa(i, Record) ? i.variable : i, input)
    output = tuplemap(Variable, f(map(value, input_variables)...))
    push!(tape, Instruction(f, input_variables, output))
    return tuplemap(x -> Record(tape, x), output)
end

initderiv(x::AbstractArray) = fill!(similar(x), 0)
initderiv(x) = zero(x)

initderiv!(x) = x
initderiv!(v::Variable{<:AbstractArray}) = (fill!(v.deriv, 0); v)
initderiv!(v::Variable{<:Number}) = (v.deriv = zero(v.deriv); v)

seed!(v::Variable) = (v.deriv = one(v.deriv); nothing)
seed!(r::Record) = seed!(r.variable)

value(x) = x
value(v::Variable) = v.value
value(r::Record) = value(r.variable)

deriv(v::Variable) = v.deriv
deriv(r::Record) = deriv(r.variable)