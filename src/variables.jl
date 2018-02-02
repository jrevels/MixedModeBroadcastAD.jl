mutable struct Variable{V}
    value::V
    deriv::V
end

Variable(value::Union{AbstractArray,Real}) = Variable(value, initderiv(value))
Variable(value::AbstractArray{Bool}) = value

struct Record{V}
    tape::Tape
    variable::Variable{V}
end

Record(tape::Tape, value::AbstractArray) = value

function Record(tape::Tape, f, input...)
    input_variables = map(i -> isa(i, Record) ? i.variable : i, input)
    output = tuplemap(Variable, f(map(value, input_variables)...))
    push!(tape, Instruction(f, input_variables, output))
    return tuplemap(x -> Record(tape, x), output)
end

initderiv(x::AbstractArray) = fill!(similar(x), 0)
initderiv(x) = zero(x)

seed!(v::Variable) = (v.deriv = one(v.deriv); nothing)
seed!(r::Record) = seed!(r.variable)

value(x) = x
value(v::Variable) = v.value
value(r::Record) = value(r.variable)

deriv(v::Variable) = v.deriv
deriv(r::Record) = deriv(r.variable)
