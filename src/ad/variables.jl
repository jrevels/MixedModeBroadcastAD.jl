mutable struct Variable{V}
    value::V
    deriv::V
    downstreams::Int
    Variable(value::V) where {V} = new{V}(value, initderiv(value), 0)
end

struct Record{V}
    tape::Tape
    variable::Variable{V}
end

function Record(tape::Tape, f, input...)
    input_variables = map(i -> isa(i, Record) ? i.variable : i, input)
    output = tuplemap(Variable, f(map(value, input_variables)...))
    push!(tape, Instruction(f, input_variables, output))
    tupleforeach(incrdownstreams!, input_variables)
    return tuplemap(x -> Record(tape, x), output)
end

initderiv(x::AbstractArray) = fill!(similar(x), 0)
initderiv(x) = zero(x)

initderiv!(x) = x
initderiv!(v::Variable{<:AbstractArray}) = (fill!(v.deriv, 0); v)
initderiv!(v::Variable{<:Number}) = (v.deriv = zero(v.deriv); v)

macro propagate!(x, ∇)
    return esc(quote
        if isa($x, $Variable)
            if $(x).downstreams > 1
                $(x).deriv .+= $∇
            else
                $(x).deriv .= $∇
            end
        end
    end)
end

incrdownstreams!(x) = x
incrdownstreams!(v::Variable) = (v.downstreams += 1; v)

seed!(v::Variable{<:AbstractArray}) = (fill!(v.deriv, 1); nothing)
seed!(v::Variable{<:Number}) = (v.deriv = one(v.deriv); nothing)
seed!(r::Record) = seed!(r.variable)

value(x) = x
value(v::Variable) = v.value
value(r::Record) = value(r.variable)

deriv(v::Variable) = v.deriv
deriv(r::Record) = deriv(r.variable)
