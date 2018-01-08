using MixedModeBroadcastAD: Tape, Variable, Record, backward!
using Base.Test

function value_and_gradients(tape::Tape, f, input...)
    variables = map(x -> Record(tape, Variable(x)), input)
    output = f(input...)
    output.variable.deriv = 1f0
    backward!(tape)
    return (output, input)
end

# This function is nonsense for actual ML purposes, but has
# a similar structure to the calculation we care about.
function example(a1, a2, b1, b2, c1, c2, d1, d2)
    a = a1 * a2
    b = b1 * b2
    c = c1 * c2
    d = d1 * d2
    return sum(sin.(a) .+ log.(b) .+ exp.(c) .+ cos.(d))
end

tape = Tape()
input = Tuple(rand(Float32, 100, 100), for i in 1:8)

value_and_gradients(tape, example, input...)
