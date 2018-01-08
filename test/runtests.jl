using MixedModeBroadcastAD: Tape, Variable, Record, value, deriv, backward!, seed!
using Test

############
# autograd #
############

function autograd(tape::Tape, f, input...)
    recorded_input = map(x -> Record(tape, Variable(x)), input)
    recorded_output = f(recorded_input...)
    seed!(recorded_output)
    backward!(tape)
    return (value(recorded_output), map(deriv, recorded_input))
end

##############
# Smoke Test #
##############

smoketest(x, y) = sum(x * y)
x, y = rand(3, 3), rand(3, 3)

tape = Tape()
z, (dx, dy) = autograd(tape, smoketest, x, y)

@test z == smoketest(x, y)
@test dx == ones(3, 3) * y'
@test dy == x' * ones(3, 3)

####################
# LSTM-like kernel #
####################

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
input = Tuple(rand(Float32, 100, 100) for i in 1:8)
output, grads = autograd(tape, example, input...)
