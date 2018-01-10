using MixedModeBroadcastAD: Tape, Variable, Record, value, deriv, backward!, seed!, CuArray
using ForwardDiff
using CUDAnative
using Test

###########
# CuArray #
###########

f(x) = x .* 2

input = rand(3,3)
output = f(input)

input_dev = CuArray(input)
output_dev = f(input_dev)

@test output == Array(output_dev)

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

###########################
# Smoke Test with CuArray #
###########################

cu_x = CuArray(x)
cu_y = CuArray(y)

tape = Tape()
cu_z, (cu_dx, cu_dy) = autograd(tape, smoketest, cu_x, cu_y)

@test cu_z ≈ z
@test Array(cu_dx) ≈ dx
@test Array(cu_dy) ≈ dy

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
input = Tuple(rand(Float32, 2, 2) for i in 1:8)
output, grads = autograd(tape, example, input...)

@test output == example(input...)
@test grads[1] ≈ ForwardDiff.gradient(x -> example(x, input[2:8]...), input[1])
@test grads[2] ≈ ForwardDiff.gradient(x -> example(input[1], x, input[3:8]...), input[2])
@test grads[3] ≈ ForwardDiff.gradient(x -> example(input[1:2]..., x, input[4:8]...), input[3])
@test grads[4] ≈ ForwardDiff.gradient(x -> example(input[1:3]..., x, input[5:8]...), input[4])
@test grads[5] ≈ ForwardDiff.gradient(x -> example(input[1:4]..., x, input[6:8]...), input[5])
@test grads[6] ≈ ForwardDiff.gradient(x -> example(input[1:5]..., x, input[7:8]...), input[6])
@test grads[7] ≈ ForwardDiff.gradient(x -> example(input[1:6]..., x, input[8]), input[7])
@test grads[8] ≈ ForwardDiff.gradient(x -> example(input[1:7]..., x), input[8])

#################################
# LSTM-like kernel with CuArray #
#################################

@inline function cu_example(a1, a2, b1, b2, c1, c2, d1, d2)
    a = a1 * a2
    b = b1 * b2
    c = c1 * c2
    d = d1 * d2
    return sum(CUDAnative.sin.(a) .+ CUDAnative.log.(b) .+ CUDAnative.exp.(c) .+ CUDAnative.cos.(d))
end

tape = Tape()
cu_input = Tuple(CuArray(rand(Float32, 2, 2)) for i in 1:8)
cu_output, cu_grads = autograd(tape, cu_example, cu_input...)

@test cu_output ≈ output
@test Array(cu_grads[1]) ≈ grads[1]
@test Array(cu_grads[2]) ≈ grads[2]
@test Array(cu_grads[3]) ≈ grads[3]
@test Array(cu_grads[4]) ≈ grads[4]
@test Array(cu_grads[5]) ≈ grads[5]
@test Array(cu_grads[6]) ≈ grads[6]
@test Array(cu_grads[7]) ≈ grads[7]
@test Array(cu_grads[8]) ≈ grads[8]
