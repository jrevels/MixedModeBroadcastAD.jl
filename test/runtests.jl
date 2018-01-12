using MixedModeBroadcastAD: Tape, Variable, Record, value, deriv, backward!, seed!, CuArray
using ForwardDiff
using CUDAnative
using Test

function autograd(tape::Tape, f, input...)
    recorded_input = map(x -> Record(tape, Variable(x)), input)
    recorded_output = f(recorded_input...)
    seed!(recorded_output)
    backward!(tape)
    return (value(recorded_output), map(deriv, recorded_input))
end

@testset "smoke test" begin
    smoketest(x, y) = sum(x * y)
    x, y = rand(3, 3), rand(3, 3)

    @testset for T in [Array, CuArray]
        tape = Tape()
        z, (dx, dy) = autograd(tape, smoketest, T(x), T(y))

        @test z ≈ smoketest(x, y)
        @test Array(dx) ≈ ones(3, 3) * y'
        @test Array(dy) ≈ x' * ones(3, 3)
    end
end

@testset "LSTM-like kernel" begin
    # This function is nonsense for actual ML purposes, but has
    # a similar structure to the calculation we care about.
    function example(a1, a2, b1, b2, c1, c2, d1, d2)
        a = a1 * a2
        b = b1 * b2
        c = c1 * c2
        d = d1 * d2
        return sum(sin.(a) .+ log.(b) .+ exp.(c) .+ cos.(d))
    end
    function cuda_example(a1, a2, b1, b2, c1, c2, d1, d2)
        a = a1 * a2
        b = b1 * b2
        c = c1 * c2
        d = d1 * d2
        return sum(CUDAnative.sin.(a) .+ CUDAnative.log.(b) .+
                   CUDAnative.exp.(c) .+ CUDAnative.cos.(d))
    end
    examples = Dict(
        Array   => example,
        CuArray => cuda_example
    )

    input = Tuple(rand(Float32, 2, 2) for i in 1:8)

    @testset for T in [Array, CuArray]
        tape = Tape()
        output, grads = autograd(tape, examples[T], T.(input)...)

        @test output ≈ example(input...)
        @test Array(grads[1]) ≈ ForwardDiff.gradient(x -> example(x, input[2:8]...), input[1])
        @test Array(grads[2]) ≈ ForwardDiff.gradient(x -> example(input[1], x, input[3:8]...), input[2])
        @test Array(grads[3]) ≈ ForwardDiff.gradient(x -> example(input[1:2]..., x, input[4:8]...), input[3])
        @test Array(grads[4]) ≈ ForwardDiff.gradient(x -> example(input[1:3]..., x, input[5:8]...), input[4])
        @test Array(grads[5]) ≈ ForwardDiff.gradient(x -> example(input[1:4]..., x, input[6:8]...), input[5])
        @test Array(grads[6]) ≈ ForwardDiff.gradient(x -> example(input[1:5]..., x, input[7:8]...), input[6])
        @test Array(grads[7]) ≈ ForwardDiff.gradient(x -> example(input[1:6]..., x, input[8]), input[7])
        @test Array(grads[8]) ≈ ForwardDiff.gradient(x -> example(input[1:7]..., x), input[8])
    end
end
