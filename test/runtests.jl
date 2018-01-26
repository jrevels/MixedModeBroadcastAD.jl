using MixedModeBroadcastAD: record, autograd, CuArray
using ForwardDiff
using CUDAnative
using Test

const TEST_TYPES = [Array, CuArray]
include("kernels.jl")

@testset "smoke test" begin
    smoketest(x, y) = sum(x * y)
    x, y = rand(3, 3), rand(3, 3)

    @testset for T in TEST_TYPES
        z, (dx, dy) = autograd(smoketest, T(x), T(y))

        @test z ≈ smoketest(x, y)
        @test Array(dx) ≈ ones(3, 3) * y'
        @test Array(dy) ≈ x' * ones(3, 3)
    end
end

@testset "LSTM-like kernel" begin
    tests = Dict(Array => lstm_update_c, CuArray => cudanative_lstm_update_c)
    input = Tuple(rand(Float32, 2, 2) for i in 1:10)

    # reduce the output so we can test via autograd
    for (key,val) in tests
        test = tests[key]
        tests[key] = (args...) -> sum(test(args...))
    end

    @testset for T in TEST_TYPES
        test = tests[T]
        output, grads = autograd(test, T.(input)...)
        @test output ≈ test(T.(input)...)

        reference = tests[Array]
        for i in 1:length(input)
            testarg = x -> reference(input[1:(i - 1)]...,  x, input[(i + 1):end]...)
            @test Array(grads[i]) ≈ ForwardDiff.gradient(testarg, input[i])
        end
    end
end
