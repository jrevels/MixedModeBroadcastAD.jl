using MixedModeBroadcastAD: CuArray, autograd
using ForwardDiff
using CUDAnative
using Test
include("kernels.jl")

@testset "smoke test" begin
    smoketest(x, y) = sum(x * y)
    x, y = rand(3, 3), rand(3, 3)
    @testset for T in (Array, CuArray)
        z, (dx, dy) = autograd(smoketest, T(x), T(y))
        @test z ≈ smoketest(x, y)
        @test Array(dx) ≈ ones(3, 3) * y'
        @test Array(dy) ≈ x' * ones(3, 3)
    end
end

@testset "HM-LSTM kernels" begin
    dims = 2
    for kind in (:cpu, :gpu), precompute in (false, true)
        println("testing hmlstm kernel for kind=`", kind, "` and precompute=`", precompute, "`")
        kernel, inputs = getkernel(kind, precompute, dims)
        test = (args...) -> sum(kernel(args...))
        output, grads = autograd(test, inputs...)
        @test output ≈ test(inputs...)
        cpu_kernel, cpu_inputs = first(getkernel(:cpu, precompute, dims)), Array.(inputs)
        cpu_test = (args...) -> sum(cpu_kernel(args...))
        for i in 1:length(inputs)
            cpu_test_i = x -> cpu_test(cpu_inputs[1:(i - 1)]..., x, cpu_inputs[(i + 1):end]...)
            @test Array(grads[i]) ≈ ForwardDiff.gradient(cpu_test_i, cpu_inputs[i])
        end
    end
end
