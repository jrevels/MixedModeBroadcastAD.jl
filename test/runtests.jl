using MixedModeBroadcastAD: autograd
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

@testset "LSTM-like kernels" begin
    reference_kernel, _ = getkernel(:cpu, 0, 2)
    reference_test = (args...) -> sum(reference_kernel(args...))
    for kind in [:cpu, :cudanative, :cudaraw], fusion_level in 0:2
        println("testing lstm-like kernel for kind `", kind, "` and fusion_level `", fusion_level, "`")
        kind == :cudaraw && fusion_level > 1 && continue
        kernel, inputs = getkernel(kind, fusion_level, 2)
        test = (args...) -> sum(kernel(args...))
        reference_inputs = Array.(inputs)
        test_output = test(inputs...)
        @test test_output ≈ reference_test(reference_inputs...)
        if kind != :cudaraw
            output, grads = autograd(test, inputs...)
            @test test_output ≈ output
            for i in 1:length(inputs)
                testarg = x -> reference_test(reference_inputs[1:(i - 1)]...,  x, reference_inputs[(i + 1):end]...)
                @test Array(grads[i]) ≈ ForwardDiff.gradient(testarg, reference_inputs[i])
            end
        end
    end
end
