using ForwardDiff, CUDAnative, Test

include("kernels.jl")

@testset "smoke test" begin
    smoketest(x, y) = x * y
    x, y = rand(3, 3), rand(3, 3)
    @testset for T in (Array, CuArray)
        z, (dx, dy) = autograd(smoketest, T(x), T(y))
        @test Array(z) ≈ smoketest(x, y)
        @test Array(dx) ≈ y'
        @test Array(dy) ≈ x'
    end
end

@testset "HM-LSTM kernels" begin
    dims = 2
    for kind in (:cpu, :gpu), soa in (false, true), cache in (false, true)
        println("testing hmlstm kernel for kind=:", kind, "; soa=", soa, "; cache=", cache)
        kernel, bools, inputs = getkernel(kind, soa, dims)
        test = (args...) -> kernel(bools..., args...)
        output, grads = autograd(test, inputs...; cache = cache)
        @test output ≈ test(inputs...)
        cpu_kernel, cpu_bools, cpu_inputs = first(getkernel(:cpu, false, dims)), Array.(bools), Array.(inputs)
        cpu_test = (args...) -> cpu_kernel(cpu_bools..., args...)
        for i in 1:length(inputs)
            println("\t...checking gradient for input $i")
            cpu_test_i = x -> cpu_test(cpu_inputs[1:(i - 1)]..., x, cpu_inputs[(i + 1):end]...)
            @test_broken Array(grads[i]) ≈ ForwardDiff.gradient(cpu_test_i, cpu_inputs[i])
        end
    end
end
