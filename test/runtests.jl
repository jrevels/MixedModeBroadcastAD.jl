using ForwardDiff, Test
using MixedModeBroadcastAD: dual_broadcast!

include("kernels/hmlstm.jl")
include("kernels/tf_style_hmlstm.jl")

@testset "HM-LSTM kernels" begin
    dims = 5
    cpu_kernel = first(getkernel(:cpu, dims))
    for kind in (:cpu, :gpu)
        println("testing hmlstm kernel for kind=:", kind)
        kernel, output, inputs, derivs = getkernel(kind, dims)
        cpu_inputs = Array.(inputs)
        dual_broadcast!(kernel, output, inputs, derivs)
        @test Array(output) ≈ cpu_kernel.(cpu_inputs...)
        for (i, cpu_input) in enumerate(cpu_inputs)
            println("\t...checking gradient for input $i")
            cpu_kernel_i = x -> begin
                before = cpu_inputs[1:(i - 1)]
                after = cpu_inputs[(i + 1):end]
                return sum(broadcast(cpu_kernel, before..., x, after...))
            end
            @test Array(derivs[i]) ≈ ForwardDiff.gradient(cpu_kernel_i, cpu_input)
        end
    end
end
