using ForwardDiff, Test
using MixedModeBroadcastAD: autodiff_broadcast!

include("kernels.jl")

@testset "HM-LSTM kernels" begin
    dims = 5
    cpu_kernel = first(getkernel(:cpu, dims))
    for kind in (:cpu, :gpu)
        println("testing hmlstm kernel for kind=:", kind)
        kernel, input_values, input_derivs, output_value = getkernel(kind, dims)
        autodiff_broadcast!(kernel, input_values, input_derivs, output_value)
        @test Array(output_value) ≈ Array(kernel.(input_values...))
        cpu_input_values = Array.(input_values)
        for (i, cpu_input_value) in enumerate(cpu_input_values)
            println("\t...checking gradient for input $i")
            cpu_kernel_i = x -> begin
                before = cpu_input_values[1:(i - 1)]
                after = cpu_input_values[(i + 1):end]
                return sum(broadcast(cpu_kernel, before..., x, after...))
            end
            @test Array(input_derivs[i]) ≈ ForwardDiff.gradient(cpu_kernel_i, cpu_input_value)
        end
    end
end
