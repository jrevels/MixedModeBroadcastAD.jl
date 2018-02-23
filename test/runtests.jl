using ForwardDiff, Test

include("kernels.jl")

@testset "hmlstm kernels" begin
    dims = 5
    for kind in (:cpu, :gpu)
        println("testing hmlstm kernels for kind=:", kind)
        tfkernel!, _, tfderivs = get_kernel(kind, dims, true)
        kernel!, inputs, derivs = get_kernel(kind, dims, false)
        kernel!(inputs, derivs)
        tfkernel!(inputs, tfderivs)
        for (d, tfd) in zip(derivs[3:end], tfderivs)
            @test Array(d) ≈ Array(tfd)
        end
        cpu_inputs = Array.(inputs)
        for (i, cpu_input) in enumerate(cpu_inputs)
            println("\t...checking gradient for input $i")
            cpu_kernel_i = x -> begin
                before = cpu_inputs[1:(i - 1)]
                after = cpu_inputs[(i + 1):end]
                return sum(broadcast(cpu_hmlstm_update_c_scalar, before..., x, after...))
            end
            @test Array(derivs[i]) ≈ ForwardDiff.gradient(cpu_kernel_i, cpu_input)
        end
    end
end
