using ForwardDiff, Test
using MixedModeBroadcastAD: unwrt

include("kernels.jl")

@testset "hmlstm kernels" begin
    dims = 128
    for usegpu in (false, true)
        println("testing hmlstm kernels for usegpu=", usegpu)
        tfkernel!, _, tfderivs, tfbuffers = get_hmlstm_kernel(true, usegpu, dims)
        kernel!, inputs, derivs, buffers = get_hmlstm_kernel(false, usegpu, dims)
        kernel!(inputs, derivs, buffers)
        tfkernel!(unwrt.(inputs), tfderivs, tfbuffers)
        @test Array(derivs[1]) ≈ Array(tfderivs[1])
        @test Array(derivs[2]) ≈ Array(tfderivs[2])
        @test Array(derivs[3]) ≈ Array(tfderivs[3])
        @test Array(derivs[4]) ≈ Array(tfderivs[4])
        cpu_inputs = Array.(unwrt.(inputs))
        deriv_i = 1
        for i in 1:length(inputs)
            if isa(inputs[i], Wrt)
                println("\t...checking gradient for input $i")
                cpu_kernel_i = x -> begin
                    before = cpu_inputs[1:(i - 1)]
                    after = cpu_inputs[(i + 1):end]
                    return sum(broadcast(cpu_hmlstm_update_c_scalar, before..., x, after...))
                end
                @test Array(derivs[deriv_i]) ≈ ForwardDiff.gradient(cpu_kernel_i, cpu_inputs[i])
                deriv_i += 1
            end
        end
    end
end

@testset "arity scaling kernels" begin
    dims = 128
    for arity in 1:3, usegpu in (false, true)
        println("testing arity scaling kernel for usegpu=", usegpu)
        kernel!, inputs, derivs, buffers = get_arity_scaling_kernel(usegpu, dims, arity)
        kernel!(inputs, derivs, buffers)
        cpu_inputs = Array.(unwrt.(inputs))
        for (i, cpu_input) in enumerate(cpu_inputs)
            println("\t...checking gradient for input $i")
            cpu_kernel_i = x -> begin
                before = cpu_inputs[1:(i - 1)]
                after = cpu_inputs[(i + 1):end]
                return sum(broadcast(cpu_arity_scaling, before..., x, after...))
            end
            @test Array(derivs[i]) ≈ ForwardDiff.gradient(cpu_kernel_i, cpu_input)
        end
    end
end
