using MixedModeBroadcastAD: CuArray, sigm, cuda_sigm, cuda_tanh

sigm(x) = 1 / (1 + exp(-x))
cuda_sigm(x) = 1 / (1 + CUDAnative.exp(-x))
cuda_tanh(x) = CUDAnative.tanh(x)

########################
# fine-grained kernels #
########################

function cpu_hmlstm_update_c_scalar(z_t, # = z_{t}^{l-1}
                                    z_l, # = z_{t-1}^{l}
                                    c, f, i, g)
    if z_l == 1 # FLUSH
        return sigm(i) * tanh(g)
    elseif z_t == 1 # UPDATE
        return sigm(f) * c + sigm(i) * tanh(g)
    else # COPY
        return c
    end
end

function gpu_hmlstm_update_c_scalar(z_t, # = z_{t}^{l-1}
                                    z_l, # = z_{t-1}^{l}
                                    c, f, i, g)
    if z_l == 1 # FLUSH
        return cuda_sigm(i) * cuda_tanh(g)
    elseif z_t == 1 # UPDATE
        return cuda_sigm(f) * c + cuda_sigm(i) * cuda_tanh(g)
    else # COPY
        return c
    end
end

##########################
# coarse-grained kernels #
##########################

# TODO: port over something comparable to what hmlstm.py is doing

###################
# kernel selector #
###################

function getkernel(kind::Symbol, dims::Int = 2048)
    @assert kind == :cpu || kind == :gpu
    if kind == :cpu
        kernel = cpu_hmlstm_update_c
        A = Array
    else
        kernel = gpu_hmlstm_update_c
        A = CuArray
    end
    # TODO: the control values should just be vectors; see matching TODO in src/ad/broadcast.jl
    control = (convert(A, round.(rand(Float32, dims, dims))) for _ in 1:2)
    compute = (convert(A, rand(Float32, dims, dims)) for _ in 1:4)
    input_values = (control..., compute...)
    input_derivs = similar.(input_values)
    output_value = rand(Float32, dims, dims)
    return kernel, input_values, input_derivs, output_value
end
