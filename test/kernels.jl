using MixedModeBroadcastAD: CuArray

sigm(x) = 1 / (1 + exp(-x))
cuda_sigm(x) = 1 / (1 + CUDAnative.exp(-x))
cuda_tanh(x) = CUDAnative.tanh(x)

########################
# fine-grained kernels #
########################

function cpu_hmlstm_update_c(z, zb, c, f, i, g)
    if z == 1.0f0 # FLUSH
        return sigm(i) * tanh(g)
    elseif zb == 0.0f0 # COPY
        return c
    else # UPDATE
        return sigm(f) * c + sigm(i) * tanh(g)
    end
end

function gpu_hmlstm_update_c(z, zb, c, f, i, g)
    if z == 1.0f0 # FLUSH
        return cuda_sigm(i) * cuda_tanh(g)
    elseif zb == 0.0f0 # COPY
        return c
    else # UPDATE
        return cuda_sigm(f) * c + cuda_sigm(i) * cuda_tanh(g)
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
    control = (convert(A, round.(rand(Float32, dims))) for _ in 1:2)
    compute = (convert(A, rand(Float32, dims, dims)) for _ in 1:4)
    input_values = (control..., compute...)
    input_derivs = similar.(input_values)
    output_value = rand(Float32, dims, dims)
    return kernel, input_values, input_derivs, output_value
end
