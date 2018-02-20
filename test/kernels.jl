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
    @assert dims >= 3
    @assert kind == :cpu || kind == :gpu

    # select kernel/array kind
    if kind == :cpu
        kernel = cpu_hmlstm_update_c
        A = Array
    else
        kernel = gpu_hmlstm_update_c
        A = CuArray
    end

    # set up control variables to ensure that we will hit all three cases at least once
    control = (round.(rand(Float32, dims)), round.(rand(Float32, dims)))
    control[1][1] = 1.0f0 # FLUSH case
    control[2][1] = 0.0f0
    control[1][2] = 0.0f0 # COPY case
    control[2][2] = 0.0f0
    control[1][3] = 0.0f0 # UPDATE case
    control[2][3] = 1.0f0
    control = (convert(A, control[1]), convert(A, control[2]))

    # set up the rest of our values/buffers
    input_values = (control..., (convert(A, rand(Float32, dims, dims)) for _ in 1:4)...)
    input_derivs = similar.(input_values)
    output_value = rand(Float32, dims, dims)

    return kernel, input_values, input_derivs, output_value
end
