using MixedModeBroadcastAD: CuArray, StructOfArrays, record, forward!, backward!,
                            sigm, cuda_sigm, cuda_tanh

########################
# fine-grained kernels #
########################

cpu_hmlstm_update_c(inputs...) = cpu_hmlstm_update_c_scalar.(inputs...)

function cpu_hmlstm_update_c_scalar(z_t, # = z_{t}^{l-1}
                                    z_l, # = z_{t-1}^{l}
                                    c,
                                    W_f, R_f, b_f,
                                    W_i, R_i, b_i,
                                    W_g, R_g, b_g)
    if z_l == 1 # FLUSH
        return sigm(W_i + R_i + b_i) * tanh(W_g + R_g + b_g)
    elseif z_t == 1 # UPDATE
        return sigm(W_f + R_f + b_f) * c +
               sigm(W_i + R_i + b_i) * tanh(W_g + R_g + b_g)
    else # COPY
        return c
    end
end

gpu_hmlstm_update_c(inputs...) = gpu_hmlstm_update_c_scalar.(inputs...)

function gpu_hmlstm_update_c_scalar(z_t, # = z_{t}^{l-1}
                                    z_l, # = z_{t-1}^{l}
                                    c,
                                    W_f, R_f, b_f,
                                    W_i, R_i, b_i,
                                    W_g, R_g, b_g)
    if z_l == 1 # FLUSH
        return cuda_sigm(W_i + R_i + b_i) * cuda_tanh(W_g + R_g + b_g)
    elseif z_t == 1 # UPDATE
        return cuda_sigm(W_f + R_f + b_f) * c +
               cuda_sigm(W_i + R_i + b_i) * cuda_tanh(W_g + R_g + b_g)
    else # COPY
        return c
    end
end

hmlstm_update_c_precomputed(inputs...) = hmlstm_update_c_precomputed_scalar.(inputs...)

function hmlstm_update_c_precomputed_scalar(z_t, # = z_{t}^{l-1}
                                            z_l, # = z_{t-1}^{l}
                                            c, f, i, g)
    if z_l == 1 # FLUSH
        return i * g
    elseif z_t == 1 # UPDATE
        return f * c + i * g
    else # COPY
        return c
    end
end

##########################
# coarse-grained kernels #
##########################

#=
TODO: port over something more similar to the TF HM-LSTM implementation, e.g:

new_c = tf.where(
    tf.equal(z, tf.constant(1., dtype=tf.float32)),
    tf.multiply(i, g, name='c'),
    tf.where(
        tf.equal(zb, tf.constant(0., dtype=tf.float32)),
        tf.identity(c),
        tf.add(tf.multiply(f, c), tf.multiply(i, g))
    )
)
=#

########################
# kernel/tape selector #
########################

tosoa(::Type{A}, x::AbstractArray{T,N}) where {A,T,N} = convert(StructOfArrays{T,N,A}, convert(StructOfArrays, x))

function getkernel(kind::Symbol, precomputed::Bool = false, dims::Int = 2048, soa::Bool = true)
    @assert kind == :cpu || kind == :gpu
    if kind == :cpu
        kernel = cpu_hmlstm_update_c
        A = Array
    else
        kernel = gpu_hmlstm_update_c
        A = CuArray
    end
    if precomputed
        kernel = hmlstm_update_c_precomputed
        n = 4
    else
        n = 10
    end
    bools = (A(rand(Bool, dims)), A(rand(Bool, dims)))
    inputs = Tuple(A(rand(Float32, dims, dims)) for _ in 1:n)
    if soa
        inputs = Tuple(tosoa(A, x) for x in inputs)
    end
    return kernel, bools, inputs
end

function gettape(args...)
    f, bools, inputs = getkernel(args...)
    tape = first(record((xs...) -> f(bools..., xs...), inputs...))
    forward!(tape)  # "precompile" forwards pass
    backward!(tape) # "precompile" backwards pass
    return tape
end
