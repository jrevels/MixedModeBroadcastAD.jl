using MixedModeBroadcastAD: record, forward!, backward!, sigm, cuda_sigm, cuda_tanh

########################
# fine-grained kernels #
########################

#=
This section provides scalar kernel implementations for updating `c` in an HM-LSTM. To
broadcast these kernels over array inputs, one simply uses Julia's broadcast syntax
(e.g. `f.(array_inputs...)`).
=#

function cpu_hmlstm_update_c(c,
                             z_t, # = z_{t}^{l-1}
                             z_l, # = z_{t-1}^{l}
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

function gpu_hmlstm_update_c(c,
                             z_t, # = z_{t}^{l-1}
                             z_l, # = z_{t-1}^{l}
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

function hmlstm_update_c_precomputed(c,
                                     z_t, # = z_{t}^{l-1}
                                     z_l, # = z_{t-1}^{l}
                                     f, i, g)
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

function getkernel(kind::Symbol, precomputed::Bool = false, dims::Int = 2048)
    @assert kind == :cpu || kind == :gpu
    if kind == :cpu
        kernel = cpu_hmlstm_update_c
        T = Array
    else
        kernel = gpu_hmlstm_update_c
        T = CuArray
    end
    if precomputed
        kernel = hmlstm_update_c_precomputed
        n = 3
    else
        n = 9
    end
    inputs = map(T, Tuple(rand(Float32, dims, dims),
                          rand(Bool, dims), rand(Bool, dims),
                          (rand(Float32, dims, dims) for _ in 1:n)...))
    return kernel, inputs
end

function gettape(args...)
    f, inputs = getkernel(args...)
    tape = first(record(f, inputs...))
    forward!(tape)  # "precompile" forwards pass
    backward!(tape) # "precompile" backwards pass
    return tape
end
