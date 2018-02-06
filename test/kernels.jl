using MixedModeBroadcastAD: CuArray, StructOfArrays, Record, Tape, Variable,
                            forward!, backward!, seed!, value, deriv,
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

#########################################
# record/autograd convenience functions #
#########################################

function record(f, input...)
    tape = Tape()
    recorded_input = map(x -> Record(tape, Variable(x)), input)
    recorded_output = f(recorded_input...)
    return tape, recorded_output, recorded_input
end

function autograd(f, input...)
    tape, recorded_output, recorded_input = record(f, input...)
    forward!(tape)
    seed!(recorded_output)
    backward!(tape)
    return (value(recorded_output), deriv.(recorded_input))
end

########################
# kernel/tape selector #
########################

tosoa(::Type{A}, x::AbstractArray{T,N}) where {A,T,N} = convert(StructOfArrays{T,N,A}, convert(StructOfArrays, x))

function getkernel(kind::Symbol, precomputed::Bool = false, soa::Bool = true, dims::Int = 2048)
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
    bools = (convert(A, rand(Bool, dims)),
             convert(A, rand(Bool, dims)))
    if soa
        inputs = Tuple(tosoa(A{Float32,2}, rand(Float32, dims, dims)) for _ in 1:n)
    else
        inputs = Tuple(convert(A, rand(Float32, dims, dims)) for _ in 1:n)
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
