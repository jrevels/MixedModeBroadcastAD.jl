
#=
Since our technique only affects broadcast performance, we have limited our kernel to
execute only a small part of the LSTM layer which stresses fused broadcast operations. This
allows us tobenchmark/test our technique in isolation from the other parts of our ad-hoc
AD framework which are not necessarily performant and only exist to facilitate the
demonstration of the broadcast technique.
=#

σ(x) = 1 / (1 + exp(-x))
cuda_σ(x) = 1 / (1 + CUDAnative.exp(-x))

function lstm_update_c(c,
                       Wx_f, Wx_i, Wx_c,
                       Rh_f, Rh_i, Rh_c,
                       b_f,  b_i,  b_c)
    return σ.(Wx_f .+ Rh_f .+ b_f) .* c .+
           σ.(Wx_i .+ Rh_i .+ b_i) .* tanh.(Wx_c .+ Rh_c .+ b_c)
end

function cuda_lstm_update_c(c,
                            Wx_f, Wx_i, Wx_c,
                            Rh_f, Rh_i, Rh_c,
                            b_f,  b_i,  b_c)
    return cuda_σ.(Wx_f .+ Rh_f .+ b_f) .* c .+
           cuda_σ.(Wx_i .+ Rh_i .+ b_i) .* CUDAnative.tanh.(Wx_c .+ Rh_c .+ b_c)
end

# NOTE: unfused kernels work with an explicit out param cfr. the CUDA implementation,
#       as well as two pre-allocated arrays for temporaries

function unfused_lstm_update_c(out, tmp1, tmp2, c,
                               Wx_f, Wx_i, Wx_c,
                               Rh_f, Rh_i, Rh_c,
                               b_f,  b_i,  b_c)
    # σ.(Wx_f .+ Rh_f .+ b_f) .* c
    tmp1 .= Wx_f .+ Rh_f
    tmp1 .= tmp1 .+ b_f
    tmp1 .= σ.(tmp1)
    tmp1 .= tmp1 .* c

    # σ.(Wx_i .+ Rh_i .+ b_i)
    tmp2 = Wx_i .+ Rh_i
    tmp2 .= tmp2 .+ b_i
    tmp2 .= σ.(tmp2)

    # tanh.(Wx_c .+ Rh_c .+ b_c)
    out .= Wx_c .+ Rh_c
    out .= out .+ b_c
    out .= tanh.(out)

    # σ.(...) * tanh.(...)
    out .= out .* tmp2

    # σ.(...) + σ.(...) * tanh.(...)
    out .= out .+ tmp1

    return
end

function unfused_cuda_lstm_update_c(out, tmp1, tmp2, c,
                                    Wx_f, Wx_i, Wx_c,
                                    Rh_f, Rh_i, Rh_c,
                                    b_f,  b_i,  b_c)
    # σ.(Wx_f .+ Rh_f .+ b_f) .* c
    tmp1 .= Wx_f .+ Rh_f
    tmp1 .= tmp1 .+ b_f
    tmp1 .= σ.(tmp1)
    tmp1 .= tmp1 .* c

    # σ.(Wx_i .+ Rh_i .+ b_i)
    tmp2 = Wx_i .+ Rh_i
    tmp2 .= tmp2 .+ b_i
    tmp2 .= cuda_σ.(tmp2)

    # tanh.(Wx_c .+ Rh_c .+ b_c)
    out .= Wx_c .+ Rh_c
    out .= out .+ b_c
    out .= CUDAnative.tanh.(out)

    # σ.(...) * tanh.(...)
    out .= out .* tmp2

    # σ.(...) + σ.(...) * tanh.(...)
    out .= out .+ tmp1

    return
end
