
#=
Since our technique only affects broadcast performance, we have limited our kernel to
execute only the part of the LSTM layer that stresses fused broadcast operations. This
allows us tobenchmark/test our technique in isolation from the other parts of our ad-hoc
AD framework which are not necessarily performant and only exist to facilitate the
demonstration of the broadcast technique.
=#

σ(x) = 1 / (1 + exp(-x))

function lstm_update(c,
                     Wx_f, Wx_i, Wx_c, Wx_o,
                     Rh_f, Rh_i, Rh_c, Rh_o,
                     b_f,  b_i,  b_c,  b_o)
    c = σ.(Wx_f .+ Rh_f .+ b_f) .* c .+ σ.(Wx_i .+ Rh_i .+ b_i) .* tanh.(Wx_c .+ Rh_c .+ b_c)
    h = σ.(Wx_o .+ Rh_o .+ b_o) .* tanh.(c)
    return c, h
end

# TODO
function cuda_lstm_update end
