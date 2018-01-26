using MixedModeBroadcastAD: σ, cuda_σ, cuda_tanh
import CUDAdrv

const cuda_lib = Libdl.dlopen(joinpath(@__DIR__, "kernels.so"))

#=
Since our technique only affects broadcast performance, we have limited our kernel to
execute only a small part of the LSTM layer which stresses fused broadcast operations. This
allows us to benchmark/test our technique in isolation from the other parts of our ad-hoc AD
framework which are not necessarily performant and only exist to facilitate the
demonstration of the broadcast technique.
=#


#
# Fused
#

function lstm_update_c(c,
                       Wx_f, Wx_i, Wx_c,
                       Rh_f, Rh_i, Rh_c,
                       b_f,  b_i,  b_c)
    return σ.(Wx_f .+ Rh_f .+ b_f) .* c .+
           σ.(Wx_i .+ Rh_i .+ b_i) .* tanh.(Wx_c .+ Rh_c .+ b_c)
end

function cudanative_lstm_update_c(c,
                                  Wx_f, Wx_i, Wx_c,
                                  Rh_f, Rh_i, Rh_c,
                                  b_f,  b_i,  b_c)
    out =  cuda_σ.(Wx_f .+ Rh_f .+ b_f) .* c .+
           cuda_σ.(Wx_i .+ Rh_i .+ b_i) .* cuda_tanh.(Wx_c .+ Rh_c .+ b_c)
    CUDAdrv.synchronize()
    return out
end

const cuda_fun = Libdl.dlsym(cuda_lib, "lstm_update_c")
function cuda_lstm_update_c(c,
                            Wx_f, Wx_i, Wx_c,
                            Rh_f, Rh_i, Rh_c,
                            b_f,  b_i,  b_c)
    out = similar(c)
    numElements = length(out)
    ccall(cuda_fun, Void,
          (Cint, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
           Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
           Ptr{Float32}, Ptr{Float32}),
          numElements, out, c, Wx_f, Wx_i, Wx_c, Rh_f, Rh_i, Rh_c, b_f, b_i, b_c)
    CUDAdrv.synchronize()
    return out
end


#
# Unfused
#

#=
Manually call `broadcast` instead of using dot syntax to avoid Julia's automatic syntatic
fusion. This is exactly what would be called in the above kernels if Julia didn't perform
syntatic fusion automatically.
=#

function unfused_lstm_update_c(c,
                               Wx_f, Wx_i, Wx_c,
                               Rh_f, Rh_i, Rh_c,
                               b_f,  b_i,  b_c)
    return broadcast(+,
                     broadcast(*,
                               broadcast(σ, broadcast(+, Wx_f, Rh_f, b_f)),
                               c),
                     broadcast(*,
                               broadcast(σ,    broadcast(+, Wx_i, Rh_i, b_i)),
                               broadcast(tanh, broadcast(+, Wx_c, Rh_c, b_c))))
end

function unfused_cudanative_lstm_update_c(c,
                                          Wx_f, Wx_i, Wx_c,
                                          Rh_f, Rh_i, Rh_c,
                                          b_f,  b_i,  b_c)
    out = broadcast(+,
                    broadcast(*,
                              broadcast(cuda_σ, broadcast(+, Wx_f, Rh_f, b_f)),
                              c),
                    broadcast(*,
                              broadcast(cuda_σ,    broadcast(+, Wx_i, Rh_i, b_i)),
                              broadcast(cuda_tanh, broadcast(+, Wx_c, Rh_c, b_c))))
    CUDAdrv.synchronize()
    return out
end

const cuda_fun_unfused = Libdl.dlsym(cuda_lib, "unfused_lstm_update_c")
function unfused_cuda_lstm_update_c(c,
                                    Wx_f, Wx_i, Wx_c,
                                    Rh_f, Rh_i, Rh_c,
                                    b_f,  b_i,  b_c)
    out = similar(c)
    temporaries = Tuple(similar(c) for i in 1:8)
    numElements = length(out)
    ccall(cuda_fun_unfused, Void,
          (Cint, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
           Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
           Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
           Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}),
          numElements, out, temporaries[1], temporaries[2], temporaries[3], temporaries[4],
          temporaries[5], temporaries[6], temporaries[7], temporaries[8],
          c, Wx_f, Wx_i, Wx_c, Rh_f, Rh_i, Rh_c, b_f, b_i, b_c)

    CUDAdrv.synchronize()
    return out
end
