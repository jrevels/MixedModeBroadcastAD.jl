#=
Since our technique only affects broadcast performance, we have limited our kernel to
execute only a small part of the LSTM layer which stresses fused broadcast operations. This
allows us tobenchmark/test our technique in isolation from the other parts of our ad-hoc
AD framework which are not necessarily performant and only exist to facilitate the
demonstration of the broadcast technique.
=#

import CUDAdrv


#
# Primitives
#

const cuda_lib = Libdl.dlopen(joinpath(@__DIR__, "kernels.so"))

σ(x) = 1 / (1 + exp(-x))
cuda_σ(x) = 1 / (1 + CUDAnative.exp(-x))


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
           cuda_σ.(Wx_i .+ Rh_i .+ b_i) .* CUDAnative.tanh.(Wx_c .+ Rh_c .+ b_c)
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

function unfused_lstm_update_c(tmp1, tmp2, c,
                               Wx_f, Wx_i, Wx_c,
                               Rh_f, Rh_i, Rh_c,
                               b_f,  b_i,  b_c)
    out = similar(c)

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

    return out
end

function unfused_cudanative_lstm_update_c(tmp1, tmp2, c,
                                          Wx_f, Wx_i, Wx_c,
                                          Rh_f, Rh_i, Rh_c,
                                          b_f,  b_i,  b_c)
    out = similar(c)

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

    CUDAdrv.synchronize()
    return out
end

const cuda_fun_unfused = Libdl.dlsym(cuda_lib, "unfused_lstm_update_c")
function unfused_cuda_lstm_update_c(tmp1, tmp2, c,
                                    Wx_f, Wx_i, Wx_c,
                                    Rh_f, Rh_i, Rh_c,
                                    b_f,  b_i,  b_c)
    out = CuArray{Float32}(size(c))
    numElements = length(out)
    ccall(cuda_fun_unfused, Void,
          (Cint, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
           Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
           Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}),
          numElements, out, tmp1, tmp2, c, Wx_f, Wx_i, Wx_c, Rh_f, Rh_i, Rh_c, b_f, b_i, b_c)

    CUDAdrv.synchronize()
    return out
end
