using MixedModeBroadcastAD: σ, cuda_σ, cuda_tanh, CuArray
import CUDAdrv

const cuda_lib = Libdl.dlopen(joinpath(@__DIR__, "kernels.so"))

#=
Since our technique only affects broadcast performance, we have limited our kernel to
execute only a small part of the LSTM layer which stresses fused broadcast operations. This
allows us to benchmark/test our technique in isolation from the other parts of our ad-hoc AD
framework which are not necessarily performant and only exist to facilitate the
demonstration of the broadcast technique.

In the below kernels, Julia will perform broadcast fusion automatically via the dot syntax.
Thus, we control where broadcast fusion happens by breaking up our dot syntax expressions
into separate statements. Note that fused, partially fused, and unfused versions of a kernel
are exactly the same otherwise.
=#

#######
# CPU #
#######

function cpu_unfused_lstm_update_c(c,
                                   Wx_f, Wx_i, Wx_c,
                                   Rh_f, Rh_i, Rh_c,
                                   b_f,  b_i,  b_c)
    tmp_f = Wx_f .+ Rh_f
    tmp_f = tmp_f .+ b_f
    tmp_f = σ.(tmp_f)

    tmp_i = Wx_i .+ Rh_i
    tmp_i = tmp_i .+ b_i
    tmp_i = σ.(tmp_i)

    tmp_c = Wx_c .+ Rh_c
    tmp_c = tmp_c .+ b_c
    tmp_c = tanh.(tmp_c)

    tmp_ic = tmp_i .* tmp_c
    tmp_fc = tmp_f .* c
    return tmp_ic + tmp_fc
end

function cpu_partially_fused_lstm_update_c(c,
                                           Wx_f, Wx_i, Wx_c,
                                           Rh_f, Rh_i, Rh_c,
                                           b_f,  b_i,  b_c)
    tmp = σ.(Wx_i .+ Rh_i .+ b_i) .* tanh.(Wx_c .+ Rh_c .+ b_c)
    return σ.(Wx_f .+ Rh_f .+ b_f) .* c .+ tmp
end

function cpu_fully_fused_lstm_update_c(c,
                                       Wx_f, Wx_i, Wx_c,
                                       Rh_f, Rh_i, Rh_c,
                                       b_f,  b_i,  b_c)
    return σ.(Wx_f .+ Rh_f .+ b_f) .* c .+
           σ.(Wx_i .+ Rh_i .+ b_i) .* tanh.(Wx_c .+ Rh_c .+ b_c)
end

#######
# GPU #
#######

#=== CUDAnative ===#

function cudanative_unfused_lstm_update_c(c,
                                          Wx_f, Wx_i, Wx_c,
                                          Rh_f, Rh_i, Rh_c,
                                          b_f,  b_i,  b_c)
    tmp_f = Wx_f .+ Rh_f
    tmp_f = tmp_f .+ b_f
    tmp_f = cuda_σ.(tmp_f)

    tmp_i = Wx_i .+ Rh_i
    tmp_i = tmp_i .+ b_i
    tmp_i = cuda_σ.(tmp_i)

    tmp_c = Wx_c .+ Rh_c
    tmp_c = tmp_c .+ b_c
    tmp_c = cuda_tanh.(tmp_c)

    tmp_ic = tmp_i .* tmp_c
    tmp_fc = tmp_f .* c
    out = tmp_ic + tmp_fc

    CUDAdrv.synchronize()
    return out
end

function cudanative_partially_fused_lstm_update_c(c,
                                                  Wx_f, Wx_i, Wx_c,
                                                  Rh_f, Rh_i, Rh_c,
                                                  b_f,  b_i,  b_c)
    tmp = cuda_σ.(Wx_i .+ Rh_i .+ b_i) .* cuda_tanh.(Wx_c .+ Rh_c .+ b_c)
    out = cuda_σ.(Wx_f .+ Rh_f .+ b_f) .* c .+ tmp

    CUDAdrv.synchronize()
    return out
end

function cudanative_fully_fused_lstm_update_c(c,
                                              Wx_f, Wx_i, Wx_c,
                                              Rh_f, Rh_i, Rh_c,
                                              b_f,  b_i,  b_c)
    out = cuda_σ.(Wx_f .+ Rh_f .+ b_f) .* c .+
          cuda_σ.(Wx_i .+ Rh_i .+ b_i) .* cuda_tanh.(Wx_c .+ Rh_c .+ b_c)

    CUDAdrv.synchronize()
    return out
end

#=== raw CUDA wrappers ===#

const cuda_fun_unfused = Libdl.dlsym(cuda_lib, "unfused_lstm_update_c")
function cudaraw_unfused_lstm_update_c(c,
                                    Wx_f, Wx_i, Wx_c,
                                    Rh_f, Rh_i, Rh_c,
                                    b_f,  b_i,  b_c)
    out = similar(c)
    temporaries = Tuple(similar(c) for i in 1:11)
    numElements = length(out)
    ccall(cuda_fun_unfused, Void,
          (Cint, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
           Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
           Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
           Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
           Ptr{Float32}, Ptr{Float32}, Ptr{Float32}),
          numElements, out, temporaries[1], temporaries[2], temporaries[3], temporaries[4],
          temporaries[5], temporaries[6], temporaries[7], temporaries[8], temporaries[9],
          temporaries[10], temporaries[11], c, Wx_f, Wx_i, Wx_c, Rh_f, Rh_i, Rh_c, b_f, b_i,
          b_c)

    CUDAdrv.synchronize()
    return out
end

const cuda_fun = Libdl.dlsym(cuda_lib, "lstm_update_c")
function cudaraw_fully_fused_lstm_update_c(c,
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

#############
# utilities #
#############

function getkernel(kind::Symbol, fusion_level::Int, dim::Int)
    if kind === :cpu
        kernels = (cpu_unfused_lstm_update_c,
                   cpu_partially_fused_lstm_update_c,
                   cpu_fully_fused_lstm_update_c)
        T = Array
    elseif kind === :cudanative
        kernels = (cudanative_unfused_lstm_update_c,
                   cudanative_partially_fused_lstm_update_c,
                   cudanative_fully_fused_lstm_update_c)
        T = CuArray
    elseif kind === :cudaraw
        fusion_level == 1 && error("partially fused kernel not yet implemented in raw CUDA")
        kernels = (cudaraw_unfused_lstm_update_c,
                   cudaraw_fully_fused_lstm_update_c,
                   cudaraw_fully_fused_lstm_update_c)
        T = CuArray
    end
    return kernels[fusion_level + 1], Tuple(T(rand(Float32, dim, dim)) for i in 1:10)
end
