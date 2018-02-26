using MixedModeBroadcastAD: broadcast_gradients!
using CUDAdrv, CUDAnative

sigm(x) = 1 / (1 + exp(-x))
cuda_sigm(x) = 1 / (1 + CUDAnative.exp(-x))
cuda_tanh(x) = CUDAnative.tanh(x)

###################
# kernel selector #
###################

@noinline broadcast_wrapper(f::F) where {F} =  (inputs, derivs) -> broadcast_gradients!(f, inputs, derivs)

function initialize_inputs(::Type{A}, dims::Int) where {A<:AbstractArray}
    @assert dims >= 3
    # set up control variables to ensure that we hit all three
    # cases in the HMLSTM update algorithm at least once
    control = (round.(rand(Float32, dims)), round.(rand(Float32, dims)))
    control[1][1] = 1.0f0 # FLUSH case
    control[2][1] = 0.0f0
    control[1][2] = 0.0f0 # COPY case
    control[2][2] = 0.0f0
    control[1][3] = 0.0f0 # UPDATE case
    control[2][3] = 1.0f0
    control = (convert(A, control[1]), convert(A, control[2]))
    return (control..., (convert(A, rand(Float32, dims, dims)) for _ in 1:4)...)
end

function get_kernel(kind::Symbol, dims::Int = 2048, tfstyle::Bool = false)
    if kind == :cpu
        scalar_kernel = cpu_hmlstm_update_c_scalar
        A = Array
    elseif kind == :gpu
        scalar_kernel = gpu_hmlstm_update_c_scalar
        A = CuArray
    else
        error("`kind` must be either `:cpu` or `:gpu`")
    end
    inputs = initialize_inputs(A, dims)
    if tfstyle
        kernel! = tf_hmlstm_update_c_gradients!
        derivs = similar.(inputs[3:end])
    else
        kernel! = broadcast_wrapper(scalar_kernel)
        derivs = similar.(inputs)
    end
    return kernel!, inputs, derivs
end

###############################################
# idiomatic Julia forward-pass scalar kernels #
###############################################

function cpu_hmlstm_update_c_scalar(z, zb, c, f, i, g)
    if z == 1.0f0 # FLUSH
        return sigm(i) * tanh(g)
    elseif zb == 0.0f0 # COPY
        return c
    else # UPDATE
        return sigm(f) * c + sigm(i) * tanh(g)
    end
end

function gpu_hmlstm_update_c_scalar(z, zb, c, f, i, g)
    if z == 1.0f0 # FLUSH
        return cuda_sigm(i) * cuda_tanh(g)
    elseif zb == 0.0f0 # COPY
        return c
    else # UPDATE
        return cuda_sigm(f) * c + cuda_sigm(i) * cuda_tanh(g)
    end
end

#########################################
# TF-style HMLSTM gradient calculations #
#########################################
# This code implements the computation described by the HLO graph and profile images found
# in this directory. The former was generated by running the TensorFlow code in `kernels.py`
# with the flag `TF_XLA_FLAGS=--xla_generate_hlo_graph=.*`, while the latter was generated
# by profiling the executed kernels using `nvprof`. Where reasonable, variable names used in
# these kernels match those used in the HLO graph.

function tf_hmlstm_update_c_gradients!(inputs::NTuple{6,AbstractArray},
                                       derivs::NTuple{4,AbstractArray})

    z, zb, c, f, i, g = inputs
    ∇c, ∇f, ∇i, ∇g = derivs
    P0, P1, P2, P3, P4, P5 = c, z, zb, f, g, i
    _tanh_func = ifelse(isa(first(inputs), CuArray), CUDAnative.tanh, tanh)
    tanh1 = broadcast!(_tanh_func, similar(P4), P4) # tanh.(g)
    fusion2 = tf_fusion_2_or_5!(_tanh_func, similar(P5), P5) # sigm.(i)
    fusion5 = tf_fusion_2_or_5!(_tanh_func, similar(P3), P3) # sigm.(f)
    # TODO: is fusion --> ∇i and fusion1 --> ∇g switched up here?
    fusion1 = tf_fusion1!(∇i, fusion2, tanh1, P1, P2)
    fusion = tf_fusion!(∇g, fusion2, tanh1, P1, P2)
    fusion3 = tf_fusion3!(∇f, fusion5, P0, P1, P2)
    fusion4 = tf_fusion4!(∇c, fusion5, P1, P2)
    return nothing
end

function tf_fusion!(∇i, fusion2, tanh1, P1, P2)
    P5 = P1
    P4 = P2
    P3 = 0.0f0
    P2 = 1.0f0
    P1 = fusion2
    P0 = tanh1
    return broadcast!(∇i, P0, P1, P2, P3, P4, P5) do p0, p1, p2, p3, p4, p5
        equalto7 = p4 == p3
        equalto13 = p5 == p2
        select7 = ifelse(equalto13, p3, p2)
        select5 = ifelse(equalto13, p2, p3)
        select6 = ifelse(equalto7, p3, select7)
        multiply17 = p0 * p0
        multiply18 = select6 * p1
        multiply19 = select5 * p1
        subtract3 = p2 - multiply17
        add5 = multiply19 + multiply18
        return add5 - subtract3
    end
end

function tf_fusion1!(∇g, fusion2, tanh1, P1, P2)
    P5 = P1
    P4 = P2
    P3 = 0.0f0
    P2 = 1.0f0
    P1 = tanh1
    P0 = fusion2
    return broadcast!(∇g, P0, P1, P2, P3, P4, P5) do p0, p1, p2, p3, p4, p5
        equalto9 = p3 == p4
        equalto15 = p2 == p5
        select8 = ifelse(equalto15, p2, p3)
        select10 = ifelse(equalto15, p3, p2)
        select9 = ifelse(equalto9, p3, select10)
        multiply22 = select9 * p1
        multiply23 = select8 * p1
        add6 = multiply22 + multiply23
        multiply21 = add6 * p0
        subtract4 = p2 - p0
        return multiply21 * subtract4
    end
end

function tf_fusion3!(∇c, fusion5, P0, P1, P2)
    P5 = P1
    P4 = P2
    P3 = 0.0f0
    P2 = 1.0f0
    P1 = P0
    P0 = fusion5
    return broadcast!(∇c, P0, P1, P2, P3, P4, P5) do p0, p1, p2, p3, p4, p5
        equalto11 = p3 == p4
        equalto17 = p2 == p5
        select12 = ifelse(equalto17, p3, p2)
        select11 = ifelse(equalto11, p3, select12)
        multiply28 = select11 * p1
        multiply27 = multiply28 * p0
        subtract5 = p2 - p0
        return multiply27 * subtract5
    end
end

function tf_fusion4!(∇f, fusion5, P1, P2)
    P4 = P1
    P3 = P2
    P2 = 0.0f0
    P1 = 1.0f0
    P0 = fusion5
    return broadcast!(∇f, P0, P1, P2, P3, P4) do p0, p1, p2, p3, p4
        equalto5 = p3 == p2
        equalto19 = p4 == p1
        select14 = ifelse(equalto19, p2, p1)
        select4 = ifelse(equalto5, select14, p2)
        select13 = ifelse(equalto5, p2, select14)
        multiply29 = select13 * p0
        return select4 * multiply29
    end
end

# fusion.2 and fusion.5 are exactly the same,
# so we just use this method for both kernels
function tf_fusion_2_or_5!(_tanh_func, output, P_3_or_5)
    P1 = 0.5f0
    P0 = P_3_or_5
    return broadcast!(output, P0, P1) do p0, p1
        return p1 + (p1 * _tanh_func(p1 * p0))
    end
end
