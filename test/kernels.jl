using MixedModeBroadcastAD: CuArray, StructOfArrays, Record, Tape, Variable,
                            forward!, backward!, seed!, initderiv!, value, deriv,
                            sigm, cuda_sigm, cuda_tanh

########################
# fine-grained kernels #
########################

cpu_hmlstm_update_c(inputs...) = cpu_hmlstm_update_c_scalar.(inputs...)

function cpu_hmlstm_update_c_scalar(z_t, # = z_{t}^{l-1}
                                    z_l, # = z_{t-1}^{l}
                                    c, f, i, g)
    if z_l == 1 # FLUSH
        return sigm(i) * tanh(g)
    elseif z_t == 1 # UPDATE
        return sigm(f) * c + sigm(i) * tanh(g)
    else # COPY
        return c
    end
end

gpu_hmlstm_update_c(inputs...) = gpu_hmlstm_update_c_scalar.(inputs...)

function gpu_hmlstm_update_c_scalar(z_t, # = z_{t}^{l-1}
                                    z_l, # = z_{t-1}^{l}
                                    c, f, i, g)
    if z_l == 1 # FLUSH
        return cuda_sigm(i) * cuda_tanh(g)
    elseif z_t == 1 # UPDATE
        return cuda_sigm(f) * c + cuda_sigm(i) * cuda_tanh(g)
    else # COPY
        return c
    end
end

##########################
# coarse-grained kernels #
##########################

#=
TODO: port over something comparable to what hmlstm.py is doing
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

function autograd(f, input...; cache::Bool = false)
    tape, recorded_output, recorded_input = record(f, input...)
    forward!(tape)
    seed!(recorded_output)
    backward!(tape)
    if cache
        initderiv!(tape)
        forward!(tape)
        seed!(recorded_output)
        backward!(tape)
    end
    return (value(recorded_output), deriv.(recorded_input))
end

########################
# kernel/tape selector #
########################

tosoa(::Type{A}, x::AbstractArray{T,N}) where {A,T,N} = convert(StructOfArrays{T,N,A}, convert(StructOfArrays, x))

function getkernel(kind::Symbol, soa::Bool = true, dims::Int = 2048)
    @assert kind == :cpu || kind == :gpu
    if kind == :cpu
        kernel = cpu_hmlstm_update_c
        A = Array
    else
        kernel = gpu_hmlstm_update_c
        A = CuArray
    end
    bools = (convert(A, rand(Bool, dims)), convert(A, rand(Bool, dims)))
    n = 4
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
    forward!(tape) # "precompile" forwards pass
    backward!(tape) # "precompile" backwards pass
    return tape
end
