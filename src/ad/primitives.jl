#############################
# non-stdlib math functions #
#############################

d_tanh(x) = sech(x)^2

sigm(x) = 1 / (1 + exp(-x))
d_sigm(x) = (expx = exp(x); expx / (1 + expx)^2)

cuda_sigm(x) = 1 / (1 + CUDAnative.exp(-x))
d_cuda_sigm(x) = (expx = CUDAnative.exp(x); expx / (1 + expx)^2)

cuda_tanh(x) = CUDAnative.tanh(x)
d_cuda_tanh(x) = 1 - CUDAnative.tanh(x)^2

###############
# Record Pass #
###############

#=== auto-defined methods ===#

const FORWARD_METHODS = [(:(Base.sum), 1), (:(Base.:*), 2), (:(Base.:+), 2)]

for (f, arity) in FORWARD_METHODS
    if arity == 1
        @eval begin
            $(f)(x::Record) = Record(x.tape, $f, x)
        end
    elseif arity == 2
        @eval begin
            $(f)(x::Record, y::Record) = Record(x.tape, $f, x, y)
            $(f)(x::Record, y) = Record(x.tape, $f, x, y)
            $(f)(x, y::Record) = Record(y.tape, $f, x, y)
        end
    else
        error("unsupported arity $arity for method $f")
    end
end

#=== broadcast ===#

struct RecordArrayStyle <: Broadcast.AbstractArrayStyle{Any} end
struct RecordOtherStyle <: Broadcast.BroadcastStyle end

Broadcast.BroadcastStyle(::Type{<:Record}) = RecordOtherStyle()
Broadcast.BroadcastStyle(::Type{<:Record{<:AbstractArray}}) = RecordArrayStyle()

function Broadcast.broadcast(f,
                             ::Union{RecordArrayStyle,RecordOtherStyle},
                             ::Nothing,
                             ::Nothing,
                             args::Union{AbstractArray,Number,Record}...)
    tape = first(arg.tape for arg in args if isa(arg, Record))
    return Record(tape, broadcast, f, args...)
end

################
# Forward Pass #
################

#=== generic fallback ===#

function forward!(i::Instruction)
    i.output.value = i.func(map(value, i.input)...)
    return nothing
end

#=== broadcast fallbacks ===#

# multiple dispatch selects these implementations for the unfused benchmarks

forward!(i::BroadcastInstruction{<:Tuple{typeof(sigm),Any}}) = invoke(forward!, Tuple{Instruction}, i)
forward!(i::BroadcastInstruction{<:Tuple{typeof(cuda_sigm),Any}}) = invoke(forward!, Tuple{Instruction}, i)
forward!(i::BroadcastInstruction{<:Tuple{typeof(tanh),Any}}) = invoke(forward!, Tuple{Instruction}, i)
forward!(i::BroadcastInstruction{<:Tuple{typeof(cuda_tanh),Any}}) = invoke(forward!, Tuple{Instruction}, i)
forward!(i::BroadcastInstruction{<:Tuple{typeof(+),Any,Any}}) = invoke(forward!, Tuple{Instruction}, i)
forward!(i::BroadcastInstruction{<:Tuple{typeof(*),Any,Any}}) = invoke(forward!, Tuple{Instruction}, i)

#=== mixed-mode broadcast optimization ===#

# multiple dispatch selects this implementation for fused benchmarks

function forward!(i::BroadcastInstruction)
    f, _input_values = first(i.input), value.(i.input[2:end])
    input_values = map(readonly, _input_values)
    if isa(i.output, Tuple) # we have pre-cached memory we can reuse
        output_variable, output_duals = i.output
        dual_eval_broadcast!(f, output_duals, value(output_variable), input_values)
    else
        @assert isa(i.output, Variable)
        output_variable = i.output
        i.output = (output_variable, dual_eval_broadcast!(f, value(output_variable), input_values))
    end
    return nothing
end

@noinline function dual_eval_broadcast!(kernel::K,
                                        output_value::AbstractArray,
                                        input_values::NTuple{N,AbstractArray}) where {K,N}
    # Use ForwardDiff's `Dual` numbers to calculate `kernel.(input_values...)` and
    # elementwise derivatives of `kernel` at the same time (`output_duals` is an array
    # of dual numbers).
    output_duals = @fastsplat(broadcast(dual_eval, kernel, input_values...))

    # Load the value of the results into the output value buffer.
    map!(ForwardDiff.value, output_value, output_duals)
    return output_duals
end

# version of dual_eval_broadcast! that reuses dual number buffer
@noinline function dual_eval_broadcast!(kernel::K,
                                        output_duals,
                                        output_value::AbstractArray,
                                        input_values::NTuple{N,AbstractArray}) where {K,N}
    @fastsplat(broadcast!(dual_eval, output_duals, kernel, input_values...))
    map!(ForwardDiff.value, output_value, output_duals)
    return nothing
end

# TODO: don't dualize Bool inputs
@inline function dual_eval(f::F, inputs...) where {F}
    dual_inputs = ForwardDiff.dualize(Nothing, StaticArrays.SVector(inputs))
    return @fastsplat(f(dual_inputs...))
end

##################
# Backwards Pass #
##################

#=== standard reverse definitions ===#

function backward!(i::Instruction{typeof(sum),<:Tuple{Any}})
    x = first(i.input)
    y = i.output
    x.deriv .+= deriv(y)
    return nothing
end

function backward!(i::Instruction{typeof(*),<:Tuple{Any,Any}})
    x, y = i.input
    z = i.output
    x.deriv .+= deriv(z) * value(y)'
    y.deriv .+= value(x)' * deriv(z)
    return nothing
end

function backward!(i::Instruction{typeof(+),<:Tuple{Any,Any}})
    x, y = i.input
    z = i.output
    x.deriv .+= deriv(z)
    y.deriv .+= deriv(z)
    return nothing
end

for (f, df) in [(:sigm, :d_sigm), (:cuda_sigm, :d_cuda_sigm),
                (:tanh, :d_tanh), (:cuda_tanh, :d_cuda_tanh)]
    @eval function backward!(i::BroadcastInstruction{<:Tuple{typeof($f),Any}})
        f, args = first(i.input), i.input[2:end]
        for j in 1:length(args)
            args[j].deriv .+= $(df).(value(args[j])) .* deriv(i.output)
        end
        return nothing
    end
end

function backward!(i::BroadcastInstruction{<:Tuple{typeof(*),Any,Any}})
    _, x, y = i.input
    z = i.output
    x.deriv .+= value(y) .* deriv(z)
    y.deriv .+= value(x) .* deriv(z)
    return nothing
end

function backward!(i::BroadcastInstruction{<:Tuple{typeof(+),Any,Any}})
    _, x, y = i.input
    z = i.output
    x.deriv .+= deriv(z)
    y.deriv .+= deriv(z)
    return nothing
end

#=== mixed-mode broadcast optimization ===#

@inline inbounds_partials(d, i) = @inbounds ForwardDiff.partials(d, i)

@inline function backprop_partial(input_deriv, output_dual, ::Val{i}, output_deriv) where i
    return input_deriv + (inbounds_partials(output_dual, i) * output_deriv)
end

function backward!(i::BroadcastInstruction)
    f, args = first(i.input), i.input[2:end]
    output, _output_duals = i.output
    output_duals = readonly(_output_duals)
    output_deriv = readonly(deriv(output))
    for (i, arg) in enumerate(args)
        isa(arg, Variable) || continue
        arg_deriv = deriv(arg)
        broadcast!(backprop_partial, arg_deriv, arg_deriv, output_duals, Val(i), output_deriv)
    end
    return nothing
end
