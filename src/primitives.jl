#############################
# non-stdlib math functions #
#############################

d_tanh(x) = sech(x)^2

σ(x) = 1 / (1 + exp(-x))
d_σ(x) = (expx = exp(x); expx / (1 + expx)^2)

cuda_σ(x) = 1 / (1 + CUDAnative.exp(-x))
d_cuda_σ(x) = (expx = CUDAnative.exp(x); expx / (1 + expx)^2)

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

Base.BroadcastStyle(::Type{<:Record}) = Broadcast.Style{Record}()

function Broadcast.broadcast(f,
                             ::Broadcast.Style{Record},
                             ::Nothing,
                             ::Nothing,
                             args::Vararg{Union{AbstractArray,Record},N}) where {N}
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

forward!(i::BroadcastInstruction{<:Tuple{typeof(σ),Any}}) = invoke(forward!, Tuple{Instruction}, i)
forward!(i::BroadcastInstruction{<:Tuple{typeof(cuda_σ),Any}}) = invoke(forward!, Tuple{Instruction}, i)
forward!(i::BroadcastInstruction{<:Tuple{typeof(tanh),Any}}) = invoke(forward!, Tuple{Instruction}, i)
forward!(i::BroadcastInstruction{<:Tuple{typeof(cuda_tanh),Any}}) = invoke(forward!, Tuple{Instruction}, i)
forward!(i::BroadcastInstruction{<:Tuple{typeof(+),Any,Any}}) = invoke(forward!, Tuple{Instruction}, i)
forward!(i::BroadcastInstruction{<:Tuple{typeof(*),Any,Any}}) = invoke(forward!, Tuple{Instruction}, i)

#=== mixed-mode broadcast optimization ===#

# multiple dispatch selects this implementation for fused benchmarks

function forward!(i::BroadcastInstruction)
    output_variable = isa(i.output, Variable) ? i.output : first(i.output)
    f, input_values = first(i.input), value.(i.input[2:end])
    output_value = value(output_variable)
    output_duals = dual_eval_broadcast!(f, output_value, input_values)
    i.output = (output_variable, output_duals)
    return nothing
end

@noinline function dual_eval_broadcast!(kernel::K,
                                        output_value::AbstractMatrix,
                                        input_values::NTuple{N,<:AbstractMatrix}) where {K,N}
    @assert all(size(iv) === size(output_value) for iv in input_values)

    # Use ForwardDiff's `Dual` numbers to calculate `kernel.(input_values...)` and
    # elementwise derivatives of `kernel` at the same time (`output_duals` is an array
    # of dual numbers).
    output_duals = @fastsplat(broadcast(dual_eval, kernel, input_values...))

    # Load the value of the results into the output value buffer. Note that this assumes all
    # arguments have the same shape, which is not generally true for broadcast operations,
    # but is good enough for our performance experiments, since all of our test kernels
    # feature arguments of homogenous shape.
    map!(ForwardDiff.value, output_value, output_duals)
    return output_duals
end

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

for (f, df) in [(:σ, :d_σ), (:cuda_σ, :d_cuda_σ),
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

@inline function backprop_partial(input_deriv, output_dual, i, output_deriv)
    return input_deriv + (ForwardDiff.partials(output_dual, i) * output_deriv)
end

function backward!(i::BroadcastInstruction)
    f, args = first(i.input), i.input[2:end]
    output, output_duals = i.output
    output_deriv = deriv(output)
    for i in 1:length(args)
        arg_i_deriv = deriv(args[i])
        broadcast!(backprop_partial, arg_i_deriv, arg_i_deriv, output_duals, i, output_deriv)
    end
    return nothing
end
