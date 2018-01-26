#############################
# non-stdlib math functions #
#############################

σ(x) = 1 / (1 + exp(-x))
d_σ(x) = (expx = exp(x); expx / (1 + expx)^2)

cuda_σ(x) = 1 / (1 + CUDAnative.exp(-x))
d_cuda_σ(x) = (expx = CUDAnative.exp(x); expx / (1 + expx)^2)

d_tanh(x) = sech(x)^2

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
            $(f)(x::Record{tag}, y::Record{tag}) where {tag} = Record(x.tape, $f, x, y)
            $(f)(x::Record{tag}, y) where {tag}              = Record(x.tape, $f, x, y)
            $(f)(x, y::Record{tag}) where {tag}              = Record(y.tape, $f, x, y)
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
                             args::Vararg{Union{AbstractArray,Record{tag}},N}) where {N,tag}
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

forward!(i::BroadcastInstruction{typeof(σ)}) = invoke(forward!, Tuple{Instruction}, i)
forward!(i::BroadcastInstruction{typeof(cuda_σ)}) = invoke(forward!, Tuple{Instruction}, i)
forward!(i::BroadcastInstruction{typeof(tanh)}) = invoke(forward!, Tuple{Instruction}, i)
forward!(i::BroadcastInstruction{typeof(cuda_tanh)}) = invoke(forward!, Tuple{Instruction}, i)
forward!(i::BroadcastInstruction{typeof(+)}) = invoke(forward!, Tuple{Instruction}, i)
forward!(i::BroadcastInstruction{typeof(*)}) = invoke(forward!, Tuple{Instruction}, i)

#=== mixed-mode broadcast optimization ===#

# multiple dispatch selects this implementation for fused benchmarks

function forward!(i::BroadcastInstruction)
    output_variable = isa(i.output, Variable) ? i.output : first(i.output)
    f, values = first(i.input), value.(i.input[2:end])
    N, T = length(values), eltype(value(output_variable))
    gradient_template = DiffResults.GradientResult(zeros(SVector{N,T}))
    i.output = _forward_broadcast!(f, output_variable, values, gradient_template)
    return nothing
end

@noinline function _forward_broadcast!(f, output_variable, values, gradient_template)
    # Use ForwardDiff to calculate `f(values[1][i], values[2][i], ...)` and
    # `∇f(values[1][i], values[2][i], ...)` from a single elementwise application.
    df = (y...) -> ForwardDiff.gradient!(gradient_template,
                                         x -> @fastsplat(f(x...)),
                                         @fastsplat(SVector(y...)))
    df_results = df.(values...)

    # `df_results` is a `AbstractArray{DiffResults.DiffResult}`, i.e. each element of
    # `df_results` corresponds to a primal value and intermediate gradient. We record
    # both the values and the gradients back to the tape, which will be used in the
    # backwards pass.
    map!(DiffResults.value, output_variable.value, df_results)
    return (output_variable, df_results)
end

##################
# Backwards Pass #
##################

#=== standard reverse definitions ===#

function backward!(i::Instruction{typeof(sum)})
    x = first(i.input)
    y = i.output
    x.deriv .+= deriv(y)
    return nothing
end

function backward!(i::Instruction{typeof(*)})
    x, y = i.input
    z = i.output
    x.deriv .+= deriv(z) * value(y)'
    y.deriv .+= value(x)' * deriv(z)
    return nothing
end

function backward!(i::Instruction{typeof(+)})
    x, y = i.input
    z = i.output
    x.deriv .+= deriv(z)
    y.deriv .+= deriv(z)
    return nothing
end

for (f, df) in [(:σ, :d_σ), (:cuda_σ, :d_cuda_σ),
                (:tanh, :d_tanh), (:cuda_tanh, :d_cuda_tanh)]
    @eval function backward!(i::BroadcastInstruction{typeof($f)})
        f, args = first(i.input), i.input[2:end]
        for j in 1:length(args)
            args[j].deriv .+= $(df).(value(args[j])) .* deriv(i.output)
        end
        return nothing
    end
end

function backward!(i::BroadcastInstruction{typeof(*)})
    _, x, y = i.input
    z = i.output
    x.deriv .+= value(y) .* deriv(z)
    y.deriv .+= value(x) .* deriv(z)
    return nothing
end

function backward!(i::BroadcastInstruction{typeof(+)})
    _, x, y = i.input
    z = i.output
    x.deriv .+= deriv(z)
    y.deriv .+= deriv(z)
    return nothing
end

#=== mixed-mode broadcast optimization ===#

# FIXME: this `@inbounds` is incorrectly placed, but it doesn't seem to have any impact
#        when placed at the `getpartial` call site
Base.@propagate_inbounds getpartial(x, i) = @inbounds DiffResults.derivative(x)[i]

# This broadcast `backward!` implementation is actually incomplete, but it doesn't matter
# for our performance experiment. Specifically, it doesn't implement the proper reduction
# and expansion semantics encountered when the arguments have different shapes. In other
# words, this implementation only works when all broadcast arguments are arrays of the same
# shape (which is sufficient for our benchmarking purposes).
function backward!(i::BroadcastInstruction)
    f, args = first(i.input), i.input[2:end]
    output, df_results = i.output
    for i in 1:length(args)
        args[i].deriv .+= getpartial.(df_results, i) .* deriv(output)
    end
    return nothing
end
