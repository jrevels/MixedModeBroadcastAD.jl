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
    f, input_values = first(i.input), value.(i.input[2:end])
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
    @propagate!(x, deriv(y))
    return nothing
end

function backward!(i::Instruction{typeof(*),<:Tuple{Any,Any}})
    x, y = i.input
    z = i.output
    @propagate!(x, deriv(z) * value(y)')
    @propagate!(y, value(x)' * deriv(z))
    return nothing
end

function backward!(i::Instruction{typeof(+),<:Tuple{Any,Any}})
    x, y = i.input
    z = i.output
    @propagate!(x, deriv(z))
    @propagate!(y, deriv(z))
    return nothing
end

for (f, df) in [(:sigm, :d_sigm), (:cuda_sigm, :d_cuda_sigm),
                (:tanh, :d_tanh), (:cuda_tanh, :d_cuda_tanh)]
    @eval function backward!(i::BroadcastInstruction{<:Tuple{typeof($f),Any}})
        f, args = first(i.input), i.input[2:end]
        for arg in args
            @propagate!(arg, $(df).(value(arg)) .* deriv(i.output))
        end
        return nothing
    end
end

function backward!(i::BroadcastInstruction{<:Tuple{typeof(*),Any,Any}})
    _, x, y = i.input
    z = i.output
    @propagate!(x, value(y) .* deriv(z))
    @propagate!(y, value(x) .* deriv(z))
    return nothing
end

function backward!(i::BroadcastInstruction{<:Tuple{typeof(+),Any,Any}})
    _, x, y = i.input
    z = i.output
    @propagate!(x, deriv(z))
    @propagate!(y, deriv(z))
    return nothing
end

#=== mixed-mode broadcast optimization ===#

@inline inbounds_partials(d, i) = @inbounds ForwardDiff.partials(d, i)

@inline function backprop_partial(output_dual, i, output_deriv)
    return inbounds_partials(output_dual, i) * output_deriv
end

@inline function multivariable_backprop_partial(input_deriv, output_dual, i, output_deriv)
    return input_deriv + backprop_partial(output_dual, i, output_deriv)
end

@noinline function backprop_partial_broadcast!(arg_derivs::Tuple, multivariable, vars, output_duals, output_derivs)
    blk, thr = cuda_dimensions(output_duals)
    @cuda blocks=blk threads=thr _backprop_partial_broadcast!(arg_derivs, Val(multivariable), Val(vars), output_duals, output_derivs)
end

@generated function _backprop_partial_broadcast!(arg_derivs::Tuple, ::Val{multivariable}, ::Val{vars}, output_duals, output_derivs) where {multivariable, vars}
    evals = Expr(:block)
    for (i,j) in enumerate(vars)
        if multivariable[i]
            push!(evals.args, :(@inbounds arg_derivs[$i][I] = multivariable_backprop_partial(arg_derivs[$i][I], output_dual, $j, output_deriv)))
        else
            push!(evals.args, :(@inbounds arg_derivs[$i][I] = backprop_partial(output_dual, $j, output_deriv)))
        end
    end

    quote
        let I = @cuda_index(output_duals) # FIXME: assumes equal shape, size, etc
            @inbounds output_dual = output_duals[I]
            @inbounds output_deriv = output_derivs[I]
            $evals
        end
        return
    end
end

function backward!(i::BroadcastInstruction)
    f, args = first(i.input), i.input[2:end]
    outputs, output_duals = i.output
    output_derivs = deriv(outputs)
    if isa(output_derivs, CuArray) ||
       isa(output_derivs, StructOfArrays{T,N,A} where {T,N,A<:CuArray}) # FIXME: dispatch
        vars = Tuple(i for i in 1:length(args) if isa(args[i], Variable))
        multivariable = Tuple(args[i].downstreams > 1  for i in vars)
        arg_derivs = Tuple(deriv(args[i]) for i in vars)
        backprop_partial_broadcast!(arg_derivs, multivariable, vars, output_duals, output_derivs)
    else
        for (i, arg) in enumerate(args)
            isa(arg, Variable) || continue
            if arg.downstreams > 1
                broadcast!(multivariable_backprop_partial, deriv(arg), deriv(arg), output_duals, i, output_derivs)
            else
                broadcast!(backprop_partial, deriv(arg), output_duals, i, output_derivs)
            end
        end
    end
    return nothing
end
