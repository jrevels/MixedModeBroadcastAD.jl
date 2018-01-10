################
# Forward Pass #
################

#=== auto-defined methods ===#

const FORWARD_METHODS = [(:(Base.sum), 1), (:(Base.:*), 2)]

for (f, arity) in FORWARD_METHODS
    if arity == 1
        @eval begin
            $(f)(x::Record) = Record(x.tape, $f, x)
        end
    elseif arity == 2
        @eval begin
            $(f)(x::Record{tag}, y::Record{tag}) where {tag} = Record(x.tape, $f, x, y)
            $(f)(x::Record{tag}, y) where {tag}            = Record(x.tape, $f, x, y)
            $(f)(x, y::Record{tag}) where {tag}            = Record(y.tape, $f, x, y)
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
    # Get the tape, underlying values, and output element type from `args`
    tape = first(arg.tape for arg in args if isa(arg, Record))
    values = map(value, args)
    S = promote_type(map(eltype, values)...)

    # Construct a broadcast kernel from `f` that uses ForwardDiff to calculate
    # `f(args[1][i], args[2][i], ...)` and `∇f(args[1][i], args[2][i], ...)`
    # from a single elementwise application.
    template = DiffResults.GradientResult(zeros(SVector{N,S}))
    nargs = length(values)
    df = if nargs == 4
        (y1, y2, y3, y4) -> ForwardDiff.gradient!(template, x -> f(x[1], x[2], x[3], x[4]), SVector(y1, y2, y3, y4))
    else
        warn("$nargs-arg splat not optimized; this will yield a GPU-incompatible apply")
        (y...) -> ForwardDiff.gradient!(template, x -> f(x...), SVector(y...))
    end

    # Apply `df` elementwise to the underlying values
    allresults = broadcast(df, values...)
    output_variable = Variable(map(DiffResults.value, allresults))

    # Record the instruction manually to the tape so that we can save `results`
    # along with `output`, since `allresults` contains the intermediate derivatives
    # that we'll propagate in the backwards pass.
    input_variables = map(x -> isa(x, Record) ? x.variable : x, args)
    push!(tape, Instruction(broadcast, (f, input_variables), (output_variable, allresults)))
    return Record(tape, output_variable)
end

##################
# Backwards Pass #
##################

getpartial(x, i) = DiffResults.derivative(x)[i]

# This broadcast `backward!` implementation is actually incomplete, but it doesn't matter
# for our performance experiment. Specifically, it doesn't implement the proper reduction
# and expansion semantics encountered when the arguments have different shapes. In
# words, this implementation only works when all broadcast arguments are arrays of
# the same shape (which is what we're benchmarking anyway).
function backward!(::typeof(broadcast), f, input, output_and_allresults)
    output, allresults = output_and_allresults
    for i in 1:length(input)
        @propagate!(input[i], getpartial.(allresults, i) .* deriv(output))
    end
end

backward!(::typeof(sum), x, y) = @propagate!(x, deriv(y))

function backward!(::typeof(*), x, y, z)
    @propagate!(x, deriv(z) * value(y)')
    @propagate!(y, value(x)' * deriv(z))
end
