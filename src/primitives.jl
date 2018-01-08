################
# Forward Pass #
################

#=== auto-defined methods ===#

const FORWARD_METHODS = [(Base.sum, 1), (Base.:+, 2), (Base.:*, 3)]

for (f, arity) in FORWARD_METHODS
    if arity == 1
        @eval begin
            $(f)(x::Record) = Record(x.tape, $f, x)
        end
    elseif arity == 2
        @eval begin
            $(f)(x::Record{T}, y::Record{T}) where {T} = Record(x.tape, $f, x, y)
            $(f)(x::Record{T}, y) where {T}            = Record(x.tape, $f, x, y)
            $(f)(x, y::Record{T}) where {T}            = Record(y.tape, $f, x, y)
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
                             args::Vararg{Union{AbstractArray,Record{T}},N}) where {N,T}
    # Get the tape, underlying values, and output element type from `args`
    tape = first(arg.tape for arg in args if isa(arg, Record))
    values = map(value, args)
    S = promote_type(map(eltype, values)...)

    # Construct a broadcast kernel from `f` that uses ForwardDiff to calculate
    # `f(args[1][i], args[2][i], ...)` and `∇f(args[1][i], args[2][i], ...)`
    # from a single elementwise application.
    template = DiffResults.GradientResult(zeros(SVector{N,S}))
    df = (y...) -> ForwardDiff.gradient!(template, x -> f(x...), SVector(y...))

    # Apply `df` elementwise to the underlying values
    results = broadcast(df, values...)
    output = Variable(map(DiffResults.value, results))

    # Record the instruction manually to the tape so that we can save `results`
    # along with `output`, since `results` contains the intermediate derivatives
    # that we'll propagate in the backwards pass.
    push!(tape, Instruction(broadcast, (f, args), (output, results)))
    return Record(tape, output)
end

##################
# Backwards Pass #
##################

backward!(::typeof(sum), x, y) = @propagate!(x, deriv(y))

function backward!(::typeof(*), x, y, z)
    @propagate!(x, A_mul_Bc(deriv(z), value(y)))
    @propagate!(y, Ac_mul_B(value(x), deriv(z)))
end
