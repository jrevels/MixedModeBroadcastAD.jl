using MixedModeBroadcastAD: Tape, Variable, Record, value, deriv, backward!, seed!, CuArray
using BenchmarkTools
using CUDAnative
using Printf

function autograd(tape::Tape, f, input...)
    recorded_input = map(x -> Record(tape, Variable(x)), input)
    recorded_output = f(recorded_input...)
    seed!(recorded_output)
    backward!(tape)
    return (value(recorded_output), map(deriv, recorded_input))
end

function example(a1, a2, b1, b2, c1, c2, d1, d2)
    a = a1 * a2
    b = b1 * b2
    c = c1 * c2
    d = d1 * d2
    return sum(sin.(a) .+ log.(b) .+ exp.(c) .+ cos.(d))
end
function cuda_example(a1, a2, b1, b2, c1, c2, d1, d2)
    a = a1 * a2
    b = b1 * b2
    c = c1 * c2
    d = d1 * d2
    return sum(CUDAnative.sin.(a) .+ CUDAnative.log.(b) .+
               CUDAnative.exp.(c) .+ CUDAnative.cos.(d))
end
const examples = Dict(
    Array   => example,
    CuArray => cuda_example
)

for N in 2.^(1:11)
    input = Tuple(rand(Float32, N, N) for i in 1:8)

    for T in [Array, CuArray]
        tape = Tape()
        output, grads = autograd(tape, examples[T], T.(input)...)
        Array.(grads)
        b = @benchmarkable begin
                output, grads = autograd(tape, fun, input_T...)
                Array.(grads)
            end setup=(
                tape = Tape();
                fun = $examples[$T];
                input_T = $T.($input)
            )
        @printf("%-24s", "$T $(N)x$(N):")
        println(run(b))
    end
    println()
end
