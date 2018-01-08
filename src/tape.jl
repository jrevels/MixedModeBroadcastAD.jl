# originally copied from https://github.com/JuliaDiff/Capstan.jl

###############
# Instruction #
###############

struct Instruction{F,I<:Tuple,O}
    func::F
    input::I
    output::O
end

Instruction(f, input, output) = Instruction(f, tuplize(input), output)

function backward!(instr::Instruction)
    backward!(instr.func, instr.input..., instr.output)
    return nothing
end

########
# Tape #
########

struct Tape{tag}
    instructions::Vector{Instruction}
end

Tape() = Tape{gensym()}(Vector{Instruction}())

Base.push!(t::Tape, i::Instruction) = push!(t.instructions, i)

Base.empty!(t::Tape) = (empty!(t.instructions); t)

function backward!(tape::Tape)
    for instr in Iterators.reverse(tape.instructions)
        backward!(instr)::Nothing
    end
    return nothing
end
