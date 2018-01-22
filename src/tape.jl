###############
# Instruction #
###############

mutable struct Instruction{F}
    func::F
    input::Tuple
    output::Any
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

function forward!(tape::Tape)
    for i in tape.instructions
        forward!(i)::Nothing
    end
    return nothing
end

function backward!(tape::Tape)
    for i in Iterators.reverse(tape.instructions)
        backward!(i)::Nothing
    end
    return nothing
end
