###############
# Instruction #
###############

mutable struct Instruction{F,I<:Tuple}
    func::F
    input::I
    output::Any
end

const BroadcastInstruction{I<:Tuple} = Instruction{typeof(broadcast),I}

########
# Tape #
########

struct Tape
    instructions::Vector{Instruction}
end

Tape() = Tape(Vector{Instruction}())

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
