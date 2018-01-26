###############
# Instruction #
###############

mutable struct Instruction{F,I<:Tuple}
    func::F
    input::I
    output::Any
end

const BroadcastInstruction{F} = Instruction{typeof(broadcast),T} where T <: Tuple{F,Vararg{Any}}

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
