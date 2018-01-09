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

function Base.show(io::IO, inst::Instruction)
    print(io, typeof(inst.func).name.mt.name, "(")
    for (i, x) in enumerate(inst.input)
        print(io, x)
        if i < length(inst.input)
            print(io, ", ")
        end
    end
    print(io, ") = $(inst.output)")
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

function Base.show(io::IO, tape::Tape)
    print(io, "Tape(")
    if length(tape.instructions) > 0
        for (i, inst) in enumerate(tape.instructions)
            if i > 1
                println(io)
                print(io, "     ")
            end
            print(io, inst)
        end
    end
    print(io, ")")
end