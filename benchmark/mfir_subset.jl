module MFIR

# Benchmark-only MFIR-compatible subset.  The operation names and program
# semantics mirror MultiFloats.jl/scripts/MFIR.jl closely enough to model the
# fused x3 arithmetic tail without making scripts/ part of the runtime package.

export MFIROperation, MFIR_ADD, MFIR_TWO_SUM, MFIR_FAST_TWO_SUM,
       MFIR_MUL, MFIR_FMA, MFIR_TWO_PROD,
       MFIRInstruction, MFIRProgram, arity, num_outputs,
       num_registers, use_counts

@enum MFIROperation::UInt16 begin
    MFIR_ADD
    MFIR_TWO_SUM
    MFIR_FAST_TWO_SUM
    MFIR_MUL
    MFIR_FMA
    MFIR_TWO_PROD
end

@inline function arity(op::MFIROperation)
    op == MFIR_FMA && return 3
    return 2
end

@inline function num_outputs(op::MFIROperation)
    return ((op == MFIR_TWO_SUM) | (op == MFIR_FAST_TWO_SUM) |
            (op == MFIR_TWO_PROD)) ? 2 : 1
end

struct MFIRInstruction
    op::MFIROperation
    args::NTuple{3,UInt16}
end

const NULL_ARG = zero(UInt16)

@inline MFIRInstruction(op::MFIROperation, a::Integer, b::Integer) =
    MFIRInstruction(op, (UInt16(a), UInt16(b), NULL_ARG))

@inline MFIRInstruction(op::MFIROperation, a::Integer, b::Integer, c::Integer) =
    MFIRInstruction(op, (UInt16(a), UInt16(b), UInt16(c)))

@inline arity(instruction::MFIRInstruction) = arity(instruction.op)
@inline num_outputs(instruction::MFIRInstruction) = num_outputs(instruction.op)

struct MFIRProgram
    num_inputs::Int
    instructions::Vector{MFIRInstruction}
    result_ranges::Vector{UnitRange{UInt16}}
    output_indices::Vector{UInt16}
end

function MFIRProgram(
    num_inputs::Integer,
    instructions::Vector{MFIRInstruction},
    output_indices::AbstractVector{UInt16},
)
    next_register = Int(num_inputs)
    ranges = Vector{UnitRange{UInt16}}(undef, length(instructions))
    @inbounds for i in eachindex(instructions)
        lo = next_register + 1
        next_register += num_outputs(instructions[i])
        ranges[i] = UInt16(lo):UInt16(next_register)
    end
    return MFIRProgram(Int(num_inputs), instructions, ranges, collect(output_indices))
end

@inline num_registers(program::MFIRProgram) =
    isempty(program.result_ranges) ? program.num_inputs : Int(last(program.result_ranges[end]))

function use_counts(program::MFIRProgram)
    counts = zeros(Int, num_registers(program))
    @inbounds for instruction in program.instructions
        for j in 1:arity(instruction)
            counts[Int(instruction.args[j])] += 1
        end
    end
    @inbounds for output in program.output_indices
        counts[Int(output)] += 1
    end
    return counts
end

end # module
