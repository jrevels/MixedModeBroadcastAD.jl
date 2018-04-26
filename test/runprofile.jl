import CUDAdrv
import NVTX

include("kernels.jl")

# NOTE: use with `--profile-from-start off`
NVTX.stop()

const TFSTYLE = length(ARGS) >= 1 ? parse(Bool, ARGS[1]) : false
const DIMS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2048
const ITERATIONS = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
const ARITY = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 0

ARITY > 0 && @assert !TFSTYLE
const KERNEL!, INPUTS, DERIVS, BUFFERS = if ARITY == 0
    get_hmlstm_kernel(TFSTYLE, true, DIMS)
else
    get_arity_scaling_kernel(true, DIMS, ARITY)
end

# to mimick TF, we'll copy back the results to the CPU.
# preallocate a page-locked memory buffer to avoid API call overhead.
# TODO: wrap in CUDAdrv
const buffer = Ref{Ptr{Cvoid}}()
const buffer_size = DIMS*DIMS*sizeof(Float32)
CUDAdrv.@apicall(:cuMemAllocHost, (Ptr{Ptr{Cvoid}}, Csize_t), buffer, buffer_size)

function benchmark(kernel!, inputs, derivs, buffers)
    NVTX.@range "kernel" begin
        kernel!(inputs, derivs, buffers)
        for i in 1:4
            # we download results asynchronously to avoid API overhead
            # TODO: `Base.copyto!(; async)` in CUDAdrv
            @assert buffer_size == sizeof(DERIVS[i])
            CUDAdrv.Mem.download!(buffer[], DERIVS[i].buf, buffer_size; async=true)
        end
        CUDAdrv.synchronize()
    end
end

benchmark(KERNEL!, INPUTS, DERIVS, BUFFERS) # warmup

NVTX.@activate CUDAdrv.@profile begin
    ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), STDERR.handle)
    for i in 1:ITERATIONS
        GC.gc()
        benchmark(KERNEL!, INPUTS, DERIVS, BUFFERS)
    end
    ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), C_NULL)
end

CUDAdrv.@apicall(:cuMemFreeHost, (Ptr{Cvoid},), buffer[])
