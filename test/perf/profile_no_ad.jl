using MixedModeBroadcastAD: forward!, backward!
using CUDAnative
import CUDAdrv
import NVTX
include("util.jl")

# NOTE: use with `--profile-from-start off`
NVTX.stop()

const N            = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2048
const fusion_level = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
const soa          = length(ARGS) >= 3 ? parse(Bool, ARGS[3]) : true
const cudaraw_resources = getkernel(:cudaraw, fusion_level, N, false)
const cudanative_resources = getkernel(:cudanative, fusion_level, N, soa)
const cudaraw_kernel    = cudaraw_resources[1]
const cudanative_kernel = cudanative_resources[1]
const cudaraw_inputs    = cudaraw_resources[2]
const cudanative_inputs = cudanative_resources[2]

# warm-up
cudaraw_kernel(cudaraw_inputs...)
cudanative_kernel(cudanative_inputs...)
gc()

# profile
NVTX.@activate CUDAdrv.@profile begin
    ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), STDERR.handle)
    NVTX.@range "cudaraw"    cudaraw_kernel(cudaraw_inputs...)
    NVTX.@range "cudanative" cudanative_kernel(cudanative_inputs...)
    ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), C_NULL)
end
