# MixedModeBroadcastAD

A research prototype demonstrating mixed-mode automatic differentiation of fused broadcast
kernels on the GPU.

Do not depend on this package's implementation; it is merely example code and will not be
maintained for any practical usage.

Last tests passed with:

| Dependency                                              | Commit SHA/Tag                           |
|---------------------------------------------------------|------------------------------------------|
| Julia                                                   | 0f95988141373cebb968a2eed022d661b440293d |
| StaticArrays.jl                                         | d3c5662aa8a8a65ab391dbb3bc58441dac3a1461 |
| ForwardDiff.jl                                          | 3bff34cb6b7b0c7ad0b65216a52e37a5fefee316 |
| BenchmarkTools.jl                                       | v0.2.4                                   |
| CUDAnative.jl                                           | 9b722c322ec082fa9939f693d2030e51ffe37b59 |
| CUDAdrv.jl                                              | v0.8.0                                   |
| CUDAapi.jl                                              | v0.4.0                                   |
| LLVM.jl                                                 | v0.9.6                                   |
| [NVTX.jl](https://github.com/maleadt/NVTX.jl)           | 0b32ea12d942119b26e0178d486e71a9fc1a30fc |
| [FastSplat.jl](https://github.com/maleadt/FastSplat.jl) | ff2287df83238433250def44ea4c60d696cd17ad |
