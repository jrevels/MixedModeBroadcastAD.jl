# MixedModeBroadcastAD

A research prototype demonstrating mixed-mode automatic differentiation of fused broadcast
kernels on the GPU.

Do not depend on this package's implementation; it is merely example code and will not be
maintained for any practical usage.

Last tests passed with:

| Dependency                                              | Commit SHA/Tag                           |
|---------------------------------------------------------|------------------------------------------|
| Julia                                                   | ac389b016702666853bdb80d521a02977705bba1 |
| StaticArrays.jl                                         | d3c5662aa8a8a65ab391dbb3bc58441dac3a1461 |
| ForwardDiff.jl                                          | 593fe6523da74456f9f52b5c314a3a57d4122313 |
| BenchmarkTools.jl                                       | v0.2.4                                   |
| CUDAnative.jl                                           | 263cc4120f970f6748b6b5c1105ff3d5f0de69cf |
| CUDAdrv.jl                                              | 8b9223e90e4919df4a5596fead889aa9a33dd9ad |
| CUDAapi.jl                                              | v0.4.0                                   |
| LLVM.jl                                                 | 0627abca3ebe67e8987cbfdaf6f2a01150be0ca4 |
| [NVTX.jl](https://github.com/maleadt/NVTX.jl)           | 0b32ea12d942119b26e0178d486e71a9fc1a30fc |
| [FastSplat.jl](https://github.com/maleadt/FastSplat.jl) | ff2287df83238433250def44ea4c60d696cd17ad |
