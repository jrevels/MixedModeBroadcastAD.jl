# MixedModeBroadcastAD

A research prototype demonstrating mixed-mode automatic differentiation of fused broadcast
kernels on the GPU.

Do not depend on this package's implementation; it is merely example code and will not be
maintained for any practical usage.

Last tests passed with:

| Dependency                                              | Commit SHA/Tag                           |
|---------------------------------------------------------|------------------------------------------|
| Julia                                                   | 4f87318ceecfaf684a56a7b4dea390ea928d36a3 |
| StaticArrays.jl                                         | a83cac63fd23b3a33fd2b3fe015b091c540e874e |
| ForwardDiff.jl                                          | 593fe6523da74456f9f52b5c314a3a57d4122313 |
| BenchmarkTools.jl                                       | v0.2.4                                   |
| CUDAnative.jl                                           | 14005e0c0c1b7802b3ee3a6cd60599b529392aad |
| CUDAdrv.jl                                              | 9ff699f65c1077fdb7039673eaf55d17f00b6160 |
| CUDAapi.jl                                              | v0.4.0                                   |
| [NVTX.jl](https://github.com/maleadt/NVTX.jl)           | 0b32ea12d942119b26e0178d486e71a9fc1a30fc |
| [FastSplat.jl](https://github.com/maleadt/FastSplat.jl) | ff2287df83238433250def44ea4c60d696cd17ad |
