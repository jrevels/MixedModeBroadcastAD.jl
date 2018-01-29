# MixedModeBroadcastAD

A research prototype demonstrating mixed-mode automatic differentiation of fused broadcast
kernels on the GPU.

Do not depend on this package's implementation; it is merely example code and will not be
maintained for any practical usage.

The package works with:

- Julia 0.7 (commit 4f87318ceecfaf684a56a7b4dea390ea928d36a3 verified to work)
- the deps from REQUIRE
- FastSplat from https://github.com/maleadt/FastSplat.jl
- NVTX from https://github.com/maleadt/NVTX.jl
