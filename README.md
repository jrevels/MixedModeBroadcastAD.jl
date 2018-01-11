# MixedModeBroadcastAD

A research prototype demonstrating mixed-mode automatic differentiation of fused broadcast
kernels on the GPU.

Do not depend on this package's implementation; it is merely example code and will not be
maintained for any practical usage.


## CUDA support

For this code to work with CUDAnative, some external changes are required.

1. Differentiation rules for CUDAnative intrinsics

Add the following to `DiffRules/src/rules.jl`:

```julia
@define_diffrule CUDAnative.log(x) = :(  CUDAnative.inv($x) )
@define_diffrule CUDAnative.sin(x) = :(  CUDAnative.cos($x) )
@define_diffrule CUDAnative.cos(x) = :( -CUDAnative.sin($x) )
@define_diffrule CUDAnative.exp(x) = :(  CUDAnative.exp($x) )
```

Add the following to `ForwardDiff/src/ForwardDiff.jl`:

```julia
using CUDAnative
```

2. Disable bounds checking

Invoke julia with `--check-bounds=no`.
