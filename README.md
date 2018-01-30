# MixedModeBroadcastAD

A research prototype demonstrating mixed-mode automatic differentiation of fused broadcast
kernels on the GPU.

Do not depend on this package's implementation; it is merely example code and will not be
maintained for any practical usage.

The dependencies on which the test last passed are recored in `deps.json`.

```julia
julia -L dependencies.jl

# Installing dependencies (first-time only)
julia> install()

# Force packages to specific SHA
julia> checkout()

julia> Pkg.build()

# If you want you can use `record()` and `verify()` to check if your state is consistent
```

Note that Julia has to be manually installed to the specified version.
