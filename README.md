# MixedModeBroadcastAD

A research prototype demonstrating mixed-mode automatic differentiation of fused broadcast
kernels on the GPU.

Do not depend on this package's implementation; it is merely example code and will not be
maintained for any practical usage.

The dependencies on which the test last passed are recored in `deps.json`. These
are processed using the `dependencies.jl` script, which understands the
following options:

- `-v`: verify the package state
- `-i`: install missing packages
- `-c`: check-out packages at the supported version
- `-b`: build all packages

These options can be combined. A recommended invocation is as follows: `julia
dependencies.jl -i -c -v -b`.

Note that Julia has to be manually installed to the specified version.
