using MixedModeBroadcastAD: record
using Printf
include("../kernels.jl")

function gettape(args...)
    f, inputs = getkernel(args...)
    return first(record(f, inputs...))
end

function timedelta(s)
    (unit, factor) = if s < 1e-6
        ("n", 1e9)
    elseif s < 1e-3
        ("u", 1e6)
    elseif s < 1
        ("m", 1e3)
    else
        ("", 1)
    end
    @sprintf("%.2f %ss", factor*s, unit)
end
