using MixedModeBroadcastAD: record, forward!, backward!

#=
Kernels benchmarked here are implemented in `src/kernels` and tested for correctness
in `test/runtests.jl`.

Since our technique is agnostic to a framework's method of tape construction, the
benchmarking code below pre-records the tape in order to avoid timing irrelevant
overhead due to tape construction.
=#

function benchmark_lstm_update_c(n::Int, usegpu::Bool)
    f = usegpu ? MixedModeBroadcastAD.cuda_lstm_update_c : MixedModeBroadcastAD.lstm_update_c
    c = rand(n)
    Wxs = (rand(n) for _ in 1:4)
    Rhs = (rand(n) for _ in 1:4)
    bs  = (rand(n) for _ in 1:4)
    tape, _, _ = record(f, c, Wxs..., Rhs..., bs...)
    println("---------------------------------------------------------")
    println("using GPU: ", usegpu)
    println("size: ", n)
    println("forward pass time: ", @elapsed(forward!(tape)), " seconds")
    println("backward pass time: ", @elapsed(backward!(tape)), " seconds")
end

for n in (2^i for i in 9:11)
    benchmark(n, false)
    benchmark(n, true)
end
