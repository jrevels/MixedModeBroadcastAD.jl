#!/usr/bin/env julia

# this script runs each runprofile.* script, with different values for each argument,
# collecting GPU and CUDA API traces in CSV files for further processing.

using Glob
using DataFrames

# run a benchmark once, returning the measurement data
function run_benchmark(cmd)
    output_file = joinpath(tempdir(), "nvprof.csv")
    output_pattern = "nvprof.csv.*"

    # delete old profile output
    rm.(glob(output_pattern, tempdir()))

    # run and measure
    @info "Running benchmark" cmd
    out = Pipe()
    cmd = ```
        nvprof
        --profile-from-start off
        --profile-child-processes
        --unified-memory-profiling off
        --print-api-trace
        --print-gpu-trace
        --normalized-time-unit us
        --csv
        --log-file $output_file.%p
        $cmd
    ```
    cmd_success = success(pipeline(ignorestatus(cmd), stdout=out, stderr=out))
    close(out.in)
    output_files = glob(output_pattern, tempdir())

    # read all data
    if !cmd_success
        println(readstring(out))
        error("benchmark did not succeed")
    else
        # when precompiling, an additional process will be spawned,
        # but the resulting CSV should not contain any data.
        output_data = []
        for output_file in output_files
            data = read_data(output_file)
            if data != nothing
                push!(output_data, data)
            end
        end

        if length(output_data) == 0
            error("no output files")
        elseif length(output_data) > 1
            error("too many output files")
        else
            return output_data[1]
        end
    end
end

# read the raw profiler output, and create a DataFrame
function read_data(input_path)
    table = mktemp() do temp_path,io
        for (line, contents) in enumerate(eachline(input_path; chomp=false))
            contains(contents, "No kernels were profiled.") && return nothing
            line < 4 && continue                        # skip nvprof banner
            line == 5 && continue                       # skip units header
            contains(contents, "cuEvent") && continue   # skip verbose TF event handling
            write(io, contents)
        end
        flush(io)
        readtable(temp_path)
    end

    return table
end


const ITERATIONS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1024

cd(@__DIR__) do
    for dims in [2048]
        python = run_benchmark(`python3 runprofile.py $dims "/device:GPU:0" $ITERATIONS`)
        writetable("python_$(dims).csv", python)

        for tfstyle in [true, false]
            julia = run_benchmark(`julia --depwarn=no runprofile.jl $dims $tfstyle $ITERATIONS`)
            writetable("julia_$(dims)_$(tfstyle ? "tf" : "notf").csv", julia)
        end
    end
end
