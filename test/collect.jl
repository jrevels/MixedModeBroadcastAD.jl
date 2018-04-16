#!/usr/bin/env julia

# this script runs each runprofile.* script, with different values for each argument,
# collecting GPU and CUDA API traces in CSV files for further processing.

using Glob
using DataFrames

# run a benchmark once, returning the measurement data
function run_nvprof(callback, cmd, args)
    output_file = joinpath(tempdir(), "nvprof.csv")
    output_pattern = "nvprof.csv.*"

    # delete old profile output
    rm.(glob(output_pattern, tempdir()))

    # run and measure
    @info "Profiling command" cmd
    out = Pipe()
    nvprof = ```
        nvprof
        $args
        --profile-from-start off
        --concurrent-kernels off
        --profile-child-processes
        --unified-memory-profiling off
        --normalized-time-unit us
        --csv
        --log-file $output_file.%p
        $cmd
    ```
    cmd_success = success(pipeline(ignorestatus(nvprof), stdout=out, stderr=out))
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
            data = callback(output_file)
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

function save(path, data)
    open(path, "w") do io
        serialize(io, data)
    end
end

function collect_trace(cmd)
    return run_nvprof(cmd, ```
        --print-api-trace
        --print-gpu-trace
    ```) do input_path
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
end
function collect_metrics(cmd)
    return run_nvprof(cmd, ```
        --metrics achieved_occupancy
    ```) do input_path
        table = mktemp() do temp_path,io
            for (line, contents) in enumerate(eachline(input_path; chomp=false))
                contains(contents, "No events/metrics were profiled.") && return nothing
                line < 5 && continue                        # skip nvprof banner
                write(io, contents)
            end
            flush(io)
            readtable(temp_path)
        end

        return table
    end
end

function collect(cmd, tag)
    let fn = "$(tag)_metrics.jls"
        if !isfile(fn)
            data = collect_metrics(cmd)
            save("$(tag)_metrics.jls", data)
        end
    end

    let fn = "$(tag)_trace.jls"
        if !isfile(fn)
            data = collect_trace(cmd)
            save("$(tag)_metrics.jls", data)
        end
    end
end

const ITERATIONS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1024

cd(@__DIR__) do
    for dims in [512,1024,2048]
        let
            collect(`python3 runprofile.py $dims $ITERATIONS`,
                    "python_$(dims)")
        end

        for tfstyle in [true, false]
            collect(`julia --depwarn=no runprofile.jl $tfstyle $dims $ITERATIONS`,
                    "julia_$(tfstyle ? "tf_" : "")$(dims)")
        end

        for arity in 1:3:10
            collect(`julia --depwarn=no runprofile.jl false $dims $ITERATIONS $arity`,
                    "julia_arity$(arity)_$(dims)")
        end
    end
end
