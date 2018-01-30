import JSON

const DEPENDENCIES = ["StaticArrays", "ForwardDiff", "BenchmarkTools", "CUDAnative", "CUDAdrv", "CUDAapi", "LLVM", "NVTX", "FastSplat"]
const UNREGISTERED = Dict("FastSplat" => "https://github.com/maleadt/FastSplat.jl", "NVTX" => "https://github.com/maleadt/NVTX.jl")

function collect()
    shas = Dict{String, String}()
    for dep in DEPENDENCIES 
        sha = cd(Pkg.dir(dep)) do
            readchomp(`git rev-parse --verify HEAD`)
        end
        shas[dep] = sha
    end
    shas["julia"] = Base.GIT_VERSION_INFO.commit
    return shas
end

function record(file="deps.json")
    shas = collect()
    open(file, "w") do io
        write(io, JSON.json(shas, 4))
    end
    return
end

function verify(file="deps.json")
    shas = collect()
    old_shas = JSON.Parser.parsefile(file)

    for key in keys(shas) ∪ keys(old_shas)
        if !haskey(shas, key)
            @error "$key not in current version of DEPENDENCIES"
        end
        if !haskey(old_shas, key)
            @error "$key not in recorded version of DEPENDENCIES"
        end
        if shas[key] != old_shas[key]
            @error "The recorded and current SHAs differ for $key"
        end
    end
    return true
end

function install()
    for dep in DEPENDENCIES
        if haskey(UNREGISTERED, dep)
            Pkg.clone(UNREGISTERED[dep])
        else
            Pkg.add(dep)
        end
    end
    return true
end

function checkout(file="deps.json")
    shas = JSON.Parser.parsefile(file)
    for (dep, sha) in shas
        if dep == "julia"
            continue
        end
        # Pkg.checkout complains about SHAs not having tracking information
        cd(Pkg.dir(dep)) do
            run(`git checkout $sha`)
        end
    end
    verify(file)
end
