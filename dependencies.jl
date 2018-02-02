import JSON
const deps = JSON.Parser.parsefile("deps.json")

import Pkg

function collect()
    pkgs = keys(Pkg.installed())
    shas = Dict{String, String}()
    for pkg in pkgs
        sha = cd(Pkg.dir(pkg)) do
            readchomp(`git rev-parse --verify HEAD`)
        end
        shas[pkg] = sha
    end
    shas["julia"] = Base.GIT_VERSION_INFO.commit
    return shas
end

function process(;install_cb=nothing, checkout_cb=nothing)
    shas = collect()
    for dep in deps["dependencies"]
        if !haskey(shas, dep)
            install_cb != nothing && install_cb(dep)
        else
            ref = deps["refs"][dep]
            sha = if length(ref) == 40
                ref
            else
                cd(Pkg.dir(dep)) do
                    run(`git fetch -q`)
                    readchomp(`git rev-parse --verify $ref`)
                end
            end
            if sha != shas[dep]
                checkout_cb != nothing && checkout_cb(dep, shas[dep], sha)
            end
        end
    end
end

function verify()
    @info "Verifying packages"

    install(dep)               = @warn "Dependency $dep not installed"
    checkout(dep, sha, oldsha) = @warn "Dependency $dep at $oldsha, does not match $sha"
    process(;install_cb=install, checkout_cb=checkout)
end

function install()
    @info "Installing packages"

    function callback(dep)
        @info "Installing $dep"
        if haskey(deps["repositories"], dep)
            repo = deps["repositories"][dep]
            Pkg.clone(repo)
        else
            Pkg.add(dep)
        end
    end
    process(;install_cb=callback)
end

function checkout()
    @info "Checking-out packages"

    function callback(dep, sha, oldsha)
        if dep == "julia"
            @warn "Cannot check-out julia at $sha"
        else
            @info "Checking-out $dep at $sha"
            # Pkg.checkout complains about refs not having tracking information
            cd(Pkg.dir(dep)) do
                run(`git checkout $sha`)
            end
        end
    end
    process(;checkout_cb=callback)
end

if length(ARGS) == 0
    println("Usage: $(PROGRAM_FILE) [-v] [-i] [-c] [-b")
else
    for arg in ARGS
        if arg == "-v"
            verify()
        elseif arg == "-i"
            install()
        elseif arg == "-c"
            checkout()
        elseif arg == "-b"
            Pkg.build()
        else
            @error "Unknown option $arg"
        end
    end
end
