module CCcovAdapter
export load_cccov

using LinearAlgebra: Symmetric, Diagonal
using CSV, DataFrames

using ..CCCov: CCObs

const DEFAULT_CCCOV_DIR = "/opt/CCcovariance/data"
const MM20_FILENAME = "data_MM20.dat"

function _read_cc_table(path::AbstractString)
    df = CSV.read(path, DataFrame; delim=',', comment="#", header=false, ignorerepeated=true)
    ncols = size(df, 2)
    @assert ncols ≥ 3 "Expected at least 3 columns (z, Hz, errHz) in $path, got $ncols"
    z    = Float64.(df[:, 1])
    H    = Float64.(df[:, 2])
    sigH = Float64.(df[:, 3])
    return z, H, sigH
end

_diag_cov(sig::AbstractVector{<:Real}) = Symmetric(Diagonal(Float64.(sig).^2))

function _resolve_data_dir(data_dir::Union{Nothing,String})
    if !isnothing(data_dir)
        @assert isdir(data_dir) "CC data directory not found: $data_dir"
        return data_dir
    end
    if isdir(DEFAULT_CCCOV_DIR)
        return DEFAULT_CCCOV_DIR
    end
    env_dir = get(ENV, "CC_DIR", "")
    @assert !isempty(env_dir) && isdir(env_dir) "CC data directory not found. Set CC_DIR, pass data_dir, or ensure $(DEFAULT_CCCOV_DIR) exists."
    return env_dir
end

function _maybe_load_mm20(dir::String)
    path = joinpath(dir, MM20_FILENAME)
    return isfile(path) ? CSV.read(path, DataFrame; comment="#", header=false, ignorerepeated=true, delim=' ') : nothing
end

function _interp_fraction(z_src::Vector{Float64}, values::Vector{Float64}, z_target::Vector{Float64})
    @assert issorted(z_src) "MM20 redshifts must be sorted"
    out = similar(z_target, Float64)
    for (i, z) in enumerate(z_target)
        if z <= first(z_src)
            out[i] = values[1]
        elseif z >= last(z_src)
            out[i] = values[end]
        else
            idx = searchsortedfirst(z_src, z)
            z1, z0 = z_src[idx], z_src[idx-1]
            v1, v0 = values[idx], values[idx-1]
            t = (z - z0)/(z1 - z0)
            out[i] = (1-t)*v0 + t*v1
        end
    end
    return out
end

function _systematic_cov(z::Vector{Float64}, Hz::Vector{Float64}, dir::String)
    df = _maybe_load_mm20(dir)
    df === nothing && return nothing
    @assert size(df,2) ≥ 5 "Unexpected format for $(joinpath(dir, MM20_FILENAME))"

    z_mod = Float64.(df[:,1])
    imf_frac    = _interp_fraction(z_mod, Float64.(df[:,2]) ./ 100, z)
    slib_frac   = _interp_fraction(z_mod, Float64.(df[:,3]) ./ 100, z)
    sps_frac    = _interp_fraction(z_mod, Float64.(df[:,4]) ./ 100, z)
    spsooo_frac = _interp_fraction(z_mod, Float64.(df[:,5]) ./ 100, z)

    cov = zeros(length(z), length(z))
    for frac in (imf_frac, slib_frac, sps_frac, spsooo_frac)
        v = Hz .* frac
        cov .+= v * transpose(v)
    end
    return Symmetric(cov)
end

function load_cccov(; data_dir::Union{Nothing,String}=nothing,
                       flavor::Symbol=:BC03,
                       files::Union{Nothing,Vector{String}}=nothing,
                       name::String="CCcov",
                       include_systematics::Bool=true)

    dir = _resolve_data_dir(data_dir)

    default_files = flavor === :BC03  ? ["HzTable_MM_BC03.dat"] :
                    flavor === :M11   ? ["HzTable_MM_M11.dat"] :
                    flavor === :both  ? ["HzTable_MM_BC03.dat", "HzTable_MM_M11.dat"] :
                    error("Unknown flavor: $flavor (use :BC03 | :M11 | :both)")

    flist = isnothing(files) ? default_files : files
    @assert !isempty(flist) "No files to load for CC."

    zs  = Float64[]
    Hs  = Float64[]
    es  = Float64[]
    for f in flist
        path = isabspath(f) ? f : joinpath(dir, f)
        @assert isfile(path) "CC file not found: $path"
        z, H, sigH = _read_cc_table(path)
        append!(zs, z)
        append!(Hs, H)
        append!(es, sigH)
    end

    order = sortperm(zs)
    z_sorted = zs[order]
    H_sorted = Hs[order]
    e_sorted = es[order]

    Cov_diag = _diag_cov(e_sorted)

    if include_systematics
        cov_sys = _systematic_cov(z_sorted, H_sorted, dir)
        if cov_sys !== nothing
            Cov = Symmetric(Matrix(Cov_diag) + Matrix(cov_sys))
            return CCObs(z=z_sorted, H=H_sorted, Cov=Cov, name=name)
        end
    end

    return CCObs(z=z_sorted, H=H_sorted, Cov=Cov_diag, name=name)
end

end
