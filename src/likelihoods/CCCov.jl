module CCCov
export CCObs, load_cccov, chi2_cccov

using LinearAlgebra
using CSV, DataFrames

# ---------------------------
# Data container (generic CC)
# ---------------------------
Base.@kwdef struct CCObs
    z::Vector{Float64}
    H::Vector{Float64}
    Cov::Symmetric{Float64,Matrix{Float64}}
    name::String = "CC"
end

# ---------------------------
# Low-level readers
# ---------------------------

"""
    _read_cc_table(path::AbstractString) -> (z, H, sigH)

Reads a comma-delimited file with a header commented out by '#', like:
# z, Hz, errHz, ...
0.1791,74.91,3.80,...

Returns vectors (z, H, sigH).
"""
function _read_cc_table(path::AbstractString)
    # Header is commented with '#', so we skip it and set header=false
    df = CSV.read(path, DataFrame; delim=',', comment="#", header=false, ignorerepeated=true)
    ncols = size(df, 2)
    @assert ncols ≥ 3 "Expected at least 3 columns (z, Hz, errHz) in $path, got $ncols"
    z    = Float64.(df[:, 1])
    H    = Float64.(df[:, 2])
    sigH = Float64.(df[:, 3])
    return z, H, sigH
end

"""
    _diag_cov(sig::AbstractVector) -> Symmetric

Builds a diagonal covariance from standard deviations.
"""
_diag_cov(sig::AbstractVector{<:Real}) = Symmetric(Diagonal(Float64.(sig).^2))

# ---------------------------
# High-level loader
# ---------------------------

"""
    load_cccov(; data_dir::Union{Nothing,String}=nothing,
                 flavor::Symbol=:BC03,
                 files::Union{Nothing,Vector{String}}=nothing,
                 name::String="CCcov")

Load Cosmic Chronometer (CC) data from the MMoresco CCcovariance repository.
- `data_dir`: directory containing the data files. If `nothing`, tries ENV["CC_DIR"].
- `flavor`: `:BC03`, `:M11`, or `:both`.
- `files`: optional explicit list of files (relative or absolute). Overrides `flavor`.

Returns `CCObs(z, H, Cov)` with a **diagonal** covariance built from `errHz`.
(Quand tu voudras brancher la covariance complète fournie par le dépôt, on
fera évoluer ce connecteur pour la lire—API inchangée côté appelant.)
"""
function load_cccov(; data_dir::Union{Nothing,String}=nothing,
                       flavor::Symbol=:BC03,
                       files::Union{Nothing,Vector{String}}=nothing,
                       name::String="CCcov")

    dir = isnothing(data_dir) ? get(ENV, "CC_DIR", "") : data_dir
    @assert !isempty(dir) && isdir(dir) "CC data directory not found. Set CC_DIR or pass data_dir"

    # Choose default files by flavor
    default_files = flavor === :BC03  ? ["HzTable_MM_BC03.dat"] :
                    flavor === :M11   ? ["HzTable_MM_M11.dat"] :
                    flavor === :both  ? ["HzTable_MM_BC03.dat", "HzTable_MM_M11.dat"] :
                    error("Unknown flavor: $flavor (use :BC03 | :M11 | :both)")

    flist = isnothing(files) ? default_files : files
    @assert !isempty(flist) "No files to load for CC."

    # Read and stack
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

    # Sort by z and drop exact duplicates if any
    order = sortperm(zs)
    z_sorted = zs[order]
    H_sorted = Hs[order]
    e_sorted = es[order]

    Cov = _diag_cov(e_sorted)
    return CCObs(z=z_sorted, H=H_sorted, Cov=Cov, name=name)
end

# ---------------------------
# Likelihood (Gaussian)
# ---------------------------

"""
    chi2_cccov(obs::CCObs, H_theory::Function) -> χ²

Likelihood for CC with a **full covariance** (here diagonal by default).
You pass a function `H_theory(z)`, which can be:
- pure geometry (ΛCDM), or
- already modified by ROFT (soft): H_obs(z) = H_geo(z) * exp(-Δg_time(z)).

The function computes χ² = (H_obs - H_theory)^T C^{-1} (H_obs - H_theory).
"""
function chi2_cccov(obs::CCObs, H_theory::Function)
    r = obs.H .- H_theory.(obs.z)
    return dot(r, obs.Cov \ r)
end

end # module
