module CCCovAdapter
export load_cccov

using LinearAlgebra: Symmetric, Diagonal
using CSV, DataFrames

using ..CCCov: CCObs

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

function load_cccov(; data_dir::Union{Nothing,String}=nothing,
                       flavor::Symbol=:BC03,
                       files::Union{Nothing,Vector{String}}=nothing,
                       name::String="CCcov")

    dir = isnothing(data_dir) ? get(ENV, "CC_DIR", "") : data_dir
    @assert !isempty(dir) && isdir(dir) "CC data directory not found. Set CC_DIR or pass data_dir"

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

    Cov = _diag_cov(e_sorted)
    return CCObs(z=z_sorted, H=H_sorted, Cov=Cov, name=name)
end

end
