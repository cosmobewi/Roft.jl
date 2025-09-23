module H0LiCOWCosmoChains
export TDH0GlobalObs, load_h0licow_cosmo_chains, chi2_td_h0_global, meff_from_meta

using LinearAlgebra
using CSV, DataFrames
using Statistics

# petite util sans Statistics
@inline function _mean_std(x::AbstractVector{<:Real})
    n = length(x)
    n == 0 && return (NaN, NaN)
    μ = sum(x) / n
    s2 = sum((xi - μ)^2 for xi in x) / max(n - 1, 1)
    return μ, sqrt(s2)
end

Base.@kwdef struct TDH0GlobalObs
    H0_mean::Float64
    H0_std::Float64
    lenses::Vector{String}                  # lens ids used (from meta)
    zl::Vector{Float64}
    zs::Vector{Float64}
    eta::Vector{Float64}
    w::Vector{Float64}                      # normalized weights
    name::String = "H0LiCOW global"
end

# --- helpers ---
# Robust read of the first numeric column (H0) from a comma-separated .dat with lines starting by '#' for header.
function _read_H0_column(path::AbstractString)::Vector{Float64}
    # Try CSV with comma & comment
    df = CSV.read(path, DataFrame; delim=',', comment="#", header=false, ignorerepeated=true)
    @assert size(df,2) >= 1 "No columns in $path"
    v = Vector{Any}(df[:,1])
    out = Float64[]
    for x in v
        x === missing && continue
        if x isa Real
            push!(out, float(x))
        else
            s = strip(String(x))
            isempty(s) && continue
            y = tryparse(Float64, s)
            y === nothing && error("Non-numeric token '$s' in $path")
            push!(out, y)
        end
    end
    return out
end

# Load meta CSV with columns: lens,zl,zs[,eta][,w]
function _load_meta(meta_path::AbstractString)
    df = CSV.read(meta_path, DataFrame)
    cn = lowercase.(string.(names(df)))
    # allow 'file' or 'lens' for the id
    idcol = findfirst(==("lens"), cn)
    idcol === nothing && (idcol = findfirst(==("file"), cn))
    @assert idcol !== nothing "Meta CSV must have a 'lens' or 'file' column"
    getcol(sym) = begin
        j = findfirst(==(sym), cn); j === nothing ? nothing : j
    end
    j_zl = getcol("zl"); j_zs = getcol("zs")
    @assert j_zl !== nothing && j_zs !== nothing "Meta CSV must have 'zl' and 'zs' columns"
    j_eta = getcol("eta"); j_w = getcol("w")

    lenses = String.(df[:, idcol])
    zl = Float64.(df[:, j_zl]); zs = Float64.(df[:, j_zs])
    eta = j_eta === nothing ? ones(Float64, length(lenses)) : Float64.(df[:, j_eta])
    w   = j_w   === nothing ? fill(1.0/length(lenses), length(lenses)) : Float64.(df[:, j_w])
    # normalize weights
    s = sum(w); s > 0 || (w .= 1.0/length(w)); w ./= sum(w)
    return lenses, zl, zs, eta, w
end

"""
    load_h0licow_cosmo_chains(dir; model=:FLCDM, meta_csv) -> TDH0GlobalObs

Reads all .dat in `joinpath(dir, String(model))`, extracts H0 samples, returns mean/std.
Also reads `meta_csv` with columns: lens,zl,zs[,eta][,w]. Weights are normalized.
"""
function load_h0licow_cosmo_chains(dir::AbstractString; model::Symbol=:FLCDM, meta_csv::AbstractString)
    subdir = joinpath(dir, String(model))
    @assert isdir(subdir) "Model directory not found: $subdir"

    files = filter(f -> endswith(f, ".dat"), readdir(subdir; join=true))
    @assert !isempty(files) "No .dat chains found in $subdir"

    # --- STRICT FILE SELECTION (no heuristics) ---
    file_sel = get(ENV, "H0LICOW_FILE", "")
    if !isempty(file_sel)
        sel_base = basename(file_sel)                      # exact basename with extension
        bases = basename.(files)
        idx = findfirst(==(sel_base), bases)
        @assert idx !== nothing "H0LiCOW file not found: '$sel_base' in $subdir. Available: $(join(bases, ", "))"
        files = [files[idx]]
    else
        if length(files) == 1
            # ok: single file available, use it
        else
            bases = basename.(files)
            
            error("Multiple .dat chain files in $subdir. Set H0LICOW_FILE to one of: $(join(bases, ", ")).")
        end
    end
    sel_base = basename(files[1])
    @info "Selected H0LiCOW chain file (strict)" file=sel_base

    # --- read H0 samples (unchanged) ---
    H0_all = Float64[]
    for f in files
        append!(H0_all, _read_H0_column(f))
    end
    @assert !isempty(H0_all) "Empty H0 after reading $sel_base"
    μ, σ = _mean_std(H0_all)

    # --- load meta (no auto-filtering), but enforce consistency with the selected filename ---
    lenses, zl, zs, eta, w = _load_meta(meta_csv)

    base_lc = lowercase(splitext(sel_base)[1])
    missing = [l for l in lenses if !occursin(lowercase(l), base_lc)]
    @assert isempty(missing) "Meta lenses not found in selected chain filename '$sel_base': $(join(missing, ", ")). " *
                             "Fix your meta CSV to match the file, or pick a matching H0LICOW_FILE."

    # normalize weights deterministically
    s = sum(w); s > 0 || (w .= 1.0/length(w)); w ./= sum(w)

    return TDH0GlobalObs(H0_mean=μ, H0_std=σ, lenses=lenses, zl=zl, zs=zs, eta=eta, w=w,
                         name="H0LiCOW $(String(model))")
end



# Effective ROFT rescaling factor from meta (no geometry needed here)
@inline function meff_from_meta(alpha::Float64, variant::Symbol,
                                zl::AbstractVector{<:Real}, zs::AbstractVector{<:Real},
                                eta::AbstractVector{<:Real}, w::AbstractVector{<:Real},
                                E_LCDM::Function)
    @assert length(zl)==length(zs)==length(eta)==length(w)
    M = 0.0
    for i in eachindex(w)
        Δg_l = variant===:energy  ? 2*alpha*log(E_LCDM(zl[i])) :
               variant===:thermal ?   alpha*log(1+zl[i]) : 0.0
        Δg_s = variant===:energy  ? 2*alpha*log(E_LCDM(zs[i])) :
               variant===:thermal ?   alpha*log(1+zs[i]) : 0.0
        M += w[i]*exp(eta[i]*Δg_l + (1-eta[i])*Δg_s)
    end
    return M
end

"""
    chi2_td_h0_global(obs::TDH0GlobalObs, H0::Real, M_eff::Real)

Gaussian 1D likelihood for global H0 inference: (H0_obs - H0*M_eff)^2 / σ^2.
"""
@inline function chi2_td_h0_global(obs::TDH0GlobalObs, H0::Real, M_eff::Real)
    r = obs.H0_mean - H0*M_eff
    return (r*r) / (obs.H0_std^2)
end

end # module
