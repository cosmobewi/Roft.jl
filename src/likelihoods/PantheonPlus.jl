module PantheonPlus
export SNData, load_pantheonplus, load_pantheon_2018,
       chi2_sn_profiledM, chi2_sn_with_M

using LinearAlgebra
using CSV, DataFrames
import CodecZlib: GzipDecompressorStream

# ---------------------------
# Data container (generic SN)
# ---------------------------
Base.@kwdef struct SNData
    z::Vector{Float64}
    mu::Vector{Float64}
    Cov::Symmetric{Float64,Matrix{Float64}}
    name::String = "SN"
end

# -------------------------------------------------
# Robust covariance loader:
#  - .gz      → dense matrix via DataFrame -> Matrix
#  - .cov     → first line is N, then either N^2 or N(N+1)/2 values
#  - fallback → read dense with CSV to DataFrame -> Matrix
# -------------------------------------------------
function _read_covariance(path::AbstractString)
    lower = lowercase(path)

    # .gz (dense) → via DataFrame -> Matrix
    if endswith(lower, ".gz")
        return open(GzipDecompressorStream, path) do io
            df = CSV.read(io, DataFrame; header=false, ignorerepeated=true, delim=' ')
            C  = Matrix{Float64}(df)
            @assert size(C,1) == size(C,2) "Gz covariance is not square: $(size(C))"
            Symmetric(C)
        end
    end

    # .cov (Pantheon+ DISTANCES_AND_COVAR):
    #  1st line: N
    #  then either:
    #   - N^2 values (full matrix, row/col order irrelevant, we'll reshape)
    #   - N(N+1)/2 values (lower/upper triangle) → rebuild symmetric
    if endswith(lower, ".cov")
        lines = readlines(path)
        @assert !isempty(lines) "Empty .cov file: $path"
        N = parse(Int, split(strip(lines[1]))[1])
        vals = parse.(Float64, strip.(lines[2:end]))
        M = length(vals)

        if M == N*N
            C = reshape(vals, N, N)
            return Symmetric(Matrix(C))
        elseif M == N*(N+1) ÷ 2
            C = zeros(N, N)
            k = 1
            for j in 1:N
                for i in j:N
                    C[i, j] = vals[k]
                    C[j, i] = vals[k]
                    k += 1
                end
            end
            return Symmetric(C)
        else
            error("Unexpected .cov content for $path: N=$N but values=$M (expected N^2=$(N^2) or N(N+1)/2=$(N*(N+1)÷2)).")
        end
    end

    # Fallback dense text → DataFrame -> Matrix
    df = CSV.read(path, DataFrame; header=false, ignorerepeated=true, delim=' ')
    C  = Matrix{Float64}(df)
    @assert size(C,1) == size(C,2) "Covariance is not square: $(size(C))"
    return Symmetric(C)
end

# ---------------------------
# Loaders
# ---------------------------
function load_pantheonplus(data_file::AbstractString, cov_file::AbstractString;
                           zcol::Symbol=:z, mucol::Symbol=:mu)
    # NOTE: 'comment' must be a String, not a Char
    df = CSV.read(data_file, DataFrame; ignorerepeated=true, delim=' ', comment="#")

    # auto-detect reasonable columns if default ones are missing
    if !hasproperty(df, zcol)
        for cand in (:zHD, :zCMB, :zcmb, :z)
            if hasproperty(df, cand); zcol = cand; break; end
        end
    end
    if !hasproperty(df, mucol)
        for cand in (:MU_SH0ES, :MU, :mu)
            if hasproperty(df, cand); mucol = cand; break; end
        end
    end
    @assert hasproperty(df, zcol)  "z column not found; available: $(names(df))"
    @assert hasproperty(df, mucol) "mu column not found; available: $(names(df))"

    z  = Vector{Float64}(df[!, zcol])
    mu = Vector{Float64}(df[!, mucol])

    Cov = _read_covariance(cov_file)
    @assert length(z) == size(Cov,1) == size(Cov,2)

    return SNData(z=z, mu=mu, Cov=Cov, name="PantheonPlus")
end

# (Optional) Pantheon 2018 variant reusing _read_covariance
function load_pantheon_2018(data_file::AbstractString, cov_file::AbstractString;
                            zcol::Symbol=:zcmb, mucol::Symbol=:mu)
    df = CSV.read(data_file, DataFrame; ignorerepeated=true, delim=' ', comment="#")
    @assert hasproperty(df, zcol)  "z column not found; available: $(names(df))"
    @assert hasproperty(df, mucol) "mu column not found; available: $(names(df))"
    z  = Vector{Float64}(df[!, zcol])
    mu = Vector{Float64}(df[!, mucol])
    Cov = _read_covariance(cov_file)
    @assert length(z) == size(Cov,1) == size(Cov,2)
    return SNData(z=z, mu=mu, Cov=Cov, name="Pantheon2018")
end

# ---------------------------
# χ² with/without profiling M
# ---------------------------
function chi2_sn_with_M(sn::SNData, mu_theory::Function; M::Real=0.0)
    μ_th0 = mu_theory.(sn.z)
    r     = sn.mu .- (μ_th0 .+ M)
    return dot(r, sn.Cov \ r)
end

function chi2_sn_profiledM(sn::SNData, mu_theory::Function;
                           M_prior::Union{Nothing,Tuple{<:Real,<:Real}}=nothing)
    μ_th0 = mu_theory.(sn.z)
    r0    = sn.mu .- μ_th0
    Cinv  = inv(sn.Cov)  # for production: cache a factorization
    one   = ones(length(r0))

    A = dot(one, Cinv * one)
    B = dot(one, Cinv * r0)
    χ2_r0 = dot(r0, Cinv * r0)

    if M_prior === nothing
        M_hat = B / A
        χ2    = χ2_r0 - (B^2)/A
        return χ2, M_hat
    else
        M0, σM = M_prior
        A′ = A + 1/σM^2
        B′ = B + M0/σM^2
        M_hat = B′ / A′
        χ2 = χ2_r0 + (M0^2/σM^2) - (B′^2)/A′
        return χ2, M_hat
    end
end

end # module
