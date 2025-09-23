module PantheonPlusAdapter
export load_pantheonplus, load_pantheon_2018

using LinearAlgebra: Symmetric
using CSV, DataFrames
import CodecZlib: GzipDecompressorStream

using ..PantheonPlus: SNData

function _read_covariance(path::AbstractString)
    lower = lowercase(path)

    if endswith(lower, ".gz")
        return open(GzipDecompressorStream, path) do io
            df = CSV.read(io, DataFrame; header=false, ignorerepeated=true, delim=' ')
            C  = Matrix{Float64}(df)
            @assert size(C,1) == size(C,2) "Gz covariance is not square: $(size(C))"
            Symmetric(C)
        end
    end

    if endswith(lower, ".cov")
        lines = readlines(path)
        @assert !isempty(lines) "Empty .cov file: $path"
        N = parse(Int, split(strip(lines[1]))[1])

        vals = Float64[]
        sizehint!(vals, length(lines) - 1)
        for raw in lines[2:end]
            stripped = strip(raw)
            isempty(stripped) && continue
            for token in split(stripped)
                push!(vals, parse(Float64, token))
            end
        end
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

    df = CSV.read(path, DataFrame; header=false, ignorerepeated=true, delim=' ')
    C  = Matrix{Float64}(df)
    @assert size(C,1) == size(C,2) "Covariance is not square: $(size(C))"
    return Symmetric(C)
end

function load_pantheonplus(data_file::AbstractString, cov_file::AbstractString;
                           zcol::Symbol=:z, mucol::Symbol=:mu)
    df = CSV.read(data_file, DataFrame; ignorerepeated=true, delim=' ', comment="#")

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

end
