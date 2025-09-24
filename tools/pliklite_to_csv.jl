#!/usr/bin/env julia

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Roft
using Roft.PlanckLiteAdapter: load_pliklite
using LinearAlgebra: diag
using DataFrames, CSV

function main(dir::AbstractString)
    cmb = load_pliklite(dir)

    tt_range = cmb.block_ranges[:TT]
    tt = cmb.vec[tt_range]
    ell = cmb.ell

    σ_tt = sqrt.(diag(Matrix(cmb.Cov)))

    data_out = joinpath(dir, "cmb_data.csv")
    df = DataFrame(ell=ell, TT=tt, TT_sigma=σ_tt)
    CSV.write(data_out, df)
    @info "Wrote TT bandpowers CSV" file=data_out rows=length(ell)

    cov_out = joinpath(dir, "cmb_cov.csv")
    CSV.write(cov_out, DataFrame(Matrix(cmb.Cov), :auto))
    @info "Wrote covariance CSV" file=cov_out size=size(cmb.Cov)
end

if abspath(PROGRAM_FILE) == @__FILE__
    @assert length(ARGS) == 1 "Usage: julia tools/pliklite_to_csv.jl /path/to/plik_lite_v22/"
    main(ARGS[1])
end
