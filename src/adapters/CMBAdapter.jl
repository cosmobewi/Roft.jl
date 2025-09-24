module CMBAdapter
export CMBData, load_cmb_data, align_data_to_ell!

using LinearAlgebra: Symmetric
using CSV, DataFrames

Base.@kwdef mutable struct CMBData
    name::String
    ell::Vector{Int}
    vec::Vector{Float64}
    Cov::Symmetric{Float64,Matrix{Float64}}
    blocks::Vector{Symbol}
    block_ranges::Dict{Symbol,UnitRange{Int}}
    as_Dell::Bool
end

function _df_to_float_matrix(df::DataFrame)
    df2 = copy(df)
    if ncol(df2) > 0 && !(eltype(df2[!, 1]) <: Real)
        select!(df2, Not(1))
    end
    for j in 1:ncol(df2)
        col = df2[!, j]
        df2[!, j] = eltype(col) <: Real ? Float64.(col) : parse.(Float64, string.(col))
    end
    Symmetric((Matrix{Float64}(df2) + Matrix{Float64}(df2)') / 2)
end

function load_cmb_data(data_csv::AbstractString, cov_csv::AbstractString;
                       blocks::Vector{Symbol} = [:TT],
                       ell_min::Int=30, ell_max::Int=2500,
                       as_Dell::Bool=true, name::String="CMB")
    @assert isfile(data_csv) "CMB data file not found: $data_csv"
    @assert isfile(cov_csv)  "CMB cov file not found:  $cov_csv"

    dfD = CSV.read(data_csv, DataFrame; delim=',', ignorerepeated=true)
    @assert hasproperty(dfD, :ell) "data_csv needs a column named 'ell'"
    TTcol = hasproperty(dfD, :TT) ? :TT : (hasproperty(dfD, :DellTT) ? :DellTT : nothing)
    @assert TTcol !== nothing "data_csv must have 'TT' or 'DellTT' column"

    mask = (dfD.ell .>= ell_min) .& (dfD.ell .<= ell_max)
    ell = collect(Int.(dfD.ell[mask]))

    spectra = Dict{Symbol,Vector{Float64}}()
    block_ranges = Dict{Symbol,UnitRange{Int}}()
    data_chunks = Vector{Float64}()
    start_idx = 1
    for block in blocks
        col = block === :TT ? (hasproperty(dfD, :TT) ? :TT : :DellTT) :
              block === :TE ? (hasproperty(dfD, :TE) ? :TE : :DellTE) :
              block === :EE ? (hasproperty(dfD, :EE) ? :EE : :DellEE) : block
        @assert hasproperty(dfD, col) "Column $(col) not found in CMB data file"
        vals = Float64.(dfD[mask, col])
        spectra[block] = vals
        push!(data_chunks, vals...)
        stop_idx = start_idx + length(vals) - 1
        block_ranges[block] = start_idx:stop_idx
        start_idx = stop_idx + 1
    end

    vec = collect(data_chunks)

    dfC = CSV.read(cov_csv, DataFrame; delim=',', ignorerepeated=true, header=false)
    Cov = _df_to_float_matrix(dfC)

    Ndata = length(vec)
    if size(Cov,1) != Ndata || size(Cov,2) != Ndata
        @warn "Covariance size $(size(Cov)) doesn't match data length $Ndata; using diagonal approximation."
        Cov = Symmetric(Diagonal(fill(1.0, Ndata)))
    end

    return CMBData(name=name, ell=ell, vec=vec, Cov=Cov,
                   blocks=blocks, block_ranges=block_ranges, as_Dell=as_Dell)
end

function align_data_to_ell!(cmb::CMBData, ell_target::Vector{Int})
    pos = Dict{Int,Int}()
    for (i, l) in enumerate(cmb.ell)
        haskey(pos, l) || (pos[l] = i)
    end
    filtered_target = Int[l for l in ell_target if haskey(pos, l)]
    idx = [pos[l] for l in filtered_target]
    @assert !isempty(idx) "Aucun ℓ commun entre data et émulateur."

    total_idx = Int[]
    new_ranges = Dict{Symbol,UnitRange{Int}}()
    for block in cmb.blocks
        block_range = cmb.block_ranges[block]
        block_idx = [block_range.start - 1 + i for i in idx]
        start_idx = length(total_idx) + 1
        append!(total_idx, block_idx)
        stop_idx = length(total_idx)
        new_ranges[block] = start_idx:stop_idx
    end

    cmb.vec = cmb.vec[total_idx]
    cov = Matrix(cmb.Cov)
    cmb.Cov = Symmetric(cov[total_idx, total_idx])
    cmb.ell = filtered_target
    cmb.block_ranges = new_ranges
    return cmb
end

end
