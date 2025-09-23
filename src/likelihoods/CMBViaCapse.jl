module CMBViaCapse
export CMBData, load_cmb_data, CapseCMBModel, make_cmb_model_from_env, chi2_cmb_soft_at

using LinearAlgebra, CSV, DataFrames
using ..CapseAdapter  # <- on passe par l’adapter maison

# --------- Types ---------
Base.@kwdef mutable struct CMBData
    name::String
    ell::Vector{Int}
    vec::Vector{Float64}                  # données concaténées (TT/TE/EE), ici TT seulement
    Cov::Symmetric{Float64,Matrix{Float64}}
    blocks::Vector{Symbol}                # [:TT], [:TT,:TE], etc.
    as_Dell::Bool                         # true si données en D_ℓ
end


Base.@kwdef struct CapseCMBModel
    blocks::Vector{Symbol}
    as_Dell_theory::Bool
    wTT::CapseAdapter.CapseState
    wTE::Any
    wEE::Any
    ell::Vector{Int}
end

# aligne la data (ell, vec, Cov) sur une grille ℓ cible (celle de l'émulateur)
function align_data_to_ell!(cmb::CMBData, ell_target::Vector{Int})
    pos = Dict(l=>i for (i,l) in enumerate(cmb.ell))
    idx = [pos[l] for l in ell_target if haskey(pos,l)]
    @assert !isempty(idx) "Aucun ℓ commun entre data et émulateur."
    cmb.ell = cmb.ell[idx]
    cmb.vec = cmb.vec[idx]
    cmb.Cov = Symmetric(cmb.Cov[idx, idx])
    return cmb
end

# --------- Helpers robustes ---------
function _df_to_float_matrix(df::DataFrame)
    df2 = copy(df)
    # drop une 1ère colonne non-numérique éventuelle
    if ncol(df2) > 0 && !(eltype(df2[!,1]) <: Real)
        select!(df2, Not(1))
    end
    for j in 1:ncol(df2)
        col = df2[!, j]
        if eltype(col) <: Real
            df2[!, j] = Float64.(col)
        else
            df2[!, j] = parse.(Float64, string.(col))
        end
    end
    Symmetric((Matrix{Float64}(df2) + Matrix{Float64}(df2)')/2)
end

# charge data/cov, coupe sur ell, vérifie dimensions (TT-only pour l’instant)
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

    m = (dfD.ell .>= ell_min) .& (dfD.ell .<= ell_max)
    ell = collect(Int.(dfD.ell[m]))
    yTT = Float64.(dfD[m, TTcol])

    dfC = CSV.read(cov_csv, DataFrame; delim=',', ignorerepeated=true, header=false)
    C = _df_to_float_matrix(dfC)

    Ndata = length(yTT)
    if size(C,1) != Ndata || size(C,2) != Ndata
        @warn "Covariance size $(size(C)) doesn't match data length $Ndata; cropping to top-left."
        N = min(Ndata, min(size(C,1), size(C,2)))
        ell = ell[1:N]
        yTT = yTT[1:N]
        C   = C[1:N, 1:N]
    end
    Cov = Symmetric(C)

    return CMBData(name=name, ell=ell, vec=yTT, Cov=Cov, blocks=blocks, as_Dell=as_Dell)
end

# charge les émulateurs Capse et prépare le mapping ℓ_data -> ℓ_emu (TT-only pour l’instant)
function make_cmb_model_from_env(cmb::CMBData; blocks::Vector{Symbol}=cmb.blocks)
    weights_dir = get(ENV, "CAPSE_WEIGHTS_DIR", "")
    @assert isdir(weights_dir) "Set CAPSE_WEIGHTS_DIR to trained_emu directory"

    stTT = (:TT in blocks) ? CapseAdapter.load_capse(joinpath(weights_dir,"TT")) : nothing
    @assert stTT !== nothing "Aucun émulateur TT chargé."

    # aligne la data sur la grille ℓ de l’émulateur
    align_data_to_ell!(cmb, stTT.ℓ)

    # stocke simplement l’état
    return CapseCMBModel(blocks=[:TT],
                         as_Dell_theory = true,
                         wTT = stTT,
                         wTE = nothing,
                         wEE = nothing,
                         ell = stTT.ℓ)
end

# projection + χ² (TT seul pour l’instant)
function chi2_cmb_soft_at(H0::Real, Om0::Real, cmb::CMBData, model::CapseCMBModel)
    # params cosmologiques minimaux
    ns   = parse(Float64, get(ENV,"CAPSE_NS","0.965"))
    As   = parse(Float64, get(ENV,"CAPSE_AS","2.1e-9"))
    tau  = parse(Float64, get(ENV,"CAPSE_TAU","0.054"))
    Obh2 = parse(Float64, get(ENV,"CAPSE_OBH2","0.02237"))
    Onuh2= parse(Float64, get(ENV,"CAPSE_ONUH2","0.00064"))
    Och2 = Om0*(H0/100)^2 - Obh2 - Onuh2

    p = CapseAdapter.CapseCMB(Obh2=Obh2, Och2=Och2, H0=H0, ns=ns, As=As, tau=tau)
    thTT = CapseAdapter.predict_cmb(p, model.wTT)   # théorie (Cℓ ou Dℓ suivant l'émulateur)

    if model.as_Dell_theory != cmb.as_Dell
        ℓ = model.ell
        fac = @. (ℓ * (ℓ + 1)) / (2pi)
        if !model.as_Dell_theory && cmb.as_Dell
            thTT = fac .* thTT
        elseif model.as_Dell_theory && !cmb.as_Dell
            thTT = thTT ./ fac
        end
    end

    r = cmb.vec .- thTT
    return dot(r, cmb.Cov \ r)
end


end # module
