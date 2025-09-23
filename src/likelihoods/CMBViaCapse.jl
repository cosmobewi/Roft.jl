module CMBViaCapse
export CapseCMBModel, make_cmb_model_from_env, chi2_cmb_soft_at

using LinearAlgebra
using ..CapseAdapter
using ..CMBAdapter

Base.@kwdef struct CapseCMBModel
    blocks::Vector{Symbol}
    as_Dell_theory::Bool = true
    states::Dict{Symbol,CapseAdapter.CapseState}
    ell::Vector{Int}
end

function make_cmb_model_from_env(cmb::CMBAdapter.CMBData; blocks::Vector{Symbol}=cmb.blocks)
    weights_dir = get(ENV, "CAPSE_WEIGHTS_DIR", "")
    @assert isdir(weights_dir) "Set CAPSE_WEIGHTS_DIR to trained_emu directory"

    states, ℓ = CapseAdapter.load_capse_root_multi(weights_dir; blocks=blocks)
    available_blocks = [b for b in blocks if haskey(states, b)]
    CMBAdapter.align_data_to_ell!(cmb, ℓ)

    return CapseCMBModel(blocks=available_blocks,
                         states=states,
                         ell=ℓ)
end

function chi2_cmb_soft_at(H0::Real, Om0::Real, cmb::CMBAdapter.CMBData, model::CapseCMBModel)
    stTT = get(model.states, :TT, nothing)
    @assert stTT !== nothing "Capse model requires a TT emulator"

    ns   = parse(Float64, get(ENV, "CAPSE_NS", "0.965"))
    As   = parse(Float64, get(ENV, "CAPSE_AS", "2.1e-9"))
    tau  = parse(Float64, get(ENV, "CAPSE_TAU", "0.054"))
    Obh2 = parse(Float64, get(ENV, "CAPSE_OBH2", "0.02237"))
    Onuh2= parse(Float64, get(ENV, "CAPSE_ONUH2", "0.00064"))
    Och2 = Om0 * (H0/100)^2 - Obh2 - Onuh2

    p = CapseAdapter.CapseCMB(Obh2=Obh2, Och2=Och2, H0=H0, ns=ns, As=As, tau=tau)
    thTT = CapseAdapter.predict_cmb(p, stTT)

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

end
