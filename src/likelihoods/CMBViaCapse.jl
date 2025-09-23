module CMBViaCapse
export CapseCMBModel, make_cmb_model_from_env, chi2_cmb_soft_at

using LinearAlgebra
using ..CapseAdapter
using ..CMBAdapter
using ..CapseEnv

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
    nuis = CapseEnv.read_nuisance_params()
    Och2 = Om0 * (H0/100)^2 - nuis.Obh2 - nuis.Onuh2
    p = CapseAdapter.CapseCMB(Obh2=nuis.Obh2, Och2=Och2, H0=H0,
                              ns=nuis.ns, As=nuis.As, tau=nuis.tau)

    prediction = similar(cmb.vec)

    for block in model.blocks
        st = get(model.states, block, nothing)
        @assert st !== nothing "No Capse emulator found for block $(block)"
        spec = CapseAdapter.predict_cmb(p, st)
        if model.as_Dell_theory != cmb.as_Dell
            ℓ = model.ell
            fac = @. (ℓ * (ℓ + 1)) / (2pi)
            if !model.as_Dell_theory && cmb.as_Dell
                spec = fac .* spec
            elseif model.as_Dell_theory && !cmb.as_Dell
                spec = spec ./ fac
            end
        end
        range = cmb.block_ranges[block]
        @assert length(spec) == length(range) "Capse emulator output length mismatch for block $(block)"
        prediction[range] = spec
    end

    r = cmb.vec .- prediction
    return dot(r, cmb.Cov \ r)
end

end
