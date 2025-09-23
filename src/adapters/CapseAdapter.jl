module CapseAdapter
export CapseCMB, CapseState, load_capse, load_capse_from_env,
       load_capse_root_multi, predict_cmb, cmb_chi2, ellgrid

using LinearAlgebra
using Capse

Base.@kwdef struct CapseCMB
    Obh2::Float64
    Och2::Float64
    H0::Float64
    ns::Float64
    As::Float64
    tau::Float64
end

struct CapseState
    emulator::Capse.CℓEmulator
    ℓ::Vector{Int}
    param_order::Vector{Symbol}
end

function load_capse(weights_dir::AbstractString;
                    param_order::Vector{Symbol}=Symbol[:ωb,:ωc,:h,:ns,:τ,:As])
    emu = Capse.load_emulator(weights_dir)
    ℓ = collect(Capse.get_ℓgrid(emu))
    return CapseState(emu, ℓ, param_order)
end

function load_capse_from_env(; envvar::AbstractString="CAPSE_WEIGHTS",
                                param_order::Vector{Symbol}=Symbol[:ωb,:ωc,:h,:ns,:τ,:As])
    dir = get(ENV, envvar, "")
    @assert !isempty(dir) && isdir(dir) "Définis $envvar vers le dossier de poids Capse."
    return load_capse(dir; param_order=param_order)
end

"Charge plusieurs émulateurs sous un répertoire racine (ex: trained_emu/TT, …)."
function load_capse_root_multi(root::AbstractString;
                               blocks::Vector{Symbol}=Symbol[:TT,:TE,:EE,:PP],
                               param_order::Vector{Symbol}=Symbol[:ωb,:ωc,:h,:ns,:τ,:As])
    @assert isdir(root) "Racine Capse introuvable: $root"
    states = Dict{Symbol,CapseState}()
    for b in blocks
        sub = joinpath(root, String(b))
        if isdir(sub)
            states[b] = load_capse(sub; param_order=param_order)
        end
    end
    @assert !isempty(states) "Aucun sous-dossier trouvé sous $root pour $(blocks)."
    # Vérifie grille ℓ commune
    ℓref = first(states).second.ℓ
    for st in values(states)
        @assert st.ℓ == ℓref "Grille ℓ incohérente entre blocs Capse."
    end
    return states, ℓref
end

function _param_vector(p::CapseCMB, order::Vector{Symbol})
    h = p.H0 / 100.0
    dict = Dict(:ωb=>p.Obh2, :ωc=>p.Och2, :h=>h, :ns=>p.ns, :τ=>p.tau, :As=>p.As)
    return Float64[dict[s] for s in order]
end

function predict_cmb(p::CapseCMB, st::CapseState)
    x = _param_vector(p, st.param_order)
    # Hypothèse volontairement simple: l’émulateur renvoie un TT sur la grille st.ℓ,
    # déjà en Dℓ (cf. §2). Pas d’autres chemins.
    return Capse.get_Cℓ(x, st.emulator)
end

function cmb_chi2(data_vec::AbstractVector{<:Real}, Cov::AbstractMatrix{<:Real},
                  p::CapseCMB, st::CapseState)
    th = predict_cmb(p, st)
    @assert length(th) == length(data_vec)
    r = data_vec .- th
    return dot(r, Cov \ r)
end

ellgrid(st::CapseState) = st.ℓ
end
