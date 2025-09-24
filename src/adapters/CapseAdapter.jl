module CapseAdapter
export CapseCMB, CapseState, load_capse, load_capse_from_env,
       load_capse_root_multi, predict_cmb, cmb_chi2, ellgrid

using LinearAlgebra
using Base: basename
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
    emulator::Any
    ℓ::Vector{Int}
    param_order::Vector{Symbol}
end

function load_capse(weights_dir::AbstractString;
                    param_order::Vector{Symbol}=Symbol[:ωb,:ωc,:h,:ns,:τ,:As])
    order = compute_param_order(weights_dir, param_order)
    post_file = ensure_postprocessing_file(weights_dir)
    emu = Capse.load_emulator(weights_dir; postprocessing_file=post_file)
    ℓ = collect(Capse.get_ℓgrid(emu))
    return CapseState(emu, ℓ, order)
end

function compute_param_order(weights_dir::AbstractString, fallback::Vector{Symbol})
    env_order = strip(get(ENV, "CAPSE_PARAM_ORDER", ""))
    if !isempty(env_order)
        return Symbol.(strip.(split(env_order, ','; keepempty=false)))
    end

    detected = detect_param_order(weights_dir)
    return detected === nothing ? fallback : detected
end

function detect_param_order(weights_dir::AbstractString)
    json_path = joinpath(weights_dir, "nn_setup.json")
    isfile(json_path) || return nothing
    for line in eachline(json_path)
        m = match(r"\"parameters\"\s*:\s*\"([^\"]+)\"", line)
        if m !== nothing
            raw = split(m.captures[1], ','; keepempty=false)
            return Symbol.(strip.(raw))
        end
    end
    return nothing
end

function ensure_postprocessing_file(weights_dir::AbstractString)
    post_jl = joinpath(weights_dir, "postprocessing.jl")
    post_py = joinpath(weights_dir, "postprocessing.py")

    if isfile(post_jl)
        ensure_postprocessing_methods!(post_jl)
        return basename(post_jl)
    elseif isfile(post_py)
        jl_path = synthesize_postprocessing(post_py, post_jl)
        ensure_postprocessing_methods!(jl_path)
        return basename(jl_path)
    else
        jl_path = write_default_postprocessing(post_jl)
        return basename(jl_path)
    end
end

function synthesize_postprocessing(py_path::AbstractString, jl_path::AbstractString)
    src = read(py_path, String)
    if occursin("Cl * jnp.exp(input[0]-3)", src)
        open(jl_path, "w") do io
            write(io, default_postprocessing_code())
        end
        return jl_path
    else
        error("Unsupported Capse postprocessing python file at $(py_path); installe JAX ou fournis une version Julia équivalente.")
    end
end

default_postprocessing_code() = """
function postprocessing(input, Cl)
    return Cl .* exp(input[1] - 3)
end

function postprocessing(input, Cl, _emu)
    return postprocessing(input, Cl)
end
"""

function write_default_postprocessing(jl_path::AbstractString)
    open(jl_path, "w") do io
        write(io, default_postprocessing_code())
    end
    return jl_path
end

function ensure_postprocessing_methods!(jl_path::AbstractString)
    content = read(jl_path, String)
    has_two = occursin(r"function\s+postprocessing\s*\(\s*input\s*,\s*Cl", content)
    has_three = occursin(r"function\s+postprocessing\s*\(\s*input\s*,\s*Cl\s*,", content)

    if !has_two
        write_default_postprocessing(jl_path)
    elseif !has_three
        open(jl_path, "a") do io
            println(io)
            println(io, "function postprocessing(input, Cl, _emu)")
            println(io, "    return postprocessing(input, Cl)")
            println(io, "end")
        end
    end
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
    ln10As = log(1e10 * p.As)
    mapping = Dict{Symbol,Float64}(
        :ωb => p.Obh2,
        :wb => p.Obh2,
        :Obh2 => p.Obh2,
        :Ωbh2 => p.Obh2,
        :omegabh2 => p.Obh2,
        :ωc => p.Och2,
        :wc => p.Och2,
        :Och2 => p.Och2,
        :Ωch2 => p.Och2,
        :omegach2 => p.Och2,
        :h => h,
        :H0 => p.H0,
        :ns => p.ns,
        :n_s => p.ns,
        :τ => p.tau,
        :tau => p.tau,
        :As => p.As,
        :ln10As => ln10As,
        :log10As => log10(p.As)
    )

    vals = Float64[]
    for sym in order
        if haskey(mapping, sym)
            push!(vals, mapping[sym])
        else
            error("Unsupported Capse parameter symbol $(sym). Configure CAPSE_PARAM_ORDER or extend _param_vector.")
        end
    end
    return vals
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
