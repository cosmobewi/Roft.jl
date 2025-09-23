module H0LiCOWCosmoChains
export TDH0GlobalObs, chi2_td_h0_global

using LinearAlgebra
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

"""
    chi2_td_h0_global(obs::TDH0GlobalObs, H0::Real, M_eff::Real)

Gaussian 1D likelihood for global H0 inference: (H0_obs - H0*M_eff)^2 / σ^2.
"""
@inline function chi2_td_h0_global(obs::TDH0GlobalObs, H0::Real, M_eff::Real)
    r = obs.H0_mean - H0*M_eff
    return (r*r) / (obs.H0_std^2)
end

end # module
