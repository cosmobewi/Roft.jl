module CCCov
export CCObs, chi2_cccov

using LinearAlgebra

# ---------------------------
# Data container (generic CC)
# ---------------------------
Base.@kwdef struct CCObs
    z::Vector{Float64}
    H::Vector{Float64}
    Cov::Symmetric{Float64,Matrix{Float64}}
    name::String = "CC"
end

# ---------------------------
# Likelihood (Gaussian)
# ---------------------------

"""
    chi2_cccov(obs::CCObs, H_theory::Function) -> χ²

Likelihood for CC with a **full covariance** (here diagonal by default).
You pass a function `H_theory(z)`, which can be:
- pure geometry (ΛCDM), or
- already modified by ROFT (soft): H_obs(z) = H_geo(z) * exp(-Δg_time(z)).

The function computes χ² = (H_obs - H_theory)^T C^{-1} (H_obs - H_theory).
"""
function chi2_cccov(obs::CCObs, H_theory::Function)
    r = obs.H .- H_theory.(obs.z)
    return dot(r, obs.Cov \ r)
end

end # module
