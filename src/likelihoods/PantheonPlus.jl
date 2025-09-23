module PantheonPlus
export SNData, chi2_sn_profiledM, chi2_sn_with_M

using LinearAlgebra

# ---------------------------
# Data container (generic SN)
# ---------------------------
Base.@kwdef struct SNData
    z::Vector{Float64}
    mu::Vector{Float64}
    Cov::Symmetric{Float64,Matrix{Float64}}
    name::String = "SN"
end

# ---------------------------
# χ² with/without profiling M
# ---------------------------
function chi2_sn_with_M(sn::SNData, mu_theory::Function; M::Real=0.0)
    μ_th0 = mu_theory.(sn.z)
    r     = sn.mu .- (μ_th0 .+ M)
    return dot(r, sn.Cov \ r)
end

function chi2_sn_profiledM(sn::SNData, mu_theory::Function;
                           M_prior::Union{Nothing,Tuple{<:Real,<:Real}}=nothing)
    μ_th0 = mu_theory.(sn.z)
    r0    = sn.mu .- μ_th0
    one   = ones(length(r0))

    # Factorize once and reuse the solves; avoids forming the dense inverse.
    chol = cholesky(sn.Cov)
    Cinv_one = chol \ one
    Cinv_r0  = chol \ r0

    A = dot(one, Cinv_one)
    B = dot(one, Cinv_r0)
    χ2_r0 = dot(r0, Cinv_r0)

    if M_prior === nothing
        M_hat = B / A
        χ2    = χ2_r0 - (B^2)/A
        return χ2, M_hat
    else
        M0, σM = M_prior
        A′ = A + 1/σM^2
        B′ = B + M0/σM^2
        M_hat = B′ / A′
        χ2 = χ2_r0 + (M0^2/σM^2) - (B′^2)/A′
        return χ2, M_hat
    end
end

end # module
