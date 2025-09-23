module TimeDomain
export CCData, TDData, CephSNData, TDH0GlobalObs, CCObs,
       chi2_cc, chi2_td, chi2_cephsn, chi2_td_h0_global, chi2_cccov

using LinearAlgebra: dot, Symmetric
using ..Backgrounds: FlatLCDM, H_of_z
using ..ROFTSoft

# ---------------------------
# Cosmic chronometers (geometry)
# ---------------------------
Base.@kwdef struct CCData
    z::Vector{Float64}
    H::Vector{Float64}
    sigH::Vector{Float64}
end

function chi2_cc(data::CCData; bg::FlatLCDM, roft::ROFTSoft.ROFTParams)
    @assert length(data.z) == length(data.H) == length(data.sigH)

    H_geo = H_of_z.(Ref(bg), data.z)
    H_th  = ROFTSoft.modify_CC.(H_geo, data.z, Ref(roft);
                                Om0=bg.Om0, Or0=bg.Or0)
    r     = (data.H .- H_th) ./ data.sigH
    return dot(r, r)
end

# ---------------------------
# Time-delay cosmography (local H0)
# ---------------------------
Base.@kwdef struct TDData
    zl::Vector{Float64}
    zs::Vector{Float64}
    eta::Vector{Float64}
    H0inf::Vector{Float64}
    sigH0::Vector{Float64}
end

function chi2_td(data::TDData; bg::FlatLCDM, roft::ROFTSoft.ROFTParams)
    n = length(data.H0inf)
    @assert n == length(data.zl) == length(data.zs) == length(data.eta) == length(data.sigH0)

    H0_th = ROFTSoft.modify_TD.(bg.H0, data.zl, data.zs, data.eta, Ref(roft);
                                Om0=bg.Om0, Or0=bg.Or0)
    r     = (data.H0inf .- H0_th) ./ data.sigH0
    return dot(r, r)
end

# ---------------------------
# Cepheid-calibrated SN host corrections
# ---------------------------
Base.@kwdef struct CephSNData
    z_host::Vector{Float64}
    dmu::Vector{Float64}
    sigmu::Vector{Float64}
end

function chi2_cephsn(data::CephSNData; bg::FlatLCDM, roft::ROFTSoft.ROFTParams, c_mu::Float64=1.0)
    @assert length(data.z_host) == length(data.dmu) == length(data.sigmu)

    th = ROFTSoft.delta_mu_host.(data.z_host, Ref(c_mu), Ref(roft);
                                 Om0=bg.Om0, Or0=bg.Or0)
    r  = (data.dmu .- th) ./ data.sigmu
    return dot(r, r)
end

# ---------------------------
# H0LiCOW global inference
# ---------------------------
Base.@kwdef struct TDH0GlobalObs
    H0_mean::Float64
    H0_std::Float64
    lenses::Vector{String}
    zl::Vector{Float64}
    zs::Vector{Float64}
    eta::Vector{Float64}
    w::Vector{Float64}
    name::String = "H0LiCOW global"
end

@inline function chi2_td_h0_global(obs::TDH0GlobalObs, H0::Real, M_eff::Real)
    r = obs.H0_mean - H0*M_eff
    return (r*r) / (obs.H0_std^2)
end

# ---------------------------
# CC with full covariance (stat + syst)
# ---------------------------
Base.@kwdef struct CCObs
    z::Vector{Float64}
    H::Vector{Float64}
    Cov::Symmetric{Float64,Matrix{Float64}}
    name::String = "CC"
end

function chi2_cccov(obs::CCObs, H_theory::Function)
    r = obs.H .- H_theory.(obs.z)
    return dot(r, obs.Cov \ r)
end

end # module
