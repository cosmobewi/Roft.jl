module CephSN
export CephSNData, chi2_cephsn

using LinearAlgebra: dot
using ..Backgrounds: FlatLCDM
using ..ROFTSoft

Base.@kwdef struct CephSNData
    z_host::Vector{Float64}
    dmu::Vector{Float64}
    sigmu::Vector{Float64}
end

"""
    chi2_cephsn(data; bg, roft, c_mu=1.0)

Gaussian χ² for Cepheid-calibrated SN residuals with ROFT host modification.
"""
function chi2_cephsn(data::CephSNData; bg::FlatLCDM, roft::ROFTSoft.ROFTParams, c_mu::Float64=1.0)
    @assert length(data.z_host) == length(data.dmu) == length(data.sigmu)

    th = ROFTSoft.delta_mu_host.(data.z_host, Ref(c_mu), Ref(roft);
                                 Om0=bg.Om0, Or0=bg.Or0)
    r  = (data.dmu .- th) ./ data.sigmu
    return dot(r, r)
end

end
