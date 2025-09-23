module CephSN
export CephSNData, chi2_cephsn

using ..Backgrounds: FlatLCDM
using ..ROFTSoft

Base.@kwdef struct CephSNData
    z_host::Vector{Float64}
    dmu::Vector{Float64}
    sigmu::Vector{Float64}
end

function chi2_cephsn(D::CephSNData; bg::FlatLCDM, roft::ROFTSoft.ROFTParams, c_mu::Float64=1.0)
    @assert length(D.z_host)==length(D.dmu)==length(D.sigmu)
    s = 0.0
    for i in eachindex(D.z_host)
        th = ROFTSoft.delta_mu_host(D.z_host[i], c_mu, roft; Om0=bg.Om0, Or0=bg.Or0)
        r  = (D.dmu[i] - th)/D.sigmu[i]
        s += r*r
    end
    return s
end

end
