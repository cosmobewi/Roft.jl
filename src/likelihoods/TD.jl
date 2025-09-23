module TD
export TDData, chi2_td

using LinearAlgebra: dot
using ..Backgrounds: FlatLCDM
using ..ROFTSoft

Base.@kwdef struct TDData
    zl::Vector{Float64}
    zs::Vector{Float64}
    eta::Vector{Float64}
    H0inf::Vector{Float64}
    sigH0::Vector{Float64}
end

"""
    chi2_td(data; bg, roft)

Gaussian χ² for time-delay cosmography inference with ROFT soft scaling of H₀.
"""
function chi2_td(data::TDData; bg::FlatLCDM, roft::ROFTSoft.ROFTParams)
    n = length(data.H0inf)
    @assert n == length(data.zl) == length(data.zs) == length(data.eta) == length(data.sigH0)

    H0_th = ROFTSoft.modify_TD.(bg.H0, data.zl, data.zs, data.eta, Ref(roft);
                                Om0=bg.Om0, Or0=bg.Or0)
    r     = (data.H0inf .- H0_th) ./ data.sigH0
    return dot(r, r)
end

end
