module CC
export CCData, chi2_cc

using LinearAlgebra: dot
using ..Backgrounds: FlatLCDM, H_of_z
using ..ROFTSoft

Base.@kwdef struct CCData
    z::Vector{Float64}
    H::Vector{Float64}
    sigH::Vector{Float64}
end

"""
    chi2_cc(data; bg, roft)

Gaussian χ² for cosmic chronometer data with ROFT soft corrections applied to
the background expansion.
"""
function chi2_cc(data::CCData; bg::FlatLCDM, roft::ROFTSoft.ROFTParams)
    @assert length(data.z) == length(data.H) == length(data.sigH)

    H_geo = H_of_z.(Ref(bg), data.z)
    H_th  = ROFTSoft.modify_CC.(H_geo, data.z, Ref(roft);
                                Om0=bg.Om0, Or0=bg.Or0)
    r     = (data.H .- H_th) ./ data.sigH
    return dot(r, r)
end

end
