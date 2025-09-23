module CC
export CCData, chi2_cc

using ..Backgrounds: FlatLCDM, H_of_z
using ..ROFTSoft

Base.@kwdef struct CCData
    z::Vector{Float64}
    H::Vector{Float64}
    sigH::Vector{Float64}
end

function chi2_cc(D::CCData; bg::FlatLCDM, roft::ROFTSoft.ROFTParams)
    @assert length(D.z)==length(D.H)==length(D.sigH)
    s = 0.0
    for i in eachindex(D.z)
        Hgeo = H_of_z(bg, D.z[i])
        Hth  = ROFTSoft.modify_CC(Hgeo, D.z[i], roft; Om0=bg.Om0, Or0=bg.Or0)
        r    = (D.H[i] - Hth)/D.sigH[i]
        s   += r*r
    end
    return s
end

end
