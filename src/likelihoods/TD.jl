module TD
export TDData, chi2_td

using ..Backgrounds: FlatLCDM
using ..ROFTSoft

Base.@kwdef struct TDData
    zl::Vector{Float64}
    zs::Vector{Float64}
    eta::Vector{Float64}
    H0inf::Vector{Float64}
    sigH0::Vector{Float64}
end

function chi2_td(D::TDData; bg::FlatLCDM, roft::ROFTSoft.ROFTParams)
    n = length(D.H0inf)
    @assert n==length(D.zl)==length(D.zs)==length(D.eta)==length(D.sigH0)
    s = 0.0
    for i in 1:n
        H0th = ROFTSoft.modify_TD(bg.H0, D.zl[i], D.zs[i], D.eta[i], roft; Om0=bg.Om0, Or0=bg.Or0)
        r    = (D.H0inf[i] - H0th)/D.sigH0[i]
        s   += r*r
    end
    return s
end

end
