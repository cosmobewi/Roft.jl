module Driver
export chi2_total, Packs

using ..Backgrounds: FlatLCDM, E_LCDM
using ..LikelihoodsCore
using ..TimeDomain
import ..ROFTSoft          # ← rend le sous-module visible ici
import ..PantheonPlus

const C_KMS = 299_792.458  # km/s (déclaré au niveau module)
const ROFTParamsType = ROFTSoft.ROFTParams  # alias de type pratique

Base.@kwdef struct Packs
    cc::Union{Nothing,TimeDomain.CCData} = nothing
    td::Union{Nothing,TimeDomain.TDData} = nothing
    ceph::Union{Nothing,TimeDomain.CephSNData} = nothing
    sn::Union{Nothing,PantheonPlus.SNData} = nothing
    cmb_data::Union{Nothing,Vector{Float64}} = nothing
    cmb_cov::Union{Nothing,AbstractMatrix{Float64}} = nothing
    cmb_params::Any = nothing
    bao_model::Function = ()->(Float64[])
    bao_data::Union{Nothing,Vector{Float64}} = nothing
    bao_cov::Union{Nothing,AbstractMatrix{Float64}} = nothing
    M_prior::Union{Nothing,Tuple{<:Real,<:Real}} = nothing
end

function chi2_total(D::Packs; bg::FlatLCDM, roft::ROFTParamsType, c_mu::Float64=1.0)
    χ2 = 0.0

    # (CMB/BAO éventuels ici — pas d’α en soft)

    # SN (Pantheon/Pantheon+) — pas d’α en soft
    if D.sn !== nothing
        E(z) = E_LCDM(z; Om0=bg.Om0, Or0=bg.Or0, Ol0=1-bg.Om0-bg.Or0)
        function DL(z)
            n = 1000
            zs = range(0, z; length=n)
            dz = step(zs)
            dc = sum((C_KMS/bg.H0) ./ E.(zs)) * dz
            return (1+z) * dc
        end
        mu_th0(z) = 5*log10(DL(z)) + 25
        χ2 += PantheonPlus.chi2_sn_profiledM(D.sn, mu_th0; M_prior=D.M_prior)[1]
    end

    # Probes “temps” — α agit ici seulement
    if D.cc !== nothing
        χ2 += TimeDomain.chi2_cc(D.cc; bg=bg, roft=roft)
    end
    if D.td !== nothing
        χ2 += TimeDomain.chi2_td(D.td; bg=bg, roft=roft)
    end
    if D.ceph !== nothing
        χ2 += TimeDomain.chi2_cephsn(D.ceph; bg=bg, roft=roft, c_mu=c_mu)
    end

    return χ2
end

end # module
