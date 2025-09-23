module ROFTSoft
export ROFTParams, delta_g_time, modify_CC, modify_TD, delta_mu_host

using ..Backgrounds: E_LCDM

Base.@kwdef struct ROFTParams
    alpha::Float64 = 0.0
    variant::Symbol = :energy   # :energy => Δg=2α ln E ; :thermal => Δg=α ln(1+z)
end

@inline function delta_g_time(p::ROFTParams, z; Om0, Or0=0.0, Ol0=1-Om0-Or0)
    p.variant === :energy  && return 2p.alpha*log(E_LCDM(z; Om0=Om0, Or0=Or0, Ol0=Ol0))
    p.variant === :thermal && return p.alpha*log(1+z)
    error("Unknown ROFT variant $(p.variant)")
end

@inline modify_CC(H_geo, z, p; Om0, Or0=0.0, Ol0=1-Om0-Or0) =
    H_geo * exp(-delta_g_time(p, z; Om0=Om0, Or0=Or0, Ol0=Ol0))

@inline function modify_TD(H0, zl, zs, eta, p; Om0, Or0=0.0, Ol0=1-Om0-Or0)
    Δgl = delta_g_time(p, zl; Om0=Om0, Or0=Or0, Ol0=Ol0)
    Δgs = delta_g_time(p, zs; Om0=Om0, Or0=Or0, Ol0=Ol0)
    H0 * exp(eta*Δgl + (1-eta)*Δgs)
end

@inline delta_mu_host(z_host, c_mu, p; Om0, Or0=0.0, Ol0=1-Om0-Or0) =
    -c_mu * delta_g_time(p, z_host; Om0=Om0, Or0=Or0, Ol0=Ol0)

end
