module Backgrounds
export FlatLCDM, H_of_z, E_LCDM

Base.@kwdef struct FlatLCDM
    H0::Float64
    Om0::Float64
    Or0::Float64 = 0.0
end

@inline E_LCDM(z; Om0, Or0=0.0, Ol0=1-Om0-Or0) =
    sqrt(Or0*(1+z)^4 + Om0*(1+z)^3 + Ol0)

@inline H_of_z(bg::FlatLCDM, z) = bg.H0 * E_LCDM(z; Om0=bg.Om0, Or0=bg.Or0, Ol0=1-bg.Om0-bg.Or0)

end