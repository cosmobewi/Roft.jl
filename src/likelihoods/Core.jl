module LikelihoodsCore
export chi2_gaussian, aic, bic

using LinearAlgebra

@inline chi2_gaussian(d::AbstractVector, m::AbstractVector, C::AbstractMatrix) = begin
    r = d .- m
    dot(r, C \ r)
end

@inline aic(chi2min::Real, k::Integer) = chi2min + 2k
@inline bic(chi2min::Real, k::Integer, N::Integer) = chi2min + k*log(N)

end
