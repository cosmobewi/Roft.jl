#!/usr/bin/env julia

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

for dep in ("LinearAlgebra","CSV","DataFrames","Distributions","QuadGK")
    @eval using $(Symbol(dep))
end

using Roft
const Backgrounds = Roft.Backgrounds
const ROFTSoft = Roft.ROFTSoft
const TimeDomain = Roft.TimeDomain
const CapseEnv = Roft.CapseEnv
const CMBAdapter = Roft.CMBAdapter
const CMBViaCapse = Roft.CMBViaCapse
const PlanckLiteAdapter = Roft.PlanckLiteAdapter

const PP = Roft.PantheonPlus
const CCcov = Roft.CCCov

using LinearAlgebra, Distributions, CSV, DataFrames, QuadGK

# ---------------- Cosmology helpers ----------------
@inline E_LCDM(z; Om0, Or0=0.0, Ol0=1-Om0-Or0) = sqrt(Or0*(1+z)^4 + Om0*(1+z)^3 + Ol0)
const C_KMS = 299_792.458

const PHI_GS = (sqrt(5) - 1) / 2

function golden_section_minimum(f, a::Float64, b::Float64;
                                tol::Float64=1e-3, maxiter::Int=100)
    a >= b && error("golden_section_minimum requires a < b")
    c = b - PHI_GS*(b - a)
    d = a + PHI_GS*(b - a)
    fc = f(c)
    fd = f(d)
    iter = 0
    while (b - a) > tol && iter < maxiter
        if fc < fd
            b = d
            d = c
            fd = fc
            c = b - PHI_GS*(b - a)
            fc = f(c)
        else
            a = c
            c = d
            fc = fd
            d = a + PHI_GS*(b - a)
            fd = f(d)
        end
        iter += 1
    end
    xmid = (a + b)/2
    fmid = f(xmid)
    best_x = xmid
    best_f = fmid
    if fc < best_f
        best_x, best_f = c, fc
    end
    if fd < best_f
        best_x, best_f = d, fd
    end
    return best_f, best_x, iter
end

function DC_LCDM(z; H0, Om0, Or0=0.0)
    integrand(zz) = 1 / E_LCDM(zz; Om0=Om0, Or0=Or0, Ol0=1-Om0-Or0)
    val, _ = quadgk(integrand, 0.0, z; rtol=1e-8)
    return (C_KMS / H0) * val
end

DL_LCDM(z; H0, Om0, Or0=0.0) = (1 + z) * DC_LCDM(z; H0=H0, Om0=Om0, Or0=Or0)
mu_of_z(z; H0, Om0, Or0=0.0) = 5*log10(DL_LCDM(z; H0=H0, Om0=Om0, Or0=Or0)) + 25

@inline function Δg_time(z::Real, alpha::Real, variant::Symbol, Om0::Real, Or0::Real)
    if variant === :energy
        return 2alpha*log(E_LCDM(z; Om0=Om0, Or0=Or0))
    elseif variant === :thermal
        return alpha*log(1+z)
    else
        return 0.0
    end
end

# ---------------- Background ----------------
bg = Backgrounds.FlatLCDM(H0=70.0, Om0=0.3, Or0=0.0)

# ---------------- Pantheon+ cache ----------------
const SN_CACHE_BUILT = Ref(false)
const SN_N = Ref(0)
const SN_A = Ref(0.0)
const SN_B = Ref(0.0)
const SN_CHI2_REF = Ref(0.0)
const H0_REF = Ref(bg.H0)
sigma_M = parse(Float64, get(ENV, "SN_SIGMA_M", "0.03"))

function build_sn_cache()
    pantheon_dir = get(ENV, "PANTHEONPLUS_DIR", joinpath(@__DIR__, "..", "DataRelease", "Pantheon+_Data"))
    data_file = joinpath(pantheon_dir, "Pantheon+SH0ES.dat")
    cov_file  = joinpath(pantheon_dir, "Pantheon+SH0ES_STAT+SYS.cov")
    if !isfile(data_file) || !isfile(cov_file)
        @warn "Pantheon+ files not found ⇒ SN block disabled" data_file cov_file
        SN_CACHE_BUILT[] = false
        return
    end

    sn = Roft.load_pantheonplus(data_file, cov_file; zcol=:zHD, mucol=:MU_SH0ES)
    N = length(sn.z)
    SN_N[] = N
    println("Pantheon+ loaded (N = $N, σ_M = $sigma_M mag)")

    mu_base = mu_of_z.(sn.z; H0=H0_REF[], Om0=bg.Om0, Or0=bg.Or0)
    r_ref = sn.mu .- mu_base
    one = ones(N)
    Cfac = cholesky(Symmetric(sn.Cov))
    u1 = Cfac \ one
    u2 = Cfac \ r_ref
    SN_A[] = dot(one, u1)
    SN_B[] = dot(one, u2)
    SN_CHI2_REF[] = dot(r_ref, u2)
    SN_CACHE_BUILT[] = true
end

build_sn_cache()

@inline function chi2_sn_profiled(H0::Float64)
    SN_CACHE_BUILT[] || return 0.0
    Δ = -5.0 * log10(H0 / H0_REF[])
    A = SN_A[]
    B = SN_B[]
    χ2ref = SN_CHI2_REF[]
    A′ = A + 1/(sigma_M^2)
    B′ = (B - Δ*A)
    χ2_r = χ2ref - 2Δ*B + Δ^2*A
    return χ2_r - (B′^2)/A′
end

# ---------------- CC datasets ----------------
cc_dir = get(ENV, "CC_DIR", "")
cc_flavor = Symbol(get(ENV, "CC_FLAVOR", "both") |> uppercase)
ccobs_list = TimeDomain.CCObs[]

if !isempty(cc_dir) && isdir(cc_dir)
    flavors = cc_flavor === :BC03 ? [:BC03] : cc_flavor === :M11 ? [:M11] : [:BC03, :M11]
    for fl in flavors
        push!(ccobs_list, Roft.load_cccov(data_dir=cc_dir, flavor=fl, name="CC $(String(fl))"))
    end
    println("CC datasets loaded: ", join(getfield.(ccobs_list, :name), ", "))
else
    @warn "CC_DIR not set ⇒ using mock CC data"
    push!(ccobs_list, TimeDomain.CCObs(z=[0.2,0.5,1.0], H=[78.0,96.0,130.0],
                                      Cov=Symmetric(Diagonal([5.0,6.0,8.0].^2)), name="MockCC"))
end

function cc_stats(alpha::Float64, variant::Symbol)
    A = 0.0; B = 0.0; y = 0.0
    for cc in ccobs_list
        z = cc.z
        mod = @. E_LCDM(z; Om0=bg.Om0, Or0=bg.Or0, Ol0=1-bg.Om0-bg.Or0) * exp(-Δg_time(z, alpha, variant, bg.Om0, bg.Or0))
        u = cc.Cov \ mod
        v = cc.Cov \ cc.H
        A += dot(mod, u)
        B += dot(mod, v)
        y += dot(cc.H, v)
    end
    return A, B, y
end

# ---------------- CMB via Capse ----------------
cmb_data_csv = get(ENV, "CAPSE_CMB_DATA", "")
cmb_cov_csv  = get(ENV, "CAPSE_CMB_COV", "")
cmb_blocks = split(get(ENV, "CAPSE_CMB_BLOCKS", "TT,TE,EE"), ','; keepempty=false) .|> Symbol
cmb_ell_min = parse(Int, get(ENV, "CAPSE_CMB_ELL_MIN", "30"))
cmb_ell_max = parse(Int, get(ENV, "CAPSE_CMB_ELL_MAX", "2500"))
cmb_as_Dell = lowercase(get(ENV, "CAPSE_CMB_AS_DELL", "true")) in ("1","true","yes")

CMB_DATA = nothing
CMB_MODEL = nothing

plik_dir = get(ENV, "PLANCK_PLIK_DIR", "")

if !isempty(plik_dir) && isdir(plik_dir)
    CMB_DATA = PlanckLiteAdapter.load_pliklite(plik_dir;
                                               ell_min=cmb_ell_min,
                                               ell_max=cmb_ell_max,
                                               as_Dell=cmb_as_Dell,
                                               name="Planck PlikLite")
    CMB_MODEL = CMBViaCapse.make_cmb_model_from_env(CMB_DATA; blocks=CMB_DATA.blocks)
    println("Planck PlikLite carregé via PlanckLiteAdapter (N = $(length(CMB_DATA.vec))).")
elseif !isempty(cmb_data_csv) && !isempty(cmb_cov_csv)
    CMB_DATA = Roft.load_cmb_data(cmb_data_csv, cmb_cov_csv;
                                 blocks=cmb_blocks, ell_min=cmb_ell_min, ell_max=cmb_ell_max,
                                 as_Dell=cmb_as_Dell, name="CMB")
    CMB_MODEL = CMBViaCapse.make_cmb_model_from_env(CMB_DATA; blocks=cmb_blocks)
    println("CMB via Capse chargé (blocs = $(cmb_blocks), N = $(length(CMB_DATA.vec)))")
else
    @info "CMB non chargée (définis PLANCK_PLIK_DIR ou CAPSE_CMB_DATA & CAPSE_CMB_COV pour activer)."
end

chi2_cmb_soft_at(H0, Om0) = CMB_DATA === nothing ? 0.0 :
    CMBViaCapse.chi2_cmb_soft_at(H0, Om0, CMB_DATA, CMB_MODEL)

function chi2_breakdown(alpha::Real, H0::Real, variant::Symbol)
    A_cc, B_cc, y_cc = cc_stats(alpha, variant)
    χ2_cc = chi2_cc_td(H0, A_cc, B_cc, y_cc)

    A_td, B_td, y_td = td_stats(alpha, variant)
    χ2_td = chi2_cc_td(H0, A_td, B_td, y_td)

    χ2_sn = chi2_sn_profiled(H0)
    χ2_cmb = chi2_cmb_soft_at(H0, bg.Om0)

    return (; sn=χ2_sn, cc=χ2_cc, td=χ2_td, cmb=χ2_cmb)
end

# ---------------- H0LiCOW ----------------
td_dir   = get(ENV, "H0LICOW_DIR", "")
td_model = Symbol(get(ENV, "H0LICOW_MODEL", "FLCDM"))
td_meta  = get(ENV, "H0LICOW_META", "")

td_obs = nothing
if !isempty(td_dir) && !isempty(td_meta) && isdir(td_dir) && isfile(td_meta)
    td_obs = Roft.load_h0licow_cosmo_chains(td_dir; model=td_model, meta_csv=td_meta)
    println("H0LiCOW global: H0 = $(td_obs.H0_mean) ± $(td_obs.H0_std) km/s/Mpc")
else
    @warn "H0LiCOW not loaded (set H0LICOW_DIR, H0LICOW_META)."
end

function td_stats(alpha::Float64, variant::Symbol)
    td_obs === nothing && return 0.0, 0.0, 0.0
    Efun = z -> E_LCDM(z; Om0=bg.Om0, Or0=bg.Or0, Ol0=1-bg.Om0-bg.Or0)
    eta_vec = td_obs.eta
    if haskey(ENV, "ETA_GLOBAL")
        eta_g = parse(Float64, ENV["ETA_GLOBAL"])
        eta_vec = fill(eta_g, length(eta_vec))
    end
    M = Roft.meff_from_meta(alpha, variant, td_obs.zl, td_obs.zs, eta_vec, td_obs.w, Efun)
    σ2 = td_obs.H0_std^2
    A = (M*M)/σ2
    B = (M*td_obs.H0_mean)/σ2
    y = (td_obs.H0_mean^2)/σ2
    return A, B, y
end

# ---------------- Quadratic combination (CC+TD) ----------------
@inline chi2_cc_td(H0, A, B, y) = A*H0*H0 - 2B*H0 + y

function chi2_total_min_over_H0(alpha::Real, variant::Symbol;
                                H0min::Real=parse(Float64, get(ENV, "ROFT_H0_MIN", "50")),
                                H0max::Real=parse(Float64, get(ENV, "ROFT_H0_MAX", "85")),
                                tol::Real=parse(Float64, get(ENV, "ROFT_H0_TOL", "1e-3")))
    A1, B1, y1 = cc_stats(alpha, variant)
    A2, B2, y2 = td_stats(alpha, variant)
    A = A1 + A2
    B = B1 + B2
    y = y1 + y2
    χ2_fun = H0 -> chi2_cc_td(H0, A, B, y) + chi2_sn_profiled(H0) + chi2_cmb_soft_at(H0, bg.Om0)
    best_χ2, best_H0, _ = golden_section_minimum(χ2_fun, float(H0min), float(H0max); tol=float(tol))
    return best_χ2, best_H0, A, B, y
end

function estimate_sigma_alpha(alphas::Vector{Float64}, chi2s::Vector{Float64}; window::Int=4)
    imin = argmin(chi2s)
    a_star = alphas[imin]
    χ2_star = chi2s[imin]
    i_lo = max(1, imin - window)
    i_hi = min(length(alphas), imin + window)
    xs = alphas[i_lo:i_hi] .- a_star
    ys = chi2s[i_lo:i_hi] .- χ2_star
    denom = sum(xs.^4)
    a_par = denom == 0 ? NaN : sum(xs.^2 .* ys) / denom
    return (isfinite(a_par) && a_par > 0) ? inv(sqrt(a_par)) : NaN
end

using Base.Threads
function scan_variant(variant::Symbol; a_min=-0.5, a_max=0.5, n=301)
    alphas = collect(range(a_min, a_max; length=n))
    chi2_tot = similar(alphas)
    H0_hats = similar(alphas)
    chi2_cc_td_min = similar(alphas)

    @threads for i in eachindex(alphas)
        a = alphas[i]
        χ2min, H0hat, A, B, y = chi2_total_min_over_H0(a, variant)
        chi2_tot[i] = χ2min
        H0_hats[i] = H0hat
        chi2_cc_td_min[i] = chi2_cc_td(H0hat, A, B, y)
    end

    χ2_LCDM, H0_hat0, _, _, _ = chi2_total_min_over_H0(0.0, variant)
    imin = argmin(chi2_tot)
    alpha_star = alphas[imin]
    chi2_star = chi2_tot[imin]
    H0_star = H0_hats[imin]
    dchi2 = χ2_LCDM - chi2_star
    pval = 1 - cdf(Chisq(1), max(0, dchi2))
    sigma_app = sqrt(max(0, dchi2))
    sigma_alpha = estimate_sigma_alpha(alphas, chi2_tot)

    N_cc = sum(length(cc.z) for cc in ccobs_list)
    N_sn = SN_CACHE_BUILT[] ? SN_N[] : 0
    N_td = td_obs === nothing ? 0 : 1
    N_tot = N_cc + N_sn + N_td

    ΔAIC = dchi2 - 2
    ΔBIC = dchi2 - log(max(N_tot, 1))

    return (; variant, alpha_star, chi2_star, χ2_LCDM, dchi2, pval, sigma_app, sigma_alpha,
             H0_star, H0_hat0, alphas, chi2_tot, chi2_cc_td_min, H0_hats,
             N_cc, N_sn, N_td, N_tot, ΔAIC, ΔBIC)
end

function print_scan_summary(res)
    n_alpha = length(res.alphas)
    a_min = res.alphas[1]
    a_max = res.alphas[end]
    println("=== Variant: $(res.variant) ===")
    println("  α range: [$a_min, $a_max] with $n_alpha samples")
    println("  α* = $(res.alpha_star) ± $(res.sigma_alpha)")
    println("  χ²* = $(res.chi2_star) (ΛCDM: $(res.χ2_LCDM), Δχ² = $(res.dchi2))")
    println("  Significance: σ_app = $(res.sigma_app), p-value = $(res.pval)")
    println("  H0* = $(res.H0_star) km/s/Mpc (ΛCDM best-fit H0 = $(res.H0_hat0))")
    println("  Data counts: N_SN = $(res.N_sn), N_CC = $(res.N_cc), N_TD = $(res.N_td), N_total = $(res.N_tot)")
    println("  Information criteria: ΔAIC = $(res.ΔAIC), ΔBIC = $(res.ΔBIC)")

    comps_star = chi2_breakdown(res.alpha_star, res.H0_star, res.variant)
    comps_lcdm = chi2_breakdown(0.0, res.H0_hat0, res.variant)
    fmt(x) = round(x, digits=3)
    println("  χ²@best: SN=$(fmt(comps_star.sn)), CC=$(fmt(comps_star.cc)), TD=$(fmt(comps_star.td)), CMB=$(fmt(comps_star.cmb))")
    println("  χ²@ΛCDM: SN=$(fmt(comps_lcdm.sn)), CC=$(fmt(comps_lcdm.cc)), TD=$(fmt(comps_lcdm.td)), CMB=$(fmt(comps_lcdm.cmb))")
end

res_energy = scan_variant(:energy)
print_scan_summary(res_energy)

res_thermal = scan_variant(:thermal)
print_scan_summary(res_thermal)

out_dir = get(ENV, "ROFT_OUT_DIR", joinpath(@__DIR__, "..", "out"))
mkpath(out_dir)

function save_profile(path::String, alphas, chi2_total, chi2_cc_td_min, H0_hat)
    df = DataFrame(alpha=alphas, chi2_total=chi2_total,
                   chi2_cc_td_at_H0hat=chi2_cc_td_min, H0_hat=H0_hat)
    mkpath(dirname(path))
    CSV.write(path, df)
end

save_profile(joinpath(out_dir, "mix_H0profile_alpha_energy.csv"),
             res_energy.alphas, res_energy.chi2_tot, res_energy.chi2_cc_td_min, res_energy.H0_hats)
save_profile(joinpath(out_dir, "mix_H0profile_alpha_thermal.csv"),
             res_thermal.alphas, res_thermal.chi2_tot, res_thermal.chi2_cc_td_min, res_thermal.H0_hats)

println("Profiles written in $(out_dir)")
