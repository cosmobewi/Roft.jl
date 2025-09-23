module Roft

# Re-exports
export ROFTParams, FlatLCDM, H_of_z,
       chi2_cc, chi2_td, chi2_cephsn, chi2_gaussian, chi2_total,
       aic, bic, CapseCMB, cmb_chi2

# Backgrounds (flat LCDM helpers)
include("backgrounds/FlatLCDM.jl")
using .Backgrounds: FlatLCDM, H_of_z, E_LCDM
export Backgrounds

# Core ROFT (soft)
include("roft/ROFTSoft.jl")
using .ROFTSoft: ROFTParams, delta_g_time, modify_CC, modify_TD, delta_mu_host
export ROFTSoft

# Adapters (Capse)
include("adapters/CapseAdapter.jl")
using .CapseAdapter: CapseCMB, cmb_chi2
export CapseAdapter

# Likelihoods
include("likelihoods/Core.jl")
include("likelihoods/CC.jl")
include("likelihoods/TD.jl")
include("likelihoods/CephSN.jl")
include("likelihoods/PantheonPlus.jl")
include("likelihoods/CCCov.jl")
include("likelihoods/H0LiCOWCosmoChains.jl")
include("likelihoods/CMBViaCapse.jl")

using .LikelihoodsCore: chi2_gaussian, aic, bic
using .CC: chi2_cc
using .TD: chi2_td
using .CephSN: chi2_cephsn
using .PantheonPlus: chi2_sn_profiledM
using .CCCov: chi2_cccov
using .H0LiCOWCosmoChains: chi2_td_h0_global
using .CMBViaCapse: chi2_cmb_soft_at

# Adapters
include("adapters/PantheonPlusAdapter.jl")
include("adapters/CCCovAdapter.jl")
include("adapters/H0LiCOWAdapter.jl")

using .PantheonPlusAdapter: load_pantheonplus, load_pantheon_2018
using .CCCovAdapter: load_cccov
using .H0LiCOWAdapter: load_h0licow_cosmo_chains, meff_from_meta

export PantheonPlusAdapter, CCCovAdapter, H0LiCOWAdapter
export load_pantheonplus, load_pantheon_2018, load_cccov,
       load_h0licow_cosmo_chains, meff_from_meta

# Driver
include("driver/Runner.jl")

end # module
