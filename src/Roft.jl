module Roft

# Re-exports
export ROFTParams, FlatLCDM, H_of_z,
       TimeDomain, CCData, TDData, CephSNData, TDH0GlobalObs, CCObs,
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
include("adapters/CMBAdapter.jl")
using .CapseAdapter: CapseCMB, cmb_chi2
using .CMBAdapter: CMBData, load_cmb_data, align_data_to_ell!
export CapseAdapter, CMBAdapter

# Likelihoods
include("likelihoods/Core.jl")
include("likelihoods/TimeDomain.jl")
include("likelihoods/PantheonPlus.jl")
include("likelihoods/CMBViaCapse.jl")

using .LikelihoodsCore: chi2_gaussian, aic, bic
using .TimeDomain: CCData, TDData, CephSNData, TDH0GlobalObs, CCObs,
                    chi2_cc, chi2_td, chi2_cephsn, chi2_td_h0_global, chi2_cccov
using .PantheonPlus: chi2_sn_profiledM
using .CMBViaCapse: CapseCMBModel, chi2_cmb_soft_at

const CC = TimeDomain
const TD = TimeDomain
const CephSN = TimeDomain
const H0LiCOWCosmoChains = TimeDomain
const CCCov = TimeDomain

# Adapters
include("adapters/PantheonPlusAdapter.jl")
include("adapters/CCcovAdapter.jl")
include("adapters/H0LiCOWAdapter.jl")

using .PantheonPlusAdapter: load_pantheonplus, load_pantheon_2018
using .CCcovAdapter: load_cccov

const CCCovAdapter = CCcovAdapter
using .H0LiCOWAdapter: load_h0licow_cosmo_chains, meff_from_meta

export PantheonPlusAdapter, CCcovAdapter, CCCovAdapter, H0LiCOWAdapter, CMBAdapter
export load_pantheonplus, load_pantheon_2018, load_cccov, load_cmb_data, align_data_to_ell!,
       load_h0licow_cosmo_chains, meff_from_meta

export CC, TD, CephSN, H0LiCOWCosmoChains, CCCov

# Driver
include("driver/Runner.jl")

end # module
