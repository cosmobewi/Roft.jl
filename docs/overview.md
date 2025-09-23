# Vue d'ensemble des modules Roft

## 1. ROFTSoft

`src/roft/ROFTSoft.jl`
- `ROFTParams(alpha=0.0, variant=:energy | :thermal)` : paramètres pour les corrections ROFT.
- Fonctions principales :
  - `delta_g_time(p, z; Om0, Or0, Ol0)`
  - `modify_CC`, `modify_TD`, `delta_mu_host`
- Tests : voir bloc `"ROFTSoft primitives"` dans `test/runtests.jl`.

## 2. TimeDomain

`src/likelihoods/TimeDomain.jl`
- Structures : `CCData`, `TDData`, `CephSNData`, `TDH0GlobalObs`, `CCObs`.
- χ² disponibles : `chi2_cc`, `chi2_td`, `chi2_cephsn`, `chi2_td_h0_global`, `chi2_cccov`.
- Les imports `Roft.CC`, `Roft.TD`, `Roft.CephSN`, `Roft.CCCov` pointent vers ce module.
- Tests dédiés dans `test/runtests.jl` (bloc "Likelihoods").

## 3. Adapters

### 3.1 Pantheon+ (`src/adapters/PantheonPlusAdapter.jl`)
- `load_pantheonplus`, `load_pantheon_2018` lisent data + covariance.

### 3.2 CCcov (`src/adapters/CCcovAdapter.jl`)
- `load_cccov` ajoute désormais la covariance systématique `data_MM20.dat` (si présente).

### 3.3 H0LiCOW (`src/adapters/H0LiCOWAdapter.jl`)
- `load_h0licow_cosmo_chains` + `meff_from_meta` (lecture stricte, normalisation des poids).

### 3.4 CMB (`src/adapters/CMBAdapter.jl`)
- `load_cmb_data` gère les blocs [:TT, :TE, :EE, :PP], range indices via `block_ranges`.
- `align_data_to_ell!` synchronise les données avec la grille ℓ.

## 4. Capse / CMBViaCapse

`src/likelihoods/CMBViaCapse.jl`
- `CapseCMBModel(blocks, states, ell; as_Dell_theory=true)` : support multi-blocs.
- `make_cmb_model_from_env(cmb; blocks=...)` : charge les émulateurs (via `CAPSE_WEIGHTS_DIR`) et aligne les données.
- `chi2_cmb_soft_at(H0, Om0, cmb, model)` : calcule le χ² complet (conversion Cℓ↔Dℓ, somme sur les blocs).

Documentation détaillée : voir `docs/cmb.md`.

## 5. CapseEnv

`src/config/CapseEnv.jl`
- `read_nuisance_params()` → `NuisanceParams(ns, As, tau, Obh2, Onuh2)`.
- Utilisé par `chi2_cmb_soft_at` pour centraliser la lecture des ENV.

## 6. Driver

`src/driver/Runner.jl`
- `Packs` contient désormais `cmb::CMBData` et `cmb_model::CapseCMBModel` (en plus des autres sondes).
- `chi2_total` additionne automatiquement la contribution CMB si ces champs sont renseignés.

## 7. Tests

`test/runtests.jl` contient des blocs ciblés :
- ROFTSoft, TimeDomain, Pantheon+, CCcov, H0LiCOW
- `CapseEnv` : vérifie la lecture ENV
- `CapseAdapter` : émulateur factice multi-blocs (TT+TE)
- Likelihoods globales.

---
Voir également `docs/cmb.md` pour les détails CMB/Capse.

## 8. Données externes (Makefile)

Un `Makefile` fournit une cible `init` téléchargeant/clonant les sources requises sous `/opt/` :


```
make init
```

Cette commande crée les dossiers et récupère :

- `/opt/DataRelease` : `PantheonPlusSH0ES/DataRelease` (SNe + covariances).
- `/opt/H0LiCOW` : `shsuyu/H0LiCOW-public` (chaînes, métadonnées).
- `/opt/CCcovariance` : `mmoresco/CCcovariance` (données CC + MM20).
- `/opt/Capse` : `CosmologicalEmulators/Capse.jl` (sources émulateur).
- `/opt/planck/COM_Likelihood_Data-baseline_R3.00` : archive Planck téléchargée depuis la PLA (`COSMOLOGY_OID=151902`).
- `/opt/trained_emu` : poids Capse (Zenodo `trained_emu.tar.gz`).

Vous pouvez pointer les variables d’environnement (`CAPSE_WEIGHTS_DIR=/opt/trained_emu`, `CC_DIR=/opt/CCcovariance/data`, etc.) directement sur ces dossiers.
