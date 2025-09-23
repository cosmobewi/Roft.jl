# CMB via Capse.jl

Ce document décrit comment configurer et utiliser le pipeline CMB basé sur l’émulateur Capse dans Roft.

## 1. Configuration de l’environnement

```bash
export CAPSE_WEIGHTS_DIR=/opt/trained_emu
# Paramètres de nuisance (facultatifs, valeurs par défaut entre parenthèses)
export CAPSE_NS=0.965      # ns
export CAPSE_AS=2.1e-9     # As
export CAPSE_TAU=0.054     # τ
export CAPSE_OBH2=0.02237  # ωb
export CAPSE_ONUH2=0.00064 # ων
# Ordre des paramètres si différent de [ωb, ωc, h, ns, τ, As]
# export CAPSE_PARAM_ORDER="ωb,ωc,h,ns,τ,As"
```

`CapseEnv.read_nuisance_params()` lit ces variables et retourne `CapseEnv.NuisanceParams`. En cas de valeur absente ou invalide, le fallback par défaut est utilisé et un warning est émis.

## 2. Adapter CMB (`CMBAdapter.jl`)

### 2.1 `load_cmb_data`

```julia
using Roft
cmb = Roft.load_cmb_data(
    "cmb.csv",            # données au format CSV
    "cmb_cov.csv";        # covariance (sans entête, matrice carrée)
    blocks=[:TT, :TE, :EE],
    ell_min=30,
    ell_max=2500,
    as_Dell=true,
    name="Planck"
)
```

- `data_csv` doit contenir une colonne `ell` et des colonnes nommées par bloc (ex. `TT`, `TE`, `EE`, ou `DellTT` …).
- Les blocs sont concaténés dans l’ordre fourni; l’adapter stocke les plages correspondantes dans `cmb.block_ranges`.

### 2.2 `align_data_to_ell!`

Aligner la data sur la grille ℓ de l’émulateur :
```julia
Roft.align_data_to_ell!(cmb, ell_target)
```
`ell_target` est la grille produite par Capse (ex. `model.ell`). Les `block_ranges` sont automatiquement ajustés.

## 3. Charger les émulateurs Capse

`CapseAdapter.load_capse_root_multi(CAPSE_WEIGHTS_DIR; blocks=[:TT,:TE,:EE,:PP])` retourne un dictionnaire de `CapseState` pour chaque bloc disponible, en vérifiant la cohérence de la grille ℓ.

## 4. Construire le modèle CMB

```julia
model = Roft.CMBViaCapse.make_cmb_model_from_env(cmb; blocks=[:TT, :TE])
```
Cette fonction :
- charge automatiquement les émulateurs présents pour les blocs demandés (en ignorant ceux manquants) ;
- aligne `cmb` sur la grille ℓ de l’émulateur ;
- retourne un `CapseCMBModel(blocks=..., states=..., ell=...)`.

### Multi-blocs
Le modèle gère plusieurs blocs (TT/TE/EE/PP). Chaque bloc est évalué via son `CapseState` et concaténé dans un vecteur théorie, puis comparé aux données.

## 5. Calcul du χ² CMB

```julia
chisq = Roft.CMBViaCapse.chi2_cmb_soft_at(H0, Om0, cmb, model)
```
- Lit les paramètres de nuisance avec `CapseEnv.read_nuisance_params()`.
- Construit `CapseAdapter.CapseCMB` (ωb, ωc, h, ns, As, τ) et évalue chaque bloc via `CapseAdapter.predict_cmb`.
- Convertit automatiquement Cℓ ↔ Dℓ selon les flags `model.as_Dell_theory` et `cmb.as_Dell`.
- Retourne le χ² complet en utilisant la covariance fournie.

## 6. Intégration au driver Roft

```julia
cmb = Roft.load_cmb_data(...)
model = Roft.CMBViaCapse.make_cmb_model_from_env(cmb; blocks=[:TT, :TE])
packs = Roft.Driver.Packs(cmb=cmb, cmb_model=model)
χ² = Roft.Driver.chi2_total(packs; bg=background, roft=params)
```
`chi2_total` ajoute automatiquement la contribution CMB si les champs `cmb` et `cmb_model` sont présents dans `Packs`.

## 7. Exemple complet

```julia
using Roft
ENV["CAPSE_WEIGHTS_DIR"] = "/opt/trained_emu"
ENV["CAPSE_NS"] = "0.9649"

cmb = Roft.load_cmb_data("cmb.csv", "cmb_cov.csv"; blocks=[:TT, :TE])
model = Roft.CMBViaCapse.make_cmb_model_from_env(cmb; blocks=[:TT, :TE])
chisq = Roft.CMBViaCapse.chi2_cmb_soft_at(70.0, 0.3, cmb, model)
println("χ² CMB = ", chisq)
```

---
En cas de poids absents ou de bloc manquant, `make_cmb_model_from_env` ignore simplement le bloc. Si vous souhaitez un contrôle plus fin, utilisez directement `CapseAdapter.load_capse_root_multi` et construisez `CapseCMBModel` à la main.
