# tools/planck_like_to_cmb_csv.jl
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "ROFT.jl"))
for dep in ("CSV","DataFrames","LinearAlgebra","NPY")
    try; @eval using $(Symbol(dep))
    catch; Pkg.add(dep); @eval using $(Symbol(dep)); end
end
using CSV, DataFrames, LinearAlgebra, NPY

# ----------------- ENV attendus -----------------
# PLANCK_DIR         : racine de ton repo (cloné en local)
# CAPSE_WEIGHTS_DIR  : racine des poids Capse (ex: trained_emu)
# PL_BLOCKS          : blocs à utiliser, ex "TT,TE,EE" (défaut idem)
# PL_TT / PL_TE / PL_EE / PL_PP : chemins (relatifs ou absolus) vers fichiers data par bloc
# PL_DATA_FORMAT     : "csv" (ell + colonne valeur) | "two-col" (ell,val) | "single" (juste val, alors on prend la grille de Capse)
# PL_ELL_COL         : nom de la colonne ell dans tes CSV (déf: "ell")
# PL_VAL_COL         : nom de la colonne valeur dans tes CSV (déf: "Dl" ; sinon "Cl")
# PL_IS_DELL         : "true|false" → tes values sont des Dℓ ? (déf true)
# COV_PATH           : chemin de la covariance (csv|npy|cov triangulaire ou dense)
# OUT_PREFIX         : préfixe de sortie (déf: ../cmb_from_repo)
#
# Note: si tu ne fournis pas PL_TT/PL_TE/… on essaie quelques motifs simples dans PLANCK_DIR.

function read_ellgrid_from_capse(root::AbstractString; prefer::Symbol=:TT)
    path = joinpath(root, String(prefer), "l.npy")
    @assert isfile(path) "Grille ℓ Capse introuvable: $path"
    vec = npyread(path)
    return collect(Int.(vec))
end

# lit un bloc data et retourne (ell, val)
function read_block(path::AbstractString; fmt::Symbol, ell_from_capse::Vector{Int},
                    ellcol::String, valcol::String)
    @assert isfile(path) "Fichier bloc introuvable: $path"
    if fmt == :csv
        df = CSV.read(path, DataFrame)
        @assert hasproperty(df, Symbol(ellcol)) "Colonne $ellcol absente de $path"
        @assert hasproperty(df, Symbol(valcol)) "Colonne $valcol absente de $path"
        ell = Int.(df[!, Symbol(ellcol)])
        val = Float64.(df[!, Symbol(valcol)])
        return ell, val
    elseif fmt == :two_col
        df = CSV.read(path, DataFrame; header=false)
        @assert ncol(df) >= 2 "two-col: besoin de 2 colonnes (ell,val) dans $path"
        ell = Int.(df[:,1]); val = Float64.(df[:,2]); return ell, val
    elseif fmt == :single
        # pas d'ell → on impose celle de Capse
        df = CSV.read(path, DataFrame; header=false)
        @assert ncol(df) >= 1 "single: besoin d'1 colonne (val) dans $path"
        val = Float64.(df[:,1])
        @assert length(val) == length(ell_from_capse) "longueur(val) ≠ longueur(ℓ Capse) dans $path"
        return ell_from_capse, val
    else
        error("format inconnu: $fmt")
    end
end

function to_Dell!(v::Vector{Float64}, ell::Vector{Int})
    @. v = ell*(ell+1)/(2π) * v
    return v
end

function _guess_file(dir::String, patts::Vector{String})
    for p in patts
        c = filter(isfile, joinpath.(dir, readdir(dir)))
        for f in c
            if occursin(lowercase(p), lowercase(basename(f)))
                return f
            end
        end
    end
    return ""
end

function read_covariance_any(path::AbstractString, N::Int)
    @assert isfile(path) "Covariance introuvable: $path"
    low = lowercase(path)
    if endswith(low,".npy")
        C = Array{Float64}(npyread(path))
        @assert size(C,1)==size(C,2)==N "cov.npy doit être NxN"
        return C
    elseif endswith(low,".csv") || endswith(low,".txt")
        C = Matrix{Float64}(CSV.read(path, DataFrame; header=false))
        @assert size(C,1)==size(C,2)==N "cov.csv doit être NxN"
        return C
    elseif endswith(low,".cov")
        lines = readlines(path)
        @assert !isempty(lines)
        Ne = parse(Int, split(strip(lines[1]))[1])
        @assert Ne==N "N dans .cov ($Ne) ≠ N attendu ($N)"
        vals = parse.(Float64, strip.(lines[2:end]))
        M = length(vals)
        if M == N*N
            return reshape(vals, N, N)
        elseif M == N*(N+1)÷2
            C = zeros(N,N); k=1
            for j in 1:N, i in j:N
                C[i,j]=vals[k]; C[j,i]=vals[k]; k+=1
            end
            return C
        else
            error(".cov inattendu: $path")
        end
    else
        error("Format covariance non supporté: $path")
    end
end

# ----------------- main -----------------
planck_dir = get(ENV,"PLANCK_DIR","")
capse_root = get(ENV,"CAPSE_WEIGHTS_DIR","")
@assert !isempty(planck_dir) && isdir(planck_dir) "PLANCK_DIR doit pointer vers ton repo cloné"
@assert !isempty(capse_root) && isdir(capse_root) "CAPSE_WEIGHTS_DIR doit pointer vers trained_emu"

blocks = Symbol.(split(get(ENV,"PL_BLOCKS","TT,TE,EE"), ','; keepempty=false))
fmt = Symbol(get(ENV,"PL_DATA_FORMAT","csv"))  # csv|two-col|single
ellcol = get(ENV,"PL_ELL_COL","ell")
valcol = get(ENV,"PL_VAL_COL","Dl")            # ou "Cl"
is_Dℓ = lowercase(get(ENV,"PL_IS_DELL","true")) in ("1","true","yes")
out_prefix = get(ENV,"OUT_PREFIX", joinpath(@__DIR__, "..", "cmb_from_repo"))

ℓemu = read_ellgrid_from_capse(capse_root)
# lit les blocs
bl2path = Dict{Symbol,String}()
for b in blocks
    envk = "PL_"*String(b)
    if haskey(ENV, envk) && !isempty(ENV[envk])
        bl2path[b] = ENV[envk]
    else
        # essais grossiers
        g = _guess_file(planck_dir, ["$(String(b))", lowercase(String(b))])
        @assert g!=""; bl2path[b]=g
    end
end

# assemble DataFrame
acc_cols = Dict{Symbol,Vector{Float64}}()
for b in blocks
    ell, val = read_block(bl2path[b]; fmt=fmt, ell_from_capse=ℓemu, ellcol=ellcol, valcol=valcol)
    # aligne à ℓ Capse
    if ell != ℓemu
        # garde uniquement les modes présents ET dans ℓemu (strict)
        map_ell = Dict(ell .=> collect(1:length(ell)))
        idx = Int[]
        vals = Float64[]
        for (k, L) in enumerate(ℓemu)
            if haskey(map_ell, L)
                push!(idx, map_ell[L])
                push!(vals, val[map_ell[L]])
            else
                error("ℓ=$L manquant dans $(bl2path[b]). Fournis la même grille que l'émulateur.")
            end
        end
        val = vals
    end
    # si tes valeurs sont en Cl mais data attend Dℓ
    if is_Dℓ == false
        val = copy(val); to_Dell!(val, ℓemu)
    end
    acc_cols[b] = val
end

df = DataFrame(ell=ℓemu)
for b in blocks
    df[!, b] = acc_cols[b]
end
data_path = out_prefix * "_data.csv"
CSV.write(data_path, df)

# covariance
cov_path_in = get(ENV,"COV_PATH","")
if isempty(cov_path_in)
    # essaie de trouver un fichier "cov" dans PLANCK_DIR
    cand = filter(f->occursin("cov", lowercase(f)), readdir(planck_dir; join=true))
    @assert !isempty(cand) "COV_PATH non fourni et aucun fichier *cov* trouvé dans $planck_dir"
    cov_path_in = cand[1]
end
# vecteur aplati doit suivre l’ordre des blocs
vec = Float64[]
for b in blocks; append!(vec, acc_cols[b]); end
N = length(vec)
C = read_covariance_any(cov_path_in, N)
cov_out = out_prefix * "_cov.csv"
CSV.write(cov_out, Tables.table(C))

println("OK :")
println("  data = $data_path")
println("  cov  = $cov_out")
println("  blocs=$(blocks), ℓ∈[$(first(ℓemu)),$(last(ℓemu))], Nvec=$N")
