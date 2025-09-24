module PlanckLiteAdapter

export load_pliklite

using ..CMBAdapter: CMBData
using LinearAlgebra: Symmetric

const DEFAULT_DATA_FILE = "cl_cmb_plik_v22.dat"
const DEFAULT_COV_FILE = "c_matrix_plik_v22.dat"

"""
    read_numeric_table(path)

Lit un fichier texte contenant un tableau numérique (tolère commentaires et
séparateurs variés) et retourne une matrice de `Float64`.
"""
function read_numeric_table(path::AbstractString)
    @assert isfile(path) "Missing file: $path"
    raw = read(path, String)
    # uniformise les séparateurs pour faciliter le split
    raw = replace(raw, ',' => ' ', ';' => ' ', '"' => ' ')
    raw = replace(raw, r"[^0-9eE\+\-\. \t\r\n]" => " ")

    rows = Vector{Vector{Float64}}()
    for ln in split(raw, '\n')
        ln = strip(ln)
        isempty(ln) && continue
        startswith(ln, '#') && continue
        tokens = split(ln)
        isempty(tokens) && continue
        vals = Float64[]
        ok = true
        for tok in tokens
            parsed = tryparse(Float64, tok)
            if parsed === nothing
                ok = false
                break
            end
            push!(vals, parsed)
        end
        ok && push!(rows, vals)
    end

    @assert !isempty(rows) "No numeric rows found in $path"
    ncol = maximum(length.(rows))
    @assert all(length(r) == ncol for r in rows) "Ragged numeric rows in $path"

    data = Matrix{Float64}(undef, length(rows), ncol)
    for (i, r) in enumerate(rows)
        data[i, :] = r
    end
    return data
end

"""
    read_covariance_auto(path, N)

Charge la covariance TT×TT du paquet Plik-lite. Détecte automatiquement le
format (ASCII ou binaire Fortran non formaté).
"""
function read_covariance_auto(path::AbstractString, N::Int)
    # Essaye d'abord le parse ASCII
    try
        Ctxt = read_numeric_table(path)
        return Ctxt[1:N, 1:N]
    catch
        # sinon bascule vers une lecture binaire brute
    end

    bytes = read(path)
    needed = N * N * sizeof(Float64)
    if length(bytes) == needed
        return reshape(copy(reinterpret(Float64, bytes)), N, N)
    elseif length(bytes) == needed + 8
        payload = @view bytes[5:end-4]
        @assert length(payload) == needed
        return reshape(copy(reinterpret(Float64, payload)), N, N)
    else
        error("Can't decode '$path' as $(N)×$(N) Float64: size=$(length(bytes)) bytes, expected $needed or $(needed + 8).")
    end
end

"""
    load_pliklite(dir; data_file="cl_cmb_plik_v22.dat", cov_file="c_matrix_plik_v22.dat",
                   ell_min=30, ell_max=2500, name="Planck PlikLite TT", as_Dell=true)

Lit un dossier Plik-lite et retourne un objet `CMBData` compatible `CMBAdapter`.
Pour l'instant seul le bloc TT est construit.
"""
function load_pliklite(dir::AbstractString;
                       data_file::AbstractString = DEFAULT_DATA_FILE,
                       cov_file::AbstractString = DEFAULT_COV_FILE,
                       ell_min::Int = 30,
                       ell_max::Int = 2500,
                       name::AbstractString = "Planck PlikLite TT",
                       as_Dell::Bool = true)
    dat_path = joinpath(dir, data_file)
    cov_path = joinpath(dir, cov_file)

    data = read_numeric_table(dat_path)
    nrow, ncol = size(data)
    @assert nrow > 0 "No data rows found in $(basename(dat_path))"
    @assert ncol >= 2 "Unexpected column count ($ncol) in $(basename(dat_path))"

    ℓ_full = Int.(round.(data[:, 1]))
    tt_full = data[:, 2]

    mask = (ℓ_full .>= ell_min) .& (ℓ_full .<= ell_max)
    @assert any(mask) "No multipoles within [$(ell_min), $(ell_max)] in $(basename(dat_path))"
    sel = findall(mask)

    ℓ = ℓ_full[sel]
    tt = tt_full[sel]
    fac = @. (ℓ * (ℓ + 1)) / (2π)
    if as_Dell
        tt = fac .* tt
    end

    C = read_covariance_auto(cov_path, nrow)
    C_sel = C[sel, sel]
    if as_Dell
        scaling = fac
        C_sel = C_sel .* (scaling * scaling')
    end
    Cov = Symmetric((C_sel + C_sel') / 2)

    blocks = [:TT]
    vec = copy(tt)
    block_ranges = Dict(:TT => 1:length(vec))

    return CMBData(name=name, ell=ℓ, vec=vec, Cov=Cov,
                   blocks=blocks, block_ranges=block_ranges, as_Dell=as_Dell)
end

end
