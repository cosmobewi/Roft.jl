# ROFT.jl/tools/pliklite_to_csv.jl
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "ROFT.jl"))
for dep in ("DelimitedFiles","DataFrames","CSV")
    try
        @eval using $(Symbol(dep))
    catch
        Pkg.add(dep); @eval using $(Symbol(dep))
    end
end

using DelimitedFiles, DataFrames, CSV

# --- robust numeric table reader (tolère commentaires/texte parasites)
function read_numeric_table(path::AbstractString)
    @assert isfile(path) "Missing file: $path"
    s = read(path, String)
    s = replace(s, ',' => ' ', ';' => ' ', '"' => ' ')
    s = replace(s, r"[^0-9eE\+\-\. \t\r\n]" => " ")
    rows = Vector{Vector{Float64}}()
    for ln in split(s, '\n')
        ln = strip(ln); isempty(ln) && continue
        startswith(ln, '#') && continue
        toks = split(ln); isempty(toks) && continue
        vals = Float64[]
        ok = true
        for t in toks
            v = tryparse(Float64, t)
            if v === nothing; ok=false; break; end
            push!(vals, v)
        end
        ok && push!(rows, vals)
    end
    @assert !isempty(rows) "No numeric rows found in $path"
    ncol = maximum(length.(rows))
    @assert all(length(r)==ncol for r in rows) "Ragged numeric rows in $path"
    A = Matrix{Float64}(undef, length(rows), ncol)
    for (i,r) in enumerate(rows); A[i,:] = r; end
    return A
end

# --- nouveau : lecture auto binaire/texte pour la covariance
function read_covariance_auto(path::AbstractString, N::Int)
    # 1) tente ASCII
    try
        Ctxt = read_numeric_table(path)
        return Ctxt[1:N, 1:N]
    catch
        # tombe en binaire
    end
    bytes = read(path)                # Vector{UInt8}
    L = length(bytes)
    need = N*N*8
    if L == need
        C = reshape(copy(reinterpret(Float64, bytes)), N, N)
        return C
    elseif L == need + 8
        # Fortran unformatted: 4 bytes length + payload + 4 bytes length
        payload = @view bytes[5:end-4]
        @assert length(payload) == need
        C = reshape(copy(reinterpret(Float64, payload)), N, N)
        return C
    else
        error("Can't decode '$path' as $N×$N Float64: size=$L bytes, expected $need or $(need+8).")
    end
end


function main(dir::AbstractString)
    dat_path = joinpath(dir, "cl_cmb_plik_v22.dat")
    cov_path = joinpath(dir, "c_matrix_plik_v22.dat")

    # --- DATA (Plik_lite TT-only: ell, D_ell, [err])
    data = read_numeric_table(dat_path)
    nrow, ncol = size(data)
    @assert ncol in (2,3,4) "Unexpected column count in $(basename(dat_path)): $ncol"

    df =
        ncol == 2 ? DataFrame(ell=data[:,1], TT=data[:,2]) :
        ncol == 3 ? DataFrame(ell=data[:,1], TT=data[:,2], TT_err=data[:,3]) :
                    DataFrame(ell=data[:,1], TT=data[:,2]) # si ≥4 colonnes, on ne garde que TT
    out_data = joinpath(dir, "cmb_data.csv")
    CSV.write(out_data, df)
    @info "Wrote TT bandpowers CSV" file=out_data rows=nrow

    # --- COV (peut contenir TT+TE+EE : on prend le bloc TT en haut à gauche)
   Ntt = nrow
    C = read_covariance_auto(cov_path, Ntt)

    out_cov = joinpath(dir, "cmb_cov.csv")
    CSV.write(out_cov, DataFrame(C, :auto))
    @info "Wrote covariance CSV" file=out_cov size=size(C)
end

if abspath(PROGRAM_FILE) == @__FILE__
    @assert length(ARGS) == 1 "Usage: julia tools/pliklite_to_csv.jl /path/to/plik_lite_v22/"
    main(ARGS[1])
end
