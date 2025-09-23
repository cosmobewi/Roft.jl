import Pkg; Pkg.activate("."); 
for dep in ("DataFrames","CSV")
    try
        @eval using $(Symbol(dep))
    catch
        Pkg.add(dep); @eval using $(Symbol(dep))
    end
end

using CSV, DataFrames

infile  = ARGS[1]     # ex: datas/_external/cmb_cov.csv (avec entêtes)
outfile = ARGS[2]     # ex: datas/_external/cmb_cov_clean.csv

df = CSV.read(infile, DataFrame; delim=',', ignorerepeated=true)
C  = similar(Matrix{Float64}, size(df,1), size(df,2))
for (j,col) in enumerate(eachcol(df))
    try
        C[:,j] = Float64.(col)
    catch
        C[:,j] = parse.(Float64, string.(col))
    end
end
CSV.write(outfile, DataFrame(C, :auto))
println("Écrit: ", outfile, "  → taille ", size(C))
