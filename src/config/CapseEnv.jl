module CapseEnv
export NuisanceParams, read_nuisance_params

Base.@kwdef struct NuisanceParams
    ns::Float64 = 0.965
    As::Float64 = 2.1e-9
    tau::Float64 = 0.054
    Obh2::Float64 = 0.02237
    Onuh2::Float64 = 0.00064
end

function _parse(envkey::String, default::Float64)
    val = get(ENV, envkey, "")
    isempty(val) && return default
    try
        return parse(Float64, val)
    catch err
        @warn "Invalid value for $envkey, falling back to default" envkey val default err
        return default
    end
end

function read_nuisance_params()
    return NuisanceParams(
        ns   = _parse("CAPSE_NS",   0.965),
        As   = _parse("CAPSE_AS",   2.1e-9),
        tau  = _parse("CAPSE_TAU",  0.054),
        Obh2 = _parse("CAPSE_OBH2", 0.02237),
        Onuh2= _parse("CAPSE_ONUH2",0.00064),
    )
end

end
