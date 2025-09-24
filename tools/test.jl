import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()


using Roft, Capse

weights = "/opt/trained_emu/TT"
state = Roft.CapseAdapter.load_capse(weights)

params_h0 = Roft.CapseAdapter._param_vector(
    Roft.CapseAdapter.CapseCMB(Obh2=0.022, Och2=0.12, H0=70, ns=0.965, As=2.1e-9, tau=0.054),
    state.param_order
)

params_h = deepcopy(params_h0)
idx = findfirst(==(Symbol("H0")), state.param_order)
if idx !== nothing
    params_h[idx] = 0.70   # remplace 70 par 0.70 pour tester
end

spec_h0 = Capse.get_Cℓ(params_h0, state.emulator)
spec_h  = Capse.get_Cℓ(params_h,  state.emulator)

@show spec_h0[1:5] spec_h[1:5]

using CSV, DataFrames, Roft

cmb = Roft.load_pliklite("/opt/planck/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TTTEEE.clik/clik/lkl_0/_external"; as_Dell=true)

model = Roft.CMBViaCapse.make_cmb_model_from_env(cmb; blocks=[:TT])

p = Roft.CapseAdapter.CapseCMB(Obh2=0.02237, Och2=0.12, H0=70.0,
                               ns=0.965, As=2.1e-9, tau=0.054)

spec_full = Roft.CapseAdapter.predict_cmb(p, model.states[:TT])
keep = model.keep_indices[:TT]
spec = spec_full[keep]            # théorie alignée sur les ℓ communs

range = cmb.block_ranges[:TT]
data = cmb.vec[range]             # observations correspondantes

res = data .- spec
@show spec[1:10] data[1:10] res[1:10]