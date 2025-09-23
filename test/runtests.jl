using Test
using Roft
using LinearAlgebra

const Backgrounds = Roft.Backgrounds
const ROFTSoft = Roft.ROFTSoft

@testset "ROFTSoft primitives" begin
    bg = Backgrounds.FlatLCDM(H0=70.0, Om0=0.3, Or0=0.0)
    p_energy = ROFTSoft.ROFTParams(alpha=0.1, variant=:energy)
    p_thermal = ROFTSoft.ROFTParams(alpha=0.1, variant=:thermal)

    z = 0.5
    @test ROFTSoft.delta_g_time(p_energy, z; Om0=bg.Om0) ≈ 2p_energy.alpha * log(Backgrounds.E_LCDM(z; Om0=bg.Om0))
    @test ROFTSoft.delta_g_time(p_thermal, z; Om0=bg.Om0) ≈ p_thermal.alpha * log(1 + z)

    H_geo = Backgrounds.H_of_z(bg, z)
    @test ROFTSoft.modify_CC(H_geo, z, p_energy; Om0=bg.Om0) ≈ H_geo * exp(-ROFTSoft.delta_g_time(p_energy, z; Om0=bg.Om0))

    zl, zs, eta = 0.3, 1.2, 0.4
    @test ROFTSoft.modify_TD(bg.H0, zl, zs, eta, p_energy; Om0=bg.Om0) ≈ bg.H0 * exp(eta * ROFTSoft.delta_g_time(p_energy, zl; Om0=bg.Om0) + (1-eta) * ROFTSoft.delta_g_time(p_energy, zs; Om0=bg.Om0))

    @test ROFTSoft.delta_mu_host(z, 1.5, p_energy; Om0=bg.Om0) ≈ -1.5 * ROFTSoft.delta_g_time(p_energy, z; Om0=bg.Om0)
end

@testset "ROFT soft sanity" begin
    bg   = Backgrounds.FlatLCDM(H0=70.0, Om0=0.3, Or0=0.0)
    p0   = ROFTSoft.ROFTParams(alpha=0.0)
    pα   = ROFTSoft.ROFTParams(alpha=0.1, variant=:energy)

    # Mock CC point at z=0.5 with generous sigma
    cc = Roft.CC.CCData(z=[0.5], H=[Backgrounds.H_of_z(bg,0.5)], sigH=[5.0])
    D  = Roft.Driver.Packs(cc=cc)

    χ2_LCDM = Roft.Driver.chi2_total(D; bg=bg, roft=p0)
    χ2_ROFT = Roft.Driver.chi2_total(D; bg=bg, roft=pα)

    @test χ2_LCDM ≈ 0.0 atol=1e-8
    @test χ2_ROFT ≥ 0.0
end

@testset "Pantheon+ covariance reader" begin
    tmpdir = mktempdir()
    cov_path = joinpath(tmpdir, "toy.cov")
    open(cov_path, "w") do io
        write(io, "3\n")
        write(io, "1.0 0.1 0.0\n")
        write(io, "0.1 2.0 0.2\n")
        write(io, "0.0 0.2 3.0\n")
    end

    C = Roft.PantheonPlusAdapter._read_covariance(cov_path)
    @test size(C) == (3,3)
    @test isapprox(Matrix(C), [1.0 0.1 0.0; 0.1 2.0 0.2; 0.0 0.2 3.0]; atol=1e-12)
end

@testset "Adapters" begin
    @testset "PantheonPlusAdapter.loaders" begin
        tmpdir = mktempdir()
        data_path = joinpath(tmpdir, "data.txt")
        cov_path  = joinpath(tmpdir, "cov.cov")
        open(data_path, "w") do io
            println(io, "z mu")
            println(io, "0.1 42.0")
            println(io, "0.2 42.5")
        end
        open(cov_path, "w") do io
            println(io, "2")
            println(io, "1.0")
            println(io, "0.0")
            println(io, "0.0")
            println(io, "1.0")
        end
        sn = Roft.load_pantheonplus(data_path, cov_path)
        @test sn isa Roft.PantheonPlus.SNData
        @test sn.z ≈ [0.1, 0.2]
        @test sn.mu ≈ [42.0, 42.5]
        @test size(sn.Cov) == (2,2)

        sn18 = Roft.load_pantheon_2018(data_path, cov_path; zcol=:z, mucol=:mu)
        @test sn18.name == "Pantheon2018"
    end

    @testset "CCCovAdapter" begin
        tmpdir = mktempdir()
        file = joinpath(tmpdir, "cc.dat")
        open(file, "w") do io
            println(io, "# z, Hz, errHz")
            println(io, "0.3, 75.0, 3.0")
            println(io, "0.1, 70.0, 2.0")
        end
        obs = Roft.load_cccov(data_dir=tmpdir, files=[basename(file)], include_systematics=false)
        @test obs isa Roft.CCCov.CCObs
        @test obs.z ≈ [0.1, 0.3]
        @test obs.H ≈ [70.0, 75.0]
        @test diag(Matrix(obs.Cov)) ≈ [4.0, 9.0]

        mm20 = joinpath(tmpdir, "data_MM20.dat")
        open(mm20, "w") do io
            println(io, "# z IMF stlib mod mod_ooo")
            println(io, "0.0 1.0 2.0 3.0 4.0")
            println(io, "1.0 1.0 2.0 3.0 4.0")
        end
        obs_sys = Roft.load_cccov(data_dir=tmpdir, files=[basename(file)])
        @test obs_sys isa Roft.CCCov.CCObs
        @test obs_sys.Cov[1,2] > 0
    end

    @testset "H0LiCOWAdapter" begin
        tmpdir = mktempdir()
        model_dir = joinpath(tmpdir, "FLCDM")
        mkpath(model_dir)
        chain_path = joinpath(model_dir, "toy.dat")
        open(chain_path, "w") do io
            println(io, "# H0 samples")
            println(io, "70.0")
            println(io, "72.0")
            println(io, "71.0")
        end
        meta_path = joinpath(tmpdir, "meta.csv")
        open(meta_path, "w") do io
            println(io, "lens,zl,zs,eta,w")
            println(io, "toy,0.3,1.5,0.5,1.0")
        end
        obs = Roft.load_h0licow_cosmo_chains(tmpdir; model=:FLCDM, meta_csv=meta_path)
        @test obs isa Roft.H0LiCOWCosmoChains.TDH0GlobalObs
        @test obs.H0_mean ≈ 71.0
        @test obs.H0_std > 0
        E = z -> 1.0 + z
        M = Roft.meff_from_meta(0.1, :energy, obs.zl, obs.zs, obs.eta, obs.w, E)
        @test M > 0
    end

    @testset "CapseEnv" begin
        keys_vals = Dict(
            "CAPSE_NS" => "0.97",
            "CAPSE_AS" => "2.5e-9",
            "CAPSE_TAU" => "0.06",
            "CAPSE_OBH2" => "0.023",
            "CAPSE_ONUH2" => "0.001"
        )
        saved = Dict{String,Union{Nothing,String}}()
        for (k,v) in keys_vals
            saved[k] = get(ENV, k, nothing)
            ENV[k] = v
        end

        params = Roft.CapseEnv.read_nuisance_params()
        @test params.ns ≈ 0.97
        @test params.As ≈ 2.5e-9
        @test params.tau ≈ 0.06
        @test params.Obh2 ≈ 0.023
        @test params.Onuh2 ≈ 0.001

        for (k,val) in saved
            val === nothing ? delete!(ENV, k) : (ENV[k] = val)
        end
    end

    @testset "CapseAdapter" begin
        import Capse
        @eval Capse begin
            struct DummyEmulator
                ℓ::Vector{Int}
                response::Matrix{Float64}
            end
            get_ℓgrid(emu::DummyEmulator) = emu.ℓ
            get_Cℓ(x::AbstractVector{<:Real}, emu::DummyEmulator) = emu.response * collect(x)
        end

        ℓ = [10, 20]
        resp = [1.0 0.0 0.0 0.0 0.0 0.0;
                0.0 1.0 0.0 0.0 0.0 0.0]
        st_tt = Roft.CapseAdapter.CapseState(Capse.DummyEmulator(ℓ, resp), ℓ, [:ωb, :ωc, :h, :ns, :τ, :As])
        st_te = Roft.CapseAdapter.CapseState(Capse.DummyEmulator(ℓ, resp .* 0.5), ℓ, [:ωb, :ωc, :h, :ns, :τ, :As])
        states = Dict(:TT => st_tt, :TE => st_te)
        @test states[:TT].ℓ == [10, 20]
        @test states[:TE].param_order == [:ωb, :ωc, :h, :ns, :τ, :As]

        p = Roft.CapseAdapter.CapseCMB(Obh2=0.022, Och2=0.12, H0=70.0, ns=0.96, As=2.1e-9, tau=0.054)
        vec = Roft.CapseAdapter._param_vector(p, st_tt.param_order)
        @test vec ≈ [0.022, 0.12, 0.7, 0.96, 0.054, 2.1e-9]

        th = Roft.CapseAdapter.predict_cmb(p, st_tt)
        @test th ≈ [0.022, 0.12]

        data_vec = th .+ [0.5, -1.0]
        cov = Symmetric(Matrix{Float64}(I, 2, 2))
        @test Roft.CapseAdapter.cmb_chi2(data_vec, cov, p, st_tt) ≈ 0.5^2 + (-1.0)^2

        tmp = mktempdir()
        data_csv = joinpath(tmp, "cmb.csv")
        cov_csv  = joinpath(tmp, "cmb_cov.csv")
        open(data_csv, "w") do io
            println(io, "ell,TT,TE")
            println(io, "10,100.0,8.0")
            println(io, "20,110.0,9.0")
        end
        open(cov_csv, "w") do io
            println(io, "1.0,0.0,0.0,0.0")
            println(io, "0.0,1.0,0.0,0.0")
            println(io, "0.0,0.0,1.0,0.0")
            println(io, "0.0,0.0,0.0,1.0")
        end
        cmb_data = Roft.load_cmb_data(data_csv, cov_csv; ell_min=5, ell_max=30, blocks=[:TT, :TE])
        @test cmb_data.ell == [10, 20]
        @test cmb_data.blocks == [:TT, :TE]
        @test cmb_data.block_ranges[:TT] == 1:2
        @test cmb_data.block_ranges[:TE] == 3:4

        model = Roft.CMBViaCapse.CapseCMBModel(blocks=[:TT, :TE], states=states, ell=ℓ)
        Roft.align_data_to_ell!(cmb_data, model.ell)
        @test model.ell == [10, 20]
        @test cmb_data.block_ranges[:TT] == 1:2
        @test cmb_data.block_ranges[:TE] == 3:4

        parms = Dict("CAPSE_NS"=>"0.96", "CAPSE_AS"=>"2.1e-9", "CAPSE_TAU"=>"0.054",
                     "CAPSE_OBH2"=>"0.022", "CAPSE_ONUH2"=>"0.00064")
        saved = Dict{String,Union{Nothing,String}}()
        for (k,v) in parms
            saved[k] = get(ENV, k, nothing)
            ENV[k] = v
        end

        cmb_data.vec = vcat(th .+ [0.2, -0.3], (0.5 .* th) .+ [0.1, -0.2])
        cmb_data.Cov = Symmetric(Matrix{Float64}(I, 4, 4))
        H0 = 70.0
        Om0 = 0.3
        chi2 = Roft.CMBViaCapse.chi2_cmb_soft_at(H0, Om0, cmb_data, model)
        @test isapprox(chi2, 0.2^2 + (-0.3)^2 + 0.1^2 + (-0.2)^2; atol=5e-3)

        for (k, val) in saved
            if val === nothing
                delete!(ENV, k)
            else
                ENV[k] = val
            end
        end
    end
end

@testset "Likelihoods" begin
    bg = Backgrounds.FlatLCDM(H0=70.0, Om0=0.3, Or0=0.0)
    roft = ROFTSoft.ROFTParams(alpha=0.05, variant=:energy)

    cc = Roft.CC.CCData(z=[0.0], H=[bg.H0], sigH=[1.0])
    @test Roft.CC.chi2_cc(cc; bg=bg, roft=ROFTSoft.ROFTParams()) ≈ 0.0 atol=1e-12

    td = Roft.TD.TDData(zl=[0.2], zs=[1.0], eta=[0.5], H0inf=[bg.H0], sigH0=[1.0])
    @test Roft.TD.chi2_td(td; bg=bg, roft=ROFTSoft.ROFTParams()) ≈ 0.0 atol=1e-12

    ceph = Roft.CephSN.CephSNData(z_host=[0.05], dmu=[0.0], sigmu=[0.1])
    @test Roft.CephSN.chi2_cephsn(ceph; bg=bg, roft=ROFTSoft.ROFTParams()) ≈ 0.0 atol=1e-12

    Cov_sn = Symmetric(Matrix{Float64}(I, 2, 2))
    sn = Roft.PantheonPlus.SNData(z=[0.1, 0.2], mu=[43.0, 43.2], Cov=Cov_sn)
    μth(z) = 42.9 + 1.0 * (z - 0.1)
    χ2_with = Roft.PantheonPlus.chi2_sn_with_M(sn, μth; M=0.1)
    χ2_prof, Mhat = Roft.PantheonPlus.chi2_sn_profiledM(sn, μth)
    @test χ2_with ≥ 0
    @test χ2_prof ≥ 0
    @test abs(Mhat) ≤ 1.0

    obs = Roft.H0LiCOWCosmoChains.TDH0GlobalObs(H0_mean=73.0, H0_std=1.8, lenses=["A"], zl=[0.5], zs=[1.7], eta=[0.4], w=[1.0])
    @test Roft.H0LiCOWCosmoChains.chi2_td_h0_global(obs, 70.0, 1.0) ≈ ((73.0-70.0)^2)/(1.8^2)

    H_theory(z) = Backgrounds.H_of_z(bg, z)
    cc_full = Roft.CCCov.CCObs(z=[0.1], H=[H_theory(0.1)], Cov=Symmetric(fill(0.5^2, 1, 1)))
    @test Roft.CCCov.chi2_cccov(cc_full, H_theory) ≈ 0.0 atol=1e-12
end
