using Test
using ROFT

@testset "ROFT soft sanity" begin
    bg   = ROFT.Backgrounds.FlatLCDM(H0=70.0, Om0=0.3, Or0=0.0)
    p0   = ROFT.ROFTSoft.ROFTParams(alpha=0.0)
    pα   = ROFT.ROFTSoft.ROFTParams(alpha=0.1, variant=:energy)

    # Mock CC point at z=0.5 with generous sigma
    cc = ROFT.CC.CCData(z=[0.5], H=[ROFT.Backgrounds.H_of_z(bg,0.5)], sigH=[5.0])
    D  = ROFT.Driver.Packs(cc=cc)

    χ2_LCDM = ROFT.Driver.chi2_total(D; bg=bg, roft=p0)
    χ2_ROFT = ROFT.Driver.chi2_total(D; bg=bg, roft=pα)

    @test χ2_LCDM ≈ 0.0 atol=1e-8
    @test χ2_ROFT ≥ 0.0
end
