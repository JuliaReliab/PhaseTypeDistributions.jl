using PhaseTypeDistributions
using PhaseTypeDistributions.Phfit
using Test

@testset "aic formula" begin
    # k=5: n_alpha=2 (3 nonzero - 1), n_rate=3 (all distinct)
    cf1 = CF1(3)
    cf1.alpha = [0.5, 0.3, 0.2]
    cf1.rate  = [1.0, 2.0, 3.0]
    r = aic(cf1, -30.0)
    @test r.n_alpha == 2
    @test r.n_rate  == 3
    @test r.k       == 5
    @test r.aic     == -2*(-30.0) + 2*5
end

@testset "aic alpha near zero" begin
    # dim=3, one alpha below threshold → n_alpha = 1 (2 nonzero - 1)
    cf1 = CF1(3)
    cf1.alpha = [0.6, 0.4, 1e-10]
    cf1.rate  = [1.0, 2.0, 3.0]
    r = aic(cf1, -30.0)
    @test r.n_alpha == 1
    @test r.n_rate  == 3
    @test r.k       == 4
end

@testset "aic rates within tolerance" begin
    # two rates nearly equal → n_rate = 2
    cf1 = CF1(3)
    cf1.alpha = [0.5, 0.3, 0.2]
    cf1.rate  = [1.0, 1.00005, 3.0]   # first two within default rate_tol=1e-4
    r = aic(cf1, -30.0; rate_tol=1e-3)
    @test r.n_rate == 2
    @test r.k      == 4
end

@testset "aic all rates distinct with tight tolerance" begin
    cf1 = CF1(3)
    cf1.alpha = [0.5, 0.3, 0.2]
    cf1.rate  = [1.0, 1.00005, 3.0]
    r = aic(cf1, -30.0; rate_tol=1e-6)  # tight: all three distinct
    @test r.n_rate == 3
end

@testset "aic dim=1 edge case" begin
    cf1 = CF1(1)
    r = aic(cf1, -10.0)
    @test r.n_alpha == 0   # 1 nonzero - 1 = 0
    @test r.n_rate  == 1
    @test r.k       == 1
    @test r.aic     == -2*(-10.0) + 2*1
end

@testset "aic custom alpha_tol" begin
    cf1 = CF1(3)
    cf1.alpha = [0.6, 0.4, 0.05]
    cf1.rate  = [1.0, 2.0, 3.0]
    # default tol=1e-8: all three nonzero → n_alpha=2
    @test aic(cf1, -30.0).n_alpha == 2
    # large tol: 0.05 treated as zero → n_alpha=1
    @test aic(cf1, -30.0; alpha_tol=0.1).n_alpha == 1
end

@testset "aic on fitted model" begin
    using Random
    dat = TimeSpanSample(rand(MersenneTwister(1), 100))
    res = phfit(CF1(5), dat; progress_init=false, progress=false)
    r   = aic(res.model, res.llf)
    @test isfinite(r.aic)
    @test r.k > 0
    @test r.k <= 2 * 5 - 1   # at most 2d-1 free parameters
    @test r.aic == -2 * res.llf + 2 * r.k
end
