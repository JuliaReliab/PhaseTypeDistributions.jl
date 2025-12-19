using Test
using Random
using Distributions
using PhaseTypeDistributions

@testset "Distributions interface CF1" begin
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 10.0]
    ph = CF1(alpha, rate)
    t = 0.5

    @test isapprox(cdf(ph, t), phcdf(ph, t))
    @test isapprox(ccdf(ph, t), phccdf(ph, t))
    @test isapprox(pdf(ph, t), phpdf(ph, t))
    @test isapprox(Distributions.mean(ph), phmean(ph))
    @test isapprox(Distributions.mean(ph, 2), phmean(ph, 2))

    rng = MersenneTwister(1234)
    samples = rand(rng, ph, 5)
    @test length(samples) == 5
    @test all(>=(zero(eltype(samples))), samples)
end

@testset "Distributions interface GPH" begin
    alpha = [0.1, 0.3, 0.6]
    T = [-1.4 1.4 0; 0 -0.4 0.4; 0 0 -10.0]
    tau = [0, 0, 10.0]
    ph = GPH(alpha, T, tau)
    t = 0.5

    @test isapprox(Distributions.cdf(ph, t), phcdf(ph, t))
    @test isapprox(Distributions.ccdf(ph, t), phccdf(ph, t))
    @test isapprox(Distributions.pdf(ph, t), phpdf(ph, t))
    @test isapprox(Distributions.mean(ph), phmean(ph))
    @test isapprox(Distributions.mean(ph, 2), phmean(ph, 2))

    rng = MersenneTwister(4321)
    samples = Distributions.rand(rng, ph, 5)
    @test length(samples) == 5
    @test all(>=(zero(eltype(samples))), samples)
end
