import PhaseTypeDistributions.Phfit: hypoerlang
using PhaseTypeDistributions.Phfit

@testset "Test of 3mom bobbio" begin
    ph = hypoerlang(;shape=[1,5], initprob=[0.4, 0.6], rate=[6.3, 5.6])
    println(ph)
    ph2 = GPH(ph)
    println(ph2.alpha)
    println(ph2.T)
    println(ph2.tau)
end

@testset "Test of 3mom bobbio" begin
    m1 = 0.31332853432887525
    m2 = 0.2500000000000002
    m3 = 0.23499640074665654
    ph = ph3mom_bobbio05(m1, m2, m3)
    println(ph)
    @test isapprox(m1, phmean(ph, 1))
    @test isapprox(m2, phmean(ph, 2))
    @test isapprox(m3, phmean(ph, 3))
end

@testset "Test of cf1mom" begin
    m1 = 0.31332853432887525
    m2 = 0.2500000000000002
    m3 = 0.23499640074665654
    f = 10.0

    ph = cf1mom_power(10, m1, f)
    @test isapprox(m1, phmean(ph, 1))
    @test isapprox(ph.rate[10]/ph.rate[1], f)
    println(phmean(ph, 2))

    ph = cf1mom_linear(10, m1, f)
    @test isapprox(m1, phmean(ph, 1))
    @test isapprox(ph.rate[10]/ph.rate[1], f)
    println(phmean(ph, 2))
end
