using SparseArrays
using SparseMatrix
using PhaseTypeDistributions.Phfit

import PhaseTypeDistributions.Phfit: Estep, estep!, mstep!

@testset "Test for wsample" begin
    t = rand(10)
    w = rand(10)
    dat = WeightedSample(t, w)
    @test dat.tdat[1] == minimum(t)
    t = rand(100)
    dat = PointSample(t)
    @test dat.tdat[1] == minimum(t)
    @test all(dat.wdat .== 1.0)
end

@testset "Test of estep" begin
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 2.0]
    cf1 = CF1(alpha, rate)
    ph = GPH(cf1, SparseCSC)

    t = rand(100)
    dat = PointSample(t)
    eres = Estep(ph)

    @time llf = estep!(cf1, dat, eres)
    @time llf = estep!(cf1, dat, eres)
    @test sum(eres.ez) ≈ sum(t)

    ph = GPH(cf1, SparseCOO)
    t = rand(100)
    dat = PointSample(t)
    eres = Estep(ph)
    @time llf = estep!(cf1, dat, eres)
    @time llf = estep!(cf1, dat, eres)
    @test sum(eres.ez) ≈ sum(t)

    ph = GPH(cf1, SparseCSR)
    t = rand(100)
    dat = PointSample(t)
    eres = Estep(ph)
    @time llf = estep!(cf1, dat, eres)
    @time llf = estep!(cf1, dat, eres)
    @test sum(eres.ez) ≈ sum(t)

    ph = GPH(cf1, SparseMatrixCSC)
    t = rand(100)
    dat = PointSample(t)
    eres = Estep(ph)
    @time llf = estep!(cf1, dat, eres)
    @time llf = estep!(cf1, dat, eres)
    @test sum(eres.ez) ≈ sum(t)
end

@testset "Test of emstep" begin
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 2.0]
    cf1 = CF1(alpha, rate)
    ph = GPH(cf1)

    t = rand(100)
    dat = PointSample(t)
    eres = Estep(ph)

    @time llf = estep!(cf1, dat, eres)
    @time llf = estep!(cf1, dat, eres)
    println(llf)
    @test sum(eres.ez) ≈ sum(t)

    mstep!(cf1, eres)
    @time llf2 = estep!(cf1, dat, eres)
    @test llf2 > llf
end

