using PhaseTypeDistributions
using SparseMatrix
using NMarkov
using Test
using Distributions

@testset "Test of CF1" begin
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 10.0]
    cf1 = CF1(alpha, rate)
    @test cf1.rate == [0.4, 1.4, 10.0]
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 10.0]
    cf1 = CF1(alpha, rate)
    gph = GPH(cf1, SparseCSC)
    @test gph.T.rowind == [1, 1, 2, 2, 3]
    @test gph.T.val == [-0.4, 0.4, -1.4, 1.4, -10.0]
    gph = GPH(cf1, SparseCSR)
    @test gph.T.colind == [1, 2, 2, 3, 3]
    @test gph.T.val == [-0.4, 0.4, -1.4, 1.4, -10.0]
end

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

@testset "Test of emstep gph Matrix" begin
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 2.0]
    ph = GPH(CF1(alpha, rate), Matrix)

    t = rand(100)
    dat = PointSample(t)
    eres = Estep(ph)

    @time llf = estep!(ph, dat, eres)
    @test sum(eres.ez) ≈ sum(t)

    mstep!(ph, eres)
    @time llf2 = estep!(ph, dat, eres)
    @test llf2 > llf
end

@testset "Test of emstep gph CSR" begin
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 2.0]
    ph = GPH(CF1(alpha, rate), SparseCSR)

    t = rand(100)
    dat = PointSample(t)
    eres = Estep(ph)

    @time llf = estep!(ph, dat, eres)
    @test sum(eres.ez) ≈ sum(t)

    mstep!(ph, eres)
    @time llf2 = estep!(ph, dat, eres)
    @test llf2 > llf
end

@testset "Test of emstep gph CSC" begin
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 2.0]
    ph = GPH(CF1(alpha, rate), SparseCSC)

    t = rand(100)
    dat = PointSample(t)
    eres = Estep(ph)

    @time llf = estep!(ph, dat, eres)
    @test sum(eres.ez) ≈ sum(t)

    mstep!(ph, eres)
    @time llf2 = estep!(ph, dat, eres)
    @test llf2 > llf
end

@testset "Test of emstep COO" begin
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 2.0]
    ph = GPH(CF1(alpha, rate), SparseCOO)

    t = rand(100)
    dat = PointSample(t)
    eres = Estep(ph)

    @time llf = estep!(ph, dat, eres)
    @test sum(eres.ez) ≈ sum(t)

    mstep!(ph, eres)
    @time llf2 = estep!(ph, dat, eres)
    @test llf2 > llf
end

@testset "Test of emstep for 10" begin
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 2.0]
    cf1 = CF1(alpha, rate)
    ph = GPH(cf1)
    t = rand(100)
    dat = PointSample(t)
    eres = Estep(ph)

    @time prevllf = estep!(cf1, dat, eres)
    mstep!(cf1, eres)
    for k = 1:10
        @time llf = estep!(cf1, dat, eres)
        mstep!(cf1, eres)
        @test llf > prevllf
        prevllf = llf
    end
end

@testset "Test of cf1init" begin
    t = rand(100)
    dat = PointSample(t)
    cf1 = initializePH(CF1(10), dat, verbose = true)
    # println(cf1)
    @test true
end

@testset "Test of cf1init 2" begin
    dat = WeightedSample((0.0, Inf64)) do x
        pdf(Weibull(2.0, 1.0), x)
    end
    cf1 = initializePH(CF1(100), dat, verbose = true)
    # println(cf1)
    @test true
end

@testset "Test of cf1init 2" begin
    dat = WeightedSample((0.0, Inf64)) do x
        pdf(Weibull(2.0, 1.0), x)
    end
    cf1 = initializePH(CF1(50), dat, verbose = true)
    phfit!(cf1, dat, verbose = true)
    # println(cf1)
    @test true
end

@testset "Test of phfit 1" begin
    res = phfit(CF1(10), verbose=[true, true]) do x
        pdf(Weibull(2.0, 1.0), x)
    end
    println(res)
    @test true
end

@testset "Test of pdf 1" begin
    res = phfit(CF1(100), verbose=[true, true]) do x
        pdf(Weibull(2.0, 1.0), x)
    end
    ph = GPH(res[1])
    @test phpdf(res[1], 0.0) == sum(ph.alpha .* ph.tau)
    @test phcdf(res[1], 0.0) == 0.0
    @test phccdf(res[1], 0.0) == 1.0
end

@testset "Test of pdf 1" begin
    res = phfit(CF1(200), verbose=[true, true]) do x
        pdf(Weibull(2.0, 1.0), x)
    end
    t = LinRange(0.0, 2.0, 100)
    x = phpdf(res[1], t)
    y = pdf.(Weibull(2.0, 1.0), t)
    for i = eachindex(x)
        println("$(i) $(x[i]) $(y[i])")
        @test x[i] - y[i] < 1.0e-4
    end
end
