using SparseArrays
using SparseMatrix
using PhaseTypeDistributions.Phfit
using Distributions
using LinearAlgebra

import PhaseTypeDistributions.Phfit: Estep, estep!, mstep!, initializePH, phfit!

@testset "Test for groupsample" begin
    t = rand(10)
    x = rand(0:10, 10)
    dat = GroupTruncSample(t, x)
    sum(t .<= dat.maxtime) == 1
end

@testset "Test of estep0" begin
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 2.0]
    cf1 = CF1(alpha, rate)
    ph = GPH(cf1, SparseMatrixCSC)
    println(ph)

    t = [0.1*u for u = 1:10]
    data = PointSample(t)
    eres = Estep(ph)

    @time llf = estep!(cf1, data, eres)
    @time llf = estep!(cf1, data, eres)
    totalt = 0.0
    ct = 0.0
    @test sum(eres.ez) ≈ sum(t)
    println(eres.eb)
    println(eres.ey)
end

@testset "Test of estep1" begin
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 2.0]
    cf1 = CF1(alpha, rate)
    ph = GPH(cf1, SparseMatrixCSC)

    t = [0.1 for u = 1:10]
    x = [0 for u = 1:10]
    i = [true for u = 1:10]
    data = GroupTruncSample(t, x, i, 0)
    eres = Estep(ph)

    @time llf = estep!(cf1, data, eres)
    @time llf = estep!(cf1, data, eres)
    totalt = 0.0
    ct = 0.0
    for i = eachindex(data.tdat)
        ct += data.tdat[i]
        if data.gdat[i] > 0
            totalt += data.gdat[i] * ct
        end
        if data.idat[i] == true
            totalt += ct
        end
    end
    @test sum(eres.ez) ≈ totalt
    println(eres.eb)
    println(eres.ey)
end

@testset "Test of estep" begin
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 2.0]
    cf1 = CF1(alpha, rate)
    ph = GPH(cf1, SparseMatrixCSC)

    t = rand(10)
    x = rand(0:10, 10)
    data = GroupTruncSample(t, x)
    eres = Estep(ph)

    @time llf = estep!(cf1, data, eres)
    @time llf = estep!(cf1, data, eres)
    totalt = 0.0
    ct = 0.0
    for i = eachindex(data.tdat)
        ct += data.tdat[i]
        if data.gdat[i] >= 0
            totalt += data.gdat[i] * ct
        end
        if !iszero(data.idat[i])
            totalt += ct
        end
    end
    @test true
end

@testset "Test of emstep" begin
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 2.0]
    cf1 = CF1(alpha, rate)
    ph = GPH(cf1, SparseMatrixCSC)

    t = rand(10)
    x = rand(0:10, 10)
    dat = GroupTruncSample(t, x)
    eres = Estep(ph)

    @time llf = estep!(cf1, dat, eres)
    @time llf = estep!(cf1, dat, eres)

    mstep!(cf1, eres)
    @time llf2 = estep!(cf1, dat, eres)
    @test llf2 > llf
end

@testset "Test of emstep gph Matrix" begin
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 2.0]
    cf1 = CF1(alpha, rate)
    ph = GPH(cf1, Matrix)

    t = rand(10)
    x = rand(0:10, 10)
    dat = GroupTruncSample(t, x)
    eres = Estep(ph)

    @time llf = estep!(cf1, dat, eres)
    @time llf = estep!(cf1, dat, eres)

    mstep!(cf1, eres)
    @time llf2 = estep!(cf1, dat, eres)
    @test llf2 > llf
end

# @testset "Test of emstep gph CSR" begin
#     alpha = [0.1, 0.3, 0.6]
#     rate = [1.4, 0.4, 2.0]
#     cf1 = CF1(alpha, rate)
#     ph = GPH(cf1, SparseCSR)

#     t = rand(10)
#     x = rand(0:10, 10)
#     dat = GroupTruncSample(t, x)
#     eres = Estep(ph)

#     @time llf = estep!(cf1, dat, eres)
#     @time llf = estep!(cf1, dat, eres)

#     mstep!(cf1, eres)
#     @time llf2 = estep!(cf1, dat, eres)
#     @test llf2 > llf
# end

@testset "Test of emstep for 10" begin
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 2.0]
    cf1 = CF1(alpha, rate)
    ph = GPH(cf1, Matrix)

    t = rand(10)
    x = rand(0:10, 10)
    dat = GroupTruncSample(t, x)
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

@testset "Test of emstep for 10 cf1" begin
    t = rand(100)
    x = rand(0:10, 100)
    dat = GroupTruncSample(t, x)

    t = rand(10)
    x = rand(0:10, 10)
    dat = GroupTruncSample(t, x)
    cf1 = initializePH(CF1(3), dat)
    println(cf1)
    ph = GPH(cf1)
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

@testset "Test of phfit 1" begin
    t = rand(100)
    x = rand(0:10, 100)
    dat = GroupTruncSample(t, x)

    res = phfit(CF1(10), dat)
    println(res)
    @test true
end

### group poi

@testset "Test for groupsample" begin
    t = rand(10)
    x = rand(0:10, 10)
    dat = GroupTruncPoiSample(t, x)
    println(dat)
    sum(t .<= dat.maxtime) == 1
end

@testset "Test of estep" begin
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 2.0]
    cf1 = CF1(alpha, rate)
    ph = GPH(cf1, SparseMatrixCSC)

    t = rand(10)
    x = rand(0:10, 10)
    data = GroupTruncPoiSample(t, x)
    eres = Estep(ph)

    @time llf = estep!(cf1, data, eres)
    @time llf = estep!(cf1, data, eres)
    totalt = 0.0
    ct = 0.0
    for i = eachindex(data.tdat)
        ct += data.tdat[i]
        if data.gdat[i] >= 0
            totalt += data.gdat[i] * ct
        end
        if !iszero(data.idat[i])
            totalt += ct
        end
    end
    @test true
end

@testset "Test of phfit 2" begin
    t = rand(100)
    x = rand(0:10, 100)
    dat = GroupTruncPoiSample(t, x)

    res = phfit(CF1(10), dat)
    println(res)
    @test true
end
