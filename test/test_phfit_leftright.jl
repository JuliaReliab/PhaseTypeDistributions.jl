using SparseArrays
using SparseMatrix
using PhaseTypeDistributions.Phfit
using Distributions
using LinearAlgebra

import PhaseTypeDistributions.Phfit: Estep, estep!, mstep!, initializePH, phfit!

@testset "Test for groupsample" begin
    tau = rand(0:1, 10) .* rand(10)
    t = tau .+ rand(10)
    delta = rand(Bool, 10)
    for r = zip(t, tau, delta)
        println(r)
    end
    dat = LeftTruncRightCensoredSample(t, tau, delta)
    println(dat)
    println(Phfit.mean(dat))
end

@testset "Test of estep0" begin
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 2.0]
    cf1 = CF1(alpha, rate)
    ph = GPH(cf1, SparseMatrixCSC)
    println(ph)

    tau = [0.0 for i = 1:10]
    t = tau .+ rand(10)
    delta = [true for i = 1:10]
    for r = zip(t, tau, delta)
        println(r)
    end
    data = LeftTruncRightCensoredSample(t, tau, delta)
    println(data)
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
    println(ph)

    tau = [0.0 for i = 1:10]
    t = tau .+ rand(10)
    delta = [true for i = 1:10]
    for r = zip(t, tau, delta)
        println(r)
    end
    data = LeftTruncRightCensoredSample(t, tau, delta)
    println(data)
    eres = Estep(ph)

    @time llf = estep!(cf1, data, eres)
    @time llf = estep!(cf1, data, eres)
    println(eres)
    @test true
end

@testset "Test of estep2" begin
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 2.0]
    cf1 = CF1(alpha, rate)
    ph = GPH(cf1, SparseMatrixCSC)
    println(ph)

    tau = rand(0:1, 10) .* rand(10)
    t = tau .+ rand(10)
    delta = rand(Bool, 10)
    for r = zip(t, tau, delta)
        println(r)
    end
    data = LeftTruncRightCensoredSample(t, tau, delta)
    println(data)
    eres = Estep(ph)

    @time llf = estep!(cf1, data, eres)
    @time llf = estep!(cf1, data, eres)
    println(eres)
    @test true
end

@testset "Test of emstep" begin
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 2.0]
    cf1 = CF1(alpha, rate)
    ph = GPH(cf1, SparseMatrixCSC)

    tau = [0.0 for i = 1:10]
    t = tau .+ [0.1 for i = 1:10]
    delta = [true for i = 1:10]
    for r = zip(t, tau, delta)
        println(r)
    end
    # dat = PointSample(t)
    dat = LeftTruncRightCensoredSample(t, tau, delta)
    eres = Estep(ph)

    @time llf = estep!(cf1, dat, eres)
    @time llf = estep!(cf1, dat, eres)
    println("llf------　　　　", llf)
    println("eres-----", eres)

    mstep!(cf1, eres)
    println("param----", cf1)
    @time llf2 = estep!(cf1, dat, eres)
    println("llf------　　　　", llf2)
    println("eres-----", eres)
    @test llf2 > llf
end

@testset "Test of emstep2" begin
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 2.0]
    cf1 = CF1(alpha, rate)
    ph = GPH(cf1, SparseMatrixCSC)

    tau = rand(0:1, 10) .* rand(10)
    t = tau .+ rand(10)
    delta = rand(Bool, 10)
    for r = zip(t, tau, delta)
        println(r)
    end
    # dat = PointSample(t)
    dat = LeftTruncRightCensoredSample(t, tau, delta)
    eres = Estep(ph)

    @time llf = estep!(cf1, dat, eres)
    @time llf = estep!(cf1, dat, eres)
    println("llf------　　　　", llf)
    println("eres-----", eres)

    mstep!(cf1, eres)
    println("param----", cf1)
    @time llf2 = estep!(cf1, dat, eres)
    println("llf------　　　　", llf2)
    println("eres-----", eres)
    @test llf2 > llf
end

@testset "Test of emstep for 10" begin
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 2.0]
    cf1 = CF1(alpha, rate)
    ph = GPH(cf1, Matrix)

    tau = rand(0:1, 10) .* rand(10)
    t = tau .+ rand(10)
    delta = rand(Bool, 10)
    for r = zip(t, tau, delta)
        println(r)
    end
    # dat = PointSample(t)
    dat = LeftTruncRightCensoredSample(t, tau, delta)
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
    tau = rand(0:1, 100) .* rand(100)
    t = tau .+ rand(100)
    delta = rand(Bool, 100)
    dat = LeftTruncRightCensoredSample(t, tau, delta)

    res = phfit(CF1(10), dat, verbose=[true, true])
    println(res)
    @test true
end

@testset "Test of emstep_error" begin
    alpha = [0.0, 0.0, 0.0]
    rate = [1.4, 0.4, 2.0]
    cf1 = CF1(alpha, rate)
    ph = GPH(cf1, SparseMatrixCSC)

    tau = rand(0:1, 10) .* rand(10)
    t = tau .+ rand(10)
    delta = rand(Bool, 10)
    for r = zip(t, tau, delta)
        println(r)
    end
    # dat = PointSample(t)
    dat = LeftTruncRightCensoredSample(t, tau, delta)
    eres = Estep(ph)

    @time llf = estep!(cf1, dat, eres)
    println("llf------　　　　", llf)
    println("eres-----", eres)
    @test !isfinite(llf)
end

# ### group poi

# @testset "Test for groupsample" begin
#     t = rand(10)
#     x = rand(0:10, 10)
#     dat = GroupTruncPoiSample(t, x)
#     println(dat)
#     sum(t .<= dat.maxtime) == 1
# end

# @testset "Test of estep" begin
#     alpha = [0.1, 0.3, 0.6]
#     rate = [1.4, 0.4, 2.0]
#     cf1 = CF1(alpha, rate)
#     ph = GPH(cf1, SparseMatrixCSC)

#     t = rand(10)
#     x = rand(0:10, 10)
#     data = GroupTruncPoiSample(t, x)
#     eres = Estep(ph)

#     @time llf = estep!(cf1, data, eres)
#     @time llf = estep!(cf1, data, eres)
#     totalt = 0.0
#     ct = 0.0
#     for i = eachindex(data.tdat)
#         ct += data.tdat[i]
#         if data.gdat[i] >= 0
#             totalt += data.gdat[i] * ct
#         end
#         if !iszero(data.idat[i])
#             totalt += ct
#         end
#     end
#     @test true
# end

# @testset "Test of phfit 2" begin
#     t = rand(100)
#     x = rand(0:10, 100)
#     dat = GroupTruncPoiSample(t, x)

#     res = phfit(CF1(10), dat, verbose=[true, true])
#     println(res)
#     @test true
# end
