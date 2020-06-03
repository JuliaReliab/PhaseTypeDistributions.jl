using PhaseTypeDistributions
using SparseMatrix
using NMarkov
using Test
using Distributions
using LinearAlgebra

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
    cf1 = initializePH(CF1(10), dat, verbose = true)
    @test true
end

@testset "Test of cf1init 2" begin
    dat = WeightedSample((0.0, Inf64)) do x
        pdf(Weibull(2.0, 1.0), x)
    end
    cf1 = initializePH(CF1(10), dat, verbose = true)
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

@testset "Test of phfit 1" begin
    cf1, llf0, = phfit(CF1(10), verbose=[true, true]) do x
        pdf(Weibull(2.0, 1.0), x)
    end
    data = WeightedSample((0.0, Inf64)) do x
        pdf(Weibull(2.0, 1.0), x)
    end
    llf = phllf(cf1, data)
    @test abs((llf0 - llf) / llf) < 1.0e-6
end

@testset "Test of phfit 2" begin
    cf1, llf0, = phfit(CF1(10), verbose=[true, true]) do x
        pdf(Weibull(2.0, 1.0), x)
    end
    data = WeightedSample((0.0, Inf64)) do x
        pdf(Weibull(2.0, 1.0), x)
    end
    println(cf1)

    deriv = Dict{Symbol,CF1{Float64}}()
    for i = 1:cf1.dim
        s = Symbol(:rate, i)
        a = zeros(Float64, cf1.dim)
        x = zeros(Float64, cf1.dim)
        x[i] = 1.0
        deriv[s] = CF1(cf1.dim, a, x)
        if i != cf1.dim
            s = Symbol(:alpha, i)
            a = zeros(Float64, cf1.dim)
            x = zeros(Float64, cf1.dim)
            a[i] = 1.0
            a[cf1.dim] = -1.0
            deriv[s] = CF1(cf1.dim, a, x)
        end
    end
    llf, llfdash = phllf(cf1, deriv, data)
    println(llfdash)

    llfdash0 = Dict()
    for i = 1:cf1.dim
        h = 0.000001
        cf1tmp = CF1(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
        cf1tmp.rate[i] -= h
        llf1 = phllf(cf1tmp, data)
        cf1tmp = CF1(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
        cf1tmp.rate[i] += h
        llf2 = phllf(cf1tmp, data)
        s = Symbol(:rate, i)
        llfdash0[s] = (llf2 - llf1) / (2*h)

        if i != cf1.dim
            h = 0.00000001
            cf1tmp = CF1(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
            cf1tmp.alpha[i] -= h
            cf1tmp.alpha[cf1.dim] += h
            llf1 = phllf(cf1tmp, data)
            cf1tmp = CF1(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
            cf1tmp.alpha[i] += h
            cf1tmp.alpha[cf1.dim] -= h
            llf2 = phllf(cf1tmp, data)
            s = Symbol(:alpha, i)
            llfdash0[s] = (llf2 - llf1) / (2*h)
        end
    end

    @test llf ≈ phllf(cf1, data)
    for i = eachindex(llfdash0)
        rerror = abs((llfdash[i] - llfdash0[i])/llfdash0[i])
        println((i, llfdash[i], llfdash0[i], rerror))
        @test rerror < 1.0e-1
    end
end

@testset "Test of phfit 3" begin
    cf1, llf0, = phfit(CF1(5), verbose=[true, true]) do x
        pdf(Weibull(2.0, 1.0), x)
    end
    data = WeightedSample((0.0, Inf64)) do x
        pdf(Weibull(2.0, 1.0), x)
    end
    println(cf1)

    deriv = Dict{Symbol,CF1{Float64}}()
    deriv2 = Dict{Tuple{Symbol,Symbol},CF1{Float64}}()
    for i = 1:cf1.dim-1
        s = Symbol(:alpha, i)
        a = zeros(Float64, cf1.dim)
        x = zeros(Float64, cf1.dim)
        a[i] = 1.0
        a[cf1.dim] = -1.0
        deriv[s] = CF1(cf1.dim, a, x)
        for j = i:cf1.dim-1
            s2 = Symbol(:alpha, j)
            a = zeros(Float64, cf1.dim)
            x = zeros(Float64, cf1.dim)
            deriv2[(s,s2)] = CF1(cf1.dim, a, x)
        end
        for j = 1:cf1.dim
            s2 = Symbol(:rate, j)
            a = zeros(Float64, cf1.dim)
            x = zeros(Float64, cf1.dim)
            deriv2[(s,s2)] = CF1(cf1.dim, a, x)
        end
    end
    for i = 1:cf1.dim
        s = Symbol(:rate, i)
        a = zeros(Float64, cf1.dim)
        x = zeros(Float64, cf1.dim)
        x[i] = 1.0
        deriv[s] = CF1(cf1.dim, a, x)
        for j = i:cf1.dim
            s2 = Symbol(:rate, j)
            a = zeros(Float64, cf1.dim)
            x = zeros(Float64, cf1.dim)
            deriv2[(s,s2)] = CF1(cf1.dim, a, x)
        end
    end
    llf, llfdash, llfdashdash = phllf(cf1, deriv, deriv2, data)
    llf0, llfdash0 = phllf(cf1, deriv, data)
    @test llf ≈ llf0
    for i = eachindex(llfdash0)
        @test llfdash0[i] ≈ llfdash[i]
    end

    k = 1
    index = Dict()
    for i = 1:cf1.dim-1
        s = Symbol(:alpha, i)
        index[s] = k
        k += 1
    end
    for i = 1:cf1.dim
        s = Symbol(:rate, i)
        index[s] = k
        k += 1
    end
    IM = zeros(length(index), length(index))
    for i = eachindex(llfdashdash)
        IM[index[i[1]], index[i[2]]] = llfdashdash[i]
        IM[index[i[2]], index[i[1]]] = llfdashdash[i]
    end
    values,vectors = eigen(-IM)
    println(values)
    @test all(values .> 0)
    println(inv(-IM))
end
