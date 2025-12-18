using SparseArrays
using NMarkov.SparseMatrix
using PhaseTypeDistributions.Phfit
using Distributions
using LinearAlgebra

import PhaseTypeDistributions.Phfit: Estep, estep!, mstep!, initializePH, phfit!

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

@testset "Test of emstep for 10 with wsample" begin
    alpha = [0.6, 0.3, 0.1]
    rate = [4, 113.0, 800.0]
    cf1 = CF1(alpha, rate)
    ph = GPH(cf1)
    t = rand(100)
    w = rand(100)
    dat = WeightedSample(t,w)
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

@testset "Test of emstep for 10 with wsample cf1" begin
    t = rand(100)
    w = rand(100)
    dat = WeightedSample(t,w)
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

@testset "Test of cf1init" begin
    t = rand(100)
    dat = PointSample(t)
    cf1 = initializePH(CF1(10), dat)
    # println(cf1)
    @test true
end

@testset "Test of cf1init 2" begin
    dat = WeightedSample((0.0, Inf64)) do x
        pdf(Weibull(2.0, 1.0), x)
    end
    cf1 = initializePH(CF1(10), dat)
    @test true
end

@testset "Test of cf1init 2" begin
    dat = WeightedSample((0.0, Inf64)) do x
        pdf(Weibull(2.0, 1.0), x)
    end
    cf1 = initializePH(CF1(10), dat)
    phfit!(cf1, dat)
    # println(cf1)
    @test true
end

@testset "Test of phfit 1" begin
    res = phfit(CF1(10)) do x
        pdf(Weibull(2.0, 1.0), x)
    end
    println(res)
    @test true
end

@testset "Test of pdf 1" begin
    res = phfit(CF1(100)) do x
        pdf(Weibull(2.0, 1.0), x)
    end
    ph = GPH(res[1])
    @test phpdf(res[1], 0.0) == sum(ph.alpha .* ph.tau)
    @test phcdf(res[1], 0.0) ≈ 0.0
    @test phccdf(res[1], 0.0) ≈ 1.0
end

@testset "Test of pdf 1" begin
    res = phfit(CF1(200)) do x
        pdf(Weibull(2.0, 1.0), x)
    end
    t = LinRange(0.0, 2.0, 100)
    x = phpdf(res[1], t)
    y = pdf.(Weibull(2.0, 1.0), t)
    for i = eachindex(x)
        println("$(i) $(x[i]) $(y[i])")
        @test x[i] - y[i] < 1.0e-3
    end
end

@testset "Test of phfit 1" begin
    cf1, llf0, = phfit(CF1(10)) do x
        pdf(Weibull(2.0, 1.0), x)
    end
    data = WeightedSample((0.0, Inf64)) do x
        pdf(Weibull(2.0, 1.0), x)
    end
    llf = phllf(cf1, data)
    @test abs((llf0 - llf) / llf) < 1.0e-4
end

@testset "Test of phfit 2" begin
    cf1, llf0, = phfit(CF1(10)) do x
        pdf(Weibull(1.5, 1.0), x)
    end
    data = WeightedSample((0.0, Inf64)) do x
        pdf(Weibull(1.5, 1.0), x)
    end
    println(cf1)

    deriv = Dict{Symbol,CF1{Float64}}()
    for i = 1:cf1.dim
        s = Symbol(:rate, i)
        a = zero(cf1.alpha)
        x = zero(cf1.rate)
        x[i] = 1.0
        deriv[s] = CF1(cf1.dim, a, x)

        s = Symbol(:alpha, i)
        a = zero(cf1.alpha)
        x = zero(cf1.rate)
        a[i] = 1.0
        deriv[s] = CF1(cf1.dim, a, x)
    end
    llf, llfdash = phllf(cf1, deriv, data)
    println(llfdash)

    h = 0.000001
    tol = 1e-6
    for i = 1:cf1.dim
        s = Symbol(:alpha, i)
        cf1h1 = CF1(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
        cf1h2 = CF1(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
        cf1h1.alpha[i] -= h
        cf1h2.alpha[i] += h
        llf1 = phllf(cf1h1, data)
        llf2 = phllf(cf1h2, data)
        tmp = 1.0 + abs((llf2 - llf1)) / (2*h)
        error = abs((abs(llfdash[s]) + 1.0 - tmp) / (1.0 + abs(llfdash[s])))
        println((s, abs((llf2 - llf1)) / (2*h), abs(llfdash[s])))
        @test error < tol
    end
    for i = 1:cf1.dim
        s = Symbol(:rate, i)
        cf1h1 = CF1(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
        cf1h2 = CF1(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
        cf1h1.rate[i] -= h
        cf1h2.rate[i] += h
        llf1 = phllf(cf1h1, data)
        llf2 = phllf(cf1h2, data)
        tmp = 1.0 + abs((llf2 - llf1)) / (2*h)
        error = abs((abs(llfdash[s]) + 1.0 - tmp) / (1.0 + abs(llfdash[s])))
        println((s, abs((llf2 - llf1)) / (2*h), abs(llfdash[s])))
        @test error < tol
    end
end

@testset "Test of phfit 3" begin
    cf1, llf0, = phfit(CF1(5)) do x
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
    # @test all(values .> 0)
    println(inv(-IM))
end

@testset "Test of phfit 4" begin
    cf1, llf0, = phfit(CF1(10)) do x
        pdf(Weibull(1.5, 1.0), x)
    end
    data = WeightedSample((0.0, Inf64)) do x
        pdf(Weibull(1.5, 1.0), x)
    end
    println(cf1)

    deriv = Dict{Symbol,CF1{Float64}}() ## １階微分
    for i = 1:cf1.dim
        s = Symbol(:alpha, i)
        a = zero(cf1.alpha)
        x = zero(cf1.rate)
        a[i] = 1.0
        deriv[s] = CF1{Float64}(cf1.dim, a, x)
    end
    for i = 1:cf1.dim
        s = Symbol(:rate, i)
        a = zero(cf1.alpha)
        x = zero(cf1.rate)
        x[i] = 1.0
        deriv[s] = CF1{Float64}(cf1.dim, a, x)
    end
    deriv2 = Dict{Tuple{Symbol,Symbol},CF1{Float64}}() ## 2階微分
    for i = 1:cf1.dim
        s1 = Symbol(:alpha, i)
        for j = i:cf1.dim
            s2 = Symbol(:alpha, j)
            a = zero(cf1.alpha)
            x = zero(cf1.rate)
            deriv2[(s1,s2)] = CF1{Float64}(cf1.dim, a, x)
        end
        for j = 1:cf1.dim
            s2 = Symbol(:rate, j)
            a = zero(cf1.alpha)
            x = zero(cf1.rate)
            deriv2[(s1,s2)] = CF1{Float64}(cf1.dim, a, x)
        end
    end
    for i = 1:cf1.dim
        s1 = Symbol(:rate, i)
        for j = i:cf1.dim
            s2 = Symbol(:rate, j)
            a = zero(cf1.alpha)
            x = zero(cf1.rate)
            deriv2[(s1,s2)] = CF1{Float64}(cf1.dim, a, x)
        end
    end
    
    delta = 0.00001
    tol = 1e-5
    llf, llfdash, llfdashdash = phllf(cf1, deriv, deriv2, data)
    for i = 1:cf1.dim
        for j = i:cf1.dim
            s1 = Symbol(:alpha, i)
            s2 = Symbol(:alpha, j)
            cf1h1 = CF1(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
            cf1h2 = CF1(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
            cf1h3 = CF1(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
            cf1h4 = CF1(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
            cf1h1.alpha[i] -= delta
            cf1h2.alpha[i] -= delta
            cf1h1.alpha[j] -= delta
            cf1h2.alpha[j] += delta
            cf1h3.alpha[i] += delta
            cf1h4.alpha[i] += delta
            cf1h3.alpha[j] -= delta
            cf1h4.alpha[j] += delta
            lf1 = phllf(cf1h1, data)
            lf2 = phllf(cf1h2, data)
            lf3 = phllf(cf1h3, data)
            lf4 = phllf(cf1h4, data)
            tmp = abs(((lf4 - lf3) - (lf2 - lf1)) / (2*delta)^2) + 1.0
            error = (abs(llfdashdash[s1,s2]) + 1.0 - tmp) / (abs(llfdashdash[s1,s2]) + 1.0)
            println((s1, s2, abs(((lf4 - lf3) - (lf2 - lf1)) / (2*delta)^2), abs(llfdashdash[s1,s2])))
            if i < cf1.dim && j < cf1.dim
                @test abs(error) < tol
            end
        end
        for j = 1:cf1.dim
            s1 = Symbol(:alpha, i)
            s2 = Symbol(:rate, j)
            cf1h1 = CF1(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
            cf1h2 = CF1(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
            cf1h3 = CF1(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
            cf1h4 = CF1(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
            cf1h1.alpha[i] -= delta
            cf1h2.alpha[i] -= delta
            cf1h1.rate[j] -= delta
            cf1h2.rate[j] += delta
            cf1h3.alpha[i] += delta
            cf1h4.alpha[i] += delta
            cf1h3.rate[j] -= delta
            cf1h4.rate[j] += delta
            lf1 = phllf(cf1h1, data)
            lf2 = phllf(cf1h2, data)
            lf3 = phllf(cf1h3, data)
            lf4 = phllf(cf1h4, data)
            tmp = abs(((lf4 - lf3) - (lf2 - lf1)) / (2*delta)^2) + 1.0
            error = (abs(llfdashdash[s1,s2]) + 1.0 - tmp) / (abs(llfdashdash[s1,s2]) + 1.0)
            println((s1, s2, abs(((lf4 - lf3) - (lf2 - lf1)) / (2*delta)^2), abs(llfdashdash[s1,s2])))
            @test abs(error) < tol
        end
    end
    for i = 1:cf1.dim
        for j = i:cf1.dim
            s1 = Symbol(:rate, i)
            s2 = Symbol(:rate, j)
            cf1h1 = CF1(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
            cf1h2 = CF1(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
            cf1h3 = CF1(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
            cf1h4 = CF1(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
            cf1h1.rate[i] -= delta
            cf1h2.rate[i] -= delta
            cf1h1.rate[j] -= delta
            cf1h2.rate[j] += delta
            cf1h3.rate[i] += delta
            cf1h4.rate[i] += delta
            cf1h3.rate[j] -= delta
            cf1h4.rate[j] += delta
            lf1 = phllf(cf1h1, data)
            lf2 = phllf(cf1h2, data)
            lf3 = phllf(cf1h3, data)
            lf4 = phllf(cf1h4, data)
            tmp = abs(((lf4 - lf3) - (lf2 - lf1)) / (2*delta)^2) + 1.0
            error = (abs(llfdashdash[s1,s2]) + 1.0 - tmp) / (abs(llfdashdash[s1,s2]) + 1.0)
            println((s1, s2, abs(((lf4 - lf3) - (lf2 - lf1)) / (2*delta)^2), abs(llfdashdash[s1,s2])))
            @test abs(error) < tol
        end
    end
end

@testset "Test of phfit 5" begin
    cf1, llf0, = phfit(CF1(10)) do x
        pdf(Weibull(1.5, 1.0), x)
    end
    data = WeightedSample((0.0, Inf64)) do x
        pdf(Weibull(1.5, 1.0), x)
    end
    println(cf1)

    deriv = Dict{Symbol,CF1{Float64}}() ## １階微分
    i = cf1.dim
    s = Symbol(:alpha, i)
    a = zero(cf1.alpha)
    x = zero(cf1.rate)
    a[i] = 1.0
    deriv[s] = CF1{Float64}(cf1.dim, a, x)
    deriv2 = Dict{Tuple{Symbol,Symbol},CF1{Float64}}() ## 2階微分
    i = cf1.dim
    j = cf1.dim
    s1 = Symbol(:alpha, i)
    s2 = Symbol(:alpha, j)
    a = zero(cf1.alpha)
    x = zero(cf1.rate)
    deriv2[(s1,s2)] = CF1{Float64}(cf1.dim, a, x)
    llf, llfdash, llfdashdash = phllf(cf1, deriv, deriv2, data)

    delta = 0.000001
    tol = 1e-5
    i = cf1.dim
    j = cf1.dim
    s1 = Symbol(:alpha, i)
    s2 = Symbol(:alpha, j)
    cf1h1 = CF1(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
    cf1h2 = CF1(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
    cf1h3 = CF1(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
    cf1h4 = CF1(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
    cf1h1.alpha[i] -= delta
    cf1h2.alpha[i] -= delta
    cf1h1.alpha[j] -= delta
    cf1h2.alpha[j] += delta
    cf1h3.alpha[i] += delta
    cf1h4.alpha[i] += delta
    cf1h3.alpha[j] -= delta
    cf1h4.alpha[j] += delta
    lf1 = phllf(cf1h1, data)
    lf2 = phllf(cf1h2, data)
    lf3 = phllf(cf1h3, data)
    lf4 = phllf(cf1h4, data)
    tmp = abs(((lf4 - lf3) - (lf2 - lf1)) / (2*delta)^2) + 1.0
    error = (abs(llfdashdash[s1,s2]) + 1.0 - tmp) / (abs(llfdashdash[s1,s2]) + 1.0)
    println((s1, s2, abs(((lf4 - lf3) - (lf2 - lf1)) / (2*delta)^2), abs(llfdashdash[s1,s2])))
    @test abs(error) < tol
end

@testset "Test of phfit 10" begin
    cf1, llf0, = phfit(CF1(10)) do x
        pdf(Weibull(1.5, 1.0), x)
    end
    data = WeightedSample((0.0, Inf64)) do x
        pdf(Weibull(1.5, 1.0), x)
    end
    println(cf1)
end