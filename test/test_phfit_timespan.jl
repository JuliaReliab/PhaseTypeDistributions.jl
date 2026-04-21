using SparseArrays
using NMarkov.SparseMatrix
using PhaseTypeDistributions
using PhaseTypeDistributions.Phfit
using Distributions
using Random

import PhaseTypeDistributions.Phfit: Estep, estep!, mstep!, phfit!

@testset "TimeSpanSample construction — point only" begin
    t = rand(10)
    dat = TimeSpanSample(t)
    @test dat.length > 0
    @test all(dat.ndat .≈ 1.0)
    @test all(dat.wdat .≈ 1.0)
    @test length(dat.rawt) == 10
end

@testset "TimeSpanSample construction — with n" begin
    t = rand(10)
    n = rand(1:5, 10) .* 1.0
    dat = TimeSpanSample(t, n)
    @test all(dat.wdat .≈ 1.0)
    @test dat.rawn == n
end

@testset "TimeSpanSample construction — with n and w" begin
    t = rand(10)
    n = rand(1:5, 10) .* 1.0
    w = rand(10)
    dat = TimeSpanSample(t, n, w)
    @test dat.rawn == n
    @test dat.raww == w
end

@testset "TimeSpanSample construction — intervals" begin
    t = [(0.5, 1.5), (0.0, 2.0), (1.0, Inf), 0.8, (0.3, 0.9)]
    dat = TimeSpanSample(t)
    @test dat.length > 0
    @test all(isfinite, dat.tdat)
end

@testset "TimeSpanSample equivalence: ones(n,w) == default" begin
    t = rand(10)
    dat1 = TimeSpanSample(t)
    dat2 = TimeSpanSample(t, ones(10), ones(10))
    @test dat1.tdat ≈ dat2.tdat
    @test dat1.ndat ≈ dat2.ndat
    @test dat1.wdat ≈ dat2.wdat
    @test dat1.zdat == dat2.zdat
end

@testset "TimeSpanSample llf scaling by n" begin
    alpha = [0.1, 0.3, 0.6]
    rate  = [1.4, 0.4, 2.0]
    cf1   = CF1(alpha, rate)
    ph    = GPH(cf1, SparseMatrixCSC)
    t     = rand(20)

    dat1 = TimeSpanSample(t)
    dat2 = TimeSpanSample(t, fill(2.0, 20))

    eres = Estep(ph)
    llf1 = estep!(cf1, dat1, eres)
    llf2 = estep!(cf1, dat2, eres)
    @test llf2 ≈ 2 * llf1
end

@testset "TimeSpanSample llf scaling by w" begin
    alpha = [0.1, 0.3, 0.6]
    rate  = [1.4, 0.4, 2.0]
    cf1   = CF1(alpha, rate)
    ph    = GPH(cf1, SparseMatrixCSC)
    t     = rand(20)

    dat1 = TimeSpanSample(t)
    dat2 = TimeSpanSample(t, ones(20), fill(2.0, 20))

    eres = Estep(ph)
    llf1 = estep!(cf1, dat1, eres)
    llf2 = estep!(cf1, dat2, eres)
    @test llf2 ≈ 2 * llf1
end

@testset "TimeSpanSample EM monotonicity" begin
    alpha = [0.1, 0.3, 0.6]
    rate  = [1.4, 0.4, 2.0]
    cf1   = CF1(alpha, rate)
    ph    = GPH(cf1, SparseMatrixCSC)
    t     = rand(30)
    dat   = TimeSpanSample(t)
    eres  = Estep(ph)

    prevllf = estep!(cf1, dat, eres)
    mstep!(cf1, eres)
    for _ in 1:10
        llf = estep!(cf1, dat, eres)
        @test llf > prevllf
        mstep!(cf1, eres)
        prevllf = llf
    end
end

@testset "TimeSpanSample EM monotonicity — mixed intervals" begin
    rng   = MersenneTwister(42)
    alpha = [0.1, 0.3, 0.6]
    rate  = [1.4, 0.4, 2.0]
    cf1   = CF1(alpha, rate)
    ph    = GPH(cf1, SparseMatrixCSC)

    t = vcat(rand(rng, 10), [(0.1 * i, 0.1 * i + 0.5) for i in 1:10], [(r, Inf) for r in rand(rng, 5)])
    dat  = TimeSpanSample(t)
    eres = Estep(ph)

    prevllf = estep!(cf1, dat, eres)
    mstep!(cf1, eres)
    for _ in 1:5
        llf = estep!(cf1, dat, eres)
        @test llf > prevllf
        mstep!(cf1, eres)
        prevllf = llf
    end
end

@testset "TimeSpanSample phfit smoke test" begin
    t   = rand(100)
    dat = TimeSpanSample(t)
    res = phfit(CF1(5), dat; progress_init=false, progress=false)
    @test isfinite(res.llf)
end

@testset "TimeSpanSample rawn is Int" begin
    t   = rand(10)
    n   = [2, 3, 1, 4, 2, 1, 3, 2, 1, 2]
    dat = TimeSpanSample(t, n)
    @test eltype(dat.rawn) == Int
    @test dat.rawn == n
end

@testset "Bootstrap produces valid TimeSpanSample" begin
    t    = rand(50)
    dat  = TimeSpanSample(t)
    bdat = bootstrap(dat)
    @test bdat.length > 0
    @test sum(bdat.rawn) == sum(dat.rawn)   # total count preserved
    @test length(bdat.rawt) == length(dat.rawt)
    @test eltype(bdat.rawn) == Int
end

@testset "Bootstrap with rng is reproducible" begin
    t    = rand(50)
    dat  = TimeSpanSample(t)
    rng1 = MersenneTwister(42)
    rng2 = MersenneTwister(42)
    bdat1 = bootstrap(rng1, dat)
    bdat2 = bootstrap(rng2, dat)
    @test bdat1.tdat == bdat2.tdat
end

@testset "Bootstrap + phfit runs" begin
    t    = rand(50)
    dat  = TimeSpanSample(t)
    bdat = bootstrap(dat)
    res  = phfit(CF1(5), bdat; progress_init=false, progress=false)
    @test isfinite(res.llf)
end

# ── EIC ──────────────────────────────────────────────────────────────────────

@testset "eic smoke test" begin
    rng = MersenneTwister(1)
    dat = TimeSpanSample(rand(rng, 100))
    res = phfit(CF1(5), dat; progress_init=false, progress=false)
    r   = eic(MersenneTwister(42), res.model, dat; bsample=30)

    @test isfinite(r.eic)
    @test isfinite(r.ci_lower)
    @test isfinite(r.ci_upper)
    @test r.nvalid > 0
    @test r.nvalid <= 30
end

@testset "eic CI brackets point estimate" begin
    # ci_lower = eic - 2*1.96*se  <  eic  <  eic + 2*1.96*se = ci_upper
    rng = MersenneTwister(1)
    dat = TimeSpanSample(rand(rng, 100))
    res = phfit(CF1(5), dat; progress_init=false, progress=false)
    r   = eic(MersenneTwister(42), res.model, dat; bsample=30)

    @test r.ci_lower <= r.eic <= r.ci_upper
end

@testset "eic reproducibility" begin
    rng = MersenneTwister(1)
    dat = TimeSpanSample(rand(rng, 80))
    res = phfit(CF1(4), dat; progress_init=false, progress=false)

    r1 = eic(MersenneTwister(7), res.model, dat; bsample=20)
    r2 = eic(MersenneTwister(7), res.model, dat; bsample=20)

    @test r1.eic      == r2.eic
    @test r1.ci_lower == r2.ci_lower
    @test r1.ci_upper == r2.ci_upper
    @test r1.nvalid   == r2.nvalid
end

@testset "eic nvalid equals bsample on well-behaved data" begin
    rng = MersenneTwister(1)
    dat = TimeSpanSample(rand(rng, 200))
    res = phfit(CF1(3), dat; progress_init=false, progress=false)
    r   = eic(MersenneTwister(42), res.model, dat; bsample=20)

    @test r.nvalid == 20
end
