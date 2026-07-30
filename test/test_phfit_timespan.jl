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

# ---------------------------------------------------------------------------
# Left truncation
# ---------------------------------------------------------------------------

@testset "TimeSpanSample left truncation — construction" begin
    tau = [0.0, 0.5, 0.0, 1.2]
    t   = tau .+ [0.3, 0.4, 0.7, 0.2]
    dat = TimeSpanSample(t; tau=tau)

    # two truncation entries were appended on top of the four observations
    @test dat.length == 6
    @test count(==(-1), dat.zdat) == 2
    @test dat.rawtau == tau

    # tau omitted => no truncation entries, backward compatible
    dat0 = TimeSpanSample(t)
    @test dat0.length == 4
    @test !any(==(-1), dat0.zdat)
    @test all(iszero, dat0.rawtau)

    @test_throws ArgumentError TimeSpanSample(t; tau=[0.0, 0.5])
    @test_throws ArgumentError TimeSpanSample([1.0]; tau=[2.0])  # tau after the event
end

@testset "TimeSpanSample left truncation — estep! matches LeftTruncRightCensoredSample" begin
    Random.seed!(1234)
    cf1 = CF1([0.1, 0.3, 0.6], [1.4, 0.4, 2.0])
    ph  = GPH(cf1, SparseMatrixCSC)

    tau   = [0.0, 0.5, 0.0, 1.2, 0.3]
    t     = tau .+ rand(5)
    delta = [true, true, false, true, false]  # true = exact, false = right censored

    d_lr = LeftTruncRightCensoredSample(t, tau, delta)
    d_ts = TimeSpanSample([delta[i] ? t[i] : (t[i], Inf) for i in eachindex(t)]; tau=tau)

    e1 = Estep(ph); e2 = Estep(ph)
    llf1 = estep!(cf1, d_lr, e1)
    llf2 = estep!(cf1, d_ts, e2)

    @test llf1 ≈ llf2
    @test e1.etotal ≈ e2.etotal
    @test e1.eb ≈ e2.eb
    @test e1.ey ≈ e2.ey
    @test e1.ez ≈ e2.ez
    @test Matrix(e1.en) ≈ Matrix(e2.en)
end

@testset "TimeSpanSample left truncation — phfit matches LeftTruncRightCensoredSample" begin
    Random.seed!(2024)
    n     = 60
    tau   = [isodd(i) ? 0.0 : 0.2 * rand() for i in 1:n]
    t     = tau .+ rand(n)
    delta = rand(Bool, n)

    d_lr = LeftTruncRightCensoredSample(t, tau, delta)
    d_ts = TimeSpanSample([delta[i] ? t[i] : (t[i], Inf) for i in eachindex(t)]; tau=tau)

    r_lr = phfit!(CF1([0.2, 0.3, 0.5], [1.0, 2.0, 3.0]), d_lr; progress=false)
    r_ts = phfit!(CF1([0.2, 0.3, 0.5], [1.0, 2.0, 3.0]), d_ts; progress=false)

    @test r_lr[1] ≈ r_ts[1]   # llf
end

@testset "TimeSpanSample left truncation — llf against the closed form" begin
    cf1 = CF1([0.2, 0.3, 0.5], [1.5, 0.7, 2.5])

    # exact, interval, right-censored and [0,b] observations, all left truncated
    tt  = [1.1, (0.6, 1.4), (0.9, Inf), (0.0, 1.7)]
    tau = [0.4, 0.3, 0.5, 0.0]
    dat = TimeSpanSample(tt, [1, 1, 1, 1]; tau=tau)

    S(x) = ccdf(cf1, x)
    expected = log(pdf(cf1, 1.1))    - log(S(0.4)) +
               log(S(0.6) - S(1.4))  - log(S(0.3)) +
               log(S(0.9))           - log(S(0.5)) +
               log(S(0.0) - S(1.7))

    @test phllf(cf1, dat) ≈ expected
end

@testset "TimeSpanSample left truncation — bootstrap and eic" begin
    rng = MersenneTwister(9)
    n   = 80
    tau = [isodd(i) ? 0.0 : 0.15 * rand(rng) for i in 1:n]
    t   = tau .+ rand(rng, n)
    dat = TimeSpanSample(t; tau=tau)

    b = bootstrap(MersenneTwister(3), dat)
    @test b.rawtau == dat.rawtau
    @test sum(b.rawn) == sum(dat.rawn)

    res = phfit(CF1(3), dat; progress_init=false, progress=false)
    r   = eic(MersenneTwister(11), res.model, dat; bsample=10)
    @test r.nvalid == 10
    @test isfinite(r.eic)
    @test r.ci_lower <= r.eic <= r.ci_upper
end

@testset "TimeSpanSample adjacent same-kind observations are not paired" begin
    # Regression: two consecutive right-censored observations used to be mistaken
    # for the single interval [0.5, 0.8], and likewise for two [0, b] intervals.
    d = TimeSpanSample([(0.5, Inf), (0.8, Inf), 1.2])
    @test d.zdat == [4, 4, 3]        # m+1, m+1, exact

    d = TimeSpanSample([(0.0, 1.0), (0.0, 2.0)])
    @test d.zdat == [0, 0]

    # A real interval is still paired.
    d = TimeSpanSample([(0.5, 0.8)])
    @test d.zdat == [2, 1]

    # And the likelihood matches the closed form for adjacent censored observations.
    cf1 = CF1([0.3, 0.7], [1.0, 2.5])
    dat = TimeSpanSample([(0.5, Inf), (0.8, Inf), 1.2])
    @test phllf(cf1, dat) ≈ log(ccdf(cf1, 0.5)) + log(ccdf(cf1, 0.8)) + log(pdf(cf1, 1.2))
end
