using SparseMatrix
using SparseArrays
using NMarkov: mexpc

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
    gph = GPH(cf1, SparseMatrixCSC)
    @test gph.T.rowval == [1, 1, 2, 2, 3]
    @test gph.T.nzval == [-0.4, 0.4, -1.4, 1.4, -10.0]
end

@testset "Test pdf" begin
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 10.0]
    ph1 = CF1(alpha, rate)

    alpha = [0.1, 0.3, 0.6]
    T = [-1.4 1.4 0; 0 -0.4 0.4; 0 0 -10.0]
    tau = [0, 0, 10.0]
    ph2 = GPH(alpha, T, tau)

    ts = LinRange(0.0, 10.0, 100)
    res1 = phpdf(ph1, ts)
    res2 = phpdf(ph2, ts)
    @test isapprox(res1, res2)
    res1 = phcdf(ph1, ts)
    res2 = phcdf(ph2, ts)
    @test isapprox(res1, res2)
    res1 = phccdf(ph1, ts)
    res2 = phccdf(ph2, ts)
    @test isapprox(res1, res2)

    ts = rand(100)
    @time res1 = phpdf(ph1, ts)
    @time res2 = phpdf(ph2, ts)
    @test isapprox(res1, res2)
    @time res1 = phcdf(ph1, ts)
    @time res2 = phcdf(ph2, ts)
    @test isapprox(res1, res2)
    @time res1 = phccdf(ph1, ts)
    @time res2 = phccdf(ph2, ts)
    @test isapprox(res1, res2)
end

@testset "Test cdf" begin
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 10.0]
    ph1 = CF1(alpha, rate)

    ts = LinRange(0.0, 10.0, 100)
    res1 = phcdf(ph1, ts)
    for i = 1:length(res1)-1
        @test res1[i] <= res1[i+1]
    end
end

@testset "Test mean" begin
    alpha = [0.1, 0.3, 0.6]
    rate = [1.4, 0.4, 10.0]
    ph1 = CF1(alpha, rate)

    alpha = [0.1, 0.3, 0.6]
    T = [-1.4 1.4 0; 0 -0.4 0.4; 0 0 -10.0]
    tau = [0, 0, 10.0]
    ph2 = GPH(alpha, T, tau)

    res1 = phmean(ph1)
    res2 = phmean(ph2)
    @test isapprox(res1, res2)

    _, cx = mexpc(T, alpha, 100.0; transpose=:T, rmax=10000)
    @test isapprox(res1, sum(cx))

    res1 = phmean(ph1, 2)
    res2 = phmean(ph2, 2)
    @test res1 > 0.0
    @test isapprox(res1, res2)
end