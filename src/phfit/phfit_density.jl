using ZeroOrigin: @origin
using LinearAlgebra.BLAS: gemv!, scal!, axpy!
using SparseMatrix: spdiag, spger!
using NMarkov: itime, @dot, rightbound, poipmf!, unif
using Deformula: deint

struct WeightedSample{Tv} <: AbstractPHSample
    length::Int
    maxtime::Tv
    tdat::Vector{Tv}
    wdat::Vector{Tv}
end

function mean(data::WeightedSample{Tv}) where Tv
    totalt = Tv(0)
    totalw = Tv(0)
    ct = Tv(0)
    for i = eachindex(data.tdat)
        ct += data.tdat[i]
        totalt += data.wdat[i] * ct
        totalw += data.wdat[i]
    end
    totalt / totalw
end

function WeightedSample(t::Vector{Tv}, w::Vector{Tv}) where Tv
    i = sortperm(t)
    dt,maxt = itime(t[i])
    WeightedSample(length(dt), maxt, dt, w[i])
end

function PointSample(t::Vector{Tv}) where Tv
    i = sortperm(t)
    dt,maxt = itime(t[i])
    WeightedSample(length(dt), maxt, dt, ones(length(dt)))
end

function WeightedSample(f::Any, bounds::Tuple{Tv,Tv}; reltol::Tv = 1.0e-8, abstol::Tv = eps(Tv), d = 8, maxiter = 16) where Tv
    de = deint(f, bounds[1], bounds[2], reltol=reltol, abstol=abstol, d=d, maxiter=maxiter)
    WeightedSample(de.x, de.w * de.h)
end

# TODO: interface should be changed so that we can control deformula prameters
function phfit(f::Any, cf1::CF1{Tv}, bounds::Tuple{Tv,Tv} = (Tv(0), Tv(Inf)), ::Type{MatT} = SparseMatrixCSC;
    initialize = true, eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01),
    steps = 10,
    ratio = Tv[1, 4, 16, 64, 256, 1024], m1 = Tv[0.5, 1.0, 2.0], init_maxiter = 5,
    maxiter = 5000, abstol = Tv(1.0e-3), reltol = Tv(1.0e-6)) where {Tv,MatT}

    data = WeightedSample(f, bounds)
    phfit(cf1, data, MatT, initialize=initialize, eps=eps, ufact=ufact, ratio=ratio,
        m1=m1, init_maxiter=init_maxiter, steps=steps, maxiter=maxiter, abstol=abstol, reltol=reltol)
end

@inbounds @origin (vf => 0, vb => 0, vc => 0) function estep!(ph::GPH{Tv,MatT},
    data::WeightedSample{Tv}, eres::Estep{Tv,MatT}; eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01)) where {Tv,MatT}

    dim, alpha, tau = ph.dim, ph.alpha, ph.tau
    P, qv = unif(ph.T, ufact)
    @assert isfinite(qv)

    llf = Tv(0)
    tllf = Tv(0)
    clear!(eres)

    right = rightbound(qv * data.maxtime, eps) + 1
    poi = Vector{Tv}(undef, right + 1)

    m = data.length
    blf = Vector{Tv}(undef, m)
    vf = Vector{Vector{Tv}}(undef, m+1)
    vb = Vector{Vector{Tv}}(undef, m+1)
    vc = Vector{Vector{Tv}}(undef, m+1)
    vx = Vector{Vector{Tv}}(undef, right)
    for l = 1:right
        vx[l] = similar(alpha)
    end
    xtmp = similar(alpha)
    tmpv = similar(alpha)

    vf[0] = alpha
    vb[0] = tau
    for k = 1:m
        right = rightbound(qv * data.tdat[k], eps) + 1
        weight = poipmf!(qv * data.tdat[k], poi, left=0, right=right)
        vf[k] = zero(alpha)
        @. xtmp = vf[k-1]
        @origin (poi => 0, vf=>0) begin
            axpy!(poi[0], xtmp, vf[k])
            for i = 1:right
                gemv!('T', 1.0, P, xtmp, false, tmpv)
                @. xtmp = tmpv
                axpy!(poi[i], xtmp, vf[k])
            end
        end
        scal!(1.0/weight, vf[k])
        scale = @dot(vf[k], tau)
        scal!(1.0/scale, vf[k])
        axpy!(data.wdat[k], vf[k], eres.ey)

        blf[k] = scale
        vb[k] = zero(alpha)
        @. xtmp = vb[k-1]
        @origin (poi => 0, vb=>0) begin
            axpy!(poi[0], xtmp, vb[k])
            for i = 1:right
                gemv!('N', 1.0, P, xtmp, false, tmpv)
                @. xtmp = tmpv
                axpy!(poi[i], xtmp, vb[k])
            end
        end
        scal!(1.0/weight, vb[k])
        scale = @dot(alpha, vb[k])
        scal!(1.0/scale, vb[k])
        axpy!(data.wdat[k], vb[k], eres.eb)

        tllf += log(blf[k])
        llf += data.wdat[k] * tllf
    end

    vc[m] = zeros(Tv, dim)
    axpy!(data.wdat[m]/blf[m], alpha, vc[m])
    for k = m-1:-1:1
        right = rightbound(qv * data.tdat[k+1], eps) + 1
        weight = poipmf!(qv * data.tdat[k+1], poi, left=0, right=right)
        vc[k] = zeros(Tv, dim)
        @. xtmp = vc[k+1]
        @origin (poi => 0, vc=>0) begin
            vc[k] = zero(alpha)
            axpy!(poi[0], xtmp, vc[k])
            for i = 1:right
                gemv!('T', 1.0, P, xtmp, false, tmpv)
                @. xtmp = tmpv
                axpy!(poi[i], xtmp, vc[k])
            end
        end
        scal!(1.0/(weight*blf[k]), vc[k])
        axpy!(data.wdat[k]/blf[k], alpha, vc[k])
    end

    for k = 1:m
        right = rightbound(qv * data.tdat[k], eps) + 1
        weight = poipmf!(qv * data.tdat[k], poi, left=0, right=right)

        @origin (poi=>0, vb=>0) begin
            @. vx[right] = zero(Tv)
            axpy!(poi[right], vb[k-1], vx[right])
            for l = right-1:-1:1
                gemv!('N', 1.0, P, vx[l+1], false, vx[l])
                axpy!(poi[l], vb[k-1], vx[l])
            end
        end

        spger!(1.0/(qv*weight), vc[k], vx[1], 1.0, eres.en)
        for l = 1:right-1
            gemv!('T', 1.0, P, vc[k], false, tmpv)
            @. vc[k] = tmpv
            spger!(1.0/(qv*weight), vc[k], vx[l+1], 1.0, eres.en)
        end
    end

    @. eres.eb *= alpha
    @. eres.ey *= tau
    eres.ez = spdiag(eres.en)
    for i = 1:length(eres.en)
        eres.en[i] *= ph.T[i]
    end
    eres.etotal = sum(eres.eb)

    return llf
end

