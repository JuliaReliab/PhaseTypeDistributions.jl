using Origin: @origin
using SparseMatrix: spdiag, scal!, axpy!, gemv!, spger!
using NMarkov: itime, @dot, rightbound, poipmf!, unif
using SpecialFunctions: loggamma

export
    GroupTruncSample,
    GroupTruncPoiSample,
    mean

abstract type AbstractPHGroupSample <: AbstractPHSample end

struct GroupTruncSample <: AbstractPHGroupSample
    length::Int
    maxtime::Float64
    tdat::Vector{Float64}
    gdat::Vector{Int}
    idat::Vector{Bool}
    gdatlast::Int
end

function GroupTruncSample(t::Vector{Float64}, x::Vector{Int}, xlast::Int = 0)
    len = length(t)
    maxt = maximum(t)
    i = [false for i = 1:len]
    GroupTruncSample(len, maxt, t, x, i, xlast)
end

function GroupTruncSample(t::Vector{Float64}, x::Vector{Int}, i::Vector{Bool}, xlast::Int)
    len = length(t)
    maxt = maximum(t)
    GroupTruncSample(len, maxt, t, x, i, xlast)
end

function mean(data::AbstractPHGroupSample)
    totalt = 0.0
    totaln = 0.0
    ct = 0.0
    for i = eachindex(data.tdat)
        ct += data.tdat[i]
        if data.gdat[i] >= 0
            totalt += data.gdat[i] * ct
            totaln += data.gdat[i]
        end
        if !iszero(data.idat[i])
            totalt += ct
            totaln += 1
        end
    end
    return totalt / totaln
end

function estep!(ph::CF1{Tv}, data::GroupTruncSample, eres::Estep{Tv,MatT};
    eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01)) where {Tv,MatT}
    estep!(GPH(ph, MatT), data, eres, eps=eps, ufact=ufact)
end

@origin (barvf=>0, vb=>0, barvb=>0, wg=>1, wp=>1, vc=>0, poi=>0, vx=>0) function estep!(
    ph::GPH{Tv,MatT},
    data::GroupTruncSample,
    eres::Estep{Tv,MatT};
    eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01)) where {Tv,MatT}

    dim, alpha, tau = ph.dim, ph.alpha, ph.tau
    P, qv = unif(ph.T, ufact)
    baralpha = (-ph.T)' \ alpha
    one = ones(dim)

    m = data.length

    clear!(eres)
    right = rightbound(qv * data.maxtime, eps) + 1
    poi = Vector{Tv}(undef, right + 1)

    llf = Tv(0)
    tmpvf = Vector{Tv}(undef, dim)
    tmpvb = Vector{Tv}(undef, dim)
    tmpv = Vector{Tv}(undef, dim)

    barvf = Vector{Vector{Tv}}(undef, m+1)
    vb = Vector{Vector{Tv}}(undef, m+1)
    barvb = Vector{Vector{Tv}}(undef, m+1)
    wg = Vector{Tv}(undef, m+1)
    wp = Vector{Tv}(undef, m+1)
    vc = Vector{Vector{Tv}}(undef, m+1)
    vx = Vector{Vector{Tv}}(undef, right + 1)
    for i = 0:right
        vx[i] = zeros(Tv,dim)
    end

    barvf[0] = baralpha
    barvb[0] = one
    vb[0] = tau
    nn = Tv(0)
    uu = Tv(0)
    
    @inbounds for k = 1:m
        # barvf[k] = barvf[k-1] * exp(T * tdat[k])
        # barvb[k] = exp(T * tdat[k]) * barvb[k-1]
        begin
            right = rightbound(qv * data.tdat[k], eps) + 1
            weight = poipmf!(qv * data.tdat[k], poi, left=0, right=right)

            barvf[k] = zeros(Tv, dim)
            barvb[k] = zeros(Tv, dim)
            @. tmpvf = barvf[k-1]
            @. tmpvb = barvb[k-1]
            axpy!(poi[0], tmpvf, barvf[k])
            axpy!(poi[0], tmpvb, barvb[k])
            for u = 1:right
                gemv!('T', 1.0, P, tmpvf, false, tmpv); @. tmpvf = tmpv
                gemv!('N', 1.0, P, tmpvb, false, tmpv); @. tmpvb = tmpv
                axpy!(poi[u], tmpvf, barvf[k])
                axpy!(poi[u], tmpvb, barvb[k])
            end
            scal!(1/weight, barvf[k])
            scal!(1/weight, barvb[k])
        end
        # vb[k] = (-ph.T) * barvb[k]
        begin
            vb[k] = similar(alpha)
            gemv!('N', -1.0, ph.T, barvb[k], false, vb[k])
        end

        @. tmpvf = barvf[k-1] - barvf[k] # tildevf = barvf[k-1] - barvf[k]
        @. tmpvb = barvb[k-1] - barvb[k] # tildevf = barvb[k-1] - barvb[k]

        if data.gdat[k] >= 0 && data.tdat[k] != 0.0
            tmp = @dot(alpha, tmpvb)
            llf += data.gdat[k] * log(tmp) - loggamma(data.gdat[k]+1)
            nn += data.gdat[k]
            uu += tmp
            wg[k] = data.gdat[k] / tmp
            axpy!(wg[k], tmpvb, eres.eb)
            axpy!(wg[k], tmpvf, eres.ey)
        end
        if data.idat[k] == true
            tmp = @dot(alpha, vb[k])
            llf += log(tmp)
            nn += 1
            wp[k] = 1 / tmp
            axpy!(wp[k], vb[k], eres.eb)
            gemv!('T', -wp[k], ph.T, barvf[k], 1.0, eres.ey)
        else
            wp[k] = 0.0
        end
    end
    # for the interval [t_m, infinity)
    if data.gdatlast >= 0
        tmp = @dot(alpha, barvb[m])
        llf += data.gdatlast * log(tmp) - loggamma(data.gdatlast+1)
        nn += data.gdatlast
        uu += tmp
        wg[m+1] = data.gdatlast / tmp
        axpy!(wg[m+1], barvb[m], eres.eb)
        axpy!(wg[m+1], barvf[m], eres.ey)
    end
    # compute weights for unobserved periods
    @inbounds for k = 1:m
        if data.gdat[k] == -1
            wg[k] = nn / uu
            @. tmpvf = barvf[k-1] - barvf[k]
            @. tmpvb = barvb[k-1] - barvb[k]
            axpy!(wg[k], tmpvb, eres.eb)
            axpy!(wg[k], tmpvf, eres.ey)
        end
    end
    if data.gdatlast == -1
        wg[m+1] = nn / uu
        axpy!(wg[m+1], barvb[m], eres.eb)
        axpy!(wg[m+1], barvf[m], eres.ey)
    end
    llf += loggamma(nn + 1) - nn * log(uu)

    # compute vectors for convolution

    vc[m] = zero(alpha)
    axpy!(wg[m+1] - wg[m], baralpha, vc[m])
    if data.idat[m] == true
        axpy!(wp[m], alpha, vc[m])
    end
    @inbounds for k=m-1:-1:1
        # vc[k] = vc[k+1] * exp(T * tdat[k+1]) + (wg[k+1] - wg[k]) * baralpha + I(idat[k]==1) (wp[k] * alpha)
        begin
            right = rightbound(qv * data.tdat[k+1], eps) + 1
            weight = poipmf!(qv * data.tdat[k+1], poi, left=0, right=right)

            vc[k] = zeros(Tv, dim)
            @. tmpvf = vc[k+1]
            axpy!(poi[0], tmpvf, vc[k])
            for u = 1:right
                gemv!('T', 1.0, P, tmpvf, false, tmpv); @. tmpvf = tmpv
                axpy!(poi[u], tmpvf, vc[k])
            end
            scal!(1/weight, vc[k])
            axpy!(wg[k+1]-wg[k], baralpha, vc[k])
            if data.idat[k] == true
                axpy!(wp[k], alpha, vc[k])
            end
        end
    end

    @inbounds for k=1:m
        # compute convolution integral
        #  int_0^tdat[k] exp(T* s) * vb[k-1] * vc[k] * exp(T(tdat[k]-s)) ds
        begin
            right = rightbound(qv * data.tdat[k], eps) + 1
            weight = poipmf!(qv * data.tdat[k], poi, left=0, right=right)

            @. vx[right] = zero(Tv)
            axpy!(poi[right], vb[k-1], vx[right])
            for l = right-1:-1:1
                gemv!('N', 1.0, P, vx[l+1], false, vx[l])
                axpy!(poi[l], vb[k-1], vx[l])
            end

            spger!(1.0/(qv*weight), vc[k], vx[1], 1.0, eres.en)
            for l = 1:right-1
                gemv!('T', 1.0, P, vc[k], false, tmpv); @. vc[k] = tmpv
                spger!(1.0/(qv*weight), vc[k], vx[l+1], 1.0, eres.en)
            end
        end
        spger!(wg[k+1]-wg[k], baralpha, barvb[k], 1.0, eres.en)
    end
    spger!(wg[1], baralpha, barvb[0], 1.0, eres.en)

    eres.etotal = nn / uu
    @. eres.eb *= alpha
    @. eres.ey *= tau
    eres.ez = spdiag(eres.en)
    for i = 1:length(eres.en)
        eres.en[i] *= ph.T[i]
    end

    llf
end

## trunc poi

mutable struct GroupTruncPoiSample <: AbstractPHGroupSample
    length::Int
    maxtime::Float64
    tdat::Vector{Float64}
    gdat::Vector{Int}
    idat::Vector{Bool}
    gdatlast::Int
    omega::Float64
end

function GroupTruncPoiSample(t::Vector{Float64}, x::Vector{Int}, xlast::Int = -1)
    len = length(t)
    maxt = maximum(t)
    i = [false for i = 1:len]
    GroupTruncPoiSample(len, maxt, t, x, i, xlast, sum(x))
end

function estep!(ph::CF1{Tv}, data::GroupTruncPoiSample, eres::Estep{Tv,MatT};
    eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01)) where {Tv,MatT}
    estep!(GPH(ph, MatT), data, eres, eps=eps, ufact=ufact)
end

@origin (barvf=>0, vb=>0, barvb=>0, wg=>1, wp=>1, vc=>0, poi=>0, vx=>0) function estep!(
    ph::GPH{Tv,MatT},
    data::GroupTruncPoiSample,
    eres::Estep{Tv,MatT};
    eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01)) where {Tv,MatT}

    dim, alpha, tau = ph.dim, ph.alpha, ph.tau
    P, qv = unif(ph.T, ufact)
    omega = data.omega

    baralpha = (-ph.T)' \ alpha
    one = ones(dim)

    m = data.length

    clear!(eres)
    right = rightbound(qv * data.maxtime, eps) + 1
    poi = Vector{Tv}(undef, right + 1)

    llf = Tv(0)
    tmpvf = Vector{Tv}(undef, dim)
    tmpvb = Vector{Tv}(undef, dim)
    tmpv = Vector{Tv}(undef, dim)

    barvf = Vector{Vector{Tv}}(undef, m+1)
    vb = Vector{Vector{Tv}}(undef, m+1)
    barvb = Vector{Vector{Tv}}(undef, m+1)
    wg = Vector{Tv}(undef, m+1)
    wp = Vector{Tv}(undef, m+1)
    vc = Vector{Vector{Tv}}(undef, m+1)
    vx = Vector{Vector{Tv}}(undef, right + 1)
    for i = 0:right
        vx[i] = zeros(Tv,dim)
    end

    barvf[0] = baralpha
    barvb[0] = one
    vb[0] = tau
    nn = Tv(0)
    uu = Tv(0)
    
    @inbounds for k = 1:m
        # barvf[k] = barvf[k-1] * exp(T * tdat[k])
        # barvb[k] = exp(T * tdat[k]) * barvb[k-1]
        begin
            right = rightbound(qv * data.tdat[k], eps) + 1
            weight = poipmf!(qv * data.tdat[k], poi, left=0, right=right)

            barvf[k] = zeros(Tv, dim)
            barvb[k] = zeros(Tv, dim)
            @. tmpvf = barvf[k-1]
            @. tmpvb = barvb[k-1]
            axpy!(poi[0], tmpvf, barvf[k])
            axpy!(poi[0], tmpvb, barvb[k])
            for u = 1:right
                gemv!('T', 1.0, P, tmpvf, false, tmpv); @. tmpvf = tmpv
                gemv!('N', 1.0, P, tmpvb, false, tmpv); @. tmpvb = tmpv
                axpy!(poi[u], tmpvf, barvf[k])
                axpy!(poi[u], tmpvb, barvb[k])
            end
            scal!(1/weight, barvf[k])
            scal!(1/weight, barvb[k])
        end
        # vb[k] = (-ph.T) * barvb[k]
        begin
            vb[k] = similar(alpha)
            gemv!('N', -1.0, ph.T, barvb[k], false, vb[k])
        end

        @. tmpvf = barvf[k-1] - barvf[k] # tildevf = barvf[k-1] - barvf[k]
        @. tmpvb = barvb[k-1] - barvb[k] # tildevf = barvb[k-1] - barvb[k]

        if data.gdat[k] >= 0 && data.tdat[k] != 0.0
            tmp = @dot(alpha, tmpvb)
            llf += data.gdat[k] * log(tmp) - loggamma(data.gdat[k]+1)
            nn += data.gdat[k]
            uu += tmp
            wg[k] = data.gdat[k] / tmp
            axpy!(wg[k], tmpvb, eres.eb)
            axpy!(wg[k], tmpvf, eres.ey)
        end
        if data.idat[k] == true
            tmp = @dot(alpha, vb[k])
            llf += log(tmp)
            nn += 1
            wp[k] = 1 / tmp
            axpy!(wp[k], vb[k], eres.eb)
            gemv!('T', -wp[k], ph.T, barvf[k], 1.0, eres.ey)
        else
            wp[k] = 0.0
        end
    end
    # for the interval [t_m, infinity)
    if data.gdatlast >= 0
        tmp = @dot(alpha, barvb[m])
        llf += data.gdatlast * log(tmp) - loggamma(data.gdatlast+1)
        nn += data.gdatlast
        uu += tmp
        wg[m+1] = data.gdatlast / tmp
        axpy!(wg[m+1], barvb[m], eres.eb)
        axpy!(wg[m+1], barvf[m], eres.ey)
    end
    # compute weights for unobserved periods
    @inbounds for k = 1:m
        if data.gdat[k] == -1
            wg[k] = omega
            @. tmpvf = barvf[k-1] - barvf[k]
            @. tmpvb = barvb[k-1] - barvb[k]
            axpy!(wg[k], tmpvb, eres.eb)
            axpy!(wg[k], tmpvf, eres.ey)
        end
    end
    if data.gdatlast == -1
        wg[m+1] = omega
        axpy!(wg[m+1], barvb[m], eres.eb)
        axpy!(wg[m+1], barvf[m], eres.ey)
    end
    llf += nn * log(omega) - omega * uu

    # compute vectors for convolution

    vc[m] = zero(alpha)
    axpy!(wg[m+1] - wg[m], baralpha, vc[m])
    if data.idat[m] == true
        axpy!(wp[m], alpha, vc[m])
    end
    @inbounds for k=m-1:-1:1
        # vc[k] = vc[k+1] * exp(T * tdat[k+1]) + (wg[k+1] - wg[k]) * baralpha + I(idat[k]==1) (wp[k] * alpha)
        begin
            right = rightbound(qv * data.tdat[k+1], eps) + 1
            weight = poipmf!(qv * data.tdat[k+1], poi, left=0, right=right)

            vc[k] = zeros(Tv, dim)
            @. tmpvf = vc[k+1]
            axpy!(poi[0], tmpvf, vc[k])
            for u = 1:right
                gemv!('T', 1.0, P, tmpvf, false, tmpv); @. tmpvf = tmpv
                axpy!(poi[u], tmpvf, vc[k])
            end
            scal!(1/weight, vc[k])
            axpy!(wg[k+1]-wg[k], baralpha, vc[k])
            if data.idat[k] == true
                axpy!(wp[k], alpha, vc[k])
            end
        end
    end

    @inbounds for k=1:m
        # compute convolution integral
        #  int_0^tdat[k] exp(T* s) * vb[k-1] * vc[k] * exp(T(tdat[k]-s)) ds
        begin
            right = rightbound(qv * data.tdat[k], eps) + 1
            weight = poipmf!(qv * data.tdat[k], poi, left=0, right=right)

            @. vx[right] = zero(Tv)
            axpy!(poi[right], vb[k-1], vx[right])
            for l = right-1:-1:1
                gemv!('N', 1.0, P, vx[l+1], false, vx[l])
                axpy!(poi[l], vb[k-1], vx[l])
            end

            spger!(1.0/(qv*weight), vc[k], vx[1], 1.0, eres.en)
            for l = 1:right-1
                gemv!('T', 1.0, P, vc[k], false, tmpv); @. vc[k] = tmpv
                spger!(1.0/(qv*weight), vc[k], vx[l+1], 1.0, eres.en)
            end
        end
        spger!(wg[k+1]-wg[k], baralpha, barvb[k], 1.0, eres.en)
    end
    spger!(wg[1], baralpha, barvb[0], 1.0, eres.en)

    eres.etotal = nn + omega * (1.0 - uu)
    @. eres.eb *= alpha
    @. eres.ey *= tau
    eres.ez = spdiag(eres.en)
    for i = 1:length(eres.en)
        eres.en[i] *= ph.T[i]
    end

    # update omega
    data.omega = eres.etotal

    llf
end