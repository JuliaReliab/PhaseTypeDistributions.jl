using ZeroOrigin: @origin
using LinearAlgebra.BLAS: gemv!, scal!, axpy!
using SparseMatrix: spdiag, spger!
using NMarkov: itime, @dot, rightbound, poipmf!, unif

struct LeftTruncRightCensoredSample <: AbstractPHSample
    length::Int
    maxtime::Float64
    tdat::Vector{Float64}
    nu::Vector{Int} # indicator: 0 exact, 1 right censoring, 3 left truncation
end

function LeftTruncRightCensoredSample(t::Vector{Float64}, tau::Vector{Float64}, delta::Vector{Bool})
    dat = []
    for (t,c) = zip(t, delta)
        if c == 1
            push!(dat, (t, 0))
        else
            push!(dat, (t, 1))
        end
    end
    for t = tau
        if t > 0.0
            push!(dat, (t, 3))
        end
    end
    sort!(dat, by = x -> x[1])
    t = [x[1] for x = dat]
    nu = [x[2] for x = dat]
    s, maxtime = itime(t)
    LeftTruncRightCensoredSample(length(s), maxtime, s, nu)
end

function mean(data::LeftTruncRightCensoredSample)
    totalt = 0.0
    totaln = 0.0
    ct = 0.0
    for i = eachindex(data.tdat)
        ct += data.tdat[i]
        if data.nu[i] == 0 || data.nu[i] == 1
            totalt += ct
            totaln += 1
        end
    end
    return totalt / totaln
end

function estep!(ph::CF1{Tv}, data::LeftTruncRightCensoredSample, eres::Estep{Tv,MatT};
    eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01)) where {Tv,MatT}
    estep!(GPH(ph, MatT), data, eres, eps=eps, ufact=ufact)
end

@origin (vf=>0, barvf=>0, vb=>0, barvb=>0, wb=>1, vc=>0, poi=>0, vx=>0) function estep!(
    ph::GPH{Tv,MatT},
    data::LeftTruncRightCensoredSample,
    eres::Estep{Tv,MatT};
    eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01)) where {Tv,MatT}

    dim, alpha, tau, baralpha = ph.dim, ph.alpha, ph.tau, ph.baralpha
    P, qv = unif(ph.T, ufact)
    @assert isfinite(qv)
    one = ones(dim)

    m = data.length

    clear!(eres)
    right = rightbound(qv * data.maxtime, eps) + 1
    poi = Vector{Tv}(undef, right + 1)

    llf = Tv(0)
    nn = Tv(0)
    tmpvf = Vector{Tv}(undef, dim)
    tmpvb = Vector{Tv}(undef, dim)
    tmpv = Vector{Tv}(undef, dim)

    barvf = Vector{Vector{Tv}}(undef, m+1)
    vb = Vector{Vector{Tv}}(undef, m+1)
    barvb = Vector{Vector{Tv}}(undef, m+1)
    vf = Vector{Vector{Tv}}(undef, m+1)
    wb = Vector{Tv}(undef, m+1)
    vc = Vector{Vector{Tv}}(undef, m+1)
    vx = Vector{Vector{Tv}}(undef, right + 1)
    for i = 0:right
        vx[i] = zeros(Tv,dim)
    end

    barvf[0] = baralpha
    barvb[0] = one
    vb[0] = tau
    vf[0] = alpha
    
    @inbounds for k = 1:m
        begin
            right = rightbound(qv * data.tdat[k], eps) + 1
            weight = poipmf!(qv * data.tdat[k], poi, left=0, right=right)

            # barvb[k] = exp(T * tdat[k]) * barvb[k-1]
            barvb[k] = zeros(Tv, dim)
            @. tmpvb = barvb[k-1]
            axpy!(poi[0], tmpvb, barvb[k])
            for u = 1:right
                gemv!('N', 1.0, P, tmpvb, false, tmpv); @. tmpvb = tmpv
                axpy!(poi[u], tmpvb, barvb[k])
            end
            scal!(1/weight, barvb[k])

            # vb[k] = exp(T * tdat[k]) * vb[k-1]
            vb[k] = zeros(Tv, dim)
            @. tmpvb = vb[k-1]
            axpy!(poi[0], tmpvb, vb[k])
            for u = 1:right
                gemv!('N', 1.0, P, tmpvb, false, tmpv); @. tmpvb = tmpv
                axpy!(poi[u], tmpvb, vb[k])
            end
            scal!(1/weight, vb[k])

            # barvf[k] = barvf[k-1] * exp(T * tdat[k])
            barvf[k] = zeros(Tv, dim)
            @. tmpvf = barvf[k-1]
            axpy!(poi[0], tmpvf, barvf[k])
            for u = 1:right
                gemv!('T', 1.0, P, tmpvf, false, tmpv); @. tmpvf = tmpv
                axpy!(poi[u], tmpvf, barvf[k])
            end
            scal!(1/weight, barvf[k])

            # vf[k] = vf[k-1] * exp(T * tdat[k])
            vf[k] = zeros(Tv, dim)
            @. tmpvf = vf[k-1]
            axpy!(poi[0], tmpvf, vf[k])
            for u = 1:right
                gemv!('T', 1.0, P, tmpvf, false, tmpv); @. tmpvf = tmpv
                axpy!(poi[u], tmpvf, vf[k])
            end
            scal!(1/weight, vf[k])
        end
        # vb[k] = (-ph.T) * barvb[k]
        # begin
        #     vb[k] = similar(alpha)
        #     gemv!('N', -1.0, ph.T, barvb[k], false, vb[k])
        # end

        if data.nu[k] == 3 # left truncation time
            tmp = @dot(alpha, barvb[k])
            llf -= log(tmp)
            wb[k] = 1/tmp
            nn += wb[k] - 1.0
            axpy!(wb[k], one, eres.eb)
            axpy!(-wb[k], barvb[k], eres.eb)
            axpy!(wb[k], baralpha, eres.ey)
            axpy!(-wb[k], barvf[k], eres.ey)
        elseif data.nu[k] == 1 # right censoring time
            tmp = @dot(alpha, barvb[k])
            llf += log(tmp)
            wb[k] = 1/tmp
            nn += 1.0
            axpy!(wb[k], barvb[k], eres.eb)
            axpy!(wb[k], barvf[k], eres.ey)
        elseif data.nu[k] == 0 # observed faiure time
            tmp = @dot(alpha, vb[k])
            llf += log(tmp)
            wb[k] = 1/tmp
            nn += 1.0
            axpy!(wb[k], vb[k], eres.eb)
            # gemv!('T', -wb[k], ph.T, barvf[k], 1.0, eres.ey)
            axpy!(wb[k], vf[k], eres.ey)
        end
    end

    # compute vectors for convolution
    vc[m] = zero(tau)
    if data.nu[m] == 3
        axpy!(-wb[m], one, vc[m])
    elseif data.nu[m] == 1
        axpy!(wb[m], one, vc[m])
    elseif data.nu[m] == 0
        axpy!(wb[m], tau, vc[m])
    end
    @inbounds for k=m-1:-1:1
        # vc[k] = vc[k+1] * exp(T * tdat[k+1]) + ...
        begin
            right = rightbound(qv * data.tdat[k+1], eps) + 1
            weight = poipmf!(qv * data.tdat[k+1], poi, left=0, right=right)

            vc[k] = zeros(Tv, dim)
            @. tmpvb = vc[k+1]
            axpy!(poi[0], tmpvb, vc[k])
            for u = 1:right
                gemv!('N', 1.0, P, tmpvb, false, tmpv); @. tmpvb = tmpv
                axpy!(poi[u], tmpvb, vc[k])
            end
            scal!(1/weight, vc[k])
            if data.nu[k] == 3
                axpy!(-wb[k], one, vc[k])
            elseif data.nu[k] == 1
                axpy!(wb[k], one, vc[k])
            elseif data.nu[k] == 0
                axpy!(wb[k], tau, vc[k])
            end
        end
    end

    @inbounds for k=1:m
        # compute convolution integral
        #  int_0^tdat[k] exp(T* s) * vc[k] * vf[k-1] * exp(T(tdat[k]-s)) ds
        begin
            right = rightbound(qv * data.tdat[k], eps) + 1
            weight = poipmf!(qv * data.tdat[k], poi, left=0, right=right)

            @. vx[right] = zero(Tv)
            axpy!(poi[right], vf[k-1], vx[right])
            for l = right-1:-1:1
                gemv!('T', 1.0, P, vx[l+1], false, vx[l])
                axpy!(poi[l], vf[k-1], vx[l])
            end

            spger!(1.0/(qv*weight), vx[1], vc[k], 1.0, eres.en)
            for l = 1:right-1
                gemv!('N', 1.0, P, vc[k], false, tmpv); @. vc[k] = tmpv
                spger!(1.0/(qv*weight), vx[l+1], vc[k], 1.0, eres.en)
            end
        end
        if data.nu[k] == 3
            spger!(wb[k], baralpha, one, 1.0, eres.en)
            spger!(-wb[k], barvf[k], one, 1.0, eres.en)
        elseif data.nu[k] == 1
            spger!(wb[k], barvf[k], one, 1.0, eres.en)
        end
    end

    eres.etotal = nn
    @. eres.eb *= alpha
    @. eres.ey *= tau
    eres.ez = spdiag(eres.en)
    for i = 1:length(eres.en)
        eres.en[i] *= ph.T[i]
    end

    llf
end

