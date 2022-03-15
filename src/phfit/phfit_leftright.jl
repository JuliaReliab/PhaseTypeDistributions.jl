using Origin: @origin
using SparseMatrix: spdiag, scal!, axpy!, gemv!, spger!
using NMarkov: itime, @dot, rightbound, poipmf!, unif

export
    LeftTruncRightCensoredSample,
    mean

struct LeftTruncRightCensoredSample <: AbstractPHSample
    length::Int
    maxtime::Float64
    tdat::Vector{Float64}
    nu::Vector{Bool}
    delta::Vector{Bool}
end

function LeftTruncRightCensoredSample(t::Vector{Float64}, tau::Vector{Float64}, delta::Vector{Bool})
    d = [(t, true, c) for (t,c) = zip(t, delta)]
    append!(d, [(t, false, true) for t = tau])
    sort!(d, by = x -> x[1])
    s = [d[1][1], diff([x[1] for x = d])...]
    nu = [x[2] for x = d]
    delta = [x[3] for x = d]
    LeftTruncRightCensoredSample(length(s), maximum(s), s, nu, delta)
end

function mean(data::LeftTruncRightCensoredSample)
    totalt = 0.0
    totaln = 0.0
    ct = 0.0
    for i = eachindex(data.tdat)
        ct += data.tdat[i]
        if data.nu[i] == 1
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

@origin (barvf=>0, vb=>0, barvb=>0, wk=>1, wb=>1, vc=>0, poi=>0, vx=>0) function estep!(
    ph::GPH{Tv,MatT},
    data::LeftTruncRightCensoredSample,
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
    nn = Tv(0)
    tmpvf = Vector{Tv}(undef, dim)
    tmpvb = Vector{Tv}(undef, dim)
    tmpv = Vector{Tv}(undef, dim)

    barvf = Vector{Vector{Tv}}(undef, m+1)
    vb = Vector{Vector{Tv}}(undef, m+1)
    barvb = Vector{Vector{Tv}}(undef, m+1)
    wk = Vector{Tv}(undef, m+1)
    wb = Vector{Tv}(undef, m+1)
    vc = Vector{Vector{Tv}}(undef, m+1)
    vx = Vector{Vector{Tv}}(undef, right + 1)
    for i = 0:right
        vx[i] = zeros(Tv,dim)
    end

    barvf[0] = baralpha
    barvb[0] = one
    vb[0] = tau
    
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

        # wb[k] = 1/@dot(alpha, barvb[k])
        # wk[k] = 1/@dot(alpha, vb[k])

        if data.nu[k] == 0
            tmp = @dot(alpha, barvb[k])
            llf -= log(tmp)
            wk[k] = 0.0
            wb[k] = 1/tmp
            nn += wb[k]
            axpy!(wb[k], one, eres.eb)
            axpy!(-wb[k], barvb[k], eres.eb)
            axpy!(wb[k], baralpha, eres.ey)
            axpy!(-wb[k], barvf[k], eres.ey)
        elseif data.nu[k] == 1 && data.delta[k] == 0
            tmp = @dot(alpha, barvb[k])
            llf += log(tmp)
            wk[k] = 0.0
            wb[k] = 1/tmp
            axpy!(wb[k], barvb[k], eres.eb)
            axpy!(wb[k], barvf[k], eres.ey)
        elseif data.nu[k] == 1 && data.delta[k] == 1
            tmp = @dot(alpha, vb[k])
            llf += log(tmp)
            wk[k] = 1/tmp
            wb[k] = 0.0
            axpy!(wk[k], vb[k], eres.eb)
            gemv!('T', -wk[k], ph.T, barvf[k], 1.0, eres.ey)
        end
    end

    # compute vectors for convolution
    vc[m] = zero(alpha)
    if data.nu[m] == 0
        axpy!(-wb[m], baralpha, vc[m])
    elseif data.nu[m] == 1 && data.delta[m] == 0
        axpy!(wb[m], baralpha, vc[m])
    elseif data.nu[m] == 1 && data.delta[m]== 1
        axpy!(wk[m], alpha, vc[m])
    end
    @inbounds for k=m-1:-1:1
        # vc[k] = vc[k+1] * exp(T * tdat[k+1]) + ...
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
            if data.nu[k] == 0
                axpy!(-wb[k], baralpha, vc[k])
            elseif data.nu[k] == 1 && data.delta[k] == 0
                axpy!(wb[k], baralpha, vc[k])
            elseif data.nu[k] == 1 && data.delta[k]== 1
                axpy!(wk[k], alpha, vc[k])
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
        if data.nu[k] == 0
            spger!(wb[k], baralpha, one, 1.0, eres.en)
            spger!(-wb[k], baralpha, barvb[k], 1.0, eres.en)
        elseif data.nu[k] == 1 && data.delta[k] == 0
            spger!(wb[k], baralpha, barvb[k], 1.0, eres.en)
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
