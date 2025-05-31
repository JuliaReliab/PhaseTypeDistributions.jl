using Origin: @origin
using LinearAlgebra.BLAS: gemv!, scal!, axpy!
using SparseMatrix: spdiag, spger!
using NMarkov: itime, @dot, rightbound, poipmf!, unif

export
    TimeSpanSample,
    mean

function _promote_union_type(xs::AbstractVector)
    types = map(typeof, xs)
    T = reduce((a, b) -> Union{a, b}, types)
    convert(Vector{T}, xs)
end

struct TimeSpanSample{Tv} <: AbstractPHSample
    length::Int
    maxtime::Tv
    tdat::Vector{Tv}
    wdat::Vector{Tv}
    zdat::Vector{Int}
end

function TimeSpanSample(t::AbstractVector)
    ts = _promote_union_type(t)
    w = [1.0 for _ in t]
    createTimeSpanSample(ts, w)
end

function TimeSpanSample(t::AbstractVector, w::AbstractVector)
    ts = _promote_union_type(t)
    createTimeSpanSample(ts, w)
end

function createTimeSpanSample(t::Vector{Union{Tv,Tuple{Tv,Tv}}}, w::Vector{Tv}) where Tv
    expanded_values = Tv[]
    weights_values = Tv[]
    index_values = Int[]
    for (i, ti) in enumerate(t)
        if abs(w[i]) < 1e-12
            continue
        end
        if ti isa Tv
            push!(expanded_values, ti)
            push!(weights_values, w[i])
            push!(index_values, i)
        elseif ti isa Tuple{Tv, Tv}
            a, b = ti
            if isinf(b)
                push!(expanded_values, a)
                push!(weights_values, w[i])
                push!(index_values, -1)
            elseif a == 0
                push!(expanded_values, b)
                push!(weights_values, w[i])
                push!(index_values, 0)
            else
                push!(expanded_values, a)
                push!(expanded_values, b)
                push!(weights_values, w[i])
                push!(weights_values, w[i])
                push!(index_values, i)
                push!(index_values, i)
            end
        else
            error("Unsupported time sample type")
        end
    end

    m = length(expanded_values)
    ord = collect(1:m)
    for i in 1:m
        if index_values[i] == -1
            index_values[i] = m + 1
        end
    end
    ord = sortperm(expanded_values)
    inv_ord = zeros(Int, m)
    for (j, idx) in enumerate(ord)
        inv_ord[idx] = j
    end

    z = zeros(Int, m)
    prev_index = -2
    for i in 1:m
        idx = index_values[i]
        if idx == prev_index
            z[i-1] = inv_ord[i]
            z[i] = inv_ord[i-1]
        elseif idx == 0
            z[i] = 0
        elseif idx == m + 1
            z[i] = m + 1
        else
            z[i] = inv_ord[i]
        end
        prev_index = idx
    end

    reordered_values = [expanded_values[j] for j in ord]
    reordered_weights = [weights_values[j] for j in ord]
    reordered_z = [z[j] for j in ord]
    for i in length(reordered_values):-1:2
        reordered_values[i] -= reordered_values[i-1]
    end
    max_t = maximum(reordered_values)

    return TimeSpanSample(
        m,
        max_t,
        reordered_values,
        reordered_weights,
        reordered_z
    )
end

function mean(data::TimeSpanSample{Tv}) where Tv
    if data.length == 0
        return Tv(0)
    end
    total_weight = sum(data.wdat)
    if total_weight == 0
        return Tv(0)
    end
    ctime = cumsum(data.tdat)
    weighted_sum = sum(ctime[i] * data.wdat[i] for i in 1:data.length)
    return weighted_sum / total_weight
end

@origin (barvf=>0, vb=>0, barvb=>0, wb=>1, vc=>0, poi=>0, vx=>0) function estep!(
    ph::GPH{Tv,MatT},
    data::TimeSpanSample{Tv},
    eres::Estep{Tv,MatT};
    eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01)) where {Tv,MatT}

    dim, alpha, tau = ph.dim, ph.alpha, ph.tau
    P, qv = unif(ph.T, ufact)
    @assert isfinite(qv)
    baralpha = (-ph.T)' \ alpha
    one = ones(dim)

    m = data.length

    clear!(eres)
    right = rightbound(qv * data.maxtime, eps) + 1
    poi = Vector{Tv}(undef, right + 1)

    llf = Tv(0)
    nn = Tv(0)
    weight = Tv(0)
    tmpvf = Vector{Tv}(undef, dim)
    tmpvb = Vector{Tv}(undef, dim)
    tmpv = Vector{Tv}(undef, dim)

    barvf = Vector{Vector{Tv}}(undef, m+1)
    vb = Vector{Vector{Tv}}(undef, m+1)
    barvb = Vector{Vector{Tv}}(undef, m+1)
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
        if data.tdat[k] > 0.0
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
        else
            barvf[k] = copy(barvf[k-1])
            barvb[k] = copy(barvb[k-1])
            vb[k] = copy(vb[k-1])
        end

        if data.zdat[k] == k # observed time
            nn += data.wdat[k]
            tmp = @dot(alpha, vb[k])
            llf += data.wdat[k] * log(tmp)
            wb[k] = data.wdat[k] / tmp
            axpy!(wb[k], vb[k], eres.eb)
            gemv!('T', -wb[k], ph.T, barvf[k], 1.0, eres.ey)
        elseif data.zdat[k] == 0 # interval [0, t]
            nn += data.wdat[k]
            @. tmpv = one
            axpy!(-1.0, barvb[k], tmpv)
            tmp = @dot(alpha, tmpv)
            llf += data.wdat[k] * log(tmp)
            wb[k] = data.wdat[k] / tmp
            axpy!(wb[k], tmpv, eres.eb)
            spger!(wb[k], baralpha, tmpv, 1.0, eres.en)

            @. tmpv = baralpha
            axpy!(-1.0, barvf[k], tmpv)
            axpy!(wb[k], tmpv, eres.ey)
        elseif data.zdat[k] == m + 1 # interval [tdat[k], ∞)
            nn += data.wdat[k]
            tmp = @dot(alpha, barvb[k])
            llf += data.wdat[k] * log(tmp)
            wb[k] = data.wdat[k] / tmp
            axpy!(wb[k], barvb[k], eres.eb)
            axpy!(wb[k], barvf[k], eres.ey)
            spger!(wb[k], baralpha, barvb[k], 1.0, eres.en)
        elseif data.zdat[k] < k # interval [tdat_z, t]
            nn += data.wdat[k]
            @. tmpv = barvb[data.zdat[k]]
            axpy!(-1.0, barvb[k], tmpv)
            tmp = @dot(alpha, tmpv)
            llf += data.wdat[k] * log(tmp)
            wb[k] = data.wdat[k] / tmp
            wb[data.zdat[k]] = wb[k]
            axpy!(wb[k], tmpv, eres.eb)
            spger!(wb[k], baralpha, tmpv, 1.0, eres.en)

            @. tmpv = barvf[data.zdat[k]]
            axpy!(-1.0, barvf[k], tmpv)
            axpy!(wb[k], tmpv, eres.ey)
        end
    end

    # compute vectors for convolution
    vc[m] = zero(alpha)
    if data.zdat[m] < m
        axpy!(-wb[m], baralpha, vc[m])
    elseif data.zdat[m] > m
        axpy!(wb[m], baralpha, vc[m])
    else # data.zdat[m] == m
        axpy!(wb[m], alpha, vc[m])
    end
    @inbounds for k=m-1:-1:1
        if data.tdat[k+1] > 0.0
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
            end
        else
            @. vc[k] = vc[k+1]
        end
        if data.zdat[k] < k
            axpy!(-wb[k], baralpha, vc[k])
        elseif data.zdat[k] > k
            axpy!(wb[k], baralpha, vc[k])
        else # data.zdat[k] == k
            axpy!(wb[k], alpha, vc[k])
        end
    end

    @inbounds for k=1:m
        if data.tdat[k] > 0.0
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

