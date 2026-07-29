using ZeroOrigin: @origin
using LinearAlgebra.BLAS: gemv!, scal!, axpy!
using NMarkov.SparseMatrix: spdiag, spger!
using NMarkov: itime, @dot, rightbound, poipmf!, unif
using Random
using Distributions: Multinomial

function _promote_union_type(xs::AbstractVector)
    types = map(typeof, xs)
    T = reduce((a, b) -> Union{a, b}, types)
    convert(Vector{T}, xs)
end

function _scalar_eltype(ts::AbstractVector)
    isempty(ts) && return Float64
    x = first(ts)
    return x isa Tuple ? typeof(x[1]) : typeof(x)
end

struct TimeSpanSample{Tv} <: AbstractPHSample
    # Processed fields — used by E-step
    length::Int
    maxtime::Tv
    tdat::Vector{Tv}
    ndat::Vector{Tv}   # n_i: frequency/count (bootstrap target)
    wdat::Vector{Tv}   # w_i: analytic weight (quadrature/importance, fixed)
    zdat::Vector{Int}
    # Raw fields — used by bootstrap to reconstruct original observations
    rawt::Vector
    rawn::Vector{Int}
    raww::Vector{Tv}
    rawtau::Vector{Tv}   # left truncation time per observation (0 = not truncated)
end

function TimeSpanSample(t::AbstractVector; tau=nothing)
    ts = _promote_union_type(t)
    Tv = _scalar_eltype(ts)
    n = ones(Int, length(t))
    w = ones(Tv, length(t))
    createTimeSpanSample(ts, n, w, _totau(Tv, tau, length(t)))
end

function TimeSpanSample(t::AbstractVector, n::AbstractVector; tau=nothing)
    ts = _promote_union_type(t)
    Tv = _scalar_eltype(ts)
    w = ones(Tv, length(t))
    createTimeSpanSample(ts, Int.(n), w, _totau(Tv, tau, length(t)))
end

function TimeSpanSample(t::AbstractVector, n::AbstractVector, w::AbstractVector; tau=nothing)
    ts = _promote_union_type(t)
    Tv = _scalar_eltype(ts)
    createTimeSpanSample(ts, Int.(n), Tv.(w), _totau(Tv, tau, length(t)))
end

_totau(::Type{Tv}, ::Nothing, m::Int) where Tv = zeros(Tv, m)

function _totau(::Type{Tv}, tau::AbstractVector, m::Int) where Tv
    length(tau) == m || throw(ArgumentError("tau must have the same length as t ($(length(tau)) != $m)"))
    Tv.(tau)
end

function TimeSpanSample(data::WeightedSample{Tv}) where Tv
    t = cumsum(data.tdat)
    n = ones(Int, data.length)
    w = copy(data.wdat)
    createTimeSpanSample(t, n, w)
end

function TimeSpanSample(data::GroupTruncSample)
    Tv = Float64
    cumtimes = cumsum(data.tdat)
    ts = Union{Tv, Tuple{Tv, Tv}}[]
    ns = Int[]
    prev_t = Tv(0)
    for k in 1:data.length
        t_k = cumtimes[k]
        if data.gdat[k] >= 0 && data.tdat[k] != 0.0
            push!(ts, (prev_t, t_k))
            push!(ns, data.gdat[k])
        end
        if data.idat[k]
            push!(ts, t_k)
            push!(ns, 1)
        end
        prev_t = t_k
    end
    if data.gdatlast >= 0
        push!(ts, (prev_t, Tv(Inf)))
        push!(ns, data.gdatlast)
    end
    w = ones(Tv, length(ts))
    createTimeSpanSample(ts, ns, w)
end

_lowertime(ti) = ti isa Tuple ? ti[1] : ti

function createTimeSpanSample(t::Vector{<:Union{Tv,Tuple{Tv,Tv}}}, n::Vector{Int}, v::Vector{Tv},
                              tau::Vector{Tv} = zeros(Tv, length(t))) where Tv
    expanded_values = Tv[]
    n_values = Int[]
    v_values = Tv[]
    # Kind of each expanded entry. The two entries of a finite interval are pushed
    # consecutively as :span and are the only ones that get paired below; every other
    # kind is self-contained. Using an explicit kind instead of overloading the raw
    # observation index keeps two adjacent observations of the same kind (e.g. two
    # right-censored, or two [0,b] intervals) from being mistaken for one interval.
    kinds = Symbol[]
    for (i, ti) in enumerate(t)
        if n[i] == 0
            continue
        end
        if ti isa Tv
            push!(expanded_values, ti)
            push!(n_values, n[i])
            push!(v_values, v[i])
            push!(kinds, :exact)
        elseif ti isa Tuple{Tv, Tv}
            a, b = ti
            if isinf(b)
                push!(expanded_values, a)
                push!(n_values, n[i])
                push!(v_values, v[i])
                push!(kinds, :censored)
            elseif a == 0
                push!(expanded_values, b)
                push!(n_values, n[i])
                push!(v_values, v[i])
                push!(kinds, :fromzero)
            else
                push!(expanded_values, a)
                push!(expanded_values, b)
                push!(n_values, n[i])
                push!(n_values, n[i])
                push!(v_values, v[i])
                push!(v_values, v[i])
                push!(kinds, :span)
                push!(kinds, :span)
            end
        else
            error("Unsupported time sample type")
        end
    end

    # Left truncation times are appended after every ordinary observation so that
    # they never sit between the two entries of an interval.
    for (i, taui) in enumerate(tau)
        (n[i] == 0 || taui <= 0) && continue
        lo = _lowertime(t[i])
        taui <= lo || throw(ArgumentError(
            "left truncation time tau[$i]=$taui must not exceed the observation time $lo"))
        push!(expanded_values, taui)
        push!(n_values, n[i])
        push!(v_values, v[i])
        push!(kinds, :trunc)
    end

    m = length(expanded_values)
    ord = sortperm(expanded_values)
    inv_ord = zeros(Int, m)
    for (j, idx) in enumerate(ord)
        inv_ord[idx] = j
    end

    z = zeros(Int, m)
    pending_span = 0  # index of an unpaired :span lower bound, 0 if none
    for i in 1:m
        k = kinds[i]
        if k == :span
            if pending_span != 0
                z[pending_span] = inv_ord[i]
                z[i] = inv_ord[pending_span]
                pending_span = 0
            else
                pending_span = i
            end
        elseif k == :fromzero
            z[i] = 0
        elseif k == :censored
            z[i] = m + 1
        elseif k == :trunc
            z[i] = -1
        else # :exact
            z[i] = inv_ord[i]
        end
    end

    reordered_values = [expanded_values[j] for j in ord]
    reordered_n = Tv[n_values[j] for j in ord]
    reordered_v = [v_values[j] for j in ord]
    reordered_z = [z[j] for j in ord]
    for i in length(reordered_values):-1:2
        reordered_values[i] -= reordered_values[i-1]
    end
    max_t = maximum(reordered_values)

    return TimeSpanSample(
        m,
        max_t,
        reordered_values,
        reordered_n,
        reordered_v,
        reordered_z,
        t,
        n,
        v,
        tau
    )
end

function mean(data::TimeSpanSample{Tv}) where Tv
    if data.length == 0
        return Tv(0)
    end
    # Left truncation entries are not observations; they must not enter the mean.
    obs = [i for i in 1:data.length if data.zdat[i] != -1]
    total_weight = sum(data.ndat[i] * data.wdat[i] for i in obs; init=Tv(0))
    if total_weight == 0
        return Tv(0)
    end
    ctime = cumsum(data.tdat)
    weighted_sum = sum(ctime[i] * data.ndat[i] * data.wdat[i] for i in obs; init=Tv(0))
    return weighted_sum / total_weight
end

@origin (barvf=>0, vb=>0, barvb=>0, wb=>1, vc=>0, poi=>0, vx=>0) function estep!(
    ph::GPH{Tv,MatT},
    data::TimeSpanSample{Tv},
    eres::Estep{Tv,MatT};
    eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01)) where {Tv,MatT}

    dim, alpha, tau, baralpha = ph.dim, ph.alpha, ph.tau, ph.baralpha
    P, qv = unif(ph.T, ufact)
    @assert isfinite(qv)
    one = ones(Tv, dim)

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
        vx[i] = zeros(Tv, dim)
    end
    for i = 0:m
        barvf[i] = zeros(Tv, dim)
        barvb[i] = zeros(Tv, dim)
        vb[i] = zeros(Tv, dim)
        vc[i] = zeros(Tv, dim)
    end

    copyto!(barvf[0], baralpha)
    copyto!(barvb[0], one)
    copyto!(vb[0], tau)

    @inbounds for k = 1:m
        if data.tdat[k] > 0.0
            # barvf[k] = barvf[k-1] * exp(T * tdat[k])
            # barvb[k] = exp(T * tdat[k]) * barvb[k-1]
            begin
                right = rightbound(qv * data.tdat[k], eps) + 1
                weight = poipmf!(qv * data.tdat[k], poi, left=0, right=right)

                fill!(barvf[k], zero(Tv))
                fill!(barvb[k], zero(Tv))
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
                gemv!('N', -1.0, ph.T, barvb[k], false, vb[k])
            end
        else
            barvf[k] .= barvf[k-1]
            barvb[k] .= barvb[k-1]
            vb[k] .= vb[k-1]
        end

        if data.zdat[k] == -1 # left truncation at tdat[k]
            # The contribution of -log S(t_k) is the unconditional expectation minus
            # the expectation of a right-censored observation at t_k, both scaled by
            # wb = nw / S(t_k). This gives exactly the same vector structure as the
            # interval [0, t] branch below, with F(t) replaced by S(t) in wb.
            nw = data.ndat[k] * data.wdat[k]
            tmp = @dot(alpha, barvb[k])
            llf -= nw * log(tmp)
            wb[k] = nw / tmp
            nn += wb[k] - nw

            @. tmpv = one
            axpy!(-1.0, barvb[k], tmpv)
            axpy!(wb[k], tmpv, eres.eb)
            spger!(wb[k], baralpha, tmpv, 1.0, eres.en)

            @. tmpv = baralpha
            axpy!(-1.0, barvf[k], tmpv)
            axpy!(wb[k], tmpv, eres.ey)
        elseif data.zdat[k] == k # observed time
            nw = data.ndat[k] * data.wdat[k]
            nn += nw
            tmp = @dot(alpha, vb[k])
            llf += nw * log(tmp)
            wb[k] = nw / tmp
            axpy!(wb[k], vb[k], eres.eb)
            gemv!('T', -wb[k], ph.T, barvf[k], 1.0, eres.ey)
        elseif data.zdat[k] == 0 # interval [0, t]
            nw = data.ndat[k] * data.wdat[k]
            nn += nw
            @. tmpv = one
            axpy!(-1.0, barvb[k], tmpv)
            tmp = @dot(alpha, tmpv)
            llf += nw * log(tmp)
            wb[k] = nw / tmp
            axpy!(wb[k], tmpv, eres.eb)
            spger!(wb[k], baralpha, tmpv, 1.0, eres.en)

            @. tmpv = baralpha
            axpy!(-1.0, barvf[k], tmpv)
            axpy!(wb[k], tmpv, eres.ey)
        elseif data.zdat[k] == m + 1 # interval [tdat[k], ∞)
            nw = data.ndat[k] * data.wdat[k]
            nn += nw
            tmp = @dot(alpha, barvb[k])
            llf += nw * log(tmp)
            wb[k] = nw / tmp
            axpy!(wb[k], barvb[k], eres.eb)
            axpy!(wb[k], barvf[k], eres.ey)
            spger!(wb[k], baralpha, barvb[k], 1.0, eres.en)
        elseif data.zdat[k] < k # interval [tdat_z, t]
            nw = data.ndat[k] * data.wdat[k]
            nn += nw
            @. tmpv = barvb[data.zdat[k]]
            axpy!(-1.0, barvb[k], tmpv)
            tmp = @dot(alpha, tmpv)
            llf += nw * log(tmp)
            wb[k] = nw / tmp
            wb[data.zdat[k]] = wb[k]
            axpy!(wb[k], tmpv, eres.eb)
            spger!(wb[k], baralpha, tmpv, 1.0, eres.en)

            @. tmpv = barvf[data.zdat[k]]
            axpy!(-1.0, barvf[k], tmpv)
            axpy!(wb[k], tmpv, eres.ey)
        end
    end

    # compute vectors for convolution
    fill!(vc[m], zero(Tv))
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
                fill!(vc[k], zero(Tv))
                @. tmpvf = vc[k+1]
                axpy!(poi[0], tmpvf, vc[k])
                for u = 1:right
                    gemv!('T', 1.0, P, tmpvf, false, tmpv); @. tmpvf = tmpv
                    axpy!(poi[u], tmpvf, vc[k])
                end
                scal!(1/weight, vc[k])
            end
        else
            vc[k] .= vc[k+1]
        end
        if data.zdat[k] < k
            # Covers both the upper end of an interval and left truncation (zdat == -1),
            # which contribute -baralpha to the forward source either way.
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

function phllf(cf1::CF1{Tv}, data::TimeSpanSample{Tv}, ::Type{MatT}=SparseMatrixCSC;
    eps::Tv=Tv(1.0e-8), ufact::Tv=Tv(1.01)) where {Tv,MatT}
    eres = Estep(GPH(cf1, MatT))
    estep!(cf1, data, eres, eps=eps, ufact=ufact)
end

function bootstrap(rng::AbstractRNG, data::TimeSpanSample{Tv}) where Tv
    N = sum(data.rawn)
    n_new = rand(rng, Multinomial(N, data.rawn ./ N))
    TimeSpanSample(data.rawt, n_new, data.raww; tau=data.rawtau)
end

bootstrap(data::TimeSpanSample) = bootstrap(Random.default_rng(), data)

function eic(rng::AbstractRNG, ph0::CF1{Tv}, data::TimeSpanSample{Tv};
    bsample::Int = 100,
    maxiter::Int = 5000,
    steps::Int = 10,
    abstol::Tv = Tv(1.0e-3),
    reltol::Tv = Tv(1.0e-5)
) where Tv
    llf0 = phllf(ph0, data)
    d0 = TimeSpanSample(data.rawt, data.rawn, data.raww; tau=data.rawtau)
    d1 = [bootstrap(rng, data) for _ in 1:bsample]
    bias = Vector{Union{Tv,Nothing}}(undef, bsample)
    Threads.@threads for i in 1:bsample
        try
            res = phfit(ph0, d1[i]; initialize=false, maxiter=maxiter, steps=steps,
                        abstol=abstol, reltol=reltol, progress=false, progress_init=false)
            ph1 = res.model
            b1 = res.llf
            b2 = phllf(ph0, d1[i])
            b3 = llf0
            b4 = phllf(ph1, d0)
            bias[i] = b1 - b2 + b3 - b4
        catch e
            @warn "eic: bootstrap fit failed" i=i exception=(e, catch_backtrace())
            bias[i] = nothing
        end
    end
    v = Tv[x for x in bias if x !== nothing]
    n = length(v)
    mu = sum(v) / n
    sigma = sqrt(sum((x - mu)^2 for x in v) / (n - 1))
    se = sigma / sqrt(n)
    return (
        eic       = -2 * (llf0 - mu),
        ci_lower  = -2 * (llf0 - (mu - 1.96 * se)),
        ci_upper  = -2 * (llf0 - (mu + 1.96 * se)),
        nvalid    = n,
    )
end

eic(ph0::CF1{Tv}, data::TimeSpanSample{Tv}; kwargs...) where Tv = eic(Random.default_rng(), ph0, data; kwargs...)
