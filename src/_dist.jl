
export phcdf, phpdf, phccdf

function phpdf(ph::GPH{Tv}, t::TimeT; eps=Tv(1.0e-8), ufact=Tv(1.01)) where {TimeT,Tv}
    _phcalc(t, ph.alpha, ph.T, ph.tau, eps, ufact)
end

function phccdf(ph::GPH{Tv}, t::TimeT; eps=Tv(1.0e-8), ufact=Tv(1.01)) where {TimeT,Tv}
    _phcalc(t, ph.alpha, ph.T, ones(Tv, ph.dim), eps, ufact)
end

function phcdf(ph::GPH{Tv}, t::TimeT; eps=Tv(1.0e-8), ufact=Tv(1.01)) where {TimeT,Tv}
    alpha = zeros(Tv, ph.dim+1)
    tau = zeros(Tv, ph.dim+1)
    for i = eachindex(ph.alpha)
        alpha[i] = ph.alpha[i]
    end
    tau[ph.dim+1] = Tv(1)
    T = SparseCSC(BlockCOO(2, 2, [(1, 1, ph.T), (1, 2, reshape(ph.tau, ph.dim, 1)), (2, 2, zeros(1,1))]))
    _phcalc(t, alpha, T, tau, eps, ufact)
end

function phpdf(ph::CF1{Tv}, t::TimeT, ::Type{MatT} = SparseCSC; eps=Tv(1.0e-8), ufact=Tv(1.01)) where {TimeT,Tv,MatT}
    phpdf(GPH(ph, MatT), t, eps=eps, ufact=ufact)
end

function phcdf(ph::CF1{Tv}, t::TimeT, ::Type{MatT} = SparseCSC; eps=Tv(1.0e-8), ufact=Tv(1.01)) where {TimeT,Tv,MatT}
    phcdf(GPH(ph, MatT), t, eps=eps, ufact=ufact)
end

function phccdf(ph::CF1{Tv}, t::TimeT, ::Type{MatT} = SparseCSC; eps=Tv(1.0e-8), ufact=Tv(1.01)) where {TimeT,Tv,MatT}
    phccdf(GPH(ph, MatT), t, eps=eps, ufact=ufact)
end

function _phcalc(t::Tv, alpha::Vector{Tv}, T::AbstractMatrix{Tv}, tau::Vector{Tv}, eps::Tv, ufact::Tv) where {Tv}
    P, qv = unif(T, ufact)
    right = rightbound(qv*t, eps)
    weight, poi = poipmf(qv*t, right, left=0)
    tmpv = zero(alpha)
    unifstep!(Trans(), P, poi, (0, right), weight, copy(alpha), tmpv)
    return @dot(tmpv, tau)
end

function _phcalc(t::AbstractVector{Tv}, alpha::Vector{Tv}, T::AbstractMatrix{Tv}, tau::Vector{Tv}, eps::Tv, ufact::Tv) where {Tv}
    P, qv = unif(T, ufact)
    perm = sortperm(t)
    dt, maxt = itime(t[perm])
    right = rightbound(qv*maxt, eps)
    poi = Vector{Tv}(undef, right + 1)
    m = length(t)
    result = Vector{Tv}(undef, m)
    vf = alpha
    tmpv = similar(alpha)
    for k = eachindex(dt)
        right = rightbound(qv*dt[k], eps)
        weight = poipmf!(qv*dt[k], poi, left=0, right=right)
        tmpv .= Tv(0)
        unifstep!(Trans(), P, poi, (0, right), weight, copy(vf), tmpv)
        vf .= tmpv
        result[perm[k]] = @dot(vf, tau)
    end
    return result
end
