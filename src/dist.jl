using Origin: @origin
using LinearAlgebra.BLAS: gemv!, scal!, axpy!
using SparseMatrix: BlockCOO
using NMarkov: @dot, itime, unif, rightbound, poipmf!, poipmf

"""
phpdf
phcdf
phccdf

Compute the p.d.f. (phpdf), the c.d.f. (phcdf) and the complement c.d.f. (phccdf) of PH ddistribution
- ph: the object of PH distribution such as GPH and CF1
- t, ts: time or time series at which the p.d.f. is computed
- eps (optional): the error tolerace for uniformization. The default is 1.0e-8.
- ufact (optional): the uniformization factor. The default is 1.01
- Return value: a value or vector for the p.d.f., c.d.f., or c.c.d.f.
"""

function phpdf(ph::GPH{Tv}, t::TimeT; eps=Tv(1.0e-8), ufact=Tv(1.01)) where {TimeT,Tv}
    phcomp(t, ph.alpha, ph.T, ph.tau, eps, ufact)
end

function phccdf(ph::GPH{Tv}, t::TimeT; eps=Tv(1.0e-8), ufact=Tv(1.01)) where {TimeT,Tv}
    phcomp(t, ph.alpha, ph.T, ones(Tv, ph.dim), eps, ufact)
end

function phcdf(ph::GPH{Tv}, t::TimeT; eps=Tv(1.0e-8), ufact=Tv(1.01)) where {TimeT,Tv}
    alpha = zeros(Tv, ph.dim+1)
    tau = zeros(Tv, ph.dim+1)
    for i = eachindex(ph.alpha)
        alpha[i] = ph.alpha[i]
    end
    tau[ph.dim+1] = Tv(1)
    T = SparseCSC(BlockCOO(2, 2, [(1, 1, ph.T), (1, 2, reshape(ph.tau, ph.dim, 1)), (2, 2, ones(1,1))]))
    T.val[end] = 0.0
    phcomp(t, alpha, T, tau, eps, ufact)
end

function phpdf(ph::CF1{Tv}, t::TimeT, ::Type{MatT} = SparseMatrixCSC; eps=Tv(1.0e-8), ufact=Tv(1.01)) where {TimeT,Tv,MatT}
    phpdf(GPH(ph, MatT), t, eps=eps, ufact=ufact)
end

function phcdf(ph::CF1{Tv}, t::TimeT, ::Type{MatT} = SparseMatrixCSC; eps=Tv(1.0e-8), ufact=Tv(1.01)) where {TimeT,Tv,MatT}
    phcdf(GPH(ph, MatT), t, eps=eps, ufact=ufact)
end

function phccdf(ph::CF1{Tv}, t::TimeT, ::Type{MatT} = SparseMatrixCSC; eps=Tv(1.0e-8), ufact=Tv(1.01)) where {TimeT,Tv,MatT}
    phccdf(GPH(ph, MatT), t, eps=eps, ufact=ufact)
end

"""
phcomp

Compute
```math
\alpha \exp(T ts) \tau 
```
for time series `ts` with the uniformization. This routine is called from phpdf, phcdf and phccdf
"""

function phcomp(t::Tv, alpha::Vector{Tv}, T::AbstractMatrix{Tv}, tau::Vector{Tv}, eps::Tv, ufact::Tv) where {Tv}
    P, qv = unif(T, ufact)
    right = rightbound(qv*t, eps)
    weight, poi = poipmf(qv*t, right, left=0)

    y = zero(alpha)
    xtmp = copy(alpha)
    tmpv = similar(alpha)
    @origin (poi => 0) begin
        axpy!(poi[0], xtmp, y)
        for i = 1:right
            gemv!('T', 1.0, P, xtmp, false, tmpv)
            @. xtmp = tmpv
            axpy!(poi[i], xtmp, y)
        end
    end
    scal!(1/weight, y)
    @dot(y, tau)
end

function phcomp(ts::AbstractVector{Tv}, alpha::Vector{Tv}, T::AbstractMatrix{Tv}, tau::Vector{Tv}, eps::Tv, ufact::Tv) where {Tv}
    P, qv = unif(T, ufact)
    perm = sortperm(ts)
    dt, maxt = itime(ts[perm])
    right = rightbound(qv*maxt, eps)
    poi = Vector{Tv}(undef, right + 1)
    m = length(ts)
    result = Vector{Tv}(undef, m)
    xtmp = copy(alpha)
    y = similar(alpha)
    tmpv = similar(alpha)
    for k = eachindex(dt)
        right = rightbound(qv*dt[k], eps)
        weight = poipmf!(qv*dt[k], poi, left=0, right=right)
        @. y = zero(Tv)
        @origin (poi => 0) begin
            axpy!(poi[0], xtmp, y)
            for i = 1:right
                gemv!('T', 1.0, P, xtmp, false, tmpv); @. xtmp = tmpv
                axpy!(poi[i], xtmp, y)
            end
        end
        scal!(1/weight, y)
        result[perm[k]] = @dot(y, tau)
        @. xtmp = y
    end
    result
end

"""
phmean

Compute the n-th moment of PH distribution.
- ph: the object of PH distribution such as GPH and CF1
- n (optional): the n-th moment. The default is 1.
- Return value: the n-th moment
"""

function phmean(ph::GPH{Tv,MatT}, n::Int = 1) where {Tv,MatT}
    x = copy(ph.alpha)
    for k = 1:n
        x = (-ph.T)' \ x
    end
    factorial(n) * sum(x)
end

function phmean(ph::CF1{Tv}, n::Int = 1, ::Type{MatT} = SparseMatrixCSC) where {Tv,MatT}
    phmean(GPH(ph, MatT), n)
end
