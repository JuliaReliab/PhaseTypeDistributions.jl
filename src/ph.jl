"""
Phase-Type Distributions
"""

using SparseArrays: SparseMatrixCSC, nnz, sparse
using NMarkov.SparseMatrix: SparseCSR, SparseCSC, SparseCOO

function getbaralpha(T::AbstractMatrix, alpha::AbstractVector)
    (-T)' \ alpha
end

"""
AbstractPHDistribution

An abstract type for PH distributions.
The subtypes are GPH, CF1, etc.
"""

abstract type AbstractPHDistribution end

"""
GPH{Tv,MatT}

The model parameters for the general PH distribution (GPH). Tv is the type of elements. Usually this is Float64.
MatT is the type of matrix to express the infinitesimal generator T. MatT can take Matrix, SparseArrays.SparseMatrixCSC, SparseCSR, SparseCSC, SparseCOO.
- dim: the number of phases
- alpha: the initial probability vector
- T: the infinitesimal generator
- tau: the exit rate vector for the absorbing state

The p.d.f. of GPH is

```math
f(t) = \alpha \exp(T t) \tau
```
"""

mutable struct GPH{Tv,MatT} <: AbstractPHDistribution
    dim::Int
    alpha::Vector{Tv}
    T::MatT
    tau::Vector{Tv}
    baralpha::Vector{Tv}
end

function GPH(alpha::Vector{Tv}, T::MatT, tau::Vector{Tv}) where {Tv,MatT}
    m, n = size(T)
    @assert m == n && length(alpha) == length(tau) && length(alpha) == m
    baralpha = getbaralpha(T, alpha)
    GPH{Tv,MatT}(m, alpha, T, tau, baralpha)
end

function GPH(alpha::Vector{Tv}, T::MatT, tau::Vector{Tv}, baralpha::Vector{Tv}) where {Tv,MatT}
    m, n = size(T)
    @assert m == n && length(alpha) == length(tau) && length(alpha) == m && length(baralpha) == m
    GPH{Tv,MatT}(m, alpha, T, tau, baralpha)
end

function Base.copy(gph::GPH{Tv,MatT}) where {Tv,MatT}
    GPH{Tv,MatT}(gph.dim, copy(gph.alpha), copy(gph.T), copy(gph.tau), copy(gph.baralpha))
end

"""
BidiagonalPH(alpha, rate)

Create a GPH representation of the bidiagonal phase-type distribution.
- alpha: the initial probability vector
- rate: the stage transition rate vector
The output is a GPH representation of the bidiagonal phase-type distribution.
"""

function BidiagonalPH(alpha::Vector{Tv}, rate::Vector{Tv}) where Tv
    @assert length(alpha) == length(rate)
    dim = length(alpha)
    T = SparseMatrixCSC{Tv,Int}(dim, dim)
    tau = zeros(Tv, dim)
    for i = 1:dim
        T[i,i] = -rate[i]
        if i != dim
            T[i,i+1] = rate[i]
        else
            tau[i] = rate[i]
        end
    end
    GPH{Tv,SparseMatrixCSC{Tv,Int}}(dim, alpha, T, tau)
end

"""
CF1{Tv}

The model parameter for the canonical form 1 (CF1). Tv is the type of elements. Usually this is Float64.
- dim: the number of phases
- alpha: the initial probability vector
- rate: transition rates to the next state

CF1 has a special structure on the inifinitesimal generator. The phase transition does not have any cycle,
and the phase represents the stage such as Erlang distribution. Unlike Erlang distribution, the transition rate to next stage
is allowed to be any value (All the state transition rates of Erlang distributon are same). In addition, the start stage
is determined by the initial probability vector.

"""

mutable struct CF1{Tv} <: AbstractPHDistribution
    dim::Int
    alpha::Vector{Tv}
    rate::Vector{Tv}
end

function CF1(dim::Ti, ::Type{Tv} = Float64) where {Tv,Ti}
    CF1(dim, fill(1/dim, dim), fill(Tv(1), dim))
end

function CF1(alpha::Vector{Tv}, rate::Vector{Tv}) where Tv
    @assert length(alpha) == length(rate)
    n = length(alpha)
    alpha, rate = cf1sort(alpha, rate)
    CF1(n, alpha, rate)
end

function Base.copy(cf1::CF1{Tv}) where {Tv}
    CF1{Tv}(cf1.dim, copy(cf1.alpha), copy(cf1.rate))
end

function _togph(cf1::CF1{Tv}) where {Tv}
    alpha = copy(cf1.alpha)
    rate = cf1.rate
    tau = zero(alpha)
    elem = Vector{Tuple{Int,Int,Tv}}()
    for i = eachindex(rate)
        push!(elem, (i,i,-rate[i]))
        if i != cf1.dim
            push!(elem, (i,i+1,rate[i]))
        else
            tau[i] = rate[i]
        end
    end
    # make baralpha
    baralpha = similar(alpha)
    if rate[1] != 0.0
        baralpha[1] = alpha[1] / rate[1]
    else
        baralpha[1] = 0.0
    end
    for i = 2:cf1.dim
        if rate[i] != 0.0
            baralpha[i] = (alpha[i] + rate[i-1] * baralpha[i-1]) / rate[i]
        else
            baralpha[i] = 0.0
        end
    end
    return cf1.dim, alpha, SparseCOO(cf1.dim, cf1.dim, elem), tau, baralpha
end

function GPH(cf1::CF1{Tv}) where {Tv}
    dim, alpha, T, tau, baralpha = _togph(cf1)
    GPH(dim, alpha, sparse(T), tau, baralpha)
end

function GPH(cf1::CF1{Tv}, ::Type{Matrix{Tv}}) where {Tv}
    dim, alpha, T, tau, baralpha = _togph(cf1)
    GPH(dim, alpha, Matrix(T), tau, baralpha)
end

function GPH(cf1::CF1{Tv}, ::Type{SparseCSR{Tv,Ti}}) where {Ti,Tv}
    dim, alpha, T, tau, baralpha = _togph(cf1)
    GPH(dim, alpha, SparseCSR(T), tau, baralpha)
end

function GPH(cf1::CF1{Tv}, ::Type{SparseCSC{Tv,Ti}}) where {Ti,Tv}
    dim, alpha, T, tau, baralpha = _togph(cf1)
    GPH(dim, alpha, SparseCSC(T), tau, baralpha)
end

function GPH(cf1::CF1{Tv}, ::Type{SparseMatrixCSC{Tv,Ti}}) where {Ti,Tv}
    dim, alpha, T, tau, baralpha = _togph(cf1)
    GPH(dim, alpha, sparse(T), tau, baralpha)
end

function GPH(cf1::CF1{Tv}, ::Type{SparseCOO{Tv,Ti}}) where {Ti,Tv}
    dim, alpha, T, tau, baralpha = _togph(cf1)
    GPH(dim, alpha, T, tau, baralpha)
end

function GPH(cf1::CF1{Tv}, ::Type{Matrix}) where {Tv}
    dim, alpha, T, tau, baralpha = _togph(cf1)
    GPH(dim, alpha, Matrix(T), tau, baralpha)
end

function GPH(cf1::CF1{Tv}, ::Type{SparseCSR}) where {Tv}
    dim, alpha, T, tau, baralpha = _togph(cf1)
    GPH(dim, alpha, SparseCSR(T), tau, baralpha)
end

function GPH(cf1::CF1{Tv}, ::Type{SparseCSC}) where {Tv}
    dim, alpha, T, tau, baralpha = _togph(cf1)
    GPH(dim, alpha, SparseCSC(T), tau, baralpha)
end

function GPH(cf1::CF1{Tv}, ::Type{SparseMatrixCSC}) where {Tv}
    dim, alpha, T, tau, baralpha = _togph(cf1)
    GPH(dim, alpha, sparse(T), tau, baralpha)
end

function GPH(cf1::CF1{Tv}, ::Type{SparseCOO}) where {Tv}
    dim, alpha, T, tau, baralpha = _togph(cf1)
    GPH(dim, alpha, T, tau, baralpha)
end

"""
cf1sort
cf1sort!

Make the CF1 from bidiagonal acyclic PH (APH). The difference between the bidiagonal APH and the CF1 is
the stage transition rates are sorted or not. Then these functions provide the sorted state transition rates
and the corresponding initial probability vectors

- alpha: the initial probability vector
- rate: the stage transition rate vector

The output is the sorted state transition rates and the corresponding initial probability vectors.
In the case of cf1sort!, alpha and rate are changed directly.
"""

function cf1sort(alpha::Vector{Tv}, rate::Vector{Tv}) where {Tv}
    a = copy(alpha)
    b = copy(rate)
    cf1sort!(a, b)
    return a, b
end

function cf1sort!(a::Vector{Tv}, b::Vector{Tv}) where {Tv}
    for i = 1:length(a)-1
        if b[i] > b[i+1]
            cf1swap!(i, i+1, a, b)
            for j = i:-1:2
                if b[j-1] <= b[j]
                    break
                end
                cf1swap!(j-1, j, a, b)
            end
        end
    end
    nothing
end

function cf1swap!(i::Int, j::Int, alpha::Vector{Tv}, rate::Vector{Tv}) where {Tv}
    w = rate[j] / rate[i]
    alpha[i] += (Tv(1) - w) * alpha[j]
    alpha[j] *= w
    tmp = rate[j]
    rate[j] = rate[i]
    rate[i] = tmp
    nothing
end
