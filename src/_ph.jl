"""
Phase-Type Distributions
"""

export CF1, GPH

abstract type AbstractPHDistribution end

struct GPH{Tv,MatT} <: AbstractPHDistribution
    dim::Int
    alpha::Vector{Tv}
    T::MatT
    tau::Vector{Tv}
end

struct CF1{Tv} <: AbstractPHDistribution
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
    alpha, rate = _cf1_sort(alpha, rate)
    CF1(n, alpha, rate)
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
    return cf1.dim, alpha, SparseCOO(cf1.dim, cf1.dim, elem), tau
end

function GPH(cf1::CF1{Tv}) where {Tv}
    dim, alpha, T, tau = _togph(cf1)
    GPH(dim, alpha, SparseCSC(T), tau)
end

function GPH(cf1::CF1{Tv}, ::Type{Matrix{Tv}}) where {Tv}
    dim, alpha, T, tau = _togph(cf1)
    GPH(dim, alpha, Matrix(T), tau)
end

function GPH(cf1::CF1{Tv}, ::Type{SparseCSR{Tv,Ti}}) where {Ti,Tv}
    dim, alpha, T, tau = _togph(cf1)
    GPH(dim, alpha, SparseCSR(T), tau)
end

function GPH(cf1::CF1{Tv}, ::Type{SparseCSC{Tv,Ti}}) where {Ti,Tv}
    dim, alpha, T, tau = _togph(cf1)
    GPH(dim, alpha, SparseCSC(T), tau)
end

function GPH(cf1::CF1{Tv}, ::Type{SparseCOO{Tv,Ti}}) where {Ti,Tv}
    dim, alpha, T, tau = _togph(cf1)
    GPH(dim, alpha, T, tau)
end

function GPH(cf1::CF1{Tv}, ::Type{Matrix}) where {Tv}
    dim, alpha, T, tau = _togph(cf1)
    GPH(dim, alpha, Matrix(T), tau)
end

function GPH(cf1::CF1{Tv}, ::Type{SparseCSR}) where {Ti,Tv}
    dim, alpha, T, tau = _togph(cf1)
    GPH(dim, alpha, SparseCSR(T), tau)
end

function GPH(cf1::CF1{Tv}, ::Type{SparseCSC}) where {Ti,Tv}
    dim, alpha, T, tau = _togph(cf1)
    GPH(dim, alpha, SparseCSC(T), tau)
end

function GPH(cf1::CF1{Tv}, ::Type{SparseCOO}) where {Ti,Tv}
    dim, alpha, T, tau = _togph(cf1)
    GPH(dim, alpha, T, tau)
end

function _cf1_sort(alpha::Vector{Tv}, rate::Vector{Tv}) where {Tv}
    a = copy(alpha)
    b = copy(rate)
    _cf1_sort!(a, b)
    return a, b
end

function _cf1_sort!(a::Vector{Tv}, b::Vector{Tv}) where {Tv}
    for i = 1:length(a)-1
        if b[i] > b[i+1]
            _cf1_swap!(i, i+1, a, b)
            for j = i:-1:2
                if b[j-1] <= b[j]
                    break
                end
                _cf1_swap!(j-1, j, a, b)
            end
        end
    end
    nothing
end

function _cf1_swap!(i::Int, j::Int, alpha::Vector{Tv}, rate::Vector{Tv}) where {Tv}
    w = rate[j] / rate[i]
    alpha[i] += (Tv(1) - w) * alpha[j]
    alpha[j] *= w
    tmp = rate[j]
    rate[j] = rate[i]
    rate[i] = tmp
    nothing
end
