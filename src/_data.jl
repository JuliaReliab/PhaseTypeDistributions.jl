"""
Data structure
"""

export WeightedSample, PointSample

abstract type AbstractPHSample end

struct WeightedSample{Tv} <: AbstractPHSample
    length::Int
    maxtime::Tv
    tdat::Vector{Tv}
    wdat::Vector{Tv}
end

function _phmean(data::WeightedSample{Tv}) where Tv
    totalt = Tv(0)
    totalw = Tv(0)
    ct = Tv(0)
    for i = eachindex(data.tdat)
        ct += data.tdat[i]
        totalt += data.wdat[i] * ct
        totalw += data.wdat[i]
    end
    return totalt / totalw
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

function WeightedSample(f::Any, bounds::Tuple{Tv,Tv}) where Tv
    de = deint(f, bounds[1], bounds[2])
    WeightedSample(de.x, de.w * de.h)
end

