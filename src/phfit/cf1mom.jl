using PhaseTypeDistributions: CF1

"""
cf1mom_power
cf1mom_linear

These proved CF1 parameters fitted to a given first moment with a given number of phases.
- dim: The number of phases
- m1: The first moment to be fitted
- ratio: The fraction of the last stage transition rate over the first state gransition rate

The difference between power and linear is the increasing curves. The power draws the power curve,
and the linear draws the linear curve. These functions are used to determine the initial CF1 parameters
in the phfit.
"""

function cf1mom_power(dim::Ti, m1::Tv, ratio::Tv) where {Tv,Ti}
    rate = Vector{Tv}(undef, dim)
    p = exp(Tv(1)/(dim-1) * log(ratio))
    total = Tv(1)
    tmp = Tv(1)
    for i = 1:dim-1
        tmp *= (i+1) / (i * p)
        total += tmp
    end
    base = total / (dim * m1)
    tmp = base
    for i = 1:dim
        rate[i] = tmp
        tmp *= p
    end
    CF1(fill(1/dim, dim), rate)
end

function cf1mom_linear(dim::Ti, m1::Tv, ratio::Tv) where {Tv,Ti}
    rate = Vector{Tv}(undef, dim)
    al = (ratio - 1)/(dim-1)
    total = Tv(1)
    for i = 2:dim
        total += i/(al*(i-1)+1)
    end
    base = total / (dim * m1)
    for i = 1:dim
        rate[i] = base * (al * (i-1) + 1)
    end
    CF1(fill(1/dim, dim), rate)
end

