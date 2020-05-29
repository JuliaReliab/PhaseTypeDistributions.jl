"""
_phunif(ph, ufact)

Get a uniformed PH.
"""

function _phunif(ph::GPH{Tv,MatT}, ufact::Tv = 1.01) where {Tv,MatT}
    P, qv = unif(ph.T, ufact)
    return ph.dim, ph.alpha, P, ph.tau, qv
end

