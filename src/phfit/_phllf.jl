"""
phllf
"""

function phllf(cf1::CF1{Tv},
    data::AbstractPHSample, ::Type{MatT} = SparseMatrixCSC; eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01)) where {Tv,MatT}
    phllf(GPH(cf1, MatT), data, eps=eps, ufact=ufact)
end

function phllf(cf1::CF1{Tv}, cf1deriv::Dict{Symbol,CF1{Tv}},
    data::AbstractPHSample, ::Type{MatT} = SparseMatrixCSC; eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01)) where {Tv,MatT}
    gph = GPH(cf1, MatT)
    deriv = Dict((k, GPH(v, MatT)) for (k,v) in cf1deriv)
    phllf(gph, deriv, data, eps=eps, ufact=ufact)
end

function phllf(cf1::CF1{Tv}, cf1deriv::Dict{Symbol,CF1{Tv}}, cf1deriv2::Dict{Tuple{Symbol,Symbol},CF1{Tv}},
    data::AbstractPHSample, ::Type{MatT} = SparseMatrixCSC; eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01)) where {Tv,MatT}
    gph = GPH(cf1, MatT)
    deriv = Dict((k, GPH(v, MatT)) for (k,v) in cf1deriv)
    deriv2 = Dict((k, GPH(v, MatT)) for (k,v) in cf1deriv2)
    phllf(gph, deriv, deriv2, data, eps=eps, ufact=ufact)
end

function phllf(ph::GPH{Tv,MatT},
    data::WeightedSample{Tv};
    eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01)) where {Tv,MatT}
    dim, alpha, P, tau, qv = _phunif(ph, ufact)
    llf = Tv(0)

    right = rightbound(qv * data.maxtime, eps) + 1
    prob = Vector{Tv}(undef, right + 1)
    f0 = copy(alpha)
    f1 = similar(alpha)

    for k = eachindex(data.tdat)
        f1 .= Tv(0)
        unifforward!(f0, data.tdat[k], P, qv, prob, eps, f1)
        llf += data.wdat[k] * log(@dot(f1, tau))
        f0 .= f1
    end
    return llf
end

function phllf(ph::GPH{Tv,MatT}, phderiv::Dict{Symbol,GPH{Tv,MatT}},
    data::WeightedSample{Tv};
    eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01)) where {Tv,MatT}
    dim, alpha, P, tau, qv = _phunif(ph, ufact)
    Pdash = Dict((k,x.T/qv) for (k,x) in phderiv)
    llf = Tv(0)
    llfdash = Dict((k,Tv(0)) for (k,x) in phderiv)

    right = rightbound(qv * data.maxtime, eps) + 1
    prob = Vector{Tv}(undef, right + 1)
    f0 = copy(alpha)
    f1 = similar(alpha)
    f0dash = Dict((k,copy(x.alpha)) for (k,x) in phderiv)
    f1dash = Dict((k,similar(x.alpha)) for (k,x) in phderiv)

    for k = eachindex(data.tdat)
        f1 .= Tv(0)
        for j = eachindex(phderiv)
            f1dash[j] .= Tv(0)
        end
        unifforward!(f0, f0dash, data.tdat[k], P, Pdash, qv, prob, eps, f1, f1dash)
        tmp = @dot(f1, tau)
        llf += data.wdat[k] * log(tmp)
        for i = eachindex(phderiv)
            llfdash[i] += data.wdat[k] * (@dot(f1dash[i], tau) + @dot(f1, phderiv[i].tau)) / tmp
        end
        f0 .= f1
        for j = eachindex(phderiv)
            f0dash[j] .= f1dash[j]
        end
    end
    return llf, llfdash
end

function phllf(ph::GPH{Tv,MatT}, phderiv::Dict{Symbol,GPH{Tv,MatT}}, phderiv2::Dict{Tuple{Symbol,Symbol},GPH{Tv,MatT}},
    data::WeightedSample{Tv};
    eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01)) where {Tv,MatT}
    dim, alpha, P, tau, qv = _phunif(ph, ufact)
    Pdash = Dict((k,x.T/qv) for (k,x) in phderiv)
    Pdashdash = Dict((k,x.T/qv) for (k,x) in phderiv2)
    llf = Tv(0)
    llfdash = Dict((k,Tv(0)) for (k,x) in phderiv)
    tmp1 = Dict((k,Tv(0)) for (k,x) in phderiv)
    llfdashdash = Dict((k,Tv(0)) for (k,x) in phderiv2)

    right = rightbound(qv * data.maxtime, eps) + 1
    prob = Vector{Tv}(undef, right + 1)
    f0 = copy(alpha)
    f1 = similar(alpha)
    f0dash = Dict((k,copy(x.alpha)) for (k,x) in phderiv)
    f1dash = Dict((k,similar(x.alpha)) for (k,x) in phderiv)
    f0dashdash = Dict((k,copy(x.alpha)) for (k,x) in phderiv2)
    f1dashdash = Dict((k,similar(x.alpha)) for (k,x) in phderiv2)

    for k = eachindex(data.tdat)
        f1 .= Tv(0)
        for i = eachindex(f1dash)
            f1dash[i] .= Tv(0)
        end
        for i = eachindex(f1dashdash)
            f1dashdash[i] .= Tv(0)
        end
        unifforward!(f0, f0dash, f0dashdash, data.tdat[k],
            P, Pdash, Pdashdash, qv, prob, eps, f1, f1dash, f1dashdash)

        tmp = @dot(f1, tau)
        llf += data.wdat[k] * log(tmp)
        for i = eachindex(f1dash)
            tmp1[i] = (@dot(f1dash[i], tau) + @dot(f1, phderiv[i].tau)) / tmp
            llfdash[i] += data.wdat[k] * tmp1[i]
        end
        for i = eachindex(f1dashdash)
            k1 = i[1]
            k2 = i[2]
            llfdashdash[i] += data.wdat[k] * ((@dot(f1dashdash[i], tau) + @dot(f1dash[k1], phderiv[k2].tau) +
                            @dot(f1dash[k2], phderiv[k1].tau) + @dot(f1, phderiv2[i].tau)) / tmp - tmp1[k1] * tmp1[k2])
        end
        f0 .= f1
        for i = eachindex(f1dash)
            f0dash[i] .= f1dash[i]
        end
        for i = eachindex(f1dashdash)
            f0dashdash[i] .= f1dashdash[i]
        end
    end
    return llf, llfdash, llfdashdash
end

"""
Uniformization
"""

@origin prob => left function unifforward!(f0::Vector{Tv}, t::Tv,
    P::AbstractMatrix{Tv}, qv::Tv,
    prob::Vector{Tv}, eps::Tv, f1::Vector{Tv}) where Tv
    left = 0
    right = rightbound(qv*t, eps)
    weight = poipmf!(qv*t, prob, left=left, right=right)
    f1 .= prob[left] * f0
    for u = left+1:right
        f0 .= P' * f0
        axpy!(prob[u], f0, f1)
    end
    scal!(1/weight, f1)
    nothing
end

@origin prob => left function unifforward!(
    f0::Vector{Tv}, f0dash::Dict{Symbol,Vector{Tv}}, t::Tv,
    P::AbstractMatrix{Tv}, Pdash::Dict{Symbol,<:AbstractMatrix{Tv}}, qv::Tv,
    prob::Vector{Tv}, eps::Tv,
    f1::Vector{Tv}, f1dash::Dict{Symbol,Vector{Tv}}) where Tv
    left = 0
    right = rightbound(qv*t, eps)
    weight = poipmf!(qv*t, prob, left=left, right=right)
    f1 .= prob[left] * f0
    for k = eachindex(f1dash)
        f1dash[k] .= prob[left] * f0dash[k]
    end
    for u = left+1:right
        for k = eachindex(f0dash)
            f0dash[k] .= P' * f0dash[k] + Pdash[k]' * f0
        end
        f0 .= P' * f0
        axpy!(prob[u], f0, f1)
        for k = eachindex(f0dash)
            axpy!(prob[u], f0dash[k], f1dash[k])
        end
    end
    scal!(1/weight, f1)
    for k = eachindex(f1dash)
        scal!(1/weight, f1dash[k])
    end
    nothing
end

@origin prob => left function unifforward!(
    f0::Vector{Tv}, f0dash::Dict{Symbol,Vector{Tv}}, f0dashdash::Dict{Tuple{Symbol,Symbol},Vector{Tv}}, t::Tv,
    P::AbstractMatrix{Tv}, Pdash::Dict{Symbol,<:AbstractMatrix{Tv}}, Pdashdash::Dict{Tuple{Symbol,Symbol},<:AbstractMatrix{Tv}}, qv::Tv,
    prob::Vector{Tv}, eps::Tv,
    f1::Vector{Tv}, f1dash::Dict{Symbol,Vector{Tv}}, f1dashdash::Dict{Tuple{Symbol,Symbol},Vector{Tv}}) where Tv
    left = 0
    right = rightbound(qv*t, eps)
    weight = poipmf!(qv*t, prob, left=left, right=right)
    f1 .= prob[left] * f0
    for k = eachindex(f0dash)
        f1dash[k] .= prob[left] * f0dash[k]
    end
    for k = eachindex(f0dashdash)
        f1dashdash[k] .= prob[left] * f0dashdash[k]
    end
    for u = left+1:right
        for k = eachindex(f0dashdash)
            k1 = k[1]
            k2 = k[2]
            f0dashdash[k] .= P' * f0dashdash[k] + Pdash[k2]' * f0dash[k1] + Pdash[k1]' * f0dash[k2] + Pdashdash[k]' * f0
        end
        for k = eachindex(f0dash)
            f0dash[k] .= P' * f0dash[k] + Pdash[k]' * f0
        end
        f0 .= P' * f0

        axpy!(prob[u], f0, f1)
        for k = eachindex(f0dash)
            axpy!(prob[u], f0dash[k], f1dash[k])
        end
        for k = eachindex(f0dashdash)
            axpy!(prob[u], f0dashdash[k], f1dashdash[k])
        end
    end
    scal!(1/weight, f1)
    for k = eachindex(f1dash)
        scal!(1/weight, f1dash[k])
    end
    for k = eachindex(f1dashdash)
        scal!(1/weight, f1dashdash[k])
    end
    nothing
end
