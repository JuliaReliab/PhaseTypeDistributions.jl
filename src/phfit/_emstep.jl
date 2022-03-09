
mutable struct Estep{Tv,MatT}
    etotal::Tv
    eb::Vector{Tv}
    ey::Vector{Tv}
    ez::Vector{Tv}
    en::MatT
end

function Estep(ph::GPH{Tv,MatT}) where {Tv,MatT}
    Estep(Tv(0),
        Vector{Tv}(undef, ph.dim),
        Vector{Tv}(undef, ph.dim),
        Vector{Tv}(undef, ph.dim),
        similar(ph.T))
end

function estep!(ph::CF1{Tv}, data::WeightedSample{Tv}, eres::Estep{Tv,MatT};
    eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01)) where {Tv,MatT}
    estep!(GPH(ph, MatT), data, eres, eps=eps, ufact=ufact)
end

@origin (vf => 0, vb => 0, vc => 0) function estep!(ph::GPH{Tv,MatT},
    data::WeightedSample{Tv}, eres::Estep{Tv,MatT}; eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01)) where {Tv,MatT}

    dim, alpha, P, tau, qv = _phunif(ph, ufact)

    llf = Tv(0)
    tllf = Tv(0)
    # eres.etotal = Tv(0)
    eres.eb .= Tv(0)
    eres.ey .= Tv(0)
    # eres.ez .= Tv(0)
    for i = 1:length(eres.en)
        eres.en[i] = Tv(0)
    end
    right = rightbound(qv * data.maxtime, eps) + 1
    poi = Vector{Tv}(undef, right + 1)

    m = data.length
    blf = Vector{Tv}(undef, m)
    vf = Vector{Vector{Tv}}(undef, m+1)
    vb = Vector{Vector{Tv}}(undef, m+1)
    vc = Vector{Vector{Tv}}(undef, m+1)
    vf[0] = alpha
    vb[0] = tau
    for k = 1:m
        right = rightbound(qv * data.tdat[k], eps) + 1
        weight = poipmf!(qv * data.tdat[k], poi, left=0, right=right)
        vf[k] = zero(alpha)
        # unifstep!(:T, P, poi, (0, right), weight, copy(vf[k-1]), vf[k])
        scale = @dot(vf[k], tau)
        scal!(1.0/scale, vf[k])
        axpy!(data.wdat[k], vf[k], eres.ey)

        blf[k] = scale
        vb[k] = zero(alpha)
        # unifstep!(:N, P, poi, (0, right), weight, copy(vb[k-1]), vb[k])
        scale = @dot(alpha, vb[k])
        scal!(1.0/scale, vb[k])
        axpy!(data.wdat[k], vb[k], eres.eb)

        tllf += log(blf[k])
        llf += data.wdat[k] * tllf
    end

    vc[m] = zeros(Tv, dim)
    axpy!(data.wdat[m]/blf[m], alpha, vc[m])
    for k = m-1:-1:1
        right = rightbound(qv * data.tdat[k+1], eps) + 1
        weight = poipmf!(qv * data.tdat[k+1], poi, left=0, right=right)
        vc[k] = zeros(Tv, dim)
        # unifstep!(:T, P, poi, (0, right), weight, copy(vc[k+1]), vc[k])
        scal!(1.0/blf[k], vc[k])
        axpy!(data.wdat[k]/blf[k], alpha, vc[k])
    end

    tmpb = similar(alpha)
    tmpn = zero(P)
    for k = 1:m
        right = rightbound(qv * data.tdat[k], eps) + 1
        weight = poipmf!(qv * data.tdat[k], poi, left=0, right=right)
        tmpb .= Tv(0)
        # convunifstep!(:T, :N, P, poi, (0, right), weight, qv*weight,
        #     vc[k], vb[k-1], tmpb, tmpn)
        for i = 1:length(tmpn)
            eres.en[i] += tmpn[i]
            tmpn[i] = Tv(0)
        end
    end

    eres.eb .*= alpha
    eres.ey .*= tau
    eres.ez .= spdiag(eres.en)
    for i = 1:length(eres.en)
        eres.en[i] *= ph.T[i]
    end
    eres.etotal = sum(eres.eb)

    return llf
end

function mstep!(ph::GPH{Tv,Matrix{Tv}}, eres::Estep{Tv,Matrix{Tv}}) where Tv
    dim = ph.dim
    tmp = zeros(Tv, dim)
    for j = 1:dim
        for i = 1:dim
            if i != j
                ph.T[i,j] = eres.en[i,j] / eres.ez[i]
                tmp[i] += ph.T[i,j]
            end
        end
    end
    for i = 1:dim
        ph.alpha[i] = eres.eb[i] / eres.etotal
        ph.tau[i] = eres.ey[i] / eres.ez[i]
        tmp[i] += ph.tau[i]
        ph.T[i,i] = -tmp[i]
    end
    nothing
end

function mstep!(ph::GPH{Tv,SparseCSR{Tv,Ti}}, eres::Estep{Tv,SparseCSR{Tv,Ti}}) where {Tv,Ti}
    dim = ph.dim
    tmp = zeros(Tv, dim)
    d = zeros(Ti, dim)
    for i = 1:dim
        for z = eres.en.rowptr[i]:eres.en.rowptr[i+1]-1
            j = eres.en.colind[z]
            if i != j
                ph.T[z] = eres.en[z] / eres.ez[i]
                tmp[i] += ph.T[z]
            else
                d[i] = z
            end
        end
    end
    for i = 1:dim
        ph.alpha[i] = eres.eb[i] / eres.etotal
        ph.tau[i] = eres.ey[i] / eres.ez[i]
        tmp[i] += ph.tau[i]
        ph.T.val[d[i]] = -tmp[i]
    end
    nothing
end

function mstep!(ph::GPH{Tv,SparseCSC{Tv,Ti}}, eres::Estep{Tv,SparseCSC{Tv,Ti}}) where {Tv,Ti}
    dim = ph.dim
    tmp = zeros(Tv, dim)
    d = zeros(Ti, dim)
    for j = 1:dim
        for z = eres.en.colptr[j]:eres.en.colptr[j+1]-1
            i = eres.en.rowind[z]
            if i != j
                ph.T[z] = eres.en[z] / eres.ez[i]
                tmp[i] += ph.T[z]
            else
                d[i] = z
            end
        end
    end
    for i = 1:dim
        ph.alpha[i] = eres.eb[i] / eres.etotal
        ph.tau[i] = eres.ey[i] / eres.ez[i]
        tmp[i] += ph.tau[i]
        ph.T.val[d[i]] = -tmp[i]
    end
    nothing
end

function mstep!(ph::GPH{Tv,SparseMatrixCSC{Tv,Ti}}, eres::Estep{Tv,SparseMatrixCSC{Tv,Ti}}) where {Tv,Ti}
    dim = ph.dim
    tmp = zeros(Tv, dim)
    d = zeros(Ti, dim)
    for j = 1:dim
        for z = eres.en.colptr[j]:eres.en.colptr[j+1]-1
            i = eres.en.rowval[z]
            if i != j
                ph.T[z] = eres.en[z] / eres.ez[i]
                tmp[i] += ph.T[z]
            else
                d[i] = z
            end
        end
    end
    for i = 1:dim
        ph.alpha[i] = eres.eb[i] / eres.etotal
        ph.tau[i] = eres.ey[i] / eres.ez[i]
        tmp[i] += ph.tau[i]
        ph.T.nzval[d[i]] = -tmp[i]
    end
    nothing
end

function mstep!(ph::GPH{Tv,SparseCOO{Tv,Ti}}, eres::Estep{Tv,SparseCOO{Tv,Ti}}) where {Tv,Ti}
    dim = ph.dim
    tmp = zeros(Tv, dim)
    d = zeros(Ti, dim)
    for z = 1:nnz(eres.en)
        i = eres.en.rowind[z]
        j = eres.en.colind[z]
        if i != j
            ph.T[z] = eres.en[z] / eres.ez[i]
            tmp[i] += ph.T[z]
        else
            d[i] = z
        end
    end
    for i = 1:dim
        ph.alpha[i] = eres.eb[i] / eres.etotal
        ph.tau[i] = eres.ey[i] / eres.ez[i]
        tmp[i] += ph.tau[i]
        ph.T.val[d[i]] = -tmp[i]
    end
    nothing
end

function mstep!(cf1::CF1{Tv}, eres::Estep{Tv,MatT}; cf1sort = true) where {Tv,MatT}
    dim = cf1.dim
    total = Tv(0)
    for i = 1:dim
        total += eres.eb[i]
        cf1.alpha[i] = eres.eb[i] / eres.etotal
        cf1.rate[i] = total / eres.ez[i]
    end
    cf1sort && _cf1_sort!(cf1.alpha, cf1.rate)
    nothing
end
