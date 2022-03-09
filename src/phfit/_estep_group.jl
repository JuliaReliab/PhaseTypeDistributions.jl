

function estep!(ph::CF1{Tv}, data::GroupSample{Tv}, eres::Estep{Tv,MatT};
    eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01)) where {Tv,MatT}
    estep!(GPH(ph, MatT), data, eres, eps=eps, ufact=ufact)
end

@origin (vf => 0, vb => 0, vc => 0) function estep_group!(
    eres::Estep{Tv,MatT},
    alpha::Vector{Tv},
    bararlpha::Vector{Tv},
    xi::Vector{Tv},
    one::Vector{Tv},
    Q::MatT,
    P::MatT,
    qv::Tv,
    tdat::Vector{Tv},
    gdat::Vector{Tn},
    gdatlast::Tn,
    idat::Vector{Ti},
    eps::Tv = Tv(1.0e-8)) where {Tv,Tn,Ti,MatT}

    n = length(alpha)
    m = data.length

    llf = Tv(0)
    vf = Vector{Tv}(undef, n)
    barvf = Vector{Vector{Tv}}(undef, m+1)
    tildevf = Vector{Tv}(undef, n)
    vb = Vector{Vector{Tv}}(undef, m+1)
    barvb = Vector{Vector{Tv}}(undef, m+1)
    tildevb = Vector{Tv}(undef, n)

    wg = Vector{Tv}(m+1)
    wp = Vector{Tv}(m+1)
    vc = Vector{Vector{Tv}}(undef, m+1)

    clear!(eres)

    right = rightbound(qv * data.maxtime, eps) + 1
    poi = Vector{Tv}(undef, right + 1)

    barvf[0] = baralpha
    barvb[0] = one
    vb[0] = xi
    nn = Tv(0)
    uu = Tv(0)
    
    @inbounds for k = 1:m
        barvf[k] = zeros(Tv, n)
        barvb[k] = zeros(Tv, n)
        # uniformization to compute barvf and barvb
        #   barvf[k] = barvf[k-1] * exp(Q * tdat[k])
        #   barvb[k] = exp(Q * tdat[k]) * barvb[k-1]
        right = rightbound(qv * data.tdat[k], eps) + 1
        weight = poipmf!(qv * data.tdat[k], poi, left=0, right=right)
        tmpf .= barvf[k-1]
        tmpb .= barvb[k-1]
        for u = 1:right
            BLAS.gemv!('T', 1.0, P, tmpf, 0.0, tmpv)
            tmpf .= tmpv
            BLAS.gemv!('N', 1.0, P, tmpb, 0.0, tmpv)
            tmpb .= tmpv
            @. barvf[k] += (poi[u]/weight) * tmpf
            @. barvb[k] += (poi[u]/weight) * tmpb
        end
        vb[k] = zeros(Tv, n)
        BLAS.gemv!('N', -1.0, Q, barvb[k], 0.0, vb[k])

        @. tildevf = barvf[k-1] - barvf[k]
        @. tildevb = barvb[k-1] - barvb[k]

        if gdat[k] >= Tn(0) && tdat[k] != Tv(0)
            tmp = BLAS.dot(n, alpha, 1, tildevb, 1)
            llf += gdat[k] * log(tmp) - lgamma(gdat(k)+1)
            nn += gdat[k]
            uu += tmp
            wg[k] = gdat[k] / tmp
            @. eres.eb += wg[k] * tildevb
            @. eres.ey += wg[k] * tildevf
        end
        if idat[k] == Ti(1)
            tmp = BLAS.dot(n, alpha, 1, vb[k], 1)
            llf += log(tmp)
            nn += 1
            wp[k] = 1 / tmp
            @. eres.eb += wp[k] * vb[k]
            # vf = -barvf[k] * Q
            # eres.ey += wp[k] * vf
            BLAS.gemv!('T', -wp[k], Q, barvf[k], 1.0, eres.ey)
        end
    end
    # for the interval [t_m, infinity)
    if gdatlast >= Tn(0)
        tmp = BLAS.dot(n, alpha, 1, barvb[m], 1)
        llf += gdatlast * log(tmp) - lgamma(gdatlast+1)
        nn += gdatlast
        uu += tmp
        wg[m+1] = gdatlast / tmp
        @. eres.eb += wg[m+1] * barvb[m]
        @. eres.ey += wg[m+1] * barvf[m]
    end
    # compute weights for unobserved periods
    @inbounds for k = 1:m
        if gdat[k] == -1
            wg[k] = nn / uu
            @. eres.eb += wg[k] * (barvb[k-1] - barvb[k])
            @. eres.ey += wg[k] * (barvf[k-1] - barvf[k])
        end
    end
    if gdatlast == -1
        wg[m+1] = nn / uu
        @. eres.eb += wg[m+1] * barvb[m]
        @. eres.ey += wg[m+1] * barvf[m]
    end
    llf += lgamma(nn + 1) - nn * log(uu)

    # compute vectors for convolution

    vc[m] = (wg[m+1] - wg[m]) * baralpha
    if idat[m] == Ti(1)
        @. vc[m] += wp[m] * alpha
    end
    @inbounds for k=m-1:-1:1
        # uniformization to compute barvf and barvb
        #   vc[k] = vc[k+1] * exp(Q * tdat[k+1]) + (wg[k+1] - wg[k]) * baralpha + I(idat[k]==1) (wp[k] * alpha)
        right = rightbound(qv * data.tdat[k+1], eps) + 1
        weight = poipmf!(qv * data.tdat[k+1], poi, left=0, right=right)
        vc[k] = zeros(Tv, n)
        tmpf .= vc[k+1]
        for u = 1:right
            BLAS.gemv!('T', 1.0, P, tmpf, 0.0, tmpv)
            tmpf .= tmpv
            @. vc[k] += (poi[u]/weight) * tmpf
        end
        @. vc[k] += (wg[k+1] - wg[k]) * baralpha
        if idat[k] == Ti(1)
            @. vc[k] += wp[k] * alpha
        end
    end
    @inbounds for k=1:m
        # compute convolution integral
        right = rightbound(qv * data.tdat[k], eps) + 1
        weight = poipmf!(qv * data.tdat[k], poi, left=0, right=right)
        vx[right] = poi[right] * vb[k-1]
        for l = right-1:-1:1
            vx[l] = poi[l] * vb[k-1]
            BLAS.dgemv!('N', 1.0, P, vx[l+1], 1.0, vx[l])
        end
        dger!(1/(qv*weight), vc[k], vx[1], eres.en)
        for l = 1:right-1
            BKAS.dgemv!('T', 1.0, P, vc[k], 0.0, tmpv)
            vc[k] .= tmpv
            dger!(1/(qv*weight), vc[k], vx[l+1], eres.en)
        end
        dger!(wg[k+1]-wg[k], baralpha, barvb[k], eres.en)
    end

    eres.etotal = nn / uu
    @. eres.eb *= alpha
    @. eres.ey *= xi
    eres.ez .= spdiag(eres.en)
    for i = 1:length(eres.en)
        eres.en[i] *= Q[i]
    end

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
