using PhaseTypeDistributions: GPH, CF1, cf1sort!
using SparseMatrix: SparseCSR, SparseCSC, SparseCOO
using SparseArrays: SparseMatrixCSC, nnz

export phfit

function initializePH(cf1::CF1{Tv}, data::AbstractPHSample, ::Type{MatT} = SparseMatrixCSC;
    ratio = Tv[1, 4, 16, 64, 256, 1024],
    m1 = Tv[0.5, 1.0, 2.0],
    maxiter = 5, verbose = false, eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01)) where {Tv,MatT}
    verbose && println("Initializing CF1 ...")
    m = mean(data)
    maxllf = typemin(Tv)
    maxph = cf1
    eres = Estep(GPH(cf1, MatT))
    for fn = [cf1mom_power, cf1mom_linear]
        for x = m1, s = ratio
            try
                newcf1 = fn(cf1.dim, m*x, s)
                local llf::Tv
                for k = 1:maxiter
                    llf = estep!(newcf1, data, eres, eps=eps, ufact=ufact)
                    mstep!(newcf1, eres)
                end
                if !isfinite(llf)
                    verbose && print("-")
                else
                    if maxllf < llf
                        maxllf, maxph = llf, newcf1
                        verbose && print("o")
                    else
                        verbose && print("x")
                    end
                end
            catch
                verbose && print("-")
            end
        end
        verbose && println()
    end
    return maxph
end

function phfit(cf1::CF1{Tv}, data::AbstractPHSample, ::Type{MatT} = SparseMatrixCSC;
    initialize = true, eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01), cf1sort=true, verbose = [false, false],
    steps = 50,
    ratio = Tv[1, 4, 16, 64, 256, 1024], m1 = Tv[0.5, 1.0, 2.0], init_maxiter = 5,
    maxiter = 5000, reltol = Tv(1.0e-8)) where {Tv,MatT}

    local newcf1::CF1{Tv}
    if initialize
        newcf1 = initializePH(cf1, data, MatT, ratio=ratio, m1=m1,
            maxiter=init_maxiter, verbose = verbose[1], eps=eps, ufact=ufact)
    else
        newcf1 = copy(cf1)
    end
    llf, conv, iter, rerror = phfit!(newcf1, data, MatT, eps=eps, ufact=ufact,
        cf1sort=cf1sort, verbose = verbose[2], steps=steps, maxiter=maxiter, reltol=reltol)
    return newcf1, llf, conv, iter, rerror
end

function phfit!(cf1::CF1{Tv}, data::AbstractPHSample, ::Type{MatT} = SparseMatrixCSC;
    eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01), cf1sort = true, verbose = false,
    steps = 50, maxiter = 5000, reltol = Tv(1.0e-8)) where {Tv,MatT}
    iter = 0
    ph = GPH(cf1, MatT)
    eres = Estep(ph)
    conv = false
    local rerror::Tv

    llf = estep!(cf1, data, eres, eps=eps, ufact=ufact)
    mstep!(cf1, eres)
    while true
        prevllf = llf
        for k = 1:steps
            llf = estep!(cf1, data, eres, eps=eps, ufact=ufact)
            mstep!(cf1, eres, cf1sort=cf1sort)
        end
        iter += steps
        rerror = abs((llf - prevllf) / prevllf)
        verbose && println("iter=$(iter) llf=$(llf) rerror=$(rerror)")

        if llf < prevllf
            println("Warning: llf does not increase at iter=$(iter)")
        end

        if rerror < reltol
            conv = true
            break
        end

        if iter >= maxiter
            break
        end
    end
    return llf, conv, iter, rerror
end

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

function clear!(eres::Estep{Tv,MatT}) where {Tv,MatT}
    eres.etotal = zero(Tv)
    @. eres.eb = zero(Tv)
    @. eres.ey = zero(Tv)
    @. eres.ez = zero(Tv)
    for i = 1:length(eres.en)
        eres.en[i] = zero(Tv)
    end
end

function estep!(ph::CF1{Tv}, data::AbstractPHSample, eres::Estep{Tv,MatT};
    eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01)) where {Tv,MatT}
    estep!(GPH(ph, MatT), data, eres, eps=eps, ufact=ufact)
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
    cf1sort && cf1sort!(cf1.alpha, cf1.rate)
    nothing
end
