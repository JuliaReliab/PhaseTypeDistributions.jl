using PhaseTypeDistributions: GPH, CF1, cf1sort!
using NMarkov.SparseMatrix: SparseCSR, SparseCSC, SparseCOO
using SparseArrays: SparseMatrixCSC, nnz, sparse
using ProgressMeter

function initializePH(cf1::CF1{Tv}, data::AbstractPHSample, ::Type{MatT}=SparseMatrixCSC;
    ratio = Tv[1, 4, 16, 64, 256, 1024],
    m1    = Tv[0.5, 1.0, 2.0],
    maxiter = 5, eps::Tv = Tv(1e-8), ufact::Tv = Tv(1.01),
    progress_init::Bool = true
) where {Tv,MatT}

    m      = mean(data)
    maxllf = -Inf
    maxph  = cf1
    eres   = Estep(GPH(cf1, MatT))

    total = 2 * length(m1) * length(ratio)

    pm = progress_init ?
        Progress(total; desc="Initializing CF1", dt=1.0) :
        ProgressUnknown(; dt=Inf, output=devnull)

    for fn in (cf1mom_power, cf1mom_linear)
        for x in m1, s in ratio
            try
                newcf1 = fn(cf1.dim, m*x, s)
                llf = Tv(NaN)
                for _ in 1:maxiter
                    llf = estep!(newcf1, data, eres; eps=eps, ufact=ufact)
                    !isfinite(llf) && break
                    mstep!(newcf1, eres)
                end
                if !isfinite(llf)
                    @warn "CF1 init non-finite llf" m1=x ratio=s llf=llf
                elseif llf > maxllf
                    maxllf, maxph = llf, newcf1
                    @debug "CF1 updated" m1=x ratio=s llf=llf maxllf=maxllf
                else
                    @debug "CF1 not updated" m1=x ratio=s llf=llf maxllf=maxllf
                end
            catch err
                @warn "CF1 init exception" m1=x ratio=s exception=(err, catch_backtrace())
            end
            next!(pm)
        end
    end
    progress_init && finish!(pm)
    return maxph
end

function phfit(cf1::CF1{Tv}, data::AbstractPHSample, ::Type{MatT} = SparseMatrixCSC;
    initialize = true, eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01),
    steps = 10,
    ratio = Tv[1, 4, 16, 64, 256, 1024], m1 = Tv[0.5, 1.0, 2.0], init_maxiter = 5,
    maxiter = 5000, abstol = Tv(1.0e-3), reltol = Tv(1.0e-5),
    progress_init::Bool = true, progress::Bool = true) where {Tv,MatT}

    local newcf1::CF1{Tv}
    if initialize
        newcf1 = initializePH(cf1, data, MatT, ratio=ratio, m1=m1, maxiter=init_maxiter, eps=eps, ufact=ufact, progress_init=progress_init)
    else
        newcf1 = copy(cf1)
    end
    llf, conv, iter, data, aerror, rerror = phfit!(newcf1, data, MatT, eps=eps, ufact=ufact, steps=steps, maxiter=maxiter, abstol=abstol, reltol=reltol, progress=progress)
    return (model=newcf1, llf=llf, conv=conv, iter=iter, data=data, aerror=aerror, rerror=rerror)
end

function phfit!(cf1::CF1{Tv}, data::AbstractPHSample, ::Type{MatT} = SparseMatrixCSC;
    eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01), 
    steps = 10, maxiter = 5000, abstol = Tv(1.0e-3), reltol = Tv(1.0e-5),
    progress::Bool = true) where {Tv,MatT}

    iter = 0
    ph = GPH(cf1, MatT)
    eres = Estep(ph)
    conv = false
    local aerror::Tv = NaN
    local rerror::Tv = NaN

    llf = estep!(cf1, data, eres, eps=eps, ufact=ufact)
    !isfinite(llf) && (@warn("Initial llf is not finite at the first step: llf=$(llf)"); return llf, conv, iter, data, aerror, rerror)
    mstep!(cf1, eres)

    pm = progress ? Progress(maxiter; desc="Fitting", dt=1.0) : ProgressUnknown(; dt=Inf, output=devnull)
    while true
        prevllf = llf
        for k = 1:steps
            llf = estep!(cf1, data, eres, eps=eps, ufact=ufact)
            if !isfinite(llf)
                @warn("phfit!: llf is not finite at step $(iter+k): llf=$(llf)")
                break
            end
            mstep!(cf1, eres)
        end
        iter += steps
        aerror = abs(llf - prevllf)
        rerror = abs((llf - prevllf) / prevllf)

        @debug "phfit!: Iteration $(iter): llf=$(llf), aerror=$(aerror), rerror=$(rerror)"

        if llf < prevllf
            @warn("phfit!: llf does not increase at iter=$(iter); previous $(prevllf), current $(llf)")
        end

        if aerror < abstol || rerror < reltol
            @debug "phfit!: Converged at iteration $(iter): aerror=$(aerror), rerror=$(rerror)"
            conv = true
            break
        end

        if iter >= maxiter
            @warn("phfit!: Maximum number of iterations reached: $(maxiter)")
            break
        end

        update!(pm, iter)
    end
    progress && finish!(pm)
    return llf, conv, iter, data, aerror, rerror
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
                ph.T.nzval[z] = eres.en.nzval[z] / eres.ez[i]
                tmp[i] += ph.T.nzval[z]
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

function mstep!(cf1::CF1{Tv}, eres::Estep{Tv,MatT}) where {Tv,MatT}
    dim = cf1.dim
    total = Tv(0)
    for i = 1:dim
        total += eres.eb[i]
        cf1.alpha[i] = eres.eb[i] / eres.etotal
        cf1.rate[i] = total / eres.ez[i]
        if !isfinite(cf1.rate[i])
            cf1.alpha[i] = Tv(0)
            cf1.rate[i] = Tv(0)
        end
    end
    cf1sort!(cf1.alpha, cf1.rate)
    nothing
end
