export phfit!, phfit

function phfit(f::Any, cf1::CF1{Tv}, bounds::Tuple{Tv,Tv} = (Tv(0), Tv(Inf)), ::Type{MatT} = SparseCSC;
    initialize = true, eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01), verbose = [false, false],
    steps = 50,
    shapes = Tv[1, 4, 16, 64, 256, 1024], scales = Tv[0.5, 1.0, 2.0], init_maxiter = 5,
    maxiter = 5000, reltol = Tv(1.0e-8)) where {Tv,MatT}

    data = WeightedSample(f, bounds)
    phfit(cf1, data, MatT, initialize=initialize, eps=eps, ufact=ufact, verbose=verbose, shapes=shapes,
        scales=scales, init_maxiter=init_maxiter, steps=steps, maxiter=maxiter, reltol=reltol)
end

function phfit(cf1::CF1{Tv}, data::AbstractPHSample, ::Type{MatT} = SparseCSC;
    initialize = true, eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01), verbose = [false, false],
    steps = 50,
    shapes = Tv[1, 4, 16, 64, 256, 1024], scales = Tv[0.5, 1.0, 2.0], init_maxiter = 5,
    maxiter = 5000, reltol = Tv(1.0e-8)) where {Tv,MatT}

    local newcf1::CF1{Tv}
    if initialize
        newcf1 = initializePH(cf1, data, MatT, shapes=shapes, scales=scales,
            maxiter=init_maxiter, verbose = verbose[1], eps=eps, ufact=ufact)
    else
        newcf1 = copy(cf1)
    end
    llf, conv, iter, rerror = phfit!(newcf1, data, MatT, eps=eps, ufact=ufact,
        verbose = verbose[2], steps=steps, maxiter=maxiter, reltol=reltol)
    return newcf1, llf, conv, iter, rerror
end

function phfit!(cf1::CF1{Tv}, data::AbstractPHSample, ::Type{MatT} = SparseCSC;
    eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01), verbose = false,
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
            mstep!(cf1, eres)
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
