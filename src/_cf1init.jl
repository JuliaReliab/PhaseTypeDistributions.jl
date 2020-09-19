
function initializePH(cf1::CF1{Tv}, data::AbstractPHSample, ::Type{MatT} = SparseMatrixCSC;
    shapes = Tv[1, 4, 16, 64, 256, 1024],
    scales = Tv[0.5, 1.0, 2.0],
    maxiter = 5, verbose = false, eps::Tv = Tv(1.0e-8), ufact::Tv = Tv(1.01)) where {Tv,MatT}
    verbose && println("Initializing CF1 ...")
    m = _phmean(data)
    maxllf = typemin(Tv)
    maxph = cf1
    eres = Estep(GPH(cf1, MatT))
    for fn = [_cf1_params_power, _cf1_params_linear]
        for x = scales, s = shapes
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

function _cf1_params_power(dim::Ti, scale::Tv, shape::Tv) where {Tv,Ti}
    rate = Vector{Tv}(undef, dim)
    p = exp(Tv(1)/(dim-1) * log(shape))
    total = Tv(1)
    tmp = Tv(1)
    for i = 1:dim-1
        tmp *= (i+1) / (i * p)
        total += tmp
    end
    base = total / (dim * scale)
    tmp = base
    for i = 1:dim
        rate[i] = tmp
        tmp *= p
    end
    CF1(fill(1/dim, dim), rate)
end

function _cf1_params_linear(dim::Ti, scale::Tv, shape::Tv) where {Tv,Ti}
    rate = Vector{Tv}(undef, dim)
    al = (shape - 1)/(dim-1)
    total = Tv(1)
    for i = 1:dim
        total += (i+1)/(al*i+1)
    end
    base = total / (dim * scale)
    for i = 1:dim
        rate[i] = base * (al + i + 1)
    end
    CF1(fill(1/dim, dim), rate)
end
