using PhaseTypeDistributions: CF1
using PolynomialRoots: roots

export ph3mom_bobbio05

function hypoerlang(; shape, initprob, rate)
  phsize = sum(shape)
  alpha = zeros(phsize)
  rates = zeros(phsize)

  index = 1
  for (p,s,r) = zip(initprob,shape,rate)
    alpha[index] = p
    for k = 1:s
        rates[index] = r
        index += 1
    end
  end
  CF1(alpha, rates)
end

function lowerbound(n, n2)
    if n2<(n+1)/n
        lb = Inf
    elseif n2<(n+4)/(n+1)
      p = ((n+1)*(n2-2)) / (3*n2*(n-1)) * ((-2*sqrt(n+1)) / sqrt(-3*n*n2+4*n+4) -1)
      a = (n2-2) / (p*(1-n2) + sqrt(p^2+p*n*(n2-2)/(n-1)))
      l = ((3+a)*(n-1)+2*a) / ((n-1)*(1+a*p)) - (2*a*(n+1)) / (2*(n-1)+a*p*(n*a+2*n-2))
      lb = l
    else
      lb = (n+1)/n *n2
    end
    lb
end

function upperbound(n, n2)
    if n2<(n+1)/n
      ub = -Inf
    elseif n2<=n/(n-1)
      u = (2*(n-2)*(n*n2-n-1)*sqrt(1+(n*(n2-2))/(n-1)) + (n+2)*(3*n*n2-2*n-2)) / (n^2*n2)
      ub = u
    else
      ub = Inf
    end
    ub
end

function ph3mom_bobbio05(m1, m2, m3)
    ## normalized moments
    n1 = m1
    n2 = m2 / m1^2
    n3 = m3 / m1 / m2

    ## detect the number phases
    n = 2
    while n < 100 && ((n+1) / n > n2 || lowerbound(n, n2) >= n3 || upperbound(n, n2) <= n3)
        n += 1
    end
    if n2 < (n+1)/n
        n2 = (n+1)/n
    end
    if n3 < lowerbound(n, n2)
        n3 = lowerbound(n, n2)
    end
    if n3 > upperbound(n, n2)
        n3 = upperbound(n, n2)
    end

    if n2 > 2 || n3 < 2*n2 - 1
        b = 2*(4-n*(3*n2-4)) / (n2*(4+n-n*n3) + sqrt(n*n2)*sqrt(12*n2^2*(n+1)+16*n3*(n+1)+n2*(n*(n3-15)*(n3+1)-8*(n3+3))))
        a = (b*n2-2)*(n-1)*b / (b-1) / n
        p = (b-1) / a
        lambda = (p*a+1) / n1
        mu = (n-1)*lambda / a
        res = hypoerlang(shape=[n-1,1], initprob=[p, 1-p], rate=[mu, lambda])
    else
        c4 = n2*(3*n2-2*n3)*(n-1)^2
        c3 = 2*n2*(n3-3)*(n-1)^2
        c2 = 6*(n-1)*(n-n2)
        c1 = 4*n*(2-n)
        c0 = n*(n-2)
        fs = roots([c0, c1, c2, c3, c4])
        println(fs)
        found = 0
        for i = 1:length(fs)
            f = fs[i]
            a = 2*(f-1)*(n-1) / ((n-1)*(n2*f^2-2*f+2)-n)
            p = (f-1)*a
            lambda = (a+p) / n1
            mu = (n-1) / (n1 - p/lambda)
            if isreal(p) && isreal(lambda) && isreal(mu) && real(p)>=0 && real(p)<=1 && real(lambda)>0 && real(mu)>0
                p = real(p)
                lambda = real(lambda)
                mu = real(mu)
                res = hypoerlang(shape=[1,n-1], initprob=[p, 1-p], rate=[lambda, mu])
                found = 1
            end
        end
        if found == 0
            throw(ErrorException("Cannot find the APH with 3 moments $m1 $m2 $m3"))
        end
    end
    res
end

