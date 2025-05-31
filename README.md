# PhaseTypeDistributions

[![Build Status](https://travis-ci.com/okamumu/PhaseTypeDistributions.jl.svg?branch=master)](https://travis-ci.com/okamumu/PhaseTypeDistributions.jl)
[![Codecov](https://codecov.io/gh/okamumu/PhaseTypeDistributions.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/okamumu/PhaseTypeDistributions.jl)
[![Coveralls](https://coveralls.io/repos/github/okamumu/PhaseTypeDistributions.jl/badge.svg?branch=master)](https://coveralls.io/github/okamumu/PhaseTypeDistributions.jl?branch=master)

## Usage Examples

### Creating a Phase-Type Distribution
```julia
using PhaseTypeDistributions

# Define initial probabilities and rates
alpha = [0.1, 0.3, 0.6]
rate = [1.4, 0.4, 10.0]

# Create a CF1 phase-type distribution
cf1 = CF1(alpha, rate)
println(cf1)
```

### Computing PDF, CDF, and Complementary CDF
```julia
using PhaseTypeDistributions

# Define a GPH distribution
alpha = [0.1, 0.3, 0.6]
T = [-1.4 1.4 0; 0 -0.4 0.4; 0 0 -10.0]
tau = [0, 0, 10.0]
gph = GPH(alpha, T, tau)

# Compute PDF, CDF, and CCDF at specific times
ts = LinRange(0.0, 10.0, 100)
pdf_values = phpdf(gph, ts)
cdf_values = phcdf(gph, ts)
ccdf_values = phccdf(gph, ts)
println(pdf_values)
println(cdf_values)
println(ccdf_values)
```

### Fitting a Phase-Type Distribution
```julia
using PhaseTypeDistributions
using PhaseTypeDistributions.Phfit
using Distributions

# Generate sample data from a Weibull distribution
data = WeightedSample((0.0, Inf64)) do x
    pdf(Weibull(2.0, 1.0), x)
end

# Fit a CF1 phase-type distribution to the data
result = phfit(CF1(10), data, verbose = [true, true])
println("Log-likelihood: ", result.llf)
println("Convergence: ", result.conv)
println("Iterations: ", result.iter)
println("Relative error: ", result.rerror)
println("Data: ", result.data)
println("Fitted parameters: ", result.model.alpha, result.model.rate)
```

### Working with Mixed Point and Interval Data using `TimeSpanSample`

The `TimeSpanSample` structure supports data that includes both exact time points and time intervals, such as `(1.0, 2.0)` or `(4.0, Inf)`. This is useful when the observed data are partially censored or interval-censored.

```julia
using PhaseTypeDistributions
using PhaseTypeDistributions.Phfit

# Define point and interval data
t = [1.5, (0.0, 2.0), (1.0, Inf)]
w = [1.0, 5.0, 1.0]

# Create a TimeSpanSample object
sample = TimeSpanSample(t, w)

# Fit a CF1 model using the EM algorithm
result = phfit(CF1(5), sample, verbose = [true, true], maxiter = 2000)

# Output results
println("Log-likelihood: ", result.llf)
println("Convergence: ", result.conv)
println("Iterations: ", result.iter)
```

You can also compute the weighted mean of the sample:

```julia
println("Mean: ", mean(sample))
```

### Handling Left-Truncated and Right-Censored Data using `LeftTruncRightCensoredSample`

The `LeftTruncRightCensoredSample` structure allows you to fit phase-type distributions to survival data that are subject to **left truncation** and **right censoring**, common in medical and reliability applications.

* `t`: event or censoring times
* `delta`: censoring indicators (`true` = observed, `false` = right-censored)
* `tau`: left truncation times (must be the same length as `t`)

The construction combines these into a single dataset with annotated time points:

```julia
using PhaseTypeDistributions
using PhaseTypeDistributions.Phfit

# Example survival data
t = [2.0, 4.0, 5.0]
τ = [1.0, 0.0, 0.5]  # left truncation times (same length as t)
δ = [true, false, true]  # censoring indicators

# Create the LTRC sample
ltrc = LeftTruncRightCensoredSample(t, τ, δ)

# Fit a model
result = phfit(CF1(3), ltrc, verbose = [true, true])

# Output results
println("Log-likelihood: ", result.llf)
println("Fitted parameters: ", result.model.alpha, result.model.rate)
```

This interface ensures proper treatment of truncation and censoring during parameter estimation using the EM algorithm.
