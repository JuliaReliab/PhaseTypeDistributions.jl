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


