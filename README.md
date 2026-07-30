# PhaseTypeDistributions.jl

[![CI](https://github.com/JuliaReliab/PhaseTypeDistributions.jl/workflows/CI/badge.svg)](https://github.com/JuliaReliab/PhaseTypeDistributions.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/JuliaReliab/PhaseTypeDistributions.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaReliab/PhaseTypeDistributions.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaReliab/PhaseTypeDistributions.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaReliab/PhaseTypeDistributions.jl?branch=master)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Julia](https://img.shields.io/badge/Julia-1.10+-blue.svg)](https://julialang.org)


A Julia package for working with **Phase-Type (PH) distributions**, providing tools for creating, analyzing, and fitting these versatile probability distributions. Phase-type distributions are fundamental in stochastic modeling, queueing theory, reliability engineering, and survival analysis.

## Features

- **Multiple Representations**: Support for General Phase-Type (GPH) and Canonical Form 1 (CF1) distributions
- **Distribution Analysis**: Compute PDF, CDF, complementary CDF, moments, and generate random samples
- **Parameter Estimation**: Advanced EM algorithm-based fitting for various data types:
  - Point observations (exact measurements)
  - Interval-censored data
  - Left-truncated and right-censored data (survival analysis)
  - Weighted samples and density-based fitting
  - Grouped data
- **Model Selection**: AIC and Extended Information Criterion (EIC) with bootstrap confidence intervals
- **Efficient Computation**: Optimized implementations using sparse matrices and numerical quadrature
- **Flexible API**: Easy-to-use interface for both research and practical applications

## What are Phase-Type Distributions?

Phase-type distributions represent the time until absorption in a continuous-time Markov chain with one absorbing state. They form a dense class of distributions on the positive real line and can approximate any positive-valued distribution arbitrarily closely, making them extremely versatile for modeling real-world phenomena.

**Common applications include:**
- Modeling service times in queueing systems
- Reliability and survival analysis
- Risk assessment and insurance mathematics
- Telecommunications and network modeling
- Manufacturing and production systems

## Installation

This package depends on `NMarkov.jl` and `DEQuadrature.jl`, which live in the
JuliaReliab registry rather than General. Add that registry first — once per Julia
depot — and the dependencies resolve automatically:

```julia
using Pkg
Pkg.Registry.add(RegistrySpec(url="https://github.com/JuliaReliab/Registry.git"))
Pkg.add(url="https://github.com/JuliaReliab/PhaseTypeDistributions.jl")
```

Or in the Julia REPL package mode (press `]`):
```julia
pkg> registry add https://github.com/JuliaReliab/Registry.git
pkg> add https://github.com/JuliaReliab/PhaseTypeDistributions.jl
```

Requires Julia 1.10 or later.

## Quick Start

```julia
using PhaseTypeDistributions

# Create a simple CF1 distribution with 3 phases
alpha = [0.1, 0.3, 0.6]  # Initial probabilities
rate = [1.4, 0.4, 10.0]   # Exit rates
cf1 = CF1(alpha, rate)

# Compute statistics
println("Mean: ", phmean(cf1))
println("PDF at t=1.0: ", phpdf(cf1, 1.0))
println("CDF at t=1.0: ", phcdf(cf1, 1.0))

# Generate random samples
samples = phsample(cf1, 1000)
```

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

### Using Distributions.jl API

CF1 and GPH implement the Distributions.jl univariate continuous API. With `using Distributions`, you can use familiar functions directly on phase-type distributions. All time arguments are `Real`.

Supported functions:
- `pdf(d, t)`, `cdf(d, t)`, `ccdf(d, t)`
- `mean(d)`
- `rand(d)`, `rand(d, n)`

Example:
```julia
using PhaseTypeDistributions, Distributions

cf1 = CF1([0.1, 0.3, 0.6], [1.4, 0.4, 10.0])

println("mean(cf1): ", mean(cf1))
println("pdf(cf1, 1.0): ", pdf(cf1, 1.0))
println("cdf(cf1, 1.0): ", cdf(cf1, 1.0))
println("ccdf(cf1, 1.0): ", ccdf(cf1, 1.0))

samples = rand(cf1, 1000)
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
result = phfit(CF1(10), data, progress = true, progress_init = true)
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
result = phfit(CF1(5), sample, progress = true, progress_init = true, maxiter = 2000)

# Output results
println("Log-likelihood: ", result.llf)
println("Convergence: ", result.conv)
println("Iterations: ", result.iter)
```

You can also compute the weighted mean of the sample:

```julia
println("Mean: ", mean(sample))
```

#### Left truncation

Pass a `tau` vector to condition each observation on survival past a truncation time.
`tau[i] > 0` means observation `i` was only observable given that the event had not yet
occurred at time `tau[i]`; `tau[i] == 0` means no truncation. The likelihood of that
observation is divided by `S(tau[i])`.

```julia
t   = [1.5, (2.0, 3.0), (4.0, Inf)]
tau = [0.5, 1.0, 0.0]          # third observation is not truncated

sample = TimeSpanSample(t; tau = tau)
result = phfit(CF1(3), sample, progress = false, progress_init = false)
```

`tau` combines freely with counts and analytic weights (`TimeSpanSample(t, n, w; tau = tau)`)
and is carried through `bootstrap` and `eic`. This makes `TimeSpanSample` a superset of
`LeftTruncRightCensoredSample`, which handles left truncation and right censoring but not
interval censoring, per-observation counts, or bootstrap-based model selection.

### Bootstrap, EIC, and AIC for Model Selection

After fitting a model to `TimeSpanSample` data, you can assess model quality using the
**Extended Information Criterion (EIC)** and the **Akaike Information Criterion (AIC)**.

#### `bootstrap` — multinomial resampling

`bootstrap` draws a new `TimeSpanSample` by resampling observation counts from a Multinomial
distribution, leaving the analytic weights `wdat` unchanged.  It is used internally by `eic`
but can also be called directly.

```julia
using PhaseTypeDistributions
using PhaseTypeDistributions.Phfit
using Random

t   = [0.5, 1.2, (0.8, 1.5), (2.0, Inf), 0.3]
dat = TimeSpanSample(t)

bdat = bootstrap(dat)                          # uses default RNG
bdat = bootstrap(MersenneTwister(42), dat)     # reproducible
```

The total observation count is always preserved: `sum(bdat.rawn) == sum(dat.rawn)`.

#### `eic` — Extended Information Criterion

`eic` estimates the bias between in-sample log-likelihood and generalisation performance via
bootstrap resampling.  The returned named tuple contains:

| field | description |
|-------|-------------|
| `eic` | point estimate: −2·(llf₀ − mean(bias)) |
| `ci_lower` | upper edge of 95 % CI (larger EIC value; bias underestimated) |
| `ci_upper` | lower edge of 95 % CI (smaller EIC value; bias overestimated) |
| `nvalid` | number of bootstrap replicates that converged |

Note: the EIC scale follows −2·log-likelihood convention, so **smaller is better**.
The interval satisfies `ci_upper ≤ eic ≤ ci_lower`.

```julia
using PhaseTypeDistributions
using PhaseTypeDistributions.Phfit
using Random

# Prepare data and fit
t   = vcat(rand(MersenneTwister(1), 80), [(0.5*i, 0.5*i+1.0) for i in 1:20])
dat = TimeSpanSample(t)
res = phfit(CF1(5), dat; progress_init=false, progress=false)

# EIC with 200 bootstrap replicates (default rng)
r = eic(res.model, dat; bsample=200)
println("EIC: ",       r.eic)
println("95% CI: [",   r.ci_upper, ", ", r.ci_lower, "]")
println("Valid reps: ", r.nvalid)

# Reproducible run
r2 = eic(MersenneTwister(42), res.model, dat; bsample=200)
```

Keyword arguments:

| keyword | default | description |
|---------|---------|-------------|
| `bsample` | 100 | number of bootstrap replicates |
| `maxiter` | 5000 | max EM iterations per replicate |
| `steps` | 10 | EM steps between convergence checks |
| `abstol` | 1e-3 | absolute log-likelihood tolerance |
| `reltol` | 1e-5 | relative log-likelihood tolerance |

#### `aic` — Akaike Information Criterion

`aic` computes AIC using an effective parameter count for CF1: free `alpha` entries (nonzero
minus one, for the sum-to-one constraint) plus distinct `rate` values.

```julia
using PhaseTypeDistributions
using PhaseTypeDistributions.Phfit

t   = rand(100)
dat = TimeSpanSample(t)
res = phfit(CF1(5), dat; progress_init=false, progress=false)

r = aic(res.model, res.llf)
println("AIC:    ", r.aic)
println("k:      ", r.k,       "  (effective parameters)")
println("n_alpha:", r.n_alpha, "  (free alpha entries)")
println("n_rate: ", r.n_rate,  "  (distinct rates)")

# Custom tolerances
r2 = aic(res.model, res.llf; alpha_tol=1e-6, rate_tol=1e-3)
```

**Comparing models of different sizes:**

```julia
results = [phfit(CF1(n), dat; progress_init=false, progress=false) for n in 3:7]
for res in results
    r = aic(res.model, res.llf)
    println("dim=$(res.model.dim)  AIC=$(round(r.aic, digits=2))  k=$(r.k)")
end
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
result = phfit(CF1(3), ltrc, progress = true, progress_init = true)

# Output results
println("Log-likelihood: ", result.llf)
println("Fitted parameters: ", result.model.alpha, result.model.rate)
```

This interface ensures proper treatment of truncation and censoring during parameter estimation using the EM algorithm.

## Distribution Types

### CF1 (Canonical Form 1)
Canonical Form 1 is a simplified representation where each phase has a single exit rate. It's characterized by:
- Initial probability vector `alpha`
- Vector of exit rates `rate`

```julia
cf1 = CF1(alpha, rate)
```

### GPH (General Phase-Type)
The most general form, characterized by:
- Initial probability vector `alpha`
- Infinitesimal generator matrix `T`
- Exit rate vector `tau`

```julia
gph = GPH(alpha, T, tau)
```

## Available Functions

### Distribution Functions
- `phpdf(ph, t)` - Probability density function
- `phcdf(ph, t)` - Cumulative distribution function
- `phccdf(ph, t)` - Complementary CDF (survival function)
- `phmean(ph, n=1)` - n-th moment of the distribution
- `phsample(ph, m)` - Generate m random samples

### Parameter Estimation
- `phfit(cf1, data, ...)` - Fit PH distribution to data using EM algorithm
- Various data types supported via specialized sample structures:
  - `WeightedSample` - Density-based or point observations with weights
  - `TimeSpanSample` - Mixed point and interval data
  - `LeftTruncRightCensoredSample` - Survival analysis data
  - `GroupSample` - Grouped observations

### Model Selection
- `phllf(cf1, data)` - Log-likelihood for a fitted model
- `aic(cf1, llf; alpha_tol, rate_tol)` - AIC with effective parameter count for CF1
- `eic(ph0, data; bsample, ...)` - Extended Information Criterion via bootstrap (returns `eic`, `ci_lower`, `ci_upper`, `nvalid`)
- `bootstrap(data)` - Multinomial resampling of a `TimeSpanSample`

## Advanced Features

### Custom Matrix Types
The package supports various matrix representations for efficiency:
- Dense matrices (`Matrix`)
- Sparse matrices (`SparseMatrixCSC` from SparseArrays.jl)
- Specialized sparse formats from NMarkov.jl (`SparseCSR`, `SparseCSC`, `SparseCOO`)

### Numerical Integration
Uses the DEQuadrature package for efficient numerical integration when computing distribution functions.

## Dependencies

- [DEQuadrature.jl](https://github.com/JuliaReliab/DEQuadrature.jl) - Double exponential quadrature
- [NMarkov.jl](https://github.com/JuliaReliab/NMarkov.jl) - Markov chain utilities and sparse matrix operations
- Distributions.jl - Probability distributions
- ProgressMeter.jl - Progress tracking for iterative algorithms
- SpecialFunctions.jl - Mathematical special functions
- ZeroOrigin.jl - Zero-based array indexing support

## Performance Considerations

- Use sparse matrix representations for large-scale problems
- The EM algorithm convergence depends on initialization; the package provides automatic initialization heuristics
- For very large datasets, consider using grouped or sampled data representations
- Progress monitoring can be disabled for production use: `progress=false`

## References

For theoretical background on phase-type distributions and the EM algorithm:

1. Neuts, M. F. (1981). *Matrix-Geometric Solutions in Stochastic Models*. Johns Hopkins University Press.
2. Asmussen, S., Nerman, O., & Olsson, M. (1996). Fitting phase-type distributions via the EM algorithm. *Scandinavian Journal of Statistics*, 23(4), 419-441.
3. Okamura, H., Dohi, T., & Trivedi, K. S. (2009). A refined EM algorithm for PH distributions. *Performance Evaluation*, 68(10), 938-954.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This package is licensed under the MIT License. See the LICENSE file for details.

## Author

Hiroyuki Okamura

## See Also

- [NMarkov.jl](https://github.com/JuliaReliab/NMarkov.jl) - Markov chain analysis
- [DEQuadrature.jl](https://github.com/JuliaReliab/DEQuadrature.jl) - Numerical integration
