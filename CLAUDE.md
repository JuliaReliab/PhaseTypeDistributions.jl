# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install unregistered dependencies (required before first use):**
```julia
julia --project=. -e '
  using Pkg
  Pkg.add([
    PackageSpec(url="https://github.com/JuliaReliab/DEQuadrature.jl.git"),
    PackageSpec(url="https://github.com/JuliaReliab/NMarkov.jl.git")
  ])
'
```

**Run all tests:**
```
julia --project=. -e 'using Pkg; Pkg.test()'
```

**Run a single test file:**
```
julia --project=. test/test_ph.jl
```

**Start a Julia REPL with the package loaded:**
```
julia --project=. -e 'using PhaseTypeDistributions'
```

## Architecture

This package implements **Phase-Type (PH) distributions** — distributions of absorption times in continuous-time Markov chains. They are used in stochastic modeling, queueing theory, reliability, and survival analysis.

### Core types (`src/ph.jl`)

- `GPH{Tv,MatT}` — General PH distribution. Parameters: `alpha` (initial vector), `T` (sub-generator matrix), `tau` (exit rates). Supports dense and sparse matrix types (`SparseCSR`, `SparseCSC`, `SparseCOO` from NMarkov.jl).
- `CF1{Tv}` — Canonical Form 1: a sparse, cycle-free PH representation stored as `alpha` vector and `rate` vector. More efficient for fitting.
- Both types integrate with `Distributions.jl`, so `pdf()`, `cdf()`, `mean()`, `rand()` all work natively.

### Distribution computations (`src/dist.jl`)

Implements `phpdf`, `phcdf`, `phccdf`, `phmean`, `phsample` using the **uniformization method** (also called Jensen's method) for numerical stability. These back the `Distributions.jl` interface.

### Fitting module (`src/phfit/`)

The `Phfit` submodule fits PH parameters to data via the **EM algorithm**:

- `phfit(cf1, data)` / `phfit!(cf1, data)` — main entry point
- `phfit_common.jl` — EM algorithm core (E-step/M-step), plus `aic`
- Data type dispatch: `phfit_density.jl` (weighted density), `phfit_group.jl` (grouped/histogram), `phfit_leftright.jl` (left-truncated/right-censored), `phfit_timespan.jl` (interval-censored + bootstrap/EIC)
- Initialization strategies: `cf1mom.jl` (moment matching via `cf1mom_power`, `cf1mom_linear`) and `ph3mom_bobbio05.jl` (3-moment method)
- Log-likelihood in `phllf.jl`

### Model selection (`src/phfit/phfit_common.jl`, `src/phfit/phfit_timespan.jl`)

- `aic(cf1, llf; alpha_tol, rate_tol)` — AIC with effective parameter count for CF1 (active `alpha` entries and distinct `rate` values)
- `eic(ph0, data; bsample, ...)` — Extended Information Criterion via bootstrap (multinomial resampling); returns `(eic, ci_lower, ci_upper, nvalid)`
- `bootstrap(data::TimeSpanSample)` — resample counts `ndat` from `rawn` using a Multinomial draw, preserving `raww` weights
- `phllf(cf1, data)` — log-likelihood for `TimeSpanSample`

### Sample types

Data is passed to `phfit` as typed sample objects:
- `WeightedSample` — density/point observations with weights
- `PointSample` — i.i.d. point observations
- `TimeSpanSample` — mixed exact/interval observations; holds separate `ndat` (frequency, bootstrap target) and `wdat` (analytic weight, fixed), plus `rawt`/`rawn`/`raww` for bootstrap reconstruction
- `LeftTruncRightCensoredSample` — survival analysis (left truncation, right censoring)
- `GroupTruncSample`, `GroupTruncPoiSample` — grouped/histogram data

### Key external dependencies

- `NMarkov.jl` (unregistered) — sparse matrix formats and Markov chain utilities used throughout
- `DEQuadrature.jl` (unregistered) — double-exponential quadrature for numerical integration in fitting
- `Distributions.jl` — abstract type hierarchy that `GPH`/`CF1` extend
