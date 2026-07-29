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
- `TimeSpanSample` — mixed exact/interval observations; holds separate `ndat` (frequency, bootstrap target) and `wdat` (analytic weight, fixed), plus `rawt`/`rawn`/`raww` for bootstrap reconstruction. Constructors also accept `WeightedSample` or `GroupTruncSample` directly for conversion.
- `LeftTruncRightCensoredSample` — survival analysis (left truncation, right censoring)
- `GroupTruncSample`, `GroupTruncPoiSample` — grouped/histogram data

### `TimeSpanSample` constructors (`src/phfit/phfit_timespan.jl`)

All accept a `tau` keyword argument for left truncation (`tau[i] > 0` = truncation time of observation `i`, `tau[i] == 0` = none; stored in `rawtau`).

- `TimeSpanSample(t; tau)` — point observations
- `TimeSpanSample(t, n; tau)` — with integer counts
- `TimeSpanSample(t, n, w; tau)` — with counts and analytic weights; `t[i]` can be scalar (exact) or `Tuple{Tv,Tv}` (interval)
- `TimeSpanSample(data::WeightedSample)` — converts quadrature points; each point gets `ndat=1`, `wdat=wdat`
- `TimeSpanSample(data::GroupTruncSample)` — converts grouped data; interval obs → tuple entries, exact obs → scalar entries, last interval → `(t_m, Inf)`

### `zdat` encoding in `TimeSpanSample`

Internal field used by the E-step to identify observation type at each sorted time point:
- `-1` — left truncation at `t_k`
- `0` — interval `[0, t_k]`
- `k` — exact observation at `t_k`
- `j` (0 < j < k) — interval `[t_j, t_k]`, lower bound at index j
- `m+1` — right-censored `[t_k, ∞)`

`createTimeSpanSample` pairs the two endpoints of a finite interval by an explicit entry
kind (`:span`), not by a reused sentinel index — reusing the index silently merged two
adjacent same-kind observations into one interval (fixed in v0.7.0).

### Left truncation in the `TimeSpanSample` E-step

The contribution of `-log S(τ)` equals `wb * [unconditional expectation] - [right-censored-at-τ expectation]`
with `wb = n_i w_i / S(τ)`. In the `timespan` decomposition this collapses to exactly the
same vector structure as the `zdat == 0` (interval `[0,t]`) branch — `eb += wb*(1 - barvb)`,
`ey += wb*(ᾱ - barvf)`, `en += wb*outer(ᾱ, 1 - barvb)` — differing only in that `wb` divides
by `S(τ)` instead of `F(τ)`, and `llf` is subtracted. The `vc` backward recursion needs no
new branch: `zdat[k] < k` already contributes `-wb*ᾱ`, which is correct for `-1`. Verified
by exact numerical agreement with `LeftTruncRightCensoredSample`.

### Key external dependencies

- `NMarkov.jl` (unregistered) — sparse matrix formats and Markov chain utilities used throughout
- `DEQuadrature.jl` (unregistered) — double-exponential quadrature for numerical integration in fitting
- `Distributions.jl` — abstract type hierarchy that `GPH`/`CF1` extend

## Session log

### 2026-07-29

**Done:**
- Left truncation support for `TimeSpanSample` (`tau` keyword on all constructors, `rawtau` field, `zdat == -1`, new E-step branch, `mean`/`bootstrap`/`eic` updated). The E-step math that had blocked the previous session is resolved and documented above; agreement with `LeftTruncRightCensoredSample` is exact.
- Fixed a pre-existing bug where two adjacent same-kind observations were merged into one interval (see the `createTimeSpanSample` note above)
- Bumped to v0.7.0

### 2026-04-23 / 2026-04-24

**Done:**
- Added `TimeSpanSample(data::WeightedSample)` constructor — converts quadrature sample to TimeSpanSample (each point: ndat=1, wdat=wdat)
- Added `TimeSpanSample(data::GroupTruncSample)` constructor — converts group data; skips unobserved intervals (gdat==-1)
- Bumped version to 0.6.3, updated CHANGELOG.md, committed and pushed (e5beff1)

**Pending:**
- Left truncation for `TimeSpanSample` — **completed 2026-07-29**, see the entry above.
