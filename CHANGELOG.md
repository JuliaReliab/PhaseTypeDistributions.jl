# Changelog

## [0.6.0]

### Breaking changes
- `TimeSpanSample` struct extended: `wdat` field (formerly the sole weight) is now `ndat` (frequency/count, `Vector{Tv}`) and a new `wdat` (analytic weight, `Vector{Tv}`) field has been added. The two-argument constructor `TimeSpanSample(t, n)` is backward compatible (analytic weight defaults to 1).
- `TimeSpanSample` now stores raw observation data (`rawt::Vector`, `rawn::Vector{Int}`, `raww::Vector{Tv}`) for bootstrap support.

### New features
- `TimeSpanSample(t, n, w)`: three-argument constructor separating frequency count `n_i` from analytic weight `w_i`. Effective EM weight is `n_i * w_i`.
- `bootstrap(rng, data::TimeSpanSample)`: non-parametric bootstrap via multinomial resampling of `rawn`. Bootstrap target is `ndat` only; `wdat` is held fixed.
- `phllf(cf1, data::TimeSpanSample)`: log-likelihood evaluation for `TimeSpanSample` without running the full EM algorithm.
- `eic(rng, ph0, llf0, d0, data; bsample, ...)`: Extended Information Criterion with 95% CI, computed via parallelised bootstrap EM. Returns `(eic, ci_lower, ci_upper, nvalid)`.
- `aic(cf1, llf; alpha_tol, rate_tol)`: AIC with effective parameter count for CF1. Alpha entries ≤ `alpha_tol` are treated as zero; rate entries within `rate_tol` are treated as a single parameter. Returns `(aic, k, n_alpha, n_rate)`.

## [0.5.1]
- Distributions.jl integration: Refined CF1/GPH implementations of `pdf`/`cdf`/`ccdf`/`mean`/`rand`; unified `t::Real` signatures to resolve method ambiguities; added interface tests.
- Performance (minimal changes): Preallocated and reused working vectors in E-step of `phfit_timespan.jl`, `phfit_group.jl`, and `phfit_leftright.jl` using `fill!` and `.=`; improved type stability with `ones(Tv, dim)`. Algorithms and public API unchanged.
- Bug fix: Fixed `getbaralpha` function import in `Phfit` module; improved sparse matrix conversion handling for `SparseCSR` and other sparse formats.

## [0.5.0]
- Migrated from `Deformula` to `DEQuadrature`
- Migrated from `SparseMatrix` to `NMarkov.SparseMatrix`
- GitHub Actions CI/CD pipeline
- CompatHelper and TagBot workflows
- Enhanced README with comprehensive documentation

## [0.4.3]
- Use ZeroOrigin instead of Origin

## [0.4.2]
- GPH includes baralpha

## [0.4.1]
- Add sampling for CF1 and GPH distributions (`phsample`)

## [0.4.0]
- Add absolute error tolerance (abstol) to phfit!
- Change return values and interface of phfit

## [0.3.2]
- Skip truncation time 0 in LTRC data

## [0.3.1]
- Fix TimeSpanSample

## [0.3.0]
- Add `TimeSpanSample` for mixed point and interval data
- Add LTRC data support

## [0.2.8]
- Add examples
- Change phfit result from tuple to named tuple

## [0.2.7]
- Fix: use NMarkov 0.3.5

## [0.2.6]
- Fix: diag for SparseMatrixCSC

## [0.2.5]
- Add left truncation right censored data support

## [0.2.4]
- Fix: phcdf

## [0.2.3]
- Add phfit for group data

## [0.2.2]
- Restructure files
- Use BLAS-like routines
