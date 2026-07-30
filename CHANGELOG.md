# Changelog

## [0.8.0]

### Breaking changes
- Dependencies are now resolved from the **JuliaReliab registry** instead of git `master`. The `[sources]` section has been removed from `Project.toml`; add the registry once with `Pkg.Registry.add(RegistrySpec(url="https://github.com/JuliaReliab/Registry.git"))`.
- Minimum dependency versions: `NMarkov = "0.5"`, `DEQuadrature = "0.3"`.
- Minimum Julia version raised from 1.6 to **1.10**, as required by NMarkov 0.5.

### Bug fixes
- Adapted to NMarkov 0.5, whose sparse types (`SparseCSR`/`SparseCSC`/`SparseCOO`) now obey the `AbstractMatrix` contract: `length(A)` is `m*n` and a linear index is cartesian, where previously `length(A)` was `nnz(A)` and `A[i]` was `A.val[i]`. The old-style loops in `clear!`, in the `en .*= T` tail of every `estep!`, and in the `SparseCSR`/`SparseCSC`/`SparseCOO` `mstep!`s now go through a `nzvalues(A)` helper that returns the stored-entry vector. Without this, fitting with those matrix types raised `ArgumentError: cannot write the element (i,j)`. As a side effect, the default `SparseMatrixCSC` path no longer walks all `m*n` positions in these loops.

### New features
- `WeightedSample(f, bounds; …)` accepts a `dropzero` keyword (default `eps(Tv)`). DEQuadrature 0.3 moved the quadrature-node drop threshold out of `abstol` into this separate option, whose own default is 0; leaving it at 0 for a density on `[0, Inf)` keeps far-tail nodes and inflates `maxtime` to ~1e13, which makes `estep!` request an unusable Poisson p.m.f. buffer. The default here reproduces the pre-0.3 node set exactly.

## [0.7.0]

### New features
- Left truncation support for `TimeSpanSample`. All constructors accept a `tau` keyword argument (`TimeSpanSample(t; tau)`, `TimeSpanSample(t, n; tau)`, `TimeSpanSample(t, n, w; tau)`), where `tau[i] > 0` is the left truncation time of observation `i` and `tau[i] == 0` means no truncation. The likelihood contribution of observation `i` is divided by `S(tau[i])`. `TimeSpanSample` now covers everything `LeftTruncRightCensoredSample` does, plus interval censoring, per-observation counts/weights, and bootstrap-based model selection.
- The truncation times are stored in a new `rawtau` field and are preserved by `bootstrap` and `eic`.

### Bug fixes
- `TimeSpanSample`: two adjacent observations of the same kind were mistaken for a single interval. `[(0.5, Inf), (0.8, Inf)]` was treated as the interval `[0.5, 0.8]` instead of two right-censored observations, and `[(0.0, 1.0), (0.0, 2.0)]` as `[1.0, 2.0]` instead of two `[0, b]` intervals. Both produced a silently wrong likelihood. Interval endpoints are now paired by an explicit entry kind rather than by a reused sentinel index.

### Breaking changes
- `TimeSpanSample` gained a tenth field (`rawtau`); code constructing the struct positionally must be updated. The documented constructors are unaffected.

## [0.6.3]

### New features
- `TimeSpanSample(data::WeightedSample)`: converts a `WeightedSample` to `TimeSpanSample`. Each quadrature point becomes a point observation with count 1 and the original quadrature weight as analytic weight `wdat`.
- `TimeSpanSample(data::GroupTruncSample)`: converts a `GroupTruncSample` to `TimeSpanSample`. Group interval observations `(t_{k-1}, t_k]` become interval-censored entries; exact observations (`idat[k] == true`) become point entries; the last interval `[t_m, ∞)` becomes a right-censored entry. Unobserved intervals (`gdat[k] == -1`) are skipped.

## [0.6.1]

### Bug fixes
- `eic`: fixed `ci_lower`/`ci_upper` swap — the correct relationship is `ci_lower ≤ eic ≤ ci_upper`.

### API simplification
- `eic(ph0, data; bsample, ...)` / `eic(rng, ph0, data; bsample, ...)`: `llf0` and `d0` are no longer separate arguments; they are now computed internally from `data`. The old 5-argument form has been removed.

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
