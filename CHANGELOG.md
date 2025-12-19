# Changelog

## [0.5.1]
- Distributions.jl integration: Refined CF1/GPH implementations of `pdf`/`cdf`/`ccdf`/`mean`/`rand`; unified `t::Real` signatures to resolve method ambiguities; added interface tests.
- Performance (minimal changes): Preallocated and reused working vectors in E-step of `phfit_timespan.jl`, `phfit_group.jl`, and `phfit_leftright.jl` using `fill!` and `.=`; improved type stability with `ones(Tv, dim)`. Algorithms and public API unchanged.

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
