# TODO

## Enhancement

- [ ] Add left truncation support to `TimeSpanSample` and consolidate with `LeftTruncRightCensoredSample`
- [ ] Consolidate density and group data types into `TimeSpanSample`

## Bug fixes / Robustness

### High priority
- [ ] `phfit_common.jl` — `mstep!` (all 5 variants) performs unguarded division by `eres.ez[i]` when it is zero. `CF1.mstep!` already handles this; `GPH.mstep!` does not
- [ ] `phfit_timespan.jl:397` — `eic` divides by zero when all bootstrap samples fail (`n == 0`). Add an early return
- [ ] `phfit_group.jl` — `tmp = @dot(alpha, ...)` can be zero, causing division by zero at `wg[k] = data.gdat[k] / tmp` and similar sites

### Medium priority
- [ ] `ph.jl` — `cf1swap!` can overflow when `rate[i]` is very small (`w = rate[j] / rate[i]`). Add a guard
- [ ] `phfit_common.jl:97-98` — `rerror = abs((llf - prevllf) / prevllf)` is numerically unstable in early iterations when `prevllf` is a large negative value
- [ ] `phfit_group.jl` — `barvf[k] = zeros(Tv, dim)` inside a loop allocates on every iteration. Unify with the `fill!` pattern used elsewhere

### Low priority
- [ ] `phfit_timespan.jl` / `phfit_density.jl` — custom `mean` functions do not extend `Statistics.mean`
- [ ] `phfit_timespan.jl:95` — dead code `ord = collect(1:m)` is immediately overwritten by `sortperm`; remove it

## Documentation
- [ ] Add docstrings to public API: `GPH` and `CF1` constructors, `phfit`, `phfit!`, `eic`, `bootstrap`

## Tests

- [ ] `test/test_phfit_leftright.jl` — `GroupTruncPoiSample` testsets are commented out (decide whether to enable or remove)
- [ ] `test/test_phfit_group.jl` — `SparseCSR` testset is commented out (decide whether to enable or remove)
