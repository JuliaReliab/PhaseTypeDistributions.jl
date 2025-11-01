# 0.4.3

- use ZeroOrigin instead of Origin

# 0.4.2

- GPH includes baralpha

# 0.4.1

- enhancement: Sampling for CF1 and GPH distributions `phsample`

# 0.4.0

- enhancement: Add absolute error tolerance (abstol) to phfit!
- Change the order of return values in phfit!
- Change the interface to call phfit (remove verbose and verbose_init)

# 0.3.2

- enhancement: Skip the truncation time 0 in left-truncated right-censored data

# 0.3.1

- bugfix: TimeSpanSample
- change type of createTimeSpanSample

# 0.3.0

- add `TimeSpanSample` for mixed point and interval data
- add LTRC data support

# 0.2.8

- add examples
- change the result of phfit from tuple to named tuple

# 0.2.7

- bugfix: use NMarkov 0.3.5

# 0.2.6

- bugfix: diag for SparseMatrixCSC

# 0.2.5

- Implementation for Left truncation right censored data

# 0.2.4

- bugfix: phcdf

# 0.2.3

- add phfit for group data
- add phfit for grouptrunc and trunc poi

# 0.2.2

- change the file structure
- use blas-like routines

