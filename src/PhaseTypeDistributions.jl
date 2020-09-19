module PhaseTypeDistributions

using Origin: @origin
using SparseArrays: SparseMatrixCSC, nnz, sparse
using SparseMatrix: SparseCSR, SparseCSC, SparseCOO, spdiag, BlockCOO
using NMarkov: rightbound, poipmf!, poipmf, unif, unifstep!, @scal, @axpy, @dot, convunifstep!, itime
using Deformula: deint

export CF1, GPH
export phllf
export phfit!, phfit
export estep!, mstep!, Estep
export phcdf, phpdf, phccdf
export WeightedSample, PointSample
export initializePH

include("_ph.jl")
include("_phunif.jl")

include("_dist.jl")

include("_data.jl")
include("_emstep.jl")
include("_cf1init.jl")
include("_phfit.jl")

include("_phllf.jl")

end # module
