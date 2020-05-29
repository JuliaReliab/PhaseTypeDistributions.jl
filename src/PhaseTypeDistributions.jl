module PhaseTypeDistributions

using SparseMatrix: SparseCSR, SparseCSC, SparseCOO, spdiag, nnz, BlockCOO
using NMarkov: rightbound, poipmf!, poipmf, unif, unifstep!, Trans, NoTrans, @scal, @axpy, @dot, convunifstep!, itime
using Origin: @origin
using Deformula: deint

include("_ph.jl")
include("_phunif.jl")

include("_dist.jl")

include("_data.jl")
include("_emstep.jl")
include("_cf1init.jl")
include("_phfit.jl")

end # module
