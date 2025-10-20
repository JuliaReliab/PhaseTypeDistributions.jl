module PhaseTypeDistributions

include("ph.jl")
include("dist.jl")

include("phfit/Phfit.jl")

export
    phpdf,
    phcdf,
    phccdf,
    phcomp,
    phmean,
    CF1,
    GPH,
    BidiagonalPH,
    cf1sort,
    cf1sort!,
    phunif

end
