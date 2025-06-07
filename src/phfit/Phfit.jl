module Phfit

abstract type AbstractPHSample end

include("ph3mom_bobbio05.jl")
include("cf1mom.jl")

include("phfit_common.jl")

include("phfit_density.jl")
include("phllf.jl")

include("phfit_group.jl")
include("phfit_leftright.jl")
include("phfit_timespan.jl")

export
    cf1mom_power,
    cf1mom_linear,
    ph3mom_bobbio05,
    phfit,
    phfit!,
    mean,
    WeightedSample,
    PointSample,
    GroupTruncSample,
    GroupTruncPoiSample,
    LeftTruncRightCensoredSample,
    TimeSpanSample,
    phllf

end