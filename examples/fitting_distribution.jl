using PhaseTypeDistributions
using PhaseTypeDistributions.Phfit
using Distributions

# Generate sample data from a Weibull distribution
data = WeightedSample((0.0, Inf64)) do x
    pdf(Weibull(2.0, 1.0), x)
end

# Fit a CF1 phase-type distribution to the data
result = phfit(CF1(10), data, verbose = [true, true])
println("Log-likelihood: ", result.llf)
println("Convergence: ", result.conv)
println("Iterations: ", result.iter)
println("Relative error: ", result.rerror)
println("Data: ", result.data)
println("Fitted parameters: ", result.model.alpha, result.model.rate)
