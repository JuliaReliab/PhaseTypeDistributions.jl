using PhaseTypeDistributions

# Define a GPH distribution
alpha = [0.1, 0.3, 0.6]
T = [-1.4 1.4 0; 0 -0.4 0.4; 0 0 -10.0]
tau = [0, 0, 10.0]
gph = GPH(alpha, T, tau)

# Compute PDF, CDF, and CCDF at specific times
ts = LinRange(0.0, 10.0, 100)
pdf_values = phpdf(gph, ts)
cdf_values = phcdf(gph, ts)
ccdf_values = phccdf(gph, ts)
println(pdf_values)
println(cdf_values)
println(ccdf_values)