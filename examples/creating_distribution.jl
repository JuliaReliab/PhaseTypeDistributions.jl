using PhaseTypeDistributions

# Define initial probabilities and rates
alpha = [0.1, 0.3, 0.6]
rate = [1.4, 0.4, 10.0]

# Create a CF1 phase-type distribution
cf1 = CF1(alpha, rate)
println(cf1)