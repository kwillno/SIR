# This is the file containing the Markov-chain simulation
import numpy as np

daysInYear = 365
individuals = 1000

alpha = 0.005
beta  = 0.01
gamma = 0.1

permutationMatrix = np.array([
		[1-beta, beta, 0],
		[0, 1-gamma, gamma],
		[alpha, 0, 1-alpha]
	])

X_n = np.zeros((individuals,individuals))

def simulate(X_n, P, years):
	
	# Find the total amount of iterations that need to be completed
	totalIterations = years*daysInYear

	for i in range(totalIterations):
		for j in range(len(X_n)):
			pass


print(X_n)

