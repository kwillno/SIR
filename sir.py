# This is the file containing the Markov-chain simulation

from sirClass import SIR
import numpy as np

def problem1c():
	# Simulate for 20 years using given parameters

	sir = SIR(population=1, alpha=0.005, beta=0.01, gamma=0.1, years=20)

	sir.numericalLimitingDistributions(v=True)


def problem1e():
	# Simulate Y_n for n=300 timesteps

	sir = SIR(population=1000)

	# Set inital state
	sir.setInitialState(S=950, I=50, R=0)

	# Set timestep limit to n=300
	sir.totalDays = 300
	sir.X_n = sir.X_n[:300]

	sir.simulateWithDependence()

	sir.graphSIR()


def problem1f(n=300,sims=1000):
	# Find maximum and argmax of infected
	# individuals using 1000 simualtions with n=300

	sir = SIR(population=1000)

	# Set timestep limit to n=300
	# Assume that spike occurs before n=100
	sir.totalDays = n
	sir.X_n = sir.X_n[:n]

	sir.findMaxInfected(simulations=sims, v=True)



# problem1c()
# problem1e()
problem1f(sims = 10)