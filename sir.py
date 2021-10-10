# This is the file containing the Markov-chain simulation

from sirClass import SIR


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




#problem1c()
problem1e()