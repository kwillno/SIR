# This is the file containing the Markov-chain simulation

from sirClass import SIR


def problem1c():
	# Simulate for 20 years using given parameters

	sir = SIR(population=1, alpha=0.005, beta=0.01, gamma=0.1, years=20)

	sir.numericalLimitingDistributions(v=True)


problem1c()