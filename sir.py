# This is the file containing the Markov-chain simulation

from sirClass import SIR
import numpy as np
import matplotlib.pyplot as plt

def problem1c():
	# Simulate for 20 years using given parameters

	# Set up the simulation
	sir = SIR(population=1, alpha=0.005, beta=0.01, gamma=0.1, years=20)

	# Actually run the simulation
	sir.simulate()

	# Give output
	sir.countStateDays(v=True)
	sir.numericalLimitingDistributions(v=True)



def problem1e():
	# Simulate Y_n for n=300 timesteps

	sir = SIR(population=1000)

	# Set inital state
	sir.setInitialState(S=950, I=50, R=0)

	# Set timestep limit to n=300
	sir.totalDays = 300
	sir.X_n = sir.X_n[:300]

	# Run simulation
	sir.simulateWithDependence()

	# Plot the results of the simulation
	sir.graphSIR()


def problem1f(n=300,sims=1000):
	# Find maximum and argmax of infected
	# individuals using 1000 simualtions with n=300

	sir = SIR(population=1000)

	# Set timestep limit to n=300
	# Assume that spike occurs before n=100
	sir.totalDays = n
	sir.X_n = sir.X_n[:n]

	sir.findMaxInfectedCIs(simulations=sims, v=True)

def problem1g():
	# Simulate Y_n for n=300 timesteps

	sir = SIR(population=1000)

	# The different situations we are interested in.
	vaccinationRate = [0,100,600,800]

	# Ploting all scenarios in same figure.
	fig, axs = plt.subplots(2, 2)
	for i in range(len(vaccinationRate)):
		# Set inital state
		sir.setInitialState(I=50, R=0, V=vaccinationRate[i])

		# Set timestep limit to n=300
		sir.totalDays = 300
		sir.X_n = sir.X_n[:300]

		sir.simulateWithDependence()

		maxI, argmaxI = sir.findMaxInfected()

		print(f"Vaccinated: {vaccinationRate[i]},  Max Infected: {maxI},\t Time of max infected I: {argmaxI}")

		# Plotting in one figure
		X_i = sir.X_n

		SIRV = np.zeros((4,len(X_i)))
		SIRVlabel = ["Susceptible", "Infected", "Recovered", "Vaccinated"]
		coor = [(0,0),(0,1),(1,0),(1,1)]

		axis = np.linspace(0,len(X_i),len(X_i))

		for j in range(len(SIRV)):
			for k in range(len(X_i)):
				SIRV[j,k] = np.count_nonzero(X_i[k] == j)


		subplt = axs[coor[i]]
		subplt.set_title(f"Vaccinated: {vaccinationRate[i]}")
		for j in range(len(SIRV)):
			subplt.plot(axis, SIRV[j],label=f"{SIRVlabel[j]}")
		subplt.set_ylim([0,sir.population])
		subplt.legend()

	# Find 95% confidence intervals for peak infected and when this occurs
	# using 1000 simualtions. 
	for i in range(1,len(vaccinationRate)):
		# Set inital state
		sir.setInitialState(I=50, R=0, V=vaccinationRate[i])

		# Set timestep limit to n=300
		sir.totalDays = 300
		sir.X_n = sir.X_n[:300]

		print(f"\n\nVaccinationrate: {vaccinationRate[i]}")
		sir.findMaxInfectedCIs(simulations=1000, states=[50,0,vaccinationRate[i]])

	plt.show()


problem1c()
problem1e()
problem1f(sims = 1000)
problem1g()