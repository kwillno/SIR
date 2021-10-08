import numpy as np
from matplotlib import pyplot as plt

class SIR: 
	def __init__(self, population, alpha, beta, gamma):
		self.daysInYear = 365

		self.population = population

		self.P = np.array([
			[1-beta, beta, 0],
			[0, 1-gamma, gamma],
			[alpha, 0, 1-alpha]
		])

		self.X_n = np.array([])

		self.z_alpha = 1.96

	def updatePopulation(self, population):
		self.population = population

	def updateParams(self, alpha, beta, gamma):

		self.P = np.array([
			[1-beta, beta, 0],
			[0, 1-gamma, gamma],
			[alpha, 0, 1-alpha]
		])

	def simulate(self, years):
		# Find total amount of needed iterations
		self.totalDays = int(years*self.daysInYear)

		self.X_n = np.zeros((self.totalDays,self.population))

		for i in range(self.totalDays):
			# Double for loop solution
			for j in range(self.population):

				if np.random.random() > self.P[int(self.X_n[i-1,j]),int(self.X_n[i-1,j])]:
					self.X_n[i,j] = self.X_n[i-1,j] + 1
				else:
					self.X_n[i,j] = self.X_n[i-1,j]
				
				if self.X_n[i,j] == 3:
					self.X_n[i,j] = 0
				

		for i in range(self.totalDays):
			self.X_n[i].sort()

	def simulateDependence(self, years):

		self.totalDays = int(years*self.daysInYear)

		self.X_n = np.zeros((self.totalDays,self.population))

	def plot(self):

		plt.figure(0)
		plt.imshow(self.X_n.T)
		plt.show()

	def countStateDays(self, v=True):
		stateFirst = np.sum(self.X_n[int(self.totalDays/2):,0] == 0)
		stateSecond = np.sum(self.X_n[int(self.totalDays/2):,0] == 1)
		stateThird = np.sum(self.X_n[int(self.totalDays/2):,0] == 2)

		if v:
			print(f"Absolute numbers of days in different states: ")
			print(f"S: {stateFirst:8}, I: {stateSecond:8}, R: {stateThird:8}.")

			print(f"Relative numbers of days in different states: ")
			print(f"S: {2*stateFirst/self.totalDays:8.2f}, I: {2*stateSecond/self.totalDays:8.2f}, R: {2*stateThird/self.totalDays:8.2f}.")

		return stateFirst,stateSecond,stateThird

	def numericalLimitingDistributions(self, n=30, years=20, v=False):

		results = np.zeros((n,3))

		for i in range(n):
			self.simulate(years)
			results[i] = self.countStateDays(v=False)

		# Calculating statistical variables
		means = np.zeros(len(results[0]))

		for i in range(len(means)):
			means[i] = sum(results[:,i])/n

		stds = np.zeros(len(means))

		for i in range(len(means)):
			sm = 0
			for j in range(n):
				sm += (results[j,i] - means[i])**2

			stds[i] = np.sqrt(sm/(n-1))

		# Calculate error estimates: 
		CIs = np.zeros(len(means))

		for i in range(len(CIs)):
			CIs[i] = self.z_alpha*np.sqrt(stds[i]**2/n)

		if v:
			print(f"CIs: ")
			for i in range(len(means)):
				print(f"State: {i}, CI: {means[i]:.2f}±{CIs[i]:.2f}")

		self.means = means
		self.CI = CIs
