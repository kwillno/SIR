import numpy as np


class SIR: 
	def __init__(self, population, alpha, beta, gamma):
		self.daysInYear = 365

		self.population = population

		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma

		self.P = np.array([
			[1-beta, beta, 0],
			[0, 1-gamma, gamma],
			[alpha, 0, 1-alpha]
		])

		self.X_n = np.array([])

	def updatePopulation(self, population):
		self.population = population

	def updateParams(self, alpha, beta, gamma):
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma

		self.P = np.array([
			[1-beta, beta, 0],
			[0, 1-gamma, gamma],
			[alpha, 0, 1-alpha]
		])

	def params(self):
		return np.array([self.alpha, self.beta, self.gamma])

	def simulate(self, years):
		# Find total amount of needed iterations
		totalIter = int(years*self.daysInYear)

		self.X_n = np.zeros((totalIter,self.population))

		for i in range(totalIter):
			"""
			TODO: This needs to be fixed
			rands = np.random.rand(self.population)

			X_n[i] = np.where(rand > self.P[X_n[i],0], )
			"""

	def plot(self):
		"""
		Prøv å bruke imshow() på matrisa!!!
		"""
