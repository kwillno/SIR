import numpy as np
from matplotlib import pyplot as plt

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
		self.totalDays = int(years*self.daysInYear)

		self.X_n = np.zeros((self.totalDays,self.population))

		for i in range(self.totalDays):
			# Double for loop solution
			for j in range(self.population):

				if np.random.random() > self.P[int(self.X_n[i,j]),int(self.X_n[i,j])]:
					self.X_n[i,j] = self.X_n[i-1,j] + 1
				else:
					self.X_n[i,j] = self.X_n[i-1,j]
				
				if self.X_n[i,j] == 3:
					self.X_n[i,j] = 0
				

		for i in range(self.totalDays):
			self.X_n[i].sort()

	def plot(self):
		"""
		Prøv å bruke imshow() på matrisa!!!
		"""

		plt.figure(0)
		plt.imshow(self.X_n.T)
		plt.show()

	def countStateDays(self):
		stateFirst = np.sum(self.X_n[int(self.totalDays/2):,0] == 0)
		stateSecond = np.sum(self.X_n[int(self.totalDays/2):,0] == 1)
		stateThird = np.sum(self.X_n[int(self.totalDays/2):,0] == 2)

		print(f"Absolute numbers of days in different states: ")
		print(f"S: {stateFirst:8}, I: {stateSecond:8}, R: {stateThird:8}.")

		print(f"Relative numbers of days in different states: ")
		print(f"S: {2*stateFirst/self.totalDays:8.2f}, I: {2*stateSecond/self.totalDays:8.2f}, R: {2*stateThird/self.totalDays:8.2f}.")
		
	
	def limiting_distribution(self):
		
		# lager en null-matrise med samme dimensjoner som self
		limProbMatrix = np.zeros(self.shape())
		
		# Itererer gjennom matrisen elementvis
		for i in range(len(self)):
			for j in range(len(self)):
				
				# limeting probability er definert slik at rad-elementene i self kan brukes som kolonnelementer i limProbMatrix
				limProbMatrix[i,j] = self[j,i]
				
				# Vi må gjøre litt om på limProbMatrix for at ting skal funke optimalt. Vi trekker fra 1 i [i,i] elementet i limProbMatrix
				# for da kan vi definere løsningsmatrisen som [0,0,0] hvilket er digg
				if i==j:
					limProbMatrix[i,j] = self[i,j] - 1 
				
				# Det vi egt har gjort her er å gjøre noen lineære ligninger om til en matrise, men nå har vi lyst til å bytte ut den siste 
				# "ligningen", eller raden til limProbMatrix, med x + y + z = 1. Dette gjør vi av grunner som jeg ikke skjønner, men 
				# foreleseren mener det er bedre
				if j==(len(self)-1):
					self[j,i] = 1 
		
		# løser ligningssystemet
		solvedLimProb = np.solve(self, [0,0,1])
		
		# returnerer en array med løsningene... tror jeg
		return solvedLimProb
