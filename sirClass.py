import numpy as np
from scipy import stats as st
from matplotlib import pyplot as plt

class SIR: 
	def __init__(self, population, alpha=0.005, beta=0.01, gamma=0.1, years=20):

		self.population = population

		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma

		self.P = np.array([
			[1-beta, beta, 0],
			[0, 1-gamma, gamma],
			[alpha, 0, 1-alpha]
		])

		self.years = years
		self.totalDays = int(years*365)
		self.X_n = np.zeros((self.totalDays,self.population))
		
		self.z_alpha = 1.697

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

	def setInitialState(self,S=1000,I=0,R=0):

		if S+I+R != self.population:
			S = self.population - (I+R)

		susceptible = np.zeros(S)
		infected = np.ones(I)
		recovered = np.ones(R)*2

		self.X_n[0] = np.concatenate((susceptible,infected,recovered))

	def simulate(self):

		for i in range(1,self.totalDays):
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

	def simulateWithDependence(self):

		for i in range(1,self.totalDays):
			# Update parameters to get chance of infection dependant on amount in state 1

			self.updateParams(self.alpha, (0.5*np.sum(self.X_n[i-1] == 1))/self.population ,self.gamma)

			for j in range(self.population):

				if np.random.random() > self.P[int(self.X_n[i-1,j]),int(self.X_n[i-1,j])]:
					self.X_n[i,j] = self.X_n[i-1,j] + 1
				else:
					self.X_n[i,j] = self.X_n[i-1,j]
				
				if self.X_n[i,j] == 3:
					self.X_n[i,j] = 0


		for i in range(self.totalDays):
			self.X_n[i].sort()


	def plot(self):

		plt.figure(0)
		plt.imshow(self.X_n.T)
		plt.show()

	def graphSIR(self):

		S = np.zeros(len(self.X_n))
		I = S.copy()

		R = S.copy()

		axis = np.linspace(0,len(S),len(S))

		for i in range(len(S)):
			S[i] = np.count_nonzero(self.X_n[i] == 0)
			I[i] = np.count_nonzero(self.X_n[i] == 1)
			R[i] = np.count_nonzero(self.X_n[i] == 2)

		plt.figure("SIR")
		plt.title("SIR-plot")
		plt.plot(axis, S, label="Susceptible")
		plt.plot(axis, I, label="Infected")
		plt.plot(axis, R, label="Recovered")
		plt.plot(axis, S+I+R, label="Sum")
		plt.legend()
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

	def numericalLimitingDistributions(self, n=30, v=False):

		results = np.zeros((n,3))

		for i in range(n):
			self.simulate()
			results[i] = self.countStateDays(v=False)
			
		# Calculate error estimates: 
		CIs = np.zeros((3,2))

		for i in range(len(CIs)):
			CIs[i,0], CIs[i,1] = st.t.interval(0.95, n-1, loc=np.mean(results[:,i]), scale=st.sem(results[:,i]))


		if v:
			print(f"CIs: ")
			for i in range(len(CIs)):
				print(f"State: {i}, Lower/Upper: {2*CIs[i,0]/self.years:.2f}, {2*CIs[i,1]/self.years:.2f}, size: {np.abs(2*CIs[i,0]/self.years - 2*CIs[i,1]/self.years):.2f}")

		self.CI = CIs
