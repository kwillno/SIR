# This is the file containing the Markov-chain simulation

from sirClass import SIR

sir = SIR(10,0.005,0.01,0.1)

print(sir.P)
sir.simulate(1/50)
