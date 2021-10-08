# This is the file containing the Markov-chain simulation

from sirClass import SIR

"""
sir = SIR(1000,0.005,0.01,0.1)

sir.simulate(20)

sir.plot()
"""

lonelySir = SIR(1,0.005,0.01,0.1)

lonelySir.simulate(20)

lonelySir.countStateDays()